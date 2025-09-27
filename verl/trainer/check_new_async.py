"""
Sanity loader for ALFWorld envs: uses the same Hydra config as eval_ckpt.sh,
loads validation dataloader and ALFWorld envs, resets envs for each batch,
and prints the very first model input (prompt) derived from the initial observation.

Run with the same overrides as eval_ckpt.sh, for example:

python -m verl.trainer.check_new_traj_collectors \
  algorithm.adv_estimator=gigpo \
  data.train_files=$HOME/data/verl-agent/text/train.parquet \
  data.val_files=$HOME/data/verl-agent/text/test.parquet \
  data.train_batch_size=16 \
  data.val_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=ckpt/verlagent_alfworld \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.rollout.n=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1
"""

import os
import json
import random
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.main_ppo_for_verl_agent import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.rollout.async_server import AsyncLLMServerManager
from src.verlagent.environments import make_envs
from src.agent_loop.rollout_sync_env import TrajectoryCollectorUsingAsyncLLMServer
from verl.utils import hf_tokenizer
import numpy as np
from tqdm import tqdm


import ray
from omegaconf import DictConfig

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager


def init_async_rollout_manager(config: DictConfig) -> AsyncLLMServerManager:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout"
    )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    # =========================== 2. Create AsyncLLMServerManager  ===========================
    async_rollout_manager = AsyncLLMServerManager(
        config=config,
        worker_group=actor_rollout_wg,
    )

    return async_rollout_manager


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config):
    # Make logging deterministic and less noisy
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    OmegaConf.resolve(config)
    _set_seed(config.data.get("seed", 1))
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # Initialize tokenizer and processor (needed to build prompts)
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer
    if config.trainer.val_only:
        config.actor_rollout_ref.val_only = True
    async_rollout_manager = init_async_rollout_manager(config)
    async_rollout_manager.wake_up()
    tokenizer = hf_tokenizer("ckpt/Qwen2.5-1.5B-Memory", trust_remote_code=True)

    messages_with_memory0 = [
        {'role': 'system', 'content': "this is a system prompt"},
        {'role': 'user', 'content': [
            {'type': 'memory_text', 'memory_text': {'text': "This is a dummy memory for round 1"}},
            {'type': 'text', 'text': "dummy question for round 1"},
        ]},
        {'role': 'assistant', 'content': "dummy response"},
        {'role': 'user', 'content': [
            {'type': 'memory_text', 'memory_text': {'text': "This is a dummy memory for round 1"}},
            {'type': 'text', 'text': 'dummy question'},
        ]},
        {'role': 'assistant', 'content': "dummy response"},
        {'role': 'user', 'content': [
            {'type': 'memory_text', 'memory_text': {'text': "This is a dummy memory for round 1"}},
            {'type': 'text', 'text': 'What should I do next?'},
        ]},
    ]

    gen_batch = DataProto.from_single_dict({
        "raw_prompt": np.array([messages_with_memory0]),
    })

    llm_outputs = async_rollout_manager.generate_sequences(gen_batch)
    decoded_responses = tokenizer.batch_decode(llm_outputs.batch['responses'], skip_special_tokens=True)
    print(decoded_responses)

        

if __name__ == "__main__":
    main()

