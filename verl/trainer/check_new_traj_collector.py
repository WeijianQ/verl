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

from typing import List
from src.agent_loop.rollout_sync_env import Episode
alfworld_system_prompt = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task}

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
Once an episode ends, all past observations will be compressed into a numbered memory entry (e.g., "Memory 1", "Memory 2"). If you need to retrieve this past memory, you can recall it by number using the <recall> tag. For example: <recall>1</recall>. Then the observation will be restored.
"""
class MemoryLLMInputBuilder:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def _make_memory_to_embed_part(self, memory_content: str) -> str:
        return {'type': 'image_url', 'image_url': {'url': memory_content}}

    def __call__(self, obs: str, episodes: List["Episode"], **kwargs) -> str:
        task = kwargs.get("task", "")
        messages = [{
            "content": self.system_prompt.format(task=task),
            "role": "system",
        }]
        for episode in episodes:
            if episode.status == "finished":
                messages.append({
                    "content": [self._make_memory_to_embed_part(episode.current_observation)],
                    "role": "user",
                })
                messages.append({
                    "content": episode.llm_response,
                    "role": "assistant",
                })
            else:
                break
        messages.append({
            "content": obs,
            "role": "user",
        })
        return messages

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
    async_rollout_manager = init_async_rollout_manager(config)
    async_rollout_manager.wake_up()

    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
    tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
    processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

    # Build envs (train and val); we use val envs
    envs, val_envs = make_envs(config)
    del envs

    # Build datasets and loaders like the trainer
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    print(len(val_dataset))
    val_batch_size = config.data.val_batch_size if config.data.val_batch_size is not None else len(val_dataset)
    val_loader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if config.env.env_name == "alfworld/AlfredTWEnv":
        input_builder = MemoryLLMInputBuilder(system_prompt=alfworld_system_prompt)
    else:
        print(f"Using default LLM input builder for {config.env.name}")
        input_builder = None

    # Trajectory collector to build chat-template prompts from obs
    traj_collector = TrajectoryCollectorUsingAsyncLLMServer(config=config, tokenizer=tokenizer, processor=processor, input_builder=input_builder)

    print(f"Val dataloader batches: {len(val_loader)}; batch_size={val_batch_size}")

    record_jsons = []
    if config.trainer.val_only:
        config.actor_rollout_ref.val_only = True
    
    for bi, batch_dict in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
        # print(f"\n=== Eval batch {bi:03d} ===")
        # Construct a DataProto for the dataset batch (for metadata/raw_prompt)
        gen_batch = DataProto.from_single_dict(batch_dict)

        # Build model inputs (prompts) for the current observations
        prompt_batch = traj_collector.multi_turn_loop(gen_batch=gen_batch, envs=val_envs, async_rollout_manager=async_rollout_manager)

        print(prompt_batch)
        
        break

    import pickle
    with open("prompt_batch_webshop.pkl", "wb") as f:
        pickle.dump(prompt_batch, f)


        

if __name__ == "__main__":
    main()

