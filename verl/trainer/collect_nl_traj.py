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
from src.agent_loop.rollout_sync_env_collect_inst_data import TrajectoryCollectorUsingAsyncLLMServer

from tqdm import tqdm
import traceback

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


alfworld_system_prompt = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task}

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
Once an episode ends, all past observations will be compressed into a numbered memory entry (e.g., "Memory 1", "Memory 2"). If you need to retrieve this past memory, you can recall it by number using the <recall> tag. For example: <recall>1</recall>. Then the observation will be restored.
"""

webshop_system_prompt = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You are an expert agent operating in the WebShop e‑commerce environment.
Your task is to: {task}

You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
Once an episode ends, all past observations will be compressed into a numbered memory entry (e.g., "Memory 1", "Memory 2"). If you need to retrieve this past memory, you can recall it by number using the <recall> tag. For example: <recall>1</recall>. Then the observation will be restored.
"""

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

    # Build datasets and loaders like the trainer
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    
    train_batch_size = config.data.train_batch_size if config.data.train_batch_size is not None else len(train_dataset)
    train_loader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f"Train dataloader batches: {len(train_loader)}; batch_size={train_batch_size}")
    
    save_path = f"nl_traj_collect/{config.trainer.experiment_name}_all_trajs_{config.env.env_name}.jsonl"
    save_path = hydra.utils.to_absolute_path(save_path)
    print(f"Saving to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if config.env.env_name == "alfworld/AlfredTWEnv":
        system_prompt = alfworld_system_prompt
    elif config.env.env_name == "Webshop":
        system_prompt = webshop_system_prompt
    else:
        raise ValueError(f"Environment {config.env.env_name} not supported")
        # Trajectory collector to build chat-template prompts from obs
    traj_collector = TrajectoryCollectorUsingAsyncLLMServer(config=config, tokenizer=tokenizer, processor=processor, system_prompt=system_prompt)

    print(f"Train dataloader batches: {len(train_loader)}; batch_size={train_batch_size}")


    all_trajs = []
    for bi, batch_dict in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Saving to {save_path}"):
        # print(f"\n=== Eval batch {bi:03d} ===")
        # Construct a DataProto for the dataset batch (for metadata/raw_prompt)
        gen_batch = DataProto.from_single_dict(batch_dict)

        # Build model inputs (prompts) for the current observations

        try:
            prompt_batch = traj_collector.multi_turn_loop(gen_batch=gen_batch, envs=val_envs, async_rollout_manager=async_rollout_manager)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            continue


        this_trajs = [t.to_json() for t in prompt_batch.trajs]
        # all_trajs.extend(this_trajs)

        with open(save_path, "a") as f:
            for traj in this_trajs:
                json_line = json.dumps(traj, ensure_ascii=False)
                f.write(json_line + '\n')

    
if __name__ == "__main__":
    main()


