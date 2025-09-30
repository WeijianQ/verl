"""Lightweight evaluation loop for memory-enabled rollouts using reward managers."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Dict

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.trainer.collect_traj import init_async_rollout_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local

from src.agent_loop.sync_env_async_llm_batch_collector import TrajectoryCollectorUsingAsyncLLMServer
from src.verlagent.environments import make_envs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _prepare_dataloader(config: DictConfig, tokenizer, processor) -> StatefulDataLoader:
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
    batch_size = config.data.get("val_batch_size") or 1
    num_workers = config.data.get("dataloader_num_workers", 0)

    return StatefulDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _gather_sequence_rewards(reward_tensor: torch.Tensor) -> np.ndarray:
    if reward_tensor.ndim != 2:
        reward_tensor = reward_tensor.view(reward_tensor.size(0), -1)
    return reward_tensor.sum(dim=-1).detach().cpu().numpy()


def _merge_extra_info(collector: Dict[str, list], batch_info: Dict[str, Any]) -> None:
    for key, values in batch_info.items():
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().tolist()
        elif isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, (list, tuple)):
            values = [values]
        collector[key].extend(values)


def _print_summary(all_rewards: list[float], extra_info: Dict[str, list]) -> None:
    if not all_rewards:
        print("No evaluation samples processed.")
        return

    rewards_np = np.asarray(all_rewards, dtype=np.float32)
    summary = {
        "num_sequences": len(rewards_np),
        "mean_reward": float(rewards_np.mean()),
        "std_reward": float(rewards_np.std()),
        "min_reward": float(rewards_np.min()),
        "max_reward": float(rewards_np.max()),
    }

    print("=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if extra_info:
        print("--- Extra Info (first 5 entries per key) ---")
        for key, values in extra_info.items():
            preview = values[:5]
            print(f"{key}: {preview}")


def _cleanup(async_manager, envs, val_envs) -> None:
    try:
        async_manager.sleep()
        async_manager.worker_group.shutdown()
    except AttributeError:
        pass
    finally:
        try:
            envs.close()
        except Exception:
            pass
        try:
            val_envs.close()
        except Exception:
            pass
        ray.shutdown()


def run_eval_memory(config: DictConfig) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    OmegaConf.resolve(config)
    _set_seed(config.data.get("seed", 1))

    if not ray.is_initialized():
        ray.init(
            # runtime_env=get_ppo_ray_runtime_env(), 
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")}},
            num_cpus=config.ray_init.get("num_cpus")
        )

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
    processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

    envs, val_envs = make_envs(config)
    async_rollout_manager = init_async_rollout_manager(config)
    async_rollout_manager.wake_up()

    traj_collector = TrajectoryCollectorUsingAsyncLLMServer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
    )

    reward_kwargs = config.reward_model.get("reward_kwargs", {})
    reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)

    dataloader = _prepare_dataloader(config, tokenizer, processor)

    eval_cfg = config.get("eval", {})
    max_batches = eval_cfg.get("max_batches") if isinstance(eval_cfg, DictConfig) else None
    show_progress = True
    if isinstance(eval_cfg, DictConfig):
        show_progress = eval_cfg.get("progress", True)

    all_rewards: list[float] = []
    extra_info: Dict[str, list] = defaultdict(list)

    progress = None
    try:
        progress = tqdm(dataloader, desc="Evaluating", disable=not show_progress)
        for batch_idx, batch_dict in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                break

            gen_batch = DataProto.from_single_dict(batch_dict)
            rollout_batch = traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                envs=val_envs,
                async_rollout_manager=async_rollout_manager,
                is_train=False,
            )

            reward_result = reward_fn(rollout_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            batch_rewards = _gather_sequence_rewards(reward_tensor)
            all_rewards.extend(batch_rewards.tolist())

            reward_extra = reward_result.get("reward_extra_info", {})
            _merge_extra_info(extra_info, reward_extra)

            progress.set_postfix(mean_reward=np.mean(all_rewards) if all_rewards else 0.0)

    finally:
        if progress is not None:
            progress.close()
        _print_summary(all_rewards, extra_info)
        _cleanup(async_rollout_manager, envs, val_envs)


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config: DictConfig) -> None:  # pragma: no cover - entry point
    run_eval_memory(config)


if __name__ == "__main__":  # pragma: no cover
    main()
