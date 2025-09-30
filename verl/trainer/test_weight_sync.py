"""Sanity script to verify actor weight edits reach the async rollout engine."""

from __future__ import annotations

import os
import sys

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.trainer.collect_traj import init_async_rollout_manager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


def _build_test_batch(prompt: str) -> DataProto:
    messages = np.array(
        [
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ]
        ],
        dtype=object,
    )
    return DataProto.from_single_dict({"raw_prompt": messages})


def _decode_responses(output: DataProto, tokenizer) -> list[str]:
    prompts = output.batch["prompts"]
    responses = output.batch["responses"]
    attention_mask = output.batch["attention_mask"]
    prompt_len = prompts.shape[1]
    response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    decoded: list[str] = []
    for tokens, length in zip(responses, response_lengths, strict=True):
        valid_tokens = tokens[: int(length.item())]
        decoded.append(tokenizer.decode(valid_tokens, skip_special_tokens=True))
    return decoded


def _ensure_ray(config: DictConfig) -> None:
    if ray.is_initialized():
        return
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
        }
    }
    ray.init(runtime_env=runtime_env, num_cpus=OmegaConf.select(config, "ray_init.num_cpus"))


def _validate_gpu_layout(config: DictConfig) -> None:
    tp = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if world_size % tp != 0:
        raise ValueError(
            "Trainer GPU layout must be divisible by tensor_model_parallel_size for async rollout "
            f"(world_size={world_size}, tp={tp})."
        )


def _load_tokenizer(config: DictConfig):
    model_cfg = config.actor_rollout_ref.model
    local_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
    trust_remote_code = model_cfg.get("trust_remote_code", False) or config.data.get("trust_remote_code", False)
    return hf_tokenizer(local_path, trust_remote_code=trust_remote_code)


def _cleanup_manager(manager) -> None:
    if manager is None:
        return
    try:
        manager.sleep()
    except Exception:
        pass
    finally:
        try:
            loop = getattr(manager, "chat_scheduler_loop", None)
            if loop and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        try:
            thread = getattr(manager, "chat_scheduler_thread", None)
            if thread and thread.is_alive():
                thread.join(timeout=1)
        except Exception:
            pass
        try:
            for server in getattr(manager, "async_llm_servers", []):
                if server is not None:
                    ray.kill(server, no_restart=True)
        except Exception:
            pass


def _resolve_scale_factor(config: DictConfig) -> float:
    return 0.01


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config: DictConfig) -> None:  # pragma: no cover - script entry
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    OmegaConf.resolve(config)

    async_manager = None
    try:
        _validate_gpu_layout(config)
        _ensure_ray(config)

        async_manager = init_async_rollout_manager(config)
        actor_rollout_wg = async_manager.worker_group
        async_manager.wake_up()

        tokenizer = _load_tokenizer(config)
        batch = _build_test_batch("Say hello in one short sentence.")

        baseline = async_manager.generate_sequences(batch)
        baseline_text = _decode_responses(baseline, tokenizer)
        async_manager.sleep()

        scale_factor = _resolve_scale_factor(config)
        actor_rollout_wg.scale_actor_weights(scale=scale_factor)

        async_manager.wake_up()
        updated = async_manager.generate_sequences(batch)
        updated_text = _decode_responses(updated, tokenizer)
        async_manager.sleep()

        if torch.equal(baseline.batch["responses"], updated.batch["responses"]):
            raise RuntimeError("Inference outputs stayed identical after scaling actor weights; sync likely failed.")

        print("=== Weight Sync Sanity ===")
        print(f"Scale factor applied: {scale_factor}")
        print(f"Baseline: {baseline_text}")
        print(f"Updated : {updated_text}")

    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"Weight sync sanity check failed: {exc}", file=sys.stderr)
        raise
    finally:
        _cleanup_manager(async_manager)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
