"""Utility to sanity-check async rollout manager startup without env initialization."""

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
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


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


def _ensure_ray(config: DictConfig) -> None:
    # if ray.is_initialized():
    #     return
    # runtime_env = get_ppo_ray_runtime_env()
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")}},
            num_cpus=config.ray_init.num_cpus,
        )
    # ray.init(runtime_env=runtime_env, num_cpus=config.ray_init.get("num_cpus"))


def _load_tokenizer(config: DictConfig):
    model_cfg = config.actor_rollout_ref.model
    local_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
    trust_remote_code = model_cfg.get("trust_remote_code", False) or config.data.get("trust_remote_code", False)
    return hf_tokenizer(local_path, trust_remote_code=trust_remote_code)


def _validate_gpu_layout(config: DictConfig) -> None:
    tp = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if world_size % tp != 0:
        raise ValueError(
            "Trainer GPU layout must be divisible by tensor_model_parallel_size for async rollout "
            f"(world_size={world_size}, tp={tp})."
        )


def _cleanup_async_manager(manager) -> None:
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


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config: DictConfig) -> None:  # pragma: no cover - script entry
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    OmegaConf.resolve(config)

    try:
        _validate_gpu_layout(config)
        _ensure_ray(config)

        async_manager = init_async_rollout_manager(config)
        async_manager.wake_up()

        tokenizer = _load_tokenizer(config)
        batch = _build_test_batch("Reply with a friendly greeting in one short sentence.")

        output = async_manager.generate_sequences(batch)
        decoded = _decode_responses(output, tokenizer)

        print("=== Async Manager Sanity Check ===")
        print(f"Responses: {decoded}")
        print("Prompt tokens:", output.batch["prompts"].shape)
        print("Response tokens:", output.batch["responses"].shape)

    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"Async manager test failed: {exc}", file=sys.stderr)
        raise
    finally:
        if 'async_manager' in locals():
            _cleanup_async_manager(async_manager)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
