# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry point for PPO training with memory-enabled async rollout."""

import os
import inspect

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager

def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config):
    run_ppo_memory(config)


def run_ppo_memory(config) -> None:
    """Launch PPO training using the memory trajectory collector."""

    if not ray.is_initialized():
        # ray.init(runtime_env=get_ppo_ray_runtime_env(), num_cpus=config.ray_init.num_cpus)
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")}},
            num_cpus=config.ray_init.num_cpus
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        if config.trainer.no_memory:
            from src.agent_loop.sync_env_async_llm_batch_collector_no_mem import TrajectoryCollectorUsingAsyncLLMServer
            traj_collector_cls = TrajectoryCollectorUsingAsyncLLMServer
        else:
            from src.agent_loop.sync_env_async_llm_batch_collector import TrajectoryCollectorUsingAsyncLLMServer
            traj_collector_cls = TrajectoryCollectorUsingAsyncLLMServer

        from src.verlagent.environments import make_envs
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.vllm_utils import is_version_ge
        from verl.utils.dataset.rl_dataset import collate_fn

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.rollout.mode != "async":
            raise ValueError("main_ppo_memory requires actor_rollout_ref.rollout.mode to be 'async'.")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        envs, val_envs = make_envs(config)

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name == "vllm":
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge("vllm", "0.7.3"):
                raise NotImplementedError("PPO LoRA requires vllm>=0.7.3 when using async rollout.")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError

            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)
        print(f"TRAIN REWARD MANAGER: {reward_fn=}, VAL REWARD MANAGER: {val_reward_fn=}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("In async memory training, actor_rollout_ref.rollout.n must remain 1.")
        
        print(f"Using traj_collector_cls: {traj_collector_cls}")
        traj_collector = traj_collector_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
        )

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer_kwargs = {
            "config": config,
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": role_worker_mapping,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
            "reward_fn": reward_fn,
            "val_reward_fn": val_reward_fn,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "collate_fn": collate_fn,
            "train_sampler": train_sampler,
            "device_name": config.trainer.device,
        }

        trainer_sig = inspect.signature(RayPPOTrainer.__init__)
        if "traj_collector" in trainer_sig.parameters:
            trainer_kwargs["traj_collector"] = traj_collector
        if "envs" in trainer_sig.parameters:
            trainer_kwargs["envs"] = envs
        if "val_envs" in trainer_sig.parameters:
            trainer_kwargs["val_envs"] = val_envs

        trainer = RayPPOTrainer(**trainer_kwargs)

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
