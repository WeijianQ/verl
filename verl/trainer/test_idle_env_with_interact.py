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

from ast import Not
import os
import re
from typing import Literal
from dataclasses import dataclass
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

from typing import TypedDict, List


@dataclass
class Episode:
    traj_uid: str
    step: int
    raw_observation: str
    llm_response: str = None
    is_action_valid: bool = None
    reward: float = None
    is_recall: bool = False
    status: Literal["finished", "in_progress"]="in_progress" 
    done_reason: Literal["length", "step", "finished"] = None
    done: bool = False

@dataclass
class TrajectoryWithMemory:
    traj_uid: str

    def __init__(self, traj_uid: str, task: str, system_prompt: str, max_prompt_len: int):
        self.traj_uid = traj_uid
        self.episodes = []
        self.is_done = False
        self.end_reason = None
        self.total_reward = 0  
        self.task = task
        self.system_prompt = system_prompt
        self.max_prompt_len = max_prompt_len
        self.memory_bank = []# memory idx to episode idx
        
    def __len__(self):
        return len(self.episodes)

    def make_llm_input(self, obs: str, step_idx: int=None):
        if step_idx is None:
            step_idx = len(self.episodes)
        if step_idx > len(self.episodes):
            raise ValueError(f"step_idx {step_idx} is greater than the number of episodes {len(self.episodes)}")
        messages = [{
            "content": self.system_prompt.format(task=self.task),
            "role": "system",
        }]
        memory_cnt = 1
        for episode in self.episodes:
            if episode.status == "finished":
                if episode.is_recall:
                    messages.append({
                        "content": f"memory recalled: {episode.raw_observation}<|embed|>",
                        "role": "user",
                    })
                else:
                    messages.append({
                        "content": f"Memory {memory_cnt}: {episode.raw_observation}<|embed|>",
                        "role": "user",
                    })
                    memory_cnt += 1
                messages.append({
                    "content": episode.llm_response,
                    "role": "assistant",
                })

        messages.append({
            "content": obs,
            "role": "user",
        })
        return messages

    def update_episode(self, **kwargs):
        assert self.episodes[-1].status == "in_progress", f"episode {self.episodes[-1].step} is finished"
        for key, value in kwargs.items():
            if key == "done" and bool(value) is True:
                self.is_done = True
                self.end_reason = "finished"
            setattr(self.episodes[-1], key, value)
    
    def start_episode(
        self,
        observation: str,
        is_recall: bool = False,
    ):
        self.episodes.append(Episode(
            traj_uid=self.traj_uid,
            step=len(self.episodes),
            raw_observation=observation,
            status="in_progress",
            is_recall=is_recall,
        ))
        if not is_recall:
            self.memory_bank.append(self.episodes[-1])

alfworld_system_prompt = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You are an expert agent operating in the ALFRED Embodied Environment.
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

    # Build envs (train and val); we use val envs
    config.trainer.val_only = True
    config.data.val_batch_size = 1
    _, envs = make_envs(config)
    observations, infos = envs.reset(kwargs={})
    pattern = r"Your task is to:\s*(.+?)\."
    match = re.search(pattern, observations['text'][0])
    if match:
        task = match.group(1)
    else:
        raise ValueError(f"Task description not found in text observation: {observations['text']}")
    trajectory = TrajectoryWithMemory(
        traj_uid="traj_0",
        task=task,
        system_prompt=alfworld_system_prompt,
        max_prompt_len=8096,
    )
    from src.utils import wait_for_debugger
    wait_for_debugger()
    recall_pattern = r"<recall>(\d+)</recall>"
    # from src.utils import wait_for_debugger
    # wait_for_debugger() <recall>Memory 1</recall>
    IS_RECALL = False
    while True:
        #### input
        trajectory.start_episode(
            observation=observations['text'][0],
            is_recall=IS_RECALL,
        )
        IS_RECALL = False
        messages = trajectory.make_llm_input(observations['text'][0])
        human_input = input(f"{' '.join([str(m) for m in messages])}\n\n########## type your responese here ##########")

        trajectory.update_episode(
            llm_response=human_input,
        )
        ### parse human input
        
        if match := re.search(recall_pattern, human_input):
            ### apply recall
            memory_idx = int(match.group(1))
            episode_to_recall = trajectory.memory_bank[memory_idx-1] # cuz it starts from 1
            assert not episode_to_recall.is_recall, f"episode {episode_to_recall.step} is already a recall, should be a real env observation"

            new_observation = {'text': [f"memory recalled: {episode_to_recall.raw_observation}"]}

            trajectory.update_episode(
                reward=0,
                done=False,
                is_action_valid=True,
                status="finished",
            )
            IS_RECALL = True

            # to match the batched behaviror let the env idle one step
            text_actions = ["__IDLE__"]
            
            _, _, _, _ = envs.step(text_actions)
        else:
            ### apply action
            text_actions = [human_input]
            new_observation, rewards, dones, infos = envs.step(text_actions)
            trajectory.update_episode(
                reward=rewards[0],
                done=dones[0],
                is_action_valid=infos[0]['is_action_valid'],
                status="finished",
            )
        observations = new_observation

if __name__ == "__main__":
    main()

