"""
Sanity loader for ALFWorld envs: uses the same Hydra config as eval_ckpt.sh,
loads validation dataloader and ALFWorld envs, resets envs for each batch,
and prints the very first model input (prompt) derived from the initial observation.

Run with the same overrides as eval_ckpt.sh, for example:

python -m verl.trainer.sanity_env_loader \
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

from src.verlagent.environments import make_envs
from src.verlagent.multi_turn_rollout import TrajectoryCollector

from tqdm import tqdm


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

    # Initialize tokenizer and processor (needed to build prompts)
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer

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

    # Trajectory collector to build chat-template prompts from obs
    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

    print(f"Val dataloader batches: {len(val_loader)}; batch_size={val_batch_size}")

    record_jsons = []
    for bi, batch_dict in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
        # print(f"\n=== Eval batch {bi:03d} ===")
        # Construct a DataProto for the dataset batch (for metadata/raw_prompt)
        gen_batch = DataProto.from_single_dict(batch_dict)

        # Reset the envs to get the first observations for this batch
        obs, infos = val_envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        # Build model inputs (prompts) for the current observations
        prompt_batch = traj_collector.preprocess_batch(gen_batch=gen_batch, obs=obs)

        # Decode the very first prompt (input to LLM)

        for j_sample in range(len(prompt_batch.batch['input_ids'])):
            record_jsons.append({
                'uid': f"batch_{bi}_sample_{j_sample}",
                'prompt': tokenizer.decode(prompt_batch.batch['input_ids'][j_sample], skip_special_tokens=True),
                'file_name': infos[j_sample]['extra.gamefile'],
            })
    if 'out_of_distribution' in config.env.eval_split_alfworld:
        file_name = f"sanity_env_loader_record_jsons_for_val_unseen.jsonl"
    else:
        file_name = f"sanity_env_loader_record_jsons_for_val_seen.jsonl"
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, "w") as f:
        for record_json in record_jsons:
            f.write(json.dumps(record_json) + "\n")

if __name__ == "__main__":
    main()

