"""
Sanity check script: build train/val dataloaders with the same config
as eval_ckpt.sh, fix the seed, and print per-batch unique IDs for samples.

Run similar to eval_ckpt.sh by swapping the module:

python -m verl.trainer.sanity_dataloader \
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
  env.rollout.n=8 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1

No models or envs are created.
"""

import hashlib
import os
import random
from typing import Iterable

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.main_ppo_for_verl_agent import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _uids_from_batch(batch: dict) -> list[str]:
    """Compute deterministic unique ids for a batch.

    Prefer using token ids, optionally combined with data_source when present.
    """
    uids = []
    raw_ids_list: Iterable = batch.get("raw_prompt_ids", [])
    data_sources = batch.get("data_source", None)

    bsz = len(raw_ids_list)
    for i in range(bsz):
        raw_ids = raw_ids_list[i]
        # raw_ids may be a list[int] stored as object; normalize to list of ints
        if isinstance(raw_ids, np.ndarray):
            raw_ids = raw_ids.tolist()
        token_bytes = ",".join(str(x) for x in raw_ids).encode()
        h = hashlib.sha1(token_bytes)
        if data_sources is not None:
            h.update(str(data_sources[i]).encode())
        uids.append(h.hexdigest()[:16])
    return uids


@hydra.main(config_path="config", config_name="ppo_trainer_for_verl_agent", version_base=None)
def main(config):
    # Resolve config and set seeds (use the same data.seed as sampler)
    OmegaConf.resolve(config)
    seed = config.data.get("seed", 1)
    _set_seed(seed)

    # Instantiate tokenizer/processor for dataset tokenization
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    # Build datasets and samplers exactly as in training
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    # Dataloaders mirroring RayPPOTrainer._create_dataloader
    train_batch_size = config.data.get("gen_batch_size", config.data.train_batch_size)
    train_loader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    val_batch_size = config.data.val_batch_size if config.data.val_batch_size is not None else len(val_dataset)
    val_loader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Print per-batch unique ids
    # print("==== Train dataloader (seed={}, batch_size={}) ====".format(seed, train_batch_size))
    # for bi, batch_dict in enumerate(train_loader):
    #     print(f"Train batch {bi:03d}: Batch entries:")
    #     batch = DataProto.from_single_dict(batch_dict)
    #     for k, v in batch.batch.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == "input_ids":
    #                 decoded_ = tokenizer.batch_decode(v, skip_special_tokens=True)
    #                 print(f"  {k}: {v.shape} {v.dtype} decoded\n\t{decoded_}\n\n")
    #             else:
    #                 print(f"  {k}: {v.shape} {v.dtype}")
    #         else:
    #             print(f"  {k}: {type(v)}")
    #     print("Non-tensor batch entries:")
    #     for k, v in batch.non_tensor_batch.items():
    #         if k == "raw_prompt_ids":
    #             decoded_ = tokenizer.batch_decode(v, skip_special_tokens=True)
    #             print(f"  {k}: {v.shape} {v.dtype} decoded\n\t{decoded_}\n\n")
    #         else:
    #             print(f"  {k}: {type(v)}")
    #     print("Meta info:")
    #     print(batch.meta_info)

    #     if bi > 2:
    #         break
        # uids = _uids_from_batch(batch)
        # print(f"train batch {bi:03d}: {uids}")

    print("==== Val dataloader (batch_size={}) ====".format(val_batch_size))
    for bi, batch_dict in enumerate(val_loader):
        print(f"Val batch {bi:03d}: Batch entries:")
        batch = DataProto.from_single_dict(batch_dict)
        # for k, v in batch.batch.items():
        #     if isinstance(v, torch.Tensor):
        #         if k == "input_ids":
        #             decoded_ = tokenizer.batch_decode(v, skip_special_tokens=True)
        #             print(f"  {k}: {v.shape} decoded\n\t{decoded_}\n\n")
        #         else:
        #             print(f"  {k}: {v.shape} {v.dtype}")
        #     else:
        #         print(f"  {k}: {type(v)}")
        print("Non-tensor batch entries:")
        for k, v in batch.non_tensor_batch.items():
            # if k == "raw_prompt_ids":
            #     decoded_ = tokenizer.batch_decode(v, skip_special_tokens=True)
            #     print(f"  {k}: {v.shape} decoded\n\t{decoded_}\n\n")
            # else:
            #     print(f"  {k}: {v}")
            if k == "extra_info":
                print(' '.join([f"{d['split']}_{d['index']}" for d in v]))
        # print("Meta info:")
        # print(batch.meta_info)
        # if bi > 2:
        #     break
        # uids = _uids_from_batch(batch)
        # print(f"val   batch {bi:03d}: {uids}")


if __name__ == "__main__":
    main()
