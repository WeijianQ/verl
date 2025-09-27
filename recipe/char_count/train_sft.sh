#!/usr/bin/env bash
set -euxo pipefail

nproc_per_node=${nproc_per_node:-1}
save_path=${save_path:-./models/qwen2_5_memory_sft}

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.fsdp_sft_trainer_for_memory \
    data.train_files=nl_traj_sft_warmup/train_alfworld_traj_qwen25_1p5b_normal_messages.parquet \
    data.val_files=nl_traj_sft_warmup/val_alfworld_traj_qwen25_1p5b_normal_messages.parquet \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=8 \
    data.max_length=4096 \
    data.truncation=left \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    use_remove_padding=true \
    model.partial_pretrain=ckpt/Qwen2.5-1.5B-Memory \
    model.trust_remote_code=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name=alfworld-memory-sft \
    trainer.experiment_name=qwen25_1p5b_memory \
    trainer.total_epochs=1 \
    trainer.test_freq=200 \
    trainer.save_freq=200 \
    trainer.logger=console \
    trainer.device=cuda \
    trainer.n_gpus_per_node="${nproc_per_node}"