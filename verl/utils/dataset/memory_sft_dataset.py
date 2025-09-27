# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from copy import deepcopy


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MemorySFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], processor, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.max_memory_length = config.get("max_memory_length", 512)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(processor, str):
            processor = hf_processor(processor)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.memory_pad_token = self.tokenizer.convert_tokens_to_ids("<|mem_pad|>")

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.normal_messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        MemorySourceIndex = TypedDict("MemorySourceIndex", {"normal_index": int, "episode_index": int})

        self.message_index_mapping: dict[int, MemorySourceIndex] = {}
        i = 0
        for normal_index, normal_messages in enumerate(self.normal_messages):
            for memory_index in range(3, len(normal_messages), 2):
                self.message_index_mapping[i] = {"normal_index": normal_index, "episode_index": memory_index}
                i += 1

    def __len__(self):
        return len(self.message_index_mapping)

    def _make_message(self, message_index: int):
        memory_source_index = self.message_index_mapping[message_index]

        messages = self.normal_messages[memory_source_index["normal_index"]][:memory_source_index["episode_index"]]
        assert messages[-1]["role"] == "assistant"

        messages_processed = []
        all_memories = []
        for msg in messages[:-2]:
            if msg["role"] == "user":
                if msg["content"].startswith("memory recalled:"):
                    recalled_content = msg["content"].split("memory recalled: ")[1]
                    all_memories.append(recalled_content)
                    messages_processed.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "memory recalled: "},
                            {"type": "memory_text", "memory_text": {"text": recalled_content}},
                        ],
                    })
                else:
                    all_memories.append(msg["content"])
                    messages_processed.append({
                        "role": "user",
                        "content": [{"type": "memory_text", "memory_text": {"text": msg["content"]}}],
                    })
            else:
                messages_processed.append(deepcopy(msg))
        messages_processed.append(deepcopy(messages[-2]))
        messages_processed.append(deepcopy(messages[-1]))
        
        return messages_processed, all_memories

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages, all_memories = self._make_message(item)

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            input_ids = full_tokens[0]                        # [T]
            attention_mask = torch.ones_like(input_ids)
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}"
            )
            raise

        assert messages[-1]["role"] == "assistant"
        if isinstance(messages[-2]["content"], list):
            for cnt_item in messages[-2]["content"]:
                assert "memory" not in cnt_item['type']
        
        # Track concatenated tokens for validation
        last_assistant_idx = len(messages) - 1

        prefix = tokenizer.apply_chat_template(
            messages[:last_assistant_idx],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )
        prefix = prefix[0]
        prefix_len = prefix.shape[-1]


        loss_mask = torch.zeros_like(input_ids)
        loss_mask[prefix_len: input_ids.shape[0]] = 1


        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad_id = tokenizer.pad_token_id
            pad_n = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_n,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_n, dtype=attention_mask.dtype)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_n, dtype=loss_mask.dtype)])
        elif seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{seq_len=} > {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation {self.truncation}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = position_ids * attention_mask

        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        memory_inputs = {
            "memory_input_ids": torch.empty((0, 0)),
            "memory_attention_mask": torch.empty((0, 0)),
        }
        if len(all_memories) > 0:
            memory_inputs = self.processor(text="dummy text", memory=all_memories, return_tensors="pt")
            memory_inputs = {
                "memory_input_ids": memory_inputs["memory_input_ids"],
                "memory_attention_mask": memory_inputs["memory_attention_mask"],
            }
        return {
            **text_inputs,
            **memory_inputs,
        }

if __name__ == "__main__":
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM
    processor = hf_processor("ckpt/Qwen2.5-1.5B-Memory", trust_remote_code=True)
    dataset = MemorySFTDataset(
        parquet_files="nl_traj_sft_warmup/alfworld_traj_qwen25_1p5b_normal_messages.parquet",
        processor=processor,
        config={
            "max_length": 4096,
            "truncation": "error",
            "multiturn": {
                "messages_key": "messages",
            },
        },
    )

    def collate_fn(batch):

        batched_input_ids = torch.stack([item["input_ids"] for item in batch])
        batched_attention_mask = torch.stack([item["attention_mask"] for item in batch])
        batched_loss_mask = torch.stack([item["loss_mask"] for item in batch])
        batched_position_ids = torch.stack([item["position_ids"] for item in batch])

        max_memory_num = max([item["memory_input_ids"].shape[0] for item in batch])
        max_memory_len = max([item["memory_input_ids"].shape[1] for item in batch])
        if max_memory_num == 0 and max_memory_len == 0:
            batched_memory_input_ids = torch.empty((len(batch), 0, 0))
            batched_memory_attention_mask = torch.empty((len(batch), 0, 0))
        else:
            batched_memory_input_ids = torch.full((len(batch), max_memory_num, max_memory_len), processor.tokenizer.pad_token_id)
            batched_memory_attention_mask = torch.zeros_like(batched_memory_input_ids)
            for i_batch, item in enumerate(batch):
                memory_num = item["memory_input_ids"].shape[0]
                if memory_num > 0:
                    memory_length = item["memory_input_ids"].shape[1]
                    batched_memory_input_ids[i_batch, :memory_num, -memory_length:] = item["memory_input_ids"].unsqueeze(0)
                    batched_memory_attention_mask[i_batch, :memory_num, -memory_length:] = item["memory_attention_mask"].unsqueeze(0)
        
        return {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_mask,
            "loss_mask": batched_loss_mask,
            "position_ids": batched_position_ids,
            "memory_input_ids": batched_memory_input_ids,
            "memory_attention_mask": batched_memory_attention_mask,
        }

    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    world_size = 1
    rank = 0
    train_sampler = DistributedSampler(
        dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True, seed=42,
    )
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    for j, batch in enumerate(dataloader):
        # print(batch)
        with torch.no_grad():
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            print(outputs.logits.shape)
        if j > 10:
            break