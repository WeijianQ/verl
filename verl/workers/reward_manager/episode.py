"""Reward manager that maps per-episode scores onto token-level rewards."""

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register


@register("episode")
class EpisodeRewardManager:
    """Populate the last token of each response with a precomputed episode reward."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        reward_key: str = "reward_scores",
        normalize_by_length: bool = False,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_key = reward_key
        self.normalize_by_length = normalize_by_length

    def _decode_prompt(self, data_item) -> str:
        prompt_ids = data_item.batch["prompts"]
        attention_mask = data_item.batch.get("attention_mask")
        prompt_len = prompt_ids.shape[-1]
        if attention_mask is None:
            return self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        valid_length = int(attention_mask[:prompt_len].sum().item())
        if valid_length == 0:
            return ""
        valid_ids = prompt_ids[-valid_length:]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

    def _decode_response(self, data_item, valid_length: int) -> str:
        if valid_length <= 0:
            return ""
        response_ids = data_item.batch["responses"][:valid_length]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def _get_response_lengths(self, data: DataProto) -> torch.Tensor:
        responses = data.batch["responses"]
        device = responses.device
        if "response_mask" in data.batch.keys():
            return data.batch["response_mask"].to(device=device).sum(dim=-1).to(dtype=torch.long)

        prompts = data.batch["prompts"]
        prompt_len = prompts.shape[-1]
        attention_mask = data.batch["attention_mask"].to(device=device)
        return attention_mask[:, prompt_len:].sum(dim=-1).to(dtype=torch.long)

    def _get_scores(self, data: DataProto) -> torch.Tensor:
        device = data.batch["responses"].device

        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"].to(device=device, dtype=torch.float32)

        candidate_keys = [self.reward_key, "reward_scores", "reward_score"]
        raw_scores: Any = None
        for key in candidate_keys:
            value = data.non_tensor_batch.get(key)
            if value is not None:
                raw_scores = value
                break

        if raw_scores is None:
            raise KeyError(
                "No reward scores available. Expected one of 'rm_scores' tensor or non-tensor keys "
                "['reward_scores', 'reward_score']."
            )

        if isinstance(raw_scores, torch.Tensor):
            scores = raw_scores.to(device=device, dtype=torch.float32)
        else:
            scores = torch.as_tensor(raw_scores, dtype=torch.float32, device=device)

        if scores.ndim > 1:
            scores = scores.squeeze()
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        scores = self._get_scores(data)
        response_lengths = self._get_response_lengths(data)
        responses = data.batch["responses"]

        if scores.shape[0] != responses.shape[0]:
            raise ValueError(
                f"Reward score count {scores.shape[0]} does not match batch size {responses.shape[0]}."
            )

        if self.normalize_by_length:
            denom = response_lengths.clamp(min=1).to(dtype=torch.float32)
            scores = scores.to(dtype=torch.float32) / denom
        else:
            scores = scores.to(dtype=torch.float32)

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        valid_mask = response_lengths > 0
        if valid_mask.any():
            row_idx = torch.arange(responses.shape[0], device=responses.device)[valid_mask]
            col_idx = response_lengths[valid_mask] - 1
            reward_tensor[row_idx, col_idx] = scores[valid_mask]

        reward_extra_info = defaultdict(list)
        for idx in range(len(data)):
            reward_extra_info["episode_score"].append(float(scores[idx].item()))
            reward_extra_info["response_length"].append(int(response_lengths[idx].item()))

        if self.num_examine > 0:
            printed = 0
            for idx in range(len(data)):
                if printed >= self.num_examine:
                    break
                length = int(response_lengths[idx].item())
                if length <= 0:
                    continue
                prompt_str = self._decode_prompt(data[idx])
                response_str = self._decode_response(data[idx], length)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[episode_score]", float(scores[idx].item()))
                printed += 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
