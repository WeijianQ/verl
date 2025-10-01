"""Reward manager that maps per-episode scores onto token-level rewards."""

from collections import defaultdict
from typing import Any
from typing import Dict
import numpy as np
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

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch:
            reward_tensor = data.batch["rm_scores"]
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor

        responses: torch.Tensor = data.batch["responses"]             # [B, T_resp]
        B, T_resp = responses.shape
        device = responses.device
        dtype = torch.float32

        response_mask: torch.Tensor = data.batch.get("response_mask", None)
        if response_mask is None:
            attn = data.batch["attention_mask"]                       # [B, T_in]
            response_mask = attn[:, -T_resp:]                         # 取末 T_resp 作为 response 部分
        response_mask = response_mask.to(torch.long)

        if self.reward_key not in data.non_tensor_batch:
            raise KeyError(
                f"non_tensor_batch['{self.reward_key}'] not found. "
                f"Make sure _gather_trajectory_data filled it."
            )
        scores_np = data.non_tensor_batch[self.reward_key]
        if isinstance(scores_np, list):
            scores_np = np.array(scores_np, dtype=np.float32)
        scores = torch.as_tensor(scores_np, dtype=dtype, device=device)  # [B]
        if scores.ndim != 1 or scores.size(0) != B:
            raise ValueError(
                f"{self.reward_key} shape mismatch: expect ({B},), got {tuple(scores.shape)}"
            )

        if self.normalize_by_length:
            resp_lens = response_mask.sum(dim=1).clamp_min(1).to(scores.dtype)  # [B]
            scores = scores / resp_lens

        reward_tensor = torch.zeros((B, T_resp), dtype=dtype, device=device)
        resp_lens_long = response_mask.sum(dim=1)                                # [B], int
        has_valid = resp_lens_long > 0
        if has_valid.any():
            idx_b = torch.nonzero(has_valid, as_tuple=False).squeeze(1)          # [B_valid]
            last_pos = (resp_lens_long[idx_b] - 1).to(torch.long)                # [B_valid]
            reward_tensor[idx_b, last_pos] = scores[idx_b]

        # 5) 抽样打印（用于快速检查）
        if self.num_examine > 0:
            printed_per_source: Dict[str, int] = {}
            prompts = data.batch.get("prompts", None)
            attn_full = data.batch["attention_mask"]
            # 计算每样本 prompt 长度以便解码（假设 prompt 在前）
            prompt_len = attn_full.size(1) - T_resp
            for i in range(B):

                ds = "unknown"
                if "data_source" in data.non_tensor_batch:
                    ds_val = data.non_tensor_batch["data_source"]
                    try:
                        ds = ds_val[i] if isinstance(ds_val, (list, np.ndarray)) else str(ds_val)
                    except Exception:
                        ds = str(ds_val)

                printed_per_source.setdefault(ds, 0)
                if printed_per_source[ds] >= self.num_examine:
                    continue

                if np.random.random() >= 0.10:
                    continue

                if prompts is not None:
                    prompt_ids = prompts[i]
                    valid_prompt_len = attn_full[i, :prompt_len].sum().item()
                    pr_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]
                    prompt_str = self.tokenizer.decode(pr_ids, skip_special_tokens=False)
                else:
                    prompt_str = "<no-prompts>"

                resp_ids = responses[i]
                valid_resp_len = response_mask[i].sum().item()
                rs_ids = resp_ids[:valid_resp_len] if valid_resp_len > 0 else resp_ids[:0]
                response_str = self.tokenizer.decode(rs_ids, skip_special_tokens=False)

                print(f"[{ds}][prompt] {prompt_str}")
                print(f"[{ds}][response] {response_str}")
                print(f"[{ds}][score] {scores[i].item():.6f}")
                printed_per_source[ds] += 1

        extra: Dict[str, Any] = {}
        for k in ("episode_uids", "traj_uids", "traj_won_values"):
            if k in data.meta_info:
                extra[k] = data.meta_info[k]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
