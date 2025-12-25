# med_vqa_datasets/dataset_llava_med_alignment_500k_subset.py
# -*- coding: utf-8 -*-
"""
Stage-1: LLaVA-Med Alignment Dataset (tokens-route)
================================================

这个 Dataset 只做两件事：
1) 读取 LLaVA-Med alignment JSON（多轮对话）
2) 读取离线抽取好的 BiomedCLIP 图像 tokens（每个样本一个 .pt）

重要设计点
----------
- Dataset 不做 tokenizer，不做 padding，不做 labels mask；
  这些都交给 collator 做（更干净、更容易调试）。
- Dataset 输出的 `image_tokens` 保持 token 序列 `[T, C]` 原样，不做池化/压缩。

每条样本返回：
    {
      "id": str,
      "image_tokens": torch.Tensor[T, C]  # fp16, CPU
      "chat": List[{"role": "system/user/assistant", "content": str}]
    }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class LLaVAMedAlignmentDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "train", verbose: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split

        json_path = cfg["llava_med_json"]
        feat_dir = cfg["llava_med_image_feature_dir"]

        self.system_prompt: Optional[str] = cfg.get("system_prompt", None)
        self.max_samples: Optional[int] = cfg.get("max_samples", None)
        self.image_token_dim = int(cfg.get("image_token_dim", 768))

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[Dataset] llava_med_json not found: {json_path}")
        if not os.path.isdir(feat_dir):
            raise FileNotFoundError(f"[Dataset] llava_med_image_feature_dir not found: {feat_dir}")

        if verbose:
            print(f"[Dataset] JSON: {json_path}")
            print(f"[Dataset] Image tokens dir: {feat_dir}")
            if self.system_prompt:
                print("[Dataset] system_prompt enabled.")
            if self.max_samples:
                print(f"[Dataset] max_samples={self.max_samples} (debug)")

        with open(json_path, "r", encoding="utf-8") as f:
            all_samples: List[Dict[str, Any]] = json.load(f)

        if self.max_samples is not None:
            all_samples = all_samples[: int(self.max_samples)]

        # 预过滤没有 tokens 的样本，避免训练中频繁报错
        self.samples: List[Dict[str, Any]] = []
        self.feature_paths: List[str] = []

        missing = 0
        for sp in all_samples:
            sid = sp.get("id")
            if sid is None:
                continue
            feat_path = os.path.join(feat_dir, f"{sid}.pt")
            if os.path.exists(feat_path):
                self.samples.append(sp)
                self.feature_paths.append(feat_path)
            else:
                missing += 1

        if verbose:
            print(f"[Dataset] total in json: {len(all_samples)}")
            print(f"[Dataset] with tokens:  {len(self.samples)}")
            print(f"[Dataset] missing tokens: {missing}")

        if not self.samples:
            raise RuntimeError(
                "[Dataset] No usable samples found. "
                "Please check llava_med_image_feature_dir and whether .pt files exist."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_tokens(self, feat_path: str) -> torch.Tensor:
        """
        Tokens route: 只读取 patch token 序列，期望输出 [T, C]，其中 C == image_token_dim。

        兼容保存格式：
        - Tensor[T, C]
        - dict: {"tokens": Tensor[T, C]} / {"image_tokens": ...} 等
        """
        obj = torch.load(feat_path, map_location="cpu")

        if isinstance(obj, dict):
            # 只接受明确的 token key，避免误读 global embedding
            for k in ("tokens", "image_tokens", "token_embeddings", "patch_tokens"):
                if k in obj:
                    obj = obj[k]
                    break
            else:
                raise KeyError(f"[Dataset] Unsupported dict keys in {feat_path}: {list(obj.keys())[:10]}")

        feat = obj if torch.is_tensor(obj) else torch.as_tensor(obj)

        # 允许 [1, T, C] -> [T, C]
        if feat.dim() == 3:
            if feat.size(0) != 1:
                raise ValueError(f"[Dataset] 3D tokens must be [1,T,C], got {tuple(feat.shape)} ({feat_path})")
            feat = feat.squeeze(0)

        if feat.dim() != 2:
            raise ValueError(f"[Dataset] tokens must be [T,C], got {tuple(feat.shape)} ({feat_path})")

        if feat.size(-1) != self.image_token_dim:
            raise ValueError(
                f"[Dataset] token_dim mismatch: expect {self.image_token_dim}, got {feat.size(-1)} ({feat_path})"
            )

        # 存成 fp16 更省显存；训练端会在 autocast 下自动处理
        return feat.to(dtype=torch.float16)

    def _build_chat(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将 LLaVA-Med JSON 的对话字段转成 Qwen chat template 支持的格式。

        JSON 中字段名可能是：
        - "conversatons"（原始数据集常见拼写）
        - "conversations"

        turn["from"]:
        - human -> user
        - gpt   -> assistant
        """
        conv_list = None
        for k in ("conversatons", "conversations"):
            if k in sample:
                conv_list = sample[k]
                break
        if conv_list is None:
            raise KeyError(f"[Dataset] sample missing conversations field. keys={list(sample.keys())}")

        chat: List[Dict[str, str]] = []

        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})

        for turn in conv_list:
            src = turn.get("from", "")
            content = str(turn.get("value", ""))
            if src == "human":
                role = "user"
            elif src == "gpt":
                role = "assistant"
            else:
                role = "user"
            chat.append({"role": role, "content": content})

        return chat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        feat_path = self.feature_paths[idx]

        return {
            "id": sample["id"],
            "image_tokens": self._load_image_tokens(feat_path),  # [T, C]
            "chat": self._build_chat(sample),
        }


def build_llava_med_dataset(cfg: Dict[str, Any], split: str = "train", verbose: bool = True) -> LLaVAMedAlignmentDataset:
    """便捷构造函数：从 cfg 一步构造 Stage-1 dataset。"""
    return LLaVAMedAlignmentDataset(cfg, split=split, verbose=verbose)
