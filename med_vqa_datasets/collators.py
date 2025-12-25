# med_vqa_datasets/collators.py
# -*- coding: utf-8 -*-
"""
collators.py
============

放置本项目所有 DataCollator：

1) LlavaMedChatCollator
   - Stage-1 对齐（LLaVA-Med alignment）
   - 输入：image_tokens + chat
   - 输出：input_ids/attention_mask/labels(image answer-only) + image_tokens/padded_mask

2) VQATextDataCollator
   - Stage-2 VQA 文本-only
   - 输入：question/answer
   - 输出：input_ids/attention_mask/labels(answer-only)

3) VQAMultimodalPtTokensCollator
   - Stage-2 VQA 多模态（离线 tokens .pt 查表）
   - 输入：image(PIL)/question/answer
   - 输出：同上 + image_tokens/padded_mask

备注：为了保证“answer-only loss”严格正确，我们用 `prompt_text` 的 token 长度
去 mask `labels` 的前缀部分；其中 prompt_text 会显式带上 assistant 开头 token
（add_generation_prompt=True）。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------
# Helper: answer-only label mask
# ---------------------------------------------------------------------
def _build_answer_only_labels(full_enc: Dict[str, torch.Tensor], prompt_enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    给定:
      - full_enc : tokenizer(full_texts) 的输出（包含完整答案）
      - prompt_enc: tokenizer(prompt_texts) 的输出（停在 assistant 开头）

    返回:
      - labels: full_enc["input_ids"] 的 clone，且：
          - padding 位置为 -100
          - prompt 部分（含 system/user/assistant_start）为 -100
    """
    labels = full_enc["input_ids"].clone()

    # 1) mask padding
    labels[full_enc["attention_mask"] == 0] = -100

    # 2) mask prompt part
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1)  # [B]
    for i, l in enumerate(prompt_lens.tolist()):
        l = min(int(l), labels.size(1))
        labels[i, :l] = -100

    return labels


# ---------------------------------------------------------------------
# Stage-1: LLaVA-Med alignment (tokens route)
# ---------------------------------------------------------------------
@dataclass
class LlavaMedChatCollator:
    """
    Stage-1 tokens-route Collator。

    Dataset 输出（每条）：
      - image_tokens: [T, C]
      - chat: List[{"role","content"}]

    Collator 输出（batch）：
      - input_ids / attention_mask / labels（answer-only）
      - image_tokens: [B, T_max, C]
      - image_token_mask: [B, T_max]
    """

    tokenizer: Any
    max_length: int = 1024

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        tokens_list: List[torch.Tensor] = []

        for ex in examples:
            # ---- image tokens ----
            tok = ex["image_tokens"]
            tok = tok if torch.is_tensor(tok) else torch.as_tensor(tok)

            # 允许 [1,T,C] -> [T,C]
            if tok.dim() == 3 and tok.size(0) == 1:
                tok = tok.squeeze(0)
            if tok.dim() != 2:
                raise ValueError(f"[LlavaMedChatCollator] image_tokens should be [T,C], got {tuple(tok.shape)}")
            tokens_list.append(tok)

            # ---- chat -> text ----
            chat = ex["chat"]

            # 我们只在“最后一轮 assistant”上算 loss
            last_asst = None
            for i in range(len(chat) - 1, -1, -1):
                if chat[i].get("role") == "assistant":
                    last_asst = i
                    break

            if last_asst is None:
                # 极端情况：没有 assistant 回答
                full_msgs = chat
                prompt_msgs = chat
            else:
                full_msgs = chat[: last_asst + 1]   # 含最后一轮 assistant
                prompt_msgs = chat[: last_asst]     # 截止到 assistant 之前

            # full_text：包含答案内容
            full_texts.append(self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False))

            # prompt_text：停在 assistant 开头（用于严格 mask 掉 assistant header token）
            prompt_texts.append(self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True))

        # ---- tokenize ----
        full_enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        full_enc["labels"] = _build_answer_only_labels(full_enc, prompt_enc)

        # #数据debug
        # if not hasattr(self, "_debug_once"):
        #     self._debug_once = True
        #     lab = full_enc["labels"]
        #     att = full_enc["attention_mask"]
        #     # 每条样本真正参与 loss 的 token 数
        #     valid = (lab != -100).sum(dim=1).tolist()
        #     lens = att.sum(dim=1).tolist()
        #     print("[DEBUG] seq_len:", lens[:4])
        #     print("[DEBUG] label_tokens:", valid[:4])

        # ---- pad image tokens ----
        B = len(tokens_list)
        C = int(tokens_list[0].size(-1))
        T_max = max(int(t.size(0)) for t in tokens_list)

        image_tokens = tokens_list[0].new_zeros((B, T_max, C))
        image_token_mask = torch.zeros((B, T_max), dtype=torch.long)

        for i, t in enumerate(tokens_list):
            T = int(t.size(0))
            image_tokens[i, :T] = t
            image_token_mask[i, :T] = 1

        full_enc["image_tokens"] = image_tokens
        full_enc["image_token_mask"] = image_token_mask
        return full_enc


# ---------------------------------------------------------------------
# Stage-2: VQA text-only
# ---------------------------------------------------------------------
@dataclass
class VQATextDataCollator:
    tokenizer: Any
    max_length: int = 512
    system_prompt: str = ""
    add_image_hint: bool = False

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts: List[str] = []
        full_texts: List[str] = []

        for ex in examples:
            q = str(ex["question"])
            a = str(ex["answer"])

            sys = (self.system_prompt or "").strip()
            if self.add_image_hint:
                sys = (sys + "\nIn this training stage you will NOT see the image, only the text question.\n").strip()

            full_msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]

            full_texts.append(self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.tokenizer.apply_chat_template(full_msgs[:-1], tokenize=False, add_generation_prompt=True))

        full_enc = self.tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        full_enc["labels"] = _build_answer_only_labels(full_enc, prompt_enc)
        return full_enc


# ---------------------------------------------------------------------
# Stage-2: offline pt token store (sha1(image)->tokens)
# ---------------------------------------------------------------------
def image_sha1_key(img: Image.Image) -> str:
    """
    与 gen_vqa_rad_path_pt.py 的 key 策略完全一致：
      sha1("RGB" + size + raw_pixels)

    注意：不要对 image 做随机增强，否则 key 会变，查不到特征。
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.convert("RGB")
    raw = img.tobytes()

    h = hashlib.sha1()
    h.update(b"RGB")
    h.update(str(img.size).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


class PtTokenStore:
    """
    离线 tokens 仓库：
      - 初始化时 torch.load 一次（CPU）
      - get_by_pil(img) -> Tensor[T, C]

    兼容保存格式：
      - {"meta":..., "features": {key: Tensor[...]}, "failed":[...]}
      - {key: Tensor[...]}（极简格式）
    """

    def __init__(self, pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"[PtTokenStore] pt file not found: {pt_path}")

        payload = torch.load(pt_path, map_location="cpu")

        if isinstance(payload, dict) and "features" in payload:
            self.meta = payload.get("meta", {})
            self.features = payload["features"]
        elif isinstance(payload, dict):
            self.meta = {}
            self.features = payload
        else:
            raise TypeError(f"[PtTokenStore] Unsupported payload type: {type(payload)}")

        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise RuntimeError(f"[PtTokenStore] Empty features in pt: {pt_path}")

        any_feat = next(iter(self.features.values()))
        any_feat = any_feat if torch.is_tensor(any_feat) else torch.as_tensor(any_feat)

        # 推断 token_dim
        if any_feat.dim() == 3 and any_feat.size(0) == 1:
            any_feat = any_feat.squeeze(0)
        if any_feat.dim() != 2:
            raise RuntimeError(
                f"[PtTokenStore] This store expects tokens Tensor[T,C]. "
                f"Got shape={tuple(any_feat.shape)} from {pt_path}"
            )
        self.token_dim = int(any_feat.size(-1))

    def get_by_pil(self, img: Image.Image) -> torch.Tensor:
        key = image_sha1_key(img)
        if key not in self.features:
            raise KeyError(
                f"[PtTokenStore] Key not found in pt: {key}\n"
                f"Hint: make sure training-side key strategy is identical to pt generation."
            )
        feat = self.features[key]
        feat = feat if torch.is_tensor(feat) else torch.as_tensor(feat)

        # 允许 [1,T,C] -> [T,C]
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat.squeeze(0)
        if feat.dim() != 2:
            raise ValueError(f"[PtTokenStore] tokens must be [T,C], got {tuple(feat.shape)}")
        return feat


@dataclass
class VQAMultimodalPtTokensCollator:
    """
    Stage-2 multi：离线 tokens 版 collator（推荐）。

    - 不跑 BiomedCLIP
    - 不碰 CUDA
    - 只负责把 pt 里的 image_tokens padding 后拼进 batch
    """

    text_collator: Any
    image_tokens_pt_path: str
    strict: bool = True

    def __post_init__(self):
        self.store = PtTokenStore(self.image_tokens_pt_path)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) 文本部分：复用 VQATextDataCollator（含 answer-only mask）
        batch = self.text_collator(examples)

        # 2) 图像 tokens：CPU 查表 + padding/stack
        tokens_list: List[torch.Tensor] = []
        for ex in examples:
            try:
                img = ex["image"]  # vqa_rad_path_hf.py 返回 PIL.Image（image_transform=None）
                tok = self.store.get_by_pil(img)
            except Exception:
                if self.strict:
                    raise
                tok = torch.zeros((1, self.store.token_dim), dtype=torch.float16)  # 极简兜底
            tokens_list.append(tok)

        B = len(tokens_list)
        C = int(tokens_list[0].size(-1))
        T_max = max(int(t.size(0)) for t in tokens_list)

        image_tokens = tokens_list[0].new_zeros((B, T_max, C))
        image_token_mask = torch.zeros((B, T_max), dtype=torch.long)

        for i, t in enumerate(tokens_list):
            T = int(t.size(0))
            image_tokens[i, :T] = t
            image_token_mask[i, :T] = 1

        batch["image_tokens"] = image_tokens
        batch["image_token_mask"] = image_token_mask
        return batch
