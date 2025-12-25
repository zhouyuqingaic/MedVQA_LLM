# models/qwen_vision_patch_prefix.py
# -*- coding: utf-8 -*-
"""
QwenVisionPatchPrefixModel
==========================

这是一个“tokens-route / prefix-concat”风格的最小多模态封装：

数据流（非常关键）
----------------
1) Dataset：直接返回离线抽取的图像 patch tokens，形状 `[T, D_img]`
   - 不做池化 / 不做降维 / 不做 flatten
   - 通常 ViT-B/16@224：T=197 (CLS + 196 patches), D_img=768

2) Collator：
   - 按 batch 内最大 T 做 padding -> `[B, T_max, D_img]`
   - 生成 `image_token_mask` -> `[B, T_max]`（1=有效, 0=pad）
   - 文本部分：构造 `labels`，实现 *answer-only loss*

3) Model（本文件）：
   - 对每个 token 做 projector：`[B, T, D_img] -> [B, T, H]`
   - 将视觉前缀与文本 embedding 拼接：`[B, T+L, H]`
   - attention_mask/labels 前面拼接前缀对应的部分：
       - visual prefix 的 labels 全部设为 -100（不对视觉 token 计算 loss）

生成（generate）注意点
---------------------
HuggingFace generate 会在增量生成阶段传入 `past_key_values`。
视觉前缀只应在第一步注入 KV cache，后续步不应重复拼接前缀，
因此当 `past_key_values` 非空时，本封装会自动走纯文本 forward。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


class QwenVisionPatchPrefixModel(nn.Module):
    """Qwen CausalLM 的 tokens-route 视觉前缀封装。"""

    def __init__(
        self,
        llm: nn.Module,
        *,
        image_token_dim: int = 768,
        prefix_dropout: float = 0.0,
        use_image_tokens: bool = True,
        projector_type: str = "linear",
        projector_hidden_dim: Optional[int] = None,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.use_image_tokens = bool(use_image_tokens)

        # Qwen hidden_size（比如 4096）
        hidden_size = getattr(getattr(llm, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("[QwenVisionPatchPrefixModel] Cannot find llm.config.hidden_size")

        self.hidden_size = int(hidden_size)
        self.image_token_dim = int(image_token_dim)

        self.prefix_dropout = nn.Dropout(float(prefix_dropout))

        self.use_layer_norm = bool(layer_norm)
        self.ln = nn.LayerNorm(self.image_token_dim) if self.use_layer_norm else None

        projector_type = (projector_type or "linear").lower()
        self.projector_type = projector_type

        # projector：逐 token 映射 D_img -> hidden_size
        if projector_type == "linear":
            self.projector = nn.Linear(self.image_token_dim, self.hidden_size, bias=True)
        elif projector_type == "mlp":
            h = int(projector_hidden_dim or self.image_token_dim)
            self.projector = nn.Sequential(
                nn.Linear(self.image_token_dim, h),
                nn.GELU(),
                nn.Linear(h, self.hidden_size),
            )
        else:
            raise ValueError(f"[QwenVisionPatchPrefixModel] Unknown projector_type={projector_type!r}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        image_tokens: Optional[Tensor] = None,
        image_token_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        """
        参数
        ----
        input_ids : [B, L]
        attention_mask : [B, L]
        labels : [B, L]（通常由 collator 构造，已经做了 answer-only mask）
        image_tokens : [B, T, D_img] 或 [T, D_img]
        image_token_mask : [B, T] 或 [T]
        """

        # generate() 增量阶段会传入 past_key_values；此时前缀已在第一步进入 KV cache
        past_key_values = kwargs.get("past_key_values", None)

        # 纯文本路径（或生成阶段非首步）
        if (not self.use_image_tokens) or (image_tokens is None) or (past_key_values is not None):
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if input_ids.dim() != 2:
            raise ValueError(f"[QwenVisionPatchPrefixModel] input_ids should be [B,L], got {tuple(input_ids.shape)}")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # ---------- 1) 规范化 image_tokens 形状 ----------
        img = torch.as_tensor(image_tokens, device=input_ids.device)

        # 允许单样本 [T,C] -> [1,T,C]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.dim() != 3:
            raise ValueError(f"[QwenVisionPatchPrefixModel] image_tokens should be [B,T,C], got {tuple(img.shape)}")

        B_text = int(input_ids.size(0))
        B_img, T, C = img.shape

        if B_img != B_text:
            # 不做隐式 broadcast，避免 silent bug
            raise ValueError(
                f"[QwenVisionPatchPrefixModel] batch size mismatch: text B={B_text}, image B={B_img}. "
                f"Please ensure the collator stacks image tokens per sample."
            )

        if C != self.image_token_dim:
            raise ValueError(
                f"[QwenVisionPatchPrefixModel] image token dim mismatch: expect {self.image_token_dim}, got {C}"
            )

        # ---------- 2) image_token_mask ----------
        if image_token_mask is None:
            # 没提供 mask：默认全有效（适用于所有图的 T 一致且不 padding 的情况）
            prefix_mask = torch.ones((B_img, T), dtype=attention_mask.dtype, device=input_ids.device)
        else:
            prefix_mask = torch.as_tensor(image_token_mask, device=input_ids.device)
            if prefix_mask.dim() == 1:
                prefix_mask = prefix_mask.unsqueeze(0)
            if prefix_mask.shape != (B_img, T):
                raise ValueError(
                    f"[QwenVisionPatchPrefixModel] image_token_mask shape mismatch: "
                    f"expect {(B_img, T)}, got {tuple(prefix_mask.shape)}"
                )
            prefix_mask = prefix_mask.to(dtype=attention_mask.dtype)

        # 可选：把 padding token 清零，避免 projector 学到无意义 padding 分布
        img = img * prefix_mask.unsqueeze(-1).to(dtype=img.dtype)

        # ---------- 3) Projector：逐 token 映射到 hidden ----------
        # 让 img 的 dtype 与 projector 的权重 dtype 对齐，避免不必要的 dtype 升级
        proj_dtype = next(self.projector.parameters()).dtype
        img = img.to(dtype=proj_dtype)

        if self.ln is not None:
            img = self.ln(img)

        img_proj = self.projector(img)          # [B, T, H]
        img_proj = self.prefix_dropout(img_proj)

        # ---------- 4) 文本 embedding ----------
        embed_layer = self.llm.get_input_embeddings()
        text_embeds = embed_layer(input_ids)    # [B, L, H]

        # ---------- 5) 拼接 inputs_embeds ----------
        inputs_embeds = torch.cat([img_proj, text_embeds], dim=1)  # [B, T+L, H]

        # ---------- 6) attention_mask 拼接 ----------
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, T+L]

        # ---------- 7) labels 拼接：视觉前缀全部 -100 ----------
        new_labels = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"[QwenVisionPatchPrefixModel] labels shape {tuple(labels.shape)} != input_ids shape {tuple(input_ids.shape)}"
                )
            ignore_prefix = torch.full(
                (labels.size(0), T),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            new_labels = torch.cat([ignore_prefix, labels], dim=1)

        # ---------- 8) 调用底层 LLM（用 inputs_embeds） ----------
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    # ---- 透传一些常用接口，方便 Trainer / generate 调用 ----
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.llm.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.llm.resize_token_embeddings(new_num_tokens)

    @property
    def config(self):
        return self.llm.config




    # ---- Gradient Checkpointing: delegate to underlying HF model ----
    def _unwrap_llm_for_gc(self):
        """
        Trainer expects gradient_checkpointing_enable/disable on the *top-level* model.
        Our wrapper is nn.Module, so we forward the call to the underlying HF model.
        For PEFT LoRA model: self.llm.base_model.model is usually the HF CausalLM.
        """
        m = self.llm
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            return m.base_model.model
        return m

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        target = self._unwrap_llm_for_gc()
        if not hasattr(target, "gradient_checkpointing_enable"):
            raise AttributeError(
                f"Underlying llm ({type(target)}) does not support gradient_checkpointing_enable"
            )
        # recommended when using gradient checkpointing
        if hasattr(target, "config"):
            target.config.use_cache = False
        return target.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        target = self._unwrap_llm_for_gc()
        if hasattr(target, "gradient_checkpointing_disable"):
            return target.gradient_checkpointing_disable()
        return None

