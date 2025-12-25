# utils/builder.py
# -*- coding: utf-8 -*-
"""
builder.py
==========

目标：把“加载 Tokenizer + 加载 QLoRA 基座 + 包装 tokens-route 视觉前缀”的逻辑集中到一处，
避免 trainer 脚本里散落一堆构模细节，降低可维护性。

当前版本只保留一个明确的多模态路线：
- Vision Adapter: `patch_prefix`（逐 token projector + prefix 拼接）

这样做的好处：
- 代码路径单一，调试更容易
- 与 Stage-1 的训练目标完全对齐（answer-only loss + prefix labels=-100）

配置约定（来自 YAML）
--------------------
config:
  model_path: "/path/to/Qwen"
  image_token_dim: 768
  lora: {...}
  vision_adapter:
    type: "patch_prefix"
    use_image_feat: true/false   # false => 纯文本模式（结构仍保持一致，便于加载 stage1 ckpt）
    prefix_dropout: 0.0
    projector_type: "linear" or "mlp"
    projector_hidden_dim: 768    # projector_type=mlp 时可用
    layer_norm: false
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.nn import Module

from utils.model_utils import (
    load_tokenizer,
    build_qlora_base_model,
    print_trainable_parameters,
)
from models.qwen_vision_patch_prefix import QwenVisionPatchPrefixModel


def build_vision_llm(
    config: Dict[str, Any],
    rank: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Module, Any, int]:
    """
    构建完整的 Vision LLM（QLoRA + tokens-route Vision Prefix）以及 Tokenizer。

    返回
    ----
    model:
        QwenVisionPatchPrefixModel（内部包含 QLoRA base LLM）
    tokenizer:
        HuggingFace tokenizer（Trainer 里会作为 processing_class/tokenizer 使用）
    image_token_dim:
        图像 token 的最后一维（通常 768）
    """

    model_path = config["model_path"]
    lora_cfg = config.get("lora", {}) or {}
    vision_cfg = config.get("vision_adapter", {}) or {}

    image_token_dim = int(config.get("image_token_dim", 768))

    # ---- 1) Tokenizer ----
    tokenizer = load_tokenizer(model_path, trust_remote_code=True)

    # ---- 2) Base LLM (QLoRA + LoRA) ----
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    base_model = build_qlora_base_model(
        model_path=model_path,
        lora_cfg=lora_cfg,
        device=device,
        compute_dtype=compute_dtype,
        trust_remote_code=True,
    )

    # 避免 gradient checkpointing 时反复打印 use_cache warning
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    if rank == 0:
        print("[Builder] LoRA trainable 参数统计：")
        print_trainable_parameters(base_model, rank=rank)

    # ---- 3) Vision Adapter（只保留 patch_prefix 一条路） ----
    adapter_type = (vision_cfg.get("type") or "patch_prefix").lower()
    if adapter_type != "patch_prefix":
        raise ValueError(
            f"[Builder] Unsupported vision_adapter.type={adapter_type!r}. "
            "This refactored codebase keeps only 'patch_prefix' to reduce legacy complexity."
        )

    model = QwenVisionPatchPrefixModel(
        llm=base_model,
        image_token_dim=image_token_dim,
        prefix_dropout=float(vision_cfg.get("prefix_dropout", 0.0)),
        use_image_tokens=bool(vision_cfg.get("use_image_feat", True)),
        projector_type=str(vision_cfg.get("projector_type", "linear")),
        projector_hidden_dim=vision_cfg.get("projector_hidden_dim", None),
        layer_norm=bool(vision_cfg.get("layer_norm", False)),
    )

    # base_model 已经由 device_map 放到目标 GPU，这里只需要把 Adapter 参数移过去即可
    model.projector.to(device=device, dtype=compute_dtype)
    if model.ln is not None:
        model.ln.to(device=device, dtype=compute_dtype)

    return model, tokenizer, image_token_dim
