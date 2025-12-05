# engine/builder.py
# -*- coding: utf-8 -*-
"""
职责： 集中处理模型的所有构建和包装逻辑。
包括：加载 Tokenizer, 构建 QLoRA 基座模型，并将其包装为 Vision VLM。
"""

import torch
from typing import Dict, Any, Optional
from torch.nn import Module

# 从 utils 层导入必要的工具
from utils.model_utils import load_tokenizer, build_qlora_base_model, print_trainable_parameters
from models.qwen_vision_prefix import QwenVisionPrefixModel


def build_vision_llm(
        config: Dict[str, Any],
        rank: int,
        compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Module, Any, int]:
    """
    构建完整的 Vision LLM 模型 (QLoRA + Vision Prefix Adapter) 和 Tokenizer。

    参数
    ----
    config : dict
        完整的配置字典。
    rank : int
        当前进程的 DDP Rank。
    compute_dtype : torch.dtype
        模型使用的计算类型 (bf16/fp16)。

    返回
    ----
    (model, tokenizer, image_feat_dim)
    """

    # 1. 提取关键配置
    MODEL_PATH = config["model_path"]
    lora_cfg = config.get("lora", {})
    vision_cfg = config.get("vision_adapter", {})
    image_feat_dim = int(config.get("image_feat_dim", 512))

    vision_enabled = bool(vision_cfg.get("enabled", False))
    prefix_dropout = float(vision_cfg.get("prefix_dropout", 0.0))
    use_image_feat = bool(vision_cfg.get("use_image_feat", True))

    # 2. 准备 Tokenizer (使用 utils/model_utils)
    tokenizer = load_tokenizer(MODEL_PATH, trust_remote_code=True)

    # 3. 加载 QLoRA 基座模型 (使用 utils/model_utils)
    # 当前进程所使用的逻辑 GPU 设备
    current_device = torch.device(f"cuda:{rank}")

    base_model = build_qlora_base_model(
        model_path=MODEL_PATH,
        lora_cfg=lora_cfg,
        device=current_device,
        compute_dtype=compute_dtype,
    )

    # 打印可训练参数统计 (仅 Rank 0)
    if rank == 0:
        print("[Stage1] LoRA trainable 参数统计：")
        print_trainable_parameters(base_model, rank=rank)

    # 4. 包装 Vision Prefix Adapter
    if vision_enabled:
        if rank == 0:
            print("[Stage1] 启用 Vision Prefix Adapter (image_feat -> prefix token)")
            print(
                f"[Stage1] image_feat_dim={image_feat_dim}, prefix_dropout={prefix_dropout}, use_image_feat={use_image_feat}")

        # 将 LoRA 基座模型包装进 Vision Adapter
        num_prefix_tokens=vision_cfg["num_prefix_tokens"] #设置多prefix token模式
        model = QwenVisionPrefixModel(
            llm=base_model,
            image_feat_dim=image_feat_dim,
            prefix_dropout=prefix_dropout,
            use_image_feat=use_image_feat,
            num_prefix_tokens=num_prefix_tokens,
        ).to(current_device)
    else:
        if rank == 0:
            print("[Stage1] 未启用 Vision Adapter，退化为纯文本 Qwen + LoRA")
        model = base_model

    return model, tokenizer, image_feat_dim