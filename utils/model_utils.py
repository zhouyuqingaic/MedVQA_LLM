# utils/model_utils.py
# -*- coding: utf-8 -*-
"""
model_utils.py
==============

封装三件事：
1) Tokenizer 加载（确保 pad_token 正确）
2) QLoRA 4-bit 量化配置（bitsandbytes）
3) 基座 LLM 加载 + 注入 LoRA（PEFT）

说明
----
- 本项目使用 HuggingFace Transformers + PEFT + bitsandbytes
- 为了可读性，函数粒度保持“能单测、好定位问题”
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn import Module

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(model_path: str, trust_remote_code: bool = True):
    """加载并配置 tokenizer，确保 pad_token 可用。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Qwen2.x 有时 pad_token 为空；为了 batch padding，通常把 pad 设为 eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_qlora_bnb_config(compute_dtype: torch.dtype = torch.bfloat16) -> BitsAndBytesConfig:
    """标准 QLoRA 4-bit 量化配置。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_qlora_base_model(
    *,
    model_path: str,
    lora_cfg: Dict[str, Any],
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> Module:
    """
    加载基座 LLM（4bit 量化）并注入 LoRA Adapter。

    返回：
      - PeftModel（内部持有被量化的 base model）
    """
    bnb_config = get_qlora_bnb_config(compute_dtype=compute_dtype)

    # Transformers 推荐用 torch_dtype，而不是 dtype（dtype 有时会被忽略）
    common_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        quantization_config=bnb_config,
        device_map={"": str(device)},
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        **common_kwargs,
        dtype=compute_dtype,  # 新版
    )

    # k-bit 训练准备：冻结/转换部分模块，避免精度/梯度问题
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", []),
    )

    return get_peft_model(base_model, peft_config)


def print_trainable_parameters(model: Module, rank: int = 0) -> None:
    """打印可训练参数统计（只在 rank0 打印）。"""
    if rank != 0:
        return

    # PEFT 模型自带更详细的 print_trainable_parameters
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")
