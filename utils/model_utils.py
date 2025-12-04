# utils/model_utils.py
# -*- coding: utf-8 -*-

"""
职责： 封装大语言模型 (LLM) 的加载、量化 (QLoRA) 和 Peft/LoRA 适配器配置逻辑。
"""

import torch
from typing import Dict, Any, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from torch.nn import Module


def load_tokenizer(model_path: str, trust_remote_code: bool = True):
    """
    加载并配置 Tokenizer。
    确保 pad_token 被正确设置。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_qlora_bnb_config(compute_dtype=torch.bfloat16) -> BitsAndBytesConfig:
    """
    生成标准的 QLoRA 4-bit 量化配置。
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_qlora_base_model(
        model_path: str,
        lora_cfg: Dict[str, Any],
        device: torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
) -> Module:
    """
    加载基座 LLM (4bit 量化) 并注入 LoRA Adapter。

    参数
    ----
    model_path : str
        预训练 LLM 的路径。
    lora_cfg : dict
        从 config 中读取的 LoRA 参数字典 (r, alpha, dropout, target_modules)。
    device : torch.device
        模型要加载到的目标设备。
    compute_dtype : torch.dtype
        计算时使用的 dtype (如 torch.bfloat16)。

    返回
    ----
    model : PeftModel
        带有 LoRA Adapter 的 QLoRA 模型。
    """
    # 1. 量化配置
    bnb_config = get_qlora_bnb_config(compute_dtype=compute_dtype)

    # 2. 加载基础 LLM
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device},  # 固定到当前进程的 GPU
        trust_remote_code=trust_remote_code,
        dtype=compute_dtype,
    )

    # 3. 准备 k-bit 训练 (冻结/cast 一些模块)
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. 注入 LoRA Adapter 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg["target_modules"],
    )

    # 5. 注入 LoRA Adapter
    model = get_peft_model(base_model, peft_config)
    return model


# 如果您希望在 utils/model_utils.py 中添加一个打印可训练参数的辅助函数：
def print_trainable_parameters(model: Module, rank: int = 0):
    """
    打印模型可训练参数的统计信息（仅在 Rank 0 打印）。
    """
    if rank == 0 and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    elif rank == 0:
        # 针对非 PeftModel 的普通 nn.Module 计算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("LoRA trainable 参数统计 (非PeftModel计算)：")
        print(
            f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_params / total_params * 100:.4f}")