#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trainer_stage2_vqa.py

Stage-2 Med-VQA 训练脚本（基于 Stage-1 对齐结果继续微调）。

特性：
- 支持两种模态：
    1) text-only：只用 question/answer 文本，不使用图像特征
       （但模型结构仍是 Stage-1 的多模态结构，只是关闭图像分支）
    2) multi    ：使用 BiomedCLIP 提取图像特征 + Vision Adapter，
       在 VQA-RAD / PathVQA 上做多模态微调
- 使用与 Stage-1 相同的 DDP + HF Trainer 框架：
    - utils.ddp.launch_ddp / setup_ddp / cleanup_ddp
    - engine.builder.build_vision_llm 构建 Qwen + LoRA + Vision Adapter
- 从 Stage-1 的输出目录加载权重（包含 LoRA + Vision Adapter），在此基础上继续训练 Stage-2
"""

import os

os.environ["http_proxy"]="http://10.109.70.128:7897"
os.environ["https_proxy"]="http://10.109.70.128:7897"

import sys
from typing import Any, Dict

import torch

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from med_vqa_datasets.vqa_rad_path_hf import (
    build_hf_vqa_dataset)  # 调用 HF 封装
from med_vqa_datasets.collators import (
    VQATextDataCollator,
    VQAMultimodalDataCollator)


# 项目内模块
from utils.ddp import setup_ddp, cleanup_ddp, launch_ddp
from utils.config import load_config, get_gpus_and_world_size
from utils.builder import build_vision_llm
from backbones.biomedclip_backbone import BiomedCLIPBackbone  # 多模态模式需要


# ===========================
# 一些小工具
# ===========================

def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """小工具：从 dict 中安全取值，支持 key 不存在时返回默认值。"""
    return d[key] if key in d else default


def load_stage1_checkpoint(model: torch.nn.Module, ckpt_dir: str, device: torch.device, rank: int = 0) -> None:
    """
    从 Stage-1 输出目录加载权重，覆盖当前模型参数。

    - 优先尝试 model.safetensors（如果存在）
    - 否则尝试 pytorch_model.bin
    - 使用 strict=False 以兼容微小结构差异
    """
    if ckpt_dir is None or ckpt_dir == "":
        if rank == 0:
            print("[Stage2] 未提供 stage1_ckpt_dir，将从 base 模型重新开始训练。")
        return

    # 先尝试根目录下的权重文件
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    state_dict = None

    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                f"[Stage2] 检测到 {safetensors_path}，但未安装 safetensors。\n"
                f"请先安装: pip install safetensors"
            )
        if rank == 0:
            print(f"[Stage2] 从 Stage-1 safetensors 权重加载: {safetensors_path}")
        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(bin_path):
        if rank == 0:
            print(f"[Stage2] 从 Stage-1 PyTorch 权重加载: {bin_path}")
        state_dict = torch.load(bin_path, map_location=device)
    else:
        if rank == 0:
            print(
                f"[Stage2] 在 {ckpt_dir} 下未找到 model.safetensors 或 pytorch_model.bin，"
                f"将跳过 Stage-1 权重加载。"
            )
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[Stage2] 从 Stage-1 权重加载完成：missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing keys (前 5 个):", missing[:5])
        if unexpected:
            print("  unexpected keys (前 5 个):", unexpected[:5])

# ===========================
# Stage-2 训练主体（单进程）
# ===========================
def run_training(rank: int, world_size: int, config: Dict[str, Any]):
    """
    每个 GPU / rank 上的训练入口，由 utils.ddp.launch_ddp 启动。

    这里的 config 是整个 YAML 解析后的字典，结构类似：
        {
            "gpus": "...",
            "stage1_ckpt_dir": "...",
            "text": { ... },
            "multi": { ... },
            "modality": "text" 或 "multi"   # 由 main() 注入
        }
    """
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_warning()



    try:
        # 1) 初始化 DDP
        setup_ddp(rank, world_size)
        current_device = torch.cuda.current_device()

        modality = config.get("modality", "text").lower()
        assert modality in ("text", "multi"), f"[Stage2] 不支持的 modality: {modality}"

        stage1_ckpt_dir = config.get("stage1_ckpt_dir", None)
        stage2_cfg: Dict[str, Any] = config[modality]  # 只取 text 或 multi 这块子配置

        if rank == 0:
            print(f"[Stage2] 启动训练进程，Rank={rank}, World Size={world_size}, Modality={modality}")
            print(f"[Stage2] 使用 base 模型: {stage2_cfg['model_path']}")
            print(f"[Stage2] Stage-1 checkpoint 目录: {stage1_ckpt_dir}")
            print(f"[Stage2] 输出目录: {stage2_cfg['output_dir']}")

        # 2) 读取 Stage-2 训练相关配置
        OUTPUT_DIR = stage2_cfg["output_dir"]
        DATASET_NAME = stage2_cfg.get("dataset_name", "vqa-rad")
        TRAIN_SPLIT = stage2_cfg.get("train_split", "train")
        EVAL_SPLIT = stage2_cfg.get("eval_split", "test" if DATASET_NAME == "vqa-rad" else "validation")
        CACHE_DIR = stage2_cfg.get("cache_dir", None)

        MICRO_BATCH_SIZE = int(stage2_cfg.get("micro_batch_size", 4))
        EVAL_BATCH_SIZE = int(stage2_cfg.get("eval_batch_size", MICRO_BATCH_SIZE))
        GRAD_ACC_STEPS = int(stage2_cfg.get("gradient_accumulation_steps", 4))
        LEARNING_RATE = float(stage2_cfg.get("learning_rate", 2e-4))
        EPOCHS = float(stage2_cfg.get("epochs", 3.0))
        MAX_LENGTH = int(stage2_cfg.get("max_length", 512))
        NUM_WORKERS = int(stage2_cfg.get("num_workers", 4))
        MAX_TRAIN_SAMPLES = stage2_cfg.get("max_train_samples", None)
        MAX_EVAL_SAMPLES = stage2_cfg.get("max_eval_samples", None)
        USE_BF16 = bool(stage2_cfg.get("bf16", True))
        SEED = int(stage2_cfg.get("seed", 42))

        system_prompt = stage2_cfg.get("system_prompt", None)

        set_seed(SEED + rank)

        # 顶层 config 里的 HF 数据缓存路径
        VQA_RAD_CACHE = config.get("vqa_rad_cache", None)
        PATH_VQA_CACHE = config.get("path_vqa_cache", None)

        dataset_lower = DATASET_NAME.lower()
        if CACHE_DIR is not None:
            data_cache_dir = CACHE_DIR
        else:
            if dataset_lower == "vqa-rad":
                data_cache_dir = VQA_RAD_CACHE
            elif dataset_lower == "path-vqa":
                data_cache_dir = PATH_VQA_CACHE
            else:
                data_cache_dir = None


        # 3) 构建 Vision LLM 模型（结构与 Stage-1 一致）
        compute_dtype = torch.bfloat16 if USE_BF16 else torch.float32

        model, tokenizer, image_feat_dim = build_vision_llm(
            config=stage2_cfg,
            rank=rank,
            compute_dtype=compute_dtype,
        )

        # 4) 从 Stage-1 权重加载
        load_stage1_checkpoint(
            model=model,
            ckpt_dir=stage1_ckpt_dir,
            device=torch.device(f"cuda:{current_device}"),
            rank=rank,
        )
        # （可选）text-only 模式下冻结视觉 adapter 参数
        if modality == "text":
            if rank == 0:
                print("[Stage2] 文本-only 模式：冻结视觉 Adapter 参数，不再更新。")

            for name, param in model.named_parameters():
                if name.startswith("image_proj") or name.startswith("cross_attn") or name.startswith("gate"):
                    param.requires_grad = False

        # 5) 构建 Dataset & DataCollator
        if modality == "text":
            if rank == 0:
                print("[Stage2] 使用文本-only VQA 数据集（不加载图像特征，仅使用 question/answer 文本）。")

            train_dataset = build_hf_vqa_dataset(
                dataset_name=DATASET_NAME,
                split=TRAIN_SPLIT,
                cache_dir=data_cache_dir,
                max_samples=MAX_TRAIN_SAMPLES,
            )
            eval_dataset = build_hf_vqa_dataset(
                dataset_name=DATASET_NAME,
                split=EVAL_SPLIT,
                cache_dir=data_cache_dir,
                max_samples=MAX_EVAL_SAMPLES,
            )

            data_collator = VQATextDataCollator(
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
                system_prompt=system_prompt,
                add_image_hint=True,
            )
        else:  # modality == "multi"
            if rank == 0:
                print("[Stage2] 使用多模态 VQA 数据集（图像 + 文本）。")

            train_dataset = build_hf_vqa_dataset(
                dataset_name=DATASET_NAME,
                split=TRAIN_SPLIT,
                cache_dir=data_cache_dir,
                max_samples=MAX_TRAIN_SAMPLES,
            )
            eval_dataset = build_hf_vqa_dataset(
                dataset_name=DATASET_NAME,
                split=EVAL_SPLIT,
                cache_dir=data_cache_dir,
                max_samples=MAX_EVAL_SAMPLES,
            )

            biomed_clip = BiomedCLIPBackbone(
                model_dir=stage2_cfg["biomedclip_model_dir"],
                device=torch.device(f"cuda:{current_device}"),
            )

            data_collator = VQAMultimodalDataCollator(
                tokenizer=tokenizer,
                biomed_clip=biomed_clip,
                max_length=MAX_LENGTH,
                system_prompt=system_prompt,
            )

        if rank == 0:
            print(f"[Stage2][Dataset] 训练样本数: {len(train_dataset)}")
            print(f"[Stage2][Dataset] 验证样本数: {len(eval_dataset)}")

        # 6) 组装 TrainingArguments（与 Stage-1 保持一致的 DDP 配置）
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not USE_BF16,
            bf16=USE_BF16,
            logging_steps=10,
            save_strategy="epoch",
            # evaluation_strategy="epoch",

            # 关键：禁用 unused 参数检测，避免与 checkpoint 冲突
            ddp_find_unused_parameters=False,

            report_to="none",

            # 建议开启 gradient checkpointing，和 Stage-1 一致
            gradient_checkpointing=True,

            optim="paged_adamw_32bit",
            dataloader_num_workers=NUM_WORKERS,
            remove_unused_columns=False,  # 保留 image_feat 等字段
            local_rank=rank,
        )

        # 7) 构建 Trainer 并启动训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        metrics = trainer.evaluate()

        if rank == 0:
            print("[Stage2] Eval metrics:", metrics)
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"[Stage2] 训练完成，模型已保存至: {OUTPUT_DIR}")

        # 所有 evaluate 都结束后再清 DDP
        import torch.distributed as dist
        dist.barrier()
        cleanup_ddp()

    except Exception as e:
        print(f"[Stage2][Rank {rank}] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)


# ===========================
# main：加载配置 + 启动 DDP
# ===========================

def main():
    """
    主入口：

    1. 解析命令行参数（配置文件路径 + modality）
    2. 使用 utils.config.load_config 读取 YAML
    3. 解析 GPU 列表，设置 CUDA_VISIBLE_DEVICES
    4. 使用 utils.ddp.launch_ddp 启动 run_training
    """

    # 1) 加载配置
    cfg = load_config(r"../configs/config_stage2.yaml")

    # 2) 解析 GPU 列表和 world_size
    gpus, world_size = get_gpus_and_world_size(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"当前的cfg:{cfg}\n")

    # 4) 启动多进程训练
    launch_ddp(
        target_fn=run_training,
        world_size=world_size,
        config=cfg,
    )


if __name__ == "__main__":
    """
    快速测试示例（假设你已经写好 configs/config_stage2.yaml）：
    """
    main()
