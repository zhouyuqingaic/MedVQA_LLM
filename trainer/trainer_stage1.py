# trainer/trainer_stage1.py
# -*- coding: utf-8 -*-
"""
Stage-1 Trainer（LLaVA-Med alignment, tokens-route）
================================================

单一路线（强一致，方便排错）：
  - Dataset: LLaVAMedAlignmentDataset（离线 tokens + 原始 chat）
  - Collator: LlavaMedChatCollator（answer-only labels + tokens padding）
  - Model: QwenVisionPatchPrefixModel（逐 token projector + prefix concat）

运行：
  python trainer/trainer_stage1.py --config configs/config_stage1.yaml
"""

from __future__ import annotations

import inspect
import argparse
import os
import sys
from typing import Any, Dict

import torch
from transformers import Trainer, TrainingArguments

from utils.config import get_gpus_and_world_size, load_config
from utils.ddp import cleanup_ddp, launch_ddp, setup_ddp
from utils.builder import build_vision_llm

from med_vqa_datasets.dataset_llava_med_alignment_500k_subset import build_llava_med_dataset
from med_vqa_datasets.collators import LlavaMedChatCollator


def _build_trainer(**kwargs):
    """
    transformers 新版本引入 processing_class 并逐步弃用 tokenizer；
    但不能同时传 tokenizer 和 processing_class，否则会 ValueError。
    这里做一个兼容层：优先用 processing_class（如果 Trainer 支持），否则回退到 tokenizer。
    """
    sig = inspect.signature(Trainer.__init__)

    if "processing_class" in sig.parameters:
        # ✅ 新版：只传 processing_class，不要再让 kwargs 里残留 tokenizer
        proc = kwargs.pop("processing_class", None)
        tok = kwargs.pop("tokenizer", None)
        if proc is None:
            proc = tok
        # 确保 tokenizer 不会再被传进去
        kwargs.pop("tokenizer", None)
        return Trainer(**kwargs, processing_class=proc)

    # ✅ 旧版：Trainer 还没有 processing_class，只能传 tokenizer
    kwargs.pop("processing_class", None)
    return Trainer(**kwargs)


def _resolve_precision(cfg: Dict[str, Any]) -> tuple[bool, bool, torch.dtype]:
    """
    统一处理 bf16/fp16/fp32：
    - 优先使用显式 fp16/bf16
    - 若都没开，则走 fp32（更稳但更慢）
    """
    use_bf16 = bool(cfg.get("bf16", True))
    use_fp16 = bool(cfg.get("fp16", False))

    if use_bf16 and use_fp16:
        raise ValueError("[Stage1] bf16 and fp16 cannot both be True.")

    if use_bf16:
        return True, False, torch.bfloat16
    if use_fp16:
        return False, True, torch.float16
    return False, False, torch.float32


def run_training(rank: int, world_size: int, config: Dict[str, Any]) -> None:
    try:
        setup_ddp(rank, world_size)

        if rank == 0:
            print(f"[Stage1] Rank={rank} | world_size={world_size}")
            print(f"[Stage1] model_path: {config['model_path']}")
            print(f"[Stage1] output_dir: {config['output_dir']}")

        # ---- training hyperparams ----
        output_dir = config["output_dir"]
        micro_bs = int(config.get("micro_batch_size", 4))
        grad_acc = int(config.get("gradient_accumulation_steps", 4))
        lr = float(config.get("learning_rate", 2e-4))
        epochs = float(config.get("epochs", 1))
        max_length = int(config.get("max_length", 1024))
        num_workers = int(config.get("num_workers", 4))

        use_bf16, use_fp16, compute_dtype = _resolve_precision(config)

        # ---- model + tokenizer ----
        model, tokenizer, _image_token_dim = build_vision_llm(
            config=config,
            rank=rank,
            compute_dtype=compute_dtype,
        )

        # ---- dataset ----
        if rank == 0:
            print("[Stage1] Building dataset...")
        train_dataset = build_llava_med_dataset(config, split="train", verbose=(rank == 0))

        if rank == 0:
            print(f"[Stage1] Train samples: {len(train_dataset)}")

        # ---- collator ----
        data_collator = LlavaMedChatCollator(tokenizer=tokenizer, max_length=max_length)

        # ---- HF TrainingArguments ----
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            gradient_checkpointing = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            optim="paged_adamw_32bit",
            dataloader_num_workers=num_workers,
            remove_unused_columns=False,  # 必须保留 image_tokens / image_token_mask
            ddp_find_unused_parameters=False,
            local_rank=rank,
        )

        trainer = _build_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,  # _build_trainer 会兼容新/旧版
        )

        trainer.train()

        if rank == 0:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[Stage1] Done. Saved to: {output_dir}")

        cleanup_ddp()

    except Exception as e:
        print(f"[Stage1][Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config_stage1.yaml", help="Path to config_stage1.yaml")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_cfg = os.path.join(project_root, "configs", "config_stage1.yaml")
    cfg_path = args.config or default_cfg

    config = load_config(cfg_path)

    gpus, world_size = get_gpus_and_world_size(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"[Stage1] GPUs={gpus} | world_size={world_size}")
    launch_ddp(run_training, world_size=world_size, config=config)


if __name__ == "__main__":
    main()
