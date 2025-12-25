# trainer/trainer_stage2_vqa.py
# -*- coding: utf-8 -*-
"""
Stage-2 Trainer（VQA-RAD / PathVQA）
================================

设计目标：
- 继续在 Stage-1 checkpoint 基础上微调（LoRA + projector）
- 统一使用 tokens-route（离线 image patch tokens + patch_prefix 模型）

支持两种模式：
- text : 纯文本（保持结构一致，便于加载 stage1 权重；但 forward 不喂 image_tokens）
- multi: 多模态（从 .pt tokens cache 查表，prefix 拼接）

运行：
  python trainer/trainer_stage2_vqa.py --config configs/config_stage2.yaml --modality multi
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments, set_seed

from med_vqa_datasets.vqa_rad_path_hf import build_hf_vqa_dataset
from med_vqa_datasets.collators import VQATextDataCollator, VQAMultimodalPtTokensCollator

from utils.builder import build_vision_llm
from utils.config import get_gpus_and_world_size, load_config
from utils.ddp import cleanup_ddp, launch_ddp, setup_ddp


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
    use_bf16 = bool(cfg.get("bf16", True))
    use_fp16 = bool(cfg.get("fp16", False))
    if use_bf16 and use_fp16:
        raise ValueError("[Stage2] bf16 and fp16 cannot both be True.")
    if use_bf16:
        return True, False, torch.bfloat16
    if use_fp16:
        return False, True, torch.float16
    return False, False, torch.float32


def _load_stage1_checkpoint(model: torch.nn.Module, ckpt_dir: str, device: torch.device, rank: int) -> None:
    """从 Stage-1 输出目录加载权重（model.safetensors / pytorch_model.bin）。"""
    if not ckpt_dir:
        if rank == 0:
            print("[Stage2] stage1_ckpt_dir is empty. Train from base model.")
        return

    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    state_dict = None
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("[Stage2] Please install safetensors: pip install safetensors")
        if rank == 0:
            print(f"[Stage2] Loading Stage-1 weights: {safetensors_path}")
        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(bin_path):
        if rank == 0:
            print(f"[Stage2] Loading Stage-1 weights: {bin_path}")
        state_dict = torch.load(bin_path, map_location=device)
    else:
        if rank == 0:
            print(f"[Stage2] No weights found in {ckpt_dir}. Skip loading.")
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[Stage2] Loaded Stage-1 weights. missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing (first 5):", missing[:5])
        if unexpected:
            print("  unexpected (first 5):", unexpected[:5])


def _select_cache_dir(cfg_top: Dict[str, Any], dataset_name: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    name = dataset_name.lower()
    if name == "vqa-rad":
        return cfg_top.get("vqa_rad_cache")
    if name == "path-vqa":
        return cfg_top.get("path_vqa_cache")
    return None


def _select_pt_tokens(cfg_top: Dict[str, Any], dataset_name: str) -> str:
    name = dataset_name.lower()
    if name == "vqa-rad":
        return cfg_top["vqa_rad_pt_output"]
    if name == "path-vqa":
        return cfg_top["path_vqa_pt_output"]
    raise ValueError(f"[Stage2] Unknown dataset_name={dataset_name!r}. Expected 'vqa-rad' or 'path-vqa'.")


def run_training(rank: int, world_size: int, cfg_top: Dict[str, Any]) -> None:
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

        modality = str(cfg_top.get("modality", "text")).lower()
        if modality not in ("text", "multi"):
            raise ValueError(f"[Stage2] modality must be 'text' or 'multi', got {modality!r}")

        stage1_ckpt_dir = cfg_top.get("stage1_ckpt_dir", "")
        stage2_cfg: Dict[str, Any] = cfg_top[modality]

        if rank == 0:
            print(f"[Stage2] Rank={rank} | world_size={world_size} | modality={modality}")
            print(f"[Stage2] base model: {stage2_cfg['model_path']}")
            print(f"[Stage2] stage1_ckpt_dir: {stage1_ckpt_dir}")
            print(f"[Stage2] output_dir: {stage2_cfg['output_dir']}")

        # ---- hyperparams ----
        output_dir = stage2_cfg["output_dir"]
        dataset_name = stage2_cfg.get("dataset_name", "vqa-rad")
        train_split = stage2_cfg.get("train_split", "train")
        eval_split = stage2_cfg.get("eval_split", "test")
        cache_dir_override = stage2_cfg.get("cache_dir", None)

        micro_bs = int(stage2_cfg.get("micro_batch_size", 4))
        eval_bs = int(stage2_cfg.get("eval_batch_size", micro_bs))
        grad_acc = int(stage2_cfg.get("gradient_accumulation_steps", 4))
        lr = float(stage2_cfg.get("learning_rate", 2e-4))
        epochs = float(stage2_cfg.get("epochs", 3))
        max_length = int(stage2_cfg.get("max_length", 512))
        num_workers = int(stage2_cfg.get("num_workers", 4))
        seed = int(stage2_cfg.get("seed", 42))

        max_train_samples = stage2_cfg.get("max_train_samples", None)
        max_eval_samples = stage2_cfg.get("max_eval_samples", None)

        system_prompt = stage2_cfg.get("system_prompt", "")

        use_bf16, use_fp16, compute_dtype = _resolve_precision(stage2_cfg)

        set_seed(seed + rank)

        # ---- dataset cache dir ----
        data_cache_dir = _select_cache_dir(cfg_top, dataset_name, cache_dir_override)

        # ---- model + tokenizer ----
        model, tokenizer, _image_token_dim = build_vision_llm(
            config=stage2_cfg,
            rank=rank,
            compute_dtype=compute_dtype,
        )

        # ---- load stage1 weights ----
        _load_stage1_checkpoint(model, stage1_ckpt_dir, device=device, rank=rank)

        # text 模式：冻结视觉 adapter（projector / ln）
        if modality == "text":
            if rank == 0:
                print("[Stage2] text-only: freeze vision adapter params (projector/ln).")
            for name, p in model.named_parameters():
                if name.startswith("projector") or name.startswith("ln"):
                    p.requires_grad = False

        # ---- build dataset ----
        train_dataset = build_hf_vqa_dataset(
            dataset_name=dataset_name,
            split=train_split,
            cache_dir=data_cache_dir,
            max_samples=max_train_samples,
        )
        eval_dataset = build_hf_vqa_dataset(
            dataset_name=dataset_name,
            split=eval_split,
            cache_dir=data_cache_dir,
            max_samples=max_eval_samples,
        )

        if rank == 0:
            print(f"[Stage2] Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

        # ---- collator ----
        text_collator = VQATextDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
            add_image_hint=(modality == "text"),
        )

        if modality == "text":
            data_collator = text_collator
        else:
            pt_tokens_path = _select_pt_tokens(cfg_top, dataset_name)
            data_collator = VQAMultimodalPtTokensCollator(
                text_collator=text_collator,
                image_tokens_pt_path=pt_tokens_path,
                strict=True,
            )

        # ---- training args ----
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            num_train_epochs=epochs,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            report_to="none",
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            dataloader_num_workers=num_workers,
            remove_unused_columns=False,  # collator 需要用到 image 字段
            ddp_find_unused_parameters=False,
            local_rank=rank,
        )

        trainer = _build_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()
        metrics = trainer.evaluate()

        if rank == 0:
            print("[Stage2] Eval metrics:", metrics)
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[Stage2] Done. Saved to: {output_dir}")

        # sync then cleanup
        import torch.distributed as dist

        dist.barrier()
        cleanup_ddp()

    except Exception as e:
        print(f"[Stage2][Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config_stage2.yaml")
    parser.add_argument("--modality", type=str, default=None, choices=["text", "multi"], help="Override cfg.modality")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_cfg = os.path.join(project_root, "configs", "config_stage2.yaml")
    cfg_path = args.config or default_cfg

    cfg = load_config(cfg_path)
    if args.modality:
        cfg["modality"] = args.modality

    gpus, world_size = get_gpus_and_world_size(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"[Stage2] GPUs={gpus} | world_size={world_size} | modality={cfg.get('modality')}")
    launch_ddp(run_training, world_size=world_size, config=cfg)


if __name__ == "__main__":
    main()
