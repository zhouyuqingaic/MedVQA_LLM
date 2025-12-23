# gen_vqa_rad_path_pt.py
# -*- coding: utf-8 -*-
"""
med_vqa_datasets/gen_vqa_rad_path_pt.py
======================================

离线提取 VQA-RAD / PathVQA 的 BiomedCLIP 图像特征并保存为 .pt（图像缓存）。

本脚本已严格对齐你当前的 `backbones/biomedclip_backbone.py` 设计：

- encode_image(...) -> 全局向量 [B, embed_dim]（通常 512）
- encode_image_tokens(...) -> raw tokens [B, T, width]（例如 768），默认不做 visual.proj

输出：
torch.save({
  "meta": {...},
  "features": {"<sha1>": Tensor[T,C] 或 Tensor[D]},
  "failed": [...]
}, output_pt)
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.config import load_config
from backbones.biomedclip_backbone import BiomedCLIPBackbone
from med_vqa_datasets.vqa_rad_path_hf import build_hf_vqa_dataset


def ensure_parent_dir(file_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def require_cfg_key(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise KeyError(f"Missing required config key: '{key}' (top-level in yaml).")
    return cfg[key]


def get_biomedclip_model_dir(cfg: Dict[str, Any]) -> str:
    model_dir = None
    if isinstance(cfg.get("multi"), dict):
        model_dir = cfg["multi"].get("biomedclip_model_dir")
    if not model_dir:
        model_dir = cfg.get("biomedclip_model_dir")
    if not model_dir:
        raise KeyError(
            "Cannot find biomedclip_model_dir in config. "
            "Please set cfg['multi']['biomedclip_model_dir'] or top-level cfg['biomedclip_model_dir']."
        )
    return str(model_dir)


def image_sha1_key(img: Image.Image | Any) -> str:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    raw = img.tobytes()
    h = hashlib.sha1()
    h.update(b"RGB")
    h.update(str(img.size).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


def build_biomedclip(cfg: Dict[str, Any], device: str) -> BiomedCLIPBackbone:
    model_dir = get_biomedclip_model_dir(cfg)
    backbone = BiomedCLIPBackbone(
        model_dir=model_dir,
        device=device,
        context_length=256,
        freeze_vision=True,
        freeze_text=True,
    )
    backbone.eval()
    return backbone


@torch.no_grad()
def encode_batch(
    biomed_clip: BiomedCLIPBackbone,
    images: List[Image.Image],
    *,
    device: str,
    save_dtype: torch.dtype,
    feature_type: str,
    include_cls: bool,
) -> torch.Tensor:
    preprocess = biomed_clip.preprocess
    pixel_values = torch.stack([preprocess(im.convert("RGB")) for im in images], dim=0)
    pixel_values = pixel_values.to(device, non_blocking=True)

    use_amp = str(device).startswith("cuda") and (save_dtype == torch.float16)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        if feature_type == "global":
            feats = biomed_clip.encode_image(pixel_values)  # [B, 512]
        elif feature_type == "tokens":
            feats = biomed_clip.encode_image_tokens(pixel_values, include_cls=include_cls)  # [B, T, width]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    return feats.detach().to("cpu").to(save_dtype)


def _infer_patch_grid(num_tokens: int, include_cls: bool) -> Optional[Tuple[int, int]]:
    patch_tokens = num_tokens - (1 if include_cls else 0)
    if patch_tokens <= 0:
        return None
    g = int(math.sqrt(patch_tokens))
    return (g, g) if g * g == patch_tokens else None


@torch.no_grad()
def extract_dataset_to_pt(
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
    cache_dir: str,
    splits: List[str],
    output_pt: str,
    device: str,
    batch_size: int,
    fp16: bool,
    feature_type: str,
    include_cls: bool,
    max_samples: Optional[int] = None,
) -> None:
    ensure_parent_dir(output_pt)
    save_dtype = torch.float16 if fp16 else torch.float32

    print(f"\n[PTGen] dataset={dataset_name} | feature_type={feature_type} | include_cls={include_cls}")
    biomed_clip = build_biomedclip(cfg, device=device)

    features: Dict[str, torch.Tensor] = {}
    failed: List[Dict[str, Any]] = []

    total_samples_scanned = 0
    total_duplicates_skipped = 0

    batch_imgs: List[Image.Image] = []
    batch_keys: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_imgs, batch_keys
        if not batch_imgs:
            return
        feats = encode_batch(
            biomed_clip=biomed_clip,
            images=batch_imgs,
            device=device,
            save_dtype=save_dtype,
            feature_type=feature_type,
            include_cls=include_cls,
        )
        for k, f in zip(batch_keys, feats):
            features[k] = f
        batch_imgs = []
        batch_keys = []

    for split in splits:
        print(f"\n[PTGen] -> split='{split}'")
        try:
            ds = build_hf_vqa_dataset(
                dataset_name=dataset_name,
                split=split,
                cache_dir=cache_dir,
                max_samples=max_samples,
            )
        except Exception as e:
            print(f"[PTGen] Skip split='{split}' (not available). Reason: {repr(e)}")
            continue

        print(f"[PTGen] split size: {len(ds)}")

        for i in range(len(ds)):
            total_samples_scanned += 1
            try:
                sample = ds[i]
                img = sample["image"]

                key = image_sha1_key(img)
                if key in features:
                    total_duplicates_skipped += 1
                    continue

                batch_imgs.append(img)
                batch_keys.append(key)

                if len(batch_imgs) >= batch_size:
                    flush_batch()

                if (i + 1) % 2000 == 0:
                    print(f"[PTGen] split='{split}' processed {i+1}/{len(ds)} | unique={len(features)}")

            except Exception as e:
                failed.append({"split": split, "index": i, "error": repr(e)})

        flush_batch()
        print(f"[PTGen] split='{split}' done | unique so far: {len(features)}")

    if not features:
        raise RuntimeError(f"[PTGen] No features extracted for dataset={dataset_name}.")

    any_feat = next(iter(features.values()))
    meta: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "cache_dir": cache_dir,
        "splits": splits,
        "feature_type": feature_type,
        "dtype": str(save_dtype),
        "num_unique_images": len(features),
        "total_samples_scanned": total_samples_scanned,
        "duplicates_skipped": total_duplicates_skipped,
        "key_strategy": "sha1(RGB_pixels + size)",
        "backend": "BiomedCLIPBackbone",
        "biomedclip_model_dir": get_biomedclip_model_dir(cfg),
        "include_cls": bool(include_cls),
    }
    if feature_type == "global":
        meta["feat_dim"] = int(any_feat.numel())
    else:
        meta["num_tokens"] = int(any_feat.size(0))
        meta["token_dim"] = int(any_feat.size(1))
        meta["patch_grid"] = _infer_patch_grid(meta["num_tokens"], include_cls)

    torch.save({"meta": meta, "features": features, "failed": failed}, output_pt)
    print(f"[PTGen] Saved: {output_pt} | unique={len(features)} | failed={len(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config_stage2.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_fp16", action="store_true", help="save fp32")
    parser.add_argument("--feature_type", type=str, default="tokens", choices=["tokens", "global"])
    parser.add_argument("--no_cls", action="store_true", help="tokens mode: drop CLS")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vqa_rad_cache = require_cfg_key(cfg, "vqa_rad_cache")
    path_vqa_cache = require_cfg_key(cfg, "path_vqa_cache")
    vqa_rad_out = require_cfg_key(cfg, "vqa_rad_pt_output")
    path_vqa_out = require_cfg_key(cfg, "path_vqa_pt_output")

    extract_dataset_to_pt(
        cfg,
        dataset_name="vqa-rad",
        cache_dir=vqa_rad_cache,
        splits=["train", "test", "validation"],
        output_pt=vqa_rad_out,
        device=args.device,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
        feature_type=args.feature_type,
        include_cls=not args.no_cls,
        max_samples=args.max_samples,
    )
    extract_dataset_to_pt(
        cfg,
        dataset_name="path-vqa",
        cache_dir=path_vqa_cache,
        splits=["train", "test", "validation"],
        output_pt=path_vqa_out,
        device=args.device,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
        feature_type=args.feature_type,
        include_cls=not args.no_cls,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
