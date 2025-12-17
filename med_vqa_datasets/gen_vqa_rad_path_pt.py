# med_vqa_datasets/gen_vqa_rad_path_pt.py
# -*- coding: utf-8 -*-
"""
从 config_stage2.yaml 顶层字段读取 HF datasets cache 目录与 pt 输出路径，
离线提取 VQA-RAD / PathVQA 的 BiomedCLIP 图像特征，并分别保存到两个 .pt 文件中。

为什么必须离线？
- 你的 Stage-2 multi 之前在 collator 里跑 BiomedCLIP encode；
- 一旦 DataLoader num_workers>0，collate_fn 会在 worker 进程执行，极易触发 CUDA fork / 多份模型驻留 / OOM；
- 离线提特征后，训练时只需 CPU 读取 + torch.stack，不再触发任何 CUDA encode。

配置文件要求（config_stage2.yaml 顶层字段）：
    vqa_rad_cache: "/path/to/hf_cache/vqa-rad"
    path_vqa_cache: "/path/to/hf_cache/path-vqa"
    vqa_rad_pt_output: "/abs/path/to/cache/vqa_rad_biomedclip_imagefeat.pt"
    path_vqa_pt_output: "/abs/path/to/cache/path_vqa_biomedclip_imagefeat.pt"

运行示例（在项目根目录）：
    python med_vqa_datasets/gen_vqa_rad_path_pt.py \
        --config configs/config_stage2.yaml \
        --device cuda:0 \
        --batch_size 64

输出格式：
    torch.save({
        "meta": {...},
        "features": { "<sha1>": Tensor[D] (cpu, fp16/fp32), ... },
        "failed": [ {"split":..., "index":..., "error":...}, ... ]
    }, output_pt)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Any, Dict, List

import torch
from PIL import Image

# -------------------------
# 让脚本可直接运行（不要求 pip install -e）
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.config import load_config
from backbones.biomedclip_backbone import BiomedCLIPBackbone
from med_vqa_datasets.vqa_rad_path_hf import build_hf_vqa_dataset


# -------------------------
# 1) 文件/路径工具
# -------------------------
def ensure_parent_dir(file_path: str) -> None:
    """确保输出文件的父目录存在。"""
    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def require_cfg_key(cfg: Dict[str, Any], key: str) -> Any:
    """
    读取 config 顶层字段，缺失则报错（比 silent fallback 更安全）。
    你这次的需求就是“顶层字段直读”，所以严格一点更好。
    """
    if key not in cfg or cfg[key] in [None, ""]:
        raise KeyError(f"Missing required config key: '{key}' (top-level in yaml).")
    return cfg[key]


# -------------------------
# 2) 图像 key 设计（核心）
# -------------------------
def image_sha1_key(img: Image.Image) -> str:
    """
    用“图片内容”生成稳定 key：sha1(RGB像素 + size + mode)。

    优势：
    - HF datasets 解码出来往往是 PIL.Image，没有稳定文件路径；
    - 同一张图片在多个 QA 样本中重复出现时，可以自动去重；
    - 训练侧也可用同样方法从 PIL.Image 得到 key 查特征（纯 CPU，不涉及 CUDA）。

    注意（非常重要）：
    - key 依赖像素值；如果你在 dataset 层做了随机增强/裁剪，会改变 key；
      因此：不要在 dataset 里做随机增强；BiomedCLIP 的 preprocess 放在离线脚本里做。
    """
    if not isinstance(img, Image.Image):
        # 极端情况：如果 img 不是 PIL.Image（例如 numpy），尽量转成 PIL
        img = Image.fromarray(img)

    img = img.convert("RGB")
    raw = img.tobytes()

    h = hashlib.sha1()
    h.update(b"RGB")
    h.update(str(img.size).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


# -------------------------
# 3) 构建 BiomedCLIPBackbone（兼容不同 __init__ 签名）
# -------------------------
def build_biomedclip(cfg: Dict[str, Any], device: str) -> BiomedCLIPBackbone:
    """
    从 config 里尽量读取 biomedclip_model_dir（一般在 cfg['multi']['biomedclip_model_dir']），
    但不同工程版本 BiomedCLIPBackbone 的 __init__ 可能参数名不一致，所以这里做 try-fallback。
    """
    model_dir = None
    if isinstance(cfg.get("multi", None), dict):
        model_dir = cfg["multi"].get("biomedclip_model_dir", None)

    # 1) 先尝试带 model_dir
    if model_dir:
        try:
            return BiomedCLIPBackbone(model_dir=model_dir, device=device)
        except TypeError:
            # 参数名不同则回退
            pass

    # 2) 回退：只传 device
    return BiomedCLIPBackbone(device=device)


# -------------------------
# 4) 批量提特征
# -------------------------
@torch.no_grad()
def encode_batch(
    biomed_clip: BiomedCLIPBackbone,
    images: List[Image.Image],
    device: str,
    save_dtype: torch.dtype,
) -> torch.Tensor:
    """
    输入：PIL images list
    输出：(B, D) 的 cpu tensor（dtype=save_dtype）
    """
    if not hasattr(biomed_clip, "preprocess"):
        raise AttributeError(
            "BiomedCLIPBackbone 未暴露 preprocess。请检查 backbones/biomedclip_backbone.py"
        )

    preprocess = biomed_clip.preprocess

    # preprocess -> pixel_values: (B, 3, H, W)
    pixel_values = torch.stack([preprocess(im.convert("RGB")) for im in images], dim=0)
    pixel_values = pixel_values.to(device, non_blocking=True)

    # autocast：仅在 cuda 下启用（更省显存/更快）
    use_amp = device.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        feats = biomed_clip.encode_image(pixel_values)  # (B, D)

    feats = feats.detach().to("cpu").to(save_dtype)
    return feats


# -------------------------
# 5) 主流程：扫 splits、去重、提特征、保存 pt
# -------------------------
@torch.no_grad()
def extract_dataset_to_pt(
    cfg: Dict[str, Any],
    dataset_name: str,
    cache_dir: str,
    splits: List[str],
    output_pt: str,
    device: str,
    batch_size: int,
    fp16: bool,
) -> None:
    """
    对一个 HF 数据集（vqa-rad 或 path-vqa）全 split 提取“唯一图片”的特征并保存。
    """
    ensure_parent_dir(output_pt)

    save_dtype = torch.float16 if fp16 else torch.float32

    print(f"\n[PTGen] dataset={dataset_name}")
    print(f"[PTGen] cache_dir={cache_dir}")
    print(f"[PTGen] output_pt={output_pt}")
    print(f"[PTGen] device={device} | batch_size={batch_size} | save_dtype={save_dtype}")

    # 初始化 BiomedCLIP（只在主进程/GPU 中跑）
    biomed_clip = build_biomedclip(cfg, device=device)
    if hasattr(biomed_clip, "model"):
        biomed_clip.model.eval()

    features: Dict[str, torch.Tensor] = {}
    failed: List[Dict[str, Any]] = []

    # 统计信息（便于你 sanity check）
    total_samples_scanned = 0
    total_duplicates_skipped = 0

    # batch buffer（只放“新图片”）
    batch_imgs: List[Image.Image] = []
    batch_keys: List[str] = []

    def flush_batch() -> None:
        """把 batch buffer 里的图片 encode 并写入 features dict。"""
        nonlocal batch_imgs, batch_keys
        if not batch_imgs:
            return

        feats = encode_batch(
            biomed_clip=biomed_clip,
            images=batch_imgs,
            device=device,
            save_dtype=save_dtype,
        )  # (B, D)

        for k, f in zip(batch_keys, feats):
            features[k] = f

        batch_imgs = []
        batch_keys = []

    # 逐 split 扫描
    for split in splits:
        print(f"\n[PTGen] -> split='{split}'")
        try:
            ds = build_hf_vqa_dataset(
                dataset_name=dataset_name,
                split=split,
                cache_dir=cache_dir,
                max_samples=None,  # 全量
            )
        except Exception as e:
            # 比如 vqa-rad 没有 validation split，这里就跳过
            print(f"[PTGen] Skip split='{split}' (not available). Reason: {repr(e)}")
            continue

        print(f"[PTGen] split size: {len(ds)}")

        for i in range(len(ds)):
            total_samples_scanned += 1
            try:
                sample = ds[i]
                img = sample["image"]  # vqa_rad_path_hf.py 返回 PIL.Image（默认 image_transform=None）

                key = image_sha1_key(img)
                if key in features:
                    total_duplicates_skipped += 1
                    continue

                batch_imgs.append(img)
                batch_keys.append(key)

                if len(batch_imgs) >= batch_size:
                    flush_batch()

                # 轻量进度
                if (i + 1) % 2000 == 0:
                    print(f"[PTGen] split='{split}' processed {i+1}/{len(ds)} | unique={len(features)}")

            except Exception as e:
                failed.append({"split": split, "index": i, "error": repr(e)})

        # 每个 split 扫完就 flush 一次，便于及时落盘（也减少脚本中途失败的损失）
        flush_batch()
        print(f"[PTGen] split='{split}' done | unique so far: {len(features)}")

    if len(features) == 0:
        raise RuntimeError(
            f"[PTGen] No features extracted for dataset={dataset_name}. "
            f"Please check cache_dir='{cache_dir}' and dataset availability."
        )

    # meta 信息写全：便于你之后核对是否用对了模型/配置
    any_feat = next(iter(features.values()))
    meta = {
        "dataset_name": dataset_name,
        "cache_dir": cache_dir,
        "splits": splits,
        "feat_dim": int(any_feat.numel()),
        "dtype": str(save_dtype),
        "num_unique_images": len(features),
        "total_samples_scanned": total_samples_scanned,
        "duplicates_skipped": total_duplicates_skipped,
        "key_strategy": "sha1(RGB_pixels + size)",
        "backend": "BiomedCLIPBackbone",
        "biomedclip_model_dir": (cfg.get("multi", {}) or {}).get("biomedclip_model_dir", None),
        "note": "Training-side lookup must compute the same sha1 key from PIL image.",
    }

    payload = {
        "meta": meta,
        "features": features,
        "failed": failed,
    }

    torch.save(payload, output_pt)
    print("\n[PTGen] =========================")
    print(f"[PTGen] Saved: {output_pt}")
    print(f"[PTGen] Unique images: {len(features)} | Failed samples: {len(failed)}")
    print(f"[PTGen] Feature dim: {meta['feat_dim']} | dtype: {meta['dtype']}")
    print("[PTGen] =========================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config_stage2.yaml", help="config file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_fp16", action="store_true", help="Save features as fp32 instead of fp16.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ==========================
    # 直接读取 yaml 顶层字段（按你的要求）
    # ==========================
    vqa_rad_cache = require_cfg_key(cfg, "vqa_rad_cache")
    path_vqa_cache = require_cfg_key(cfg, "path_vqa_cache")
    vqa_rad_out = require_cfg_key(cfg, "vqa_rad_pt_output")
    path_vqa_out = require_cfg_key(cfg, "path_vqa_pt_output")

    print("====================START======================")
    print(
        f"vqa_rad_cache:{vqa_rad_cache}\n"
        f"path_vqa_cache:{path_vqa_cache}\n"
        f"vqa_rad_out:{vqa_rad_out}\n"
        f"path_vqa_out:{path_vqa_out}"
    )
    print("==================== END ======================")

    # VQA-RAD 通常只有 train/test；PathVQA 通常有 train/validation/test
    extract_dataset_to_pt(
        cfg=cfg,
        dataset_name="vqa-rad",
        cache_dir=vqa_rad_cache,
        splits=["train", "test", "validation"],   # validation 不存在会自动跳过
        output_pt=vqa_rad_out,
        device=args.device,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
    )

    extract_dataset_to_pt(
        cfg=cfg,
        dataset_name="path-vqa",
        cache_dir=path_vqa_cache,
        splits=["train", "validation", "test"],
        output_pt=path_vqa_out,
        device=args.device,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
