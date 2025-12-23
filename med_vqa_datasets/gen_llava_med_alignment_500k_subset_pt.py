# -*- coding: utf-8 -*-
"""
gen_llava_med_alignment_500k_subset_pt.py  (768 tokens-only，中文注释版)
====================================================================

你要的版本特性（按你前面的要求做的“彻底转向 768 tokens-only”）：
1) 只输出 tokens（每张图一个 Tensor[T, 768]，通常 T=197，包含 CLS）。
2) 不支持 global(512)，不做 768->512 投影，不做 fp32 保存。
3) 保留少量真正需要的命令行参数：
   --overwrite / --print_every / --max_samples / --batch_size / --gpus

输出：
- 每条样本保存一个 pt：{llava_med_image_feature_dir}/{id}.pt
- pt 内容：torch.float16 的 Tensor[T, 768]

为什么你会看到“GPU 四卡时断时续”：
- 这个任务往往 I/O 受限：读图 + 预处理 + 写文件占比很高，GPU 会在 forward 时短暂忙，
  之后又空闲等 CPU/磁盘。进度打印能帮你确认程序在持续推进。

运行示例：
1) 小规模自测（推荐）：
   python med_vqa_datasets/gen_llava_med_alignment_500k_subset_pt.py --max_samples 2000 --print_every 200

2) 多卡抽特征（4卡）：
   python med_vqa_datasets/gen_llava_med_alignment_500k_subset_pt.py --gpus 0,1,2,3 --batch_size 64 --print_every 5000

3) 强制重算（已有文件也覆盖）：
   python med_vqa_datasets/gen_llava_med_alignment_500k_subset_pt.py --gpus 0,1,2,3 --batch_size 64 --print_every 5000 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image, ImageFile

# 一些大规模数据集里会有截断的 JPEG；打开这个开关可以减少读取失败
ImageFile.LOAD_TRUNCATED_IMAGES = True

from backbones.biomedclip_backbone import BiomedCLIPBackbone  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_indices(n: int, world_size: int) -> List[range]:
    """
    把 [0, n) 平均切成 world_size 份（连续区间），用于多进程分片。
    """
    base = n // world_size
    extra = n % world_size
    out: List[range] = []
    start = 0
    for r in range(world_size):
        length = base + (1 if r < extra else 0)
        out.append(range(start, start + length))
        start += length
    return out


def safe_open_image(path: str) -> Image.Image:
    """安全读图，并转成 RGB。"""
    with Image.open(path) as im:
        return im.convert("RGB")


def atomic_save(t: torch.Tensor, out_path: str, rank: int) -> None:
    """
    原子写文件：先写 tmp，再 rename（os.replace）。
    好处：中途被 kill 也不会留下半截 .pt。
    """
    tmp = out_path + f".tmp_rank{rank}"
    torch.save(t, tmp)
    os.replace(tmp, out_path)


def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    """
    初始化 DDP（NCCL）。
    注意：这里使用 rank 作为本进程的 cuda 设备索引（配合 CUDA_VISIBLE_DEVICES）。
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    """清理 DDP。"""
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def worker(rank: int, world_size: int, cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    每个 rank 负责自己的那一段样本：
    - 读图 -> preprocess -> batch -> BiomedCLIP encode_image_tokens -> 保存
    """
    if world_size > 1:
        setup_ddp(rank, world_size, args.master_addr, args.master_port)

    device = torch.device(f"cuda:{rank}")
    torch.backends.cuda.matmul.allow_tf32 = True  # 通常更快（Ampere+）

    # 从 config_stage1.yaml 读取必要路径
    image_root: str = cfg["llava_med_image_root"]
    feature_dir: str = cfg["llava_med_image_feature_dir"]
    biomedclip_dir: str = cfg["biomedclip_model_dir"]
    json_path: str = cfg["llava_med_json"]

    os.makedirs(feature_dir, exist_ok=True)

    # 每个 rank 自己读 json，避免 mp.spawn 传超大 list 发生 pickling 开销
    with open(json_path, "r", encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)

    # 可选：截断样本数，快速自测
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]

    # 给每个 rank 分配连续区间
    my_range = split_indices(len(samples), world_size)[rank] if world_size > 1 else range(len(samples))
    my_total = len(my_range)

    # 构建 BiomedCLIP backbone（冻结）
    backbone = BiomedCLIPBackbone(
        model_dir=str(biomedclip_dir),
        device=str(device),
        context_length=256,
        freeze_vision=True,
        freeze_text=True,
    )
    backbone.eval()
    preprocess = backbone.preprocess  # PIL -> Tensor[3,H,W]

    # rank0 打印全局信息
    if rank == 0:
        print(
            f"[LLaVA-PT 768 tokens-only] total={len(samples)} world_size={world_size} "
            f"batch_size={args.batch_size} overwrite={args.overwrite} feature_dir={feature_dir}",
            flush=True,
        )

    # 统计信息（rank-local）
    seen = 0          # 处理过的样本数（含跳过）
    saved = 0         # 本次新写入文件数
    skipped = 0       # 输出已存在而跳过
    missing = 0       # 图片文件缺失
    open_failed = 0   # 读图/预处理失败
    save_failed = 0   # 写文件失败

    # batch 缓存
    batch_imgs: List[torch.Tensor] = []
    batch_ids: List[str] = []

    t0 = time.time()

    def log_progress(force: bool = False) -> None:
        """
        进度打印：百分比 / 速度 / ETA / 统计计数。
        print_every=0 时不打印。
        """
        if not args.print_every:
            return
        if (not force) and (seen % args.print_every != 0):
            return

        dt = time.time() - t0
        speed = seen / max(dt, 1e-6)  # samples/s
        remain = max(my_total - seen, 0)
        eta_s = remain / max(speed, 1e-6)
        pct = 100.0 * seen / max(my_total, 1)

        print(
            f"[Rank {rank}] {pct:5.1f}%  seen={seen}/{my_total}  "
            f"saved={saved} skipped={skipped} missing={missing} "
            f"open_failed={open_failed} save_failed={save_failed}  "
            f"{speed:6.1f} samples/s  ETA={eta_s/60:6.1f} min",
            flush=True,
        )

    def flush() -> None:
        """
        对当前 batch 做一次前向并落盘：
        - pixel_values: [B, 3, H, W]
        - feats: [B, T, 768]（期望）
        - 每条样本保存一个 feats[j]: [T, 768]
        """
        nonlocal saved, save_failed

        if not batch_imgs:
            return

        # 堆叠并搬到 GPU
        pixel_values = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)

        # tokens-only：强制 fp16 autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            feats = backbone.encode_image_tokens(pixel_values, include_cls=True)

        # 保存到 cpu fp16
        feats = feats.detach().cpu().to(torch.float16)

        for j, sid in enumerate(batch_ids):
            out_path = os.path.join(feature_dir, f"{sid}.pt")

            # 同一个 id 可能重复出现：写之前再判断一次
            if (not args.overwrite) and os.path.exists(out_path):
                continue

            try:
                atomic_save(feats[j], out_path, rank)
                saved += 1
            except Exception:
                save_failed += 1
                # 尝试清理 tmp
                tmp = out_path + f".tmp_rank{rank}"
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

        batch_imgs.clear()
        batch_ids.clear()

    # 主循环：遍历本 rank 的样本
    for idx in my_range:
        s = samples[idx]
        sid = str(s.get("id"))
        img_name = s.get("image")

        seen += 1

        out_path = os.path.join(feature_dir, f"{sid}.pt")
        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            log_progress()
            continue

        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            missing += 1
            log_progress()
            continue

        try:
            img = safe_open_image(img_path)
            batch_imgs.append(preprocess(img))
            batch_ids.append(sid)
        except Exception:
            open_failed += 1
            log_progress()
            continue

        # batch 满了就跑一轮前向
        if len(batch_imgs) >= args.batch_size:
            flush()

        log_progress()

    # 处理最后不足一个 batch 的残留
    flush()
    log_progress(force=True)

    # 本 rank 结束总结
    dt = time.time() - t0
    print(
        f"[Rank {rank}] DONE  seen={seen} saved={saved} skipped={skipped} "
        f"missing={missing} open_failed={open_failed} save_failed={save_failed} time={dt/60:.1f} min",
        flush=True,
    )

    cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser()

    # 你要求保留的参数
    parser.add_argument("--gpus", type=str, default=None, help="例如 '0,1,2,3'（覆盖 config 里的 gpus）")
    parser.add_argument("--batch_size", type=int, default=64, help="每次 forward 的 batch_size")
    parser.add_argument("--max_samples", type=int, default=None, help="只跑前 N 条样本，用于快速自测")
    parser.add_argument("--overwrite", action="store_true", help="即使输出已存在也强制重算并覆盖")
    parser.add_argument("--print_every", type=int, default=5000, help="每处理多少条样本打印一次进度（0=不打印）")

    # DDP 通信参数（一般不需要改）
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=29999)

    args = parser.parse_args()

    # 固定读取 configs/config_stage1.yaml（项目内约定）
    cfg_path = "../configs/config_stage1.yaml"
    cfg = load_config(cfg_path)

    # 设置可见 GPU 列表（重要：DDP 里 rank 会对应 CUDA_VISIBLE_DEVICES 的索引）
    gpus = args.gpus if args.gpus is not None else str(cfg.get("gpus", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    gpu_list = [g for g in gpus.split(",") if g.strip()]
    world_size = max(len(gpu_list), 1)

    if world_size == 1:
        worker(rank=0, world_size=1, cfg=cfg, args=args)
    else:
        mp.spawn(worker, args=(world_size, cfg, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()