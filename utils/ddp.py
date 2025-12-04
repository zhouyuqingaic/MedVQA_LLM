# utils/ddp.py
# -*- coding: utf-8 -*-
"""
职责： 统一 DDP (Distributed Data Parallel) 环境的初始化、清理和多进程启动逻辑
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, Any, Dict, List

# 为了避免 NCCL 在某些多机网络环境出现问题，这里做了一些保守设置
# 所有使用 DDP 的脚本都应该在最开始导入这个文件
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_ddp(rank: int, world_size: int, master_port: int = 65534):
    """
    初始化分布式训练环境 (DDP).

    参数
    ----
    rank : int
        当前进程的全局 Rank。
    world_size : int
        总进程数。
    master_port : int
        MASTER_PORT，用于不同进程间的通信。
    """
    # 自动设置环境变量，供 torch.distributed / HF Trainer 识别
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))

    # 初始化进程组
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # 绑定当前进程到对应的 GPU
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理分布式环境."""
    if dist.is_initialized():
        dist.destroy_process_group()


def launch_ddp(target_fn: Callable[[int, int, Dict[str, Any]], Any],
               world_size: int,
               config: Dict[str, Any]):
    """
    使用 torch.multiprocessing.spawn 启动多进程训练。

    参数
    ----
    target_fn : Callable
        要在每个 GPU 进程上运行的函数 (通常是 run_training)。
        签名应为 `target_fn(rank, world_size, config)`。
    world_size : int
        总进程数（即 GPU 数量）。
    config : dict
        传递给 target_fn 的配置字典。
    """
    print(f"[DDP Launch] 使用 {world_size} 个进程启动任务...")

    # mp.spawn 启动 world_size 个进程，每个进程调用 target_fn
    mp.spawn(
        target_fn,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )