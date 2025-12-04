import os
import sys
import json
import math
import yaml
import time
from typing import List, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

# 根据你的项目结构调整导入路径
from backbones.biomedclip_backbone import BiomedCLIPBackbone

# =========================================================
# 1. 全局环境设置 (防止多卡通信死锁，复用 train_qwen 的经验)
# =========================================================
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path: str) -> dict:
    """读取 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_ddp(rank, world_size):
    """初始化分布式环境 (主要用于同步 barrier，非必须但推荐)"""
    os.environ["MASTER_ADDR"] = "localhost"
    # 使用与训练脚本不同的端口，防止冲突
    os.environ["MASTER_PORT"] = "29501"

    # 即使只是推理，初始化 process group 也能让我们用 barrier() 等待所有卡结束
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def split_indices(num_items: int, world_size: int) -> List[range]:
    """将数据索引均匀分配给各个 Rank"""
    indices_per_rank = []
    base = num_items // world_size
    extra = num_items % world_size

    start = 0
    for rank in range(world_size):
        length = base + (1 if rank < extra else 0)
        end = start + length
        indices_per_rank.append(range(start, end))
        start = end
    return indices_per_rank


def worker(rank, world_size, config, samples):
    """
    每个 GPU 进程的工作逻辑
    """
    try:
        # 1. 初始化 DDP 环境
        setup_ddp(rank, world_size)

        # 获取当前逻辑设备 (由于 main 设置了 CUDA_VISIBLE_DEVICES，这里总是 cuda:rank)
        # 注意：在 set_device 后，current_device 实际上就是 rank
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            print(f"[Rank {rank}] 进程启动，开始加载模型...", flush=True)

        # 2. 解析配置
        image_root = config["llava_med_image_root"]
        feature_dir = config["llava_med_image_feature_dir"]
        biomedclip_dir = config["biomedclip_model_dir"]

        batch_size = int(config.get("feature_batch_size", 64))
        use_fp16 = bool(config.get("feature_fp16", True))

        # 确保输出目录存在 (多进程这种写法可能会有竞争，但 exist_ok=True 没问题)
        os.makedirs(feature_dir, exist_ok=True)

        # 3. 加载模型 (BiomedCLIP)
        # 注意：这里我们不需要用 DDP(model) 包装，因为只是推理，每张卡独立跑即可
        backbone = BiomedCLIPBackbone(
            model_dir=biomedclip_dir,
            device=str(device),  # 传入 "cuda:0" 等字符串
            context_length=256,
            freeze_vision=True,
            freeze_text=True,
        )
        preprocess = backbone.preprocess

        if rank == 0:
            print(f"[Rank {rank}] 模型加载完成。", flush=True)

        # 4. 分配任务
        num_items = len(samples)
        my_range = split_indices(num_items, world_size)[rank]

        print(f"[Rank {rank}] 负责处理: {len(my_range)} 条样本 (Index: {my_range.start} -> {my_range.stop})",
              flush=True)

        # 5. 推理循环
        batch_images = []
        batch_ids = []

        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        def flush_batch():
            """内部函数：执行一个 Batch 的推理和保存"""
            if not batch_images:
                return

            # Stack images: [B, 3, H, W]
            pixel_values = torch.stack(batch_images, dim=0).to(device, non_blocking=True)

            with torch.no_grad():
                if use_fp16:
                    pixel_values = pixel_values.half()  # 转 FP16

                # BiomedCLIP encode_image
                img_feats = backbone.encode_image(pixel_values)  # [B, 512]

            # 移回 CPU 并保存
            img_feats = img_feats.detach().cpu()

            for idx_in_batch, s_id in enumerate(batch_ids):
                feat = img_feats[idx_in_batch]
                out_path = os.path.join(feature_dir, f"{s_id}.pt")
                # 再次检查 (防止极端的覆盖写入，虽然不太可能)
                if not os.path.exists(out_path):
                    torch.save(feat, out_path)

            batch_images.clear()
            batch_ids.clear()

        # 开始遍历分配给我的数据
        for i in my_range:
            sample = samples[i]
            sample_id = sample["id"]
            image_name = sample["image"]

            out_path = os.path.join(feature_dir, f"{sample_id}.pt")

            # 断点续传检查
            if os.path.exists(out_path):
                skipped_count += 1
                continue

            img_path = os.path.join(image_root, image_name)

            if not os.path.exists(img_path):
                # 只有 Rank 0 打印详细警告，防止刷屏，或者记录日志
                if rank == 0:
                    print(f"Warning: 图片不存在 {img_path}")
                continue

            try:
                # 读取并预处理
                image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image)  # [3, 224, 224]

                batch_images.append(img_tensor)
                batch_ids.append(sample_id)

                # 凑够 Batch 就跑一次
                if len(batch_images) >= batch_size:
                    flush_batch()
                    processed_count += batch_size

                    # 打印进度 (每 100 个 batch 打印一次)
                    if processed_count % (batch_size * 50) == 0:
                        elapsed = time.time() - start_time
                        speed = processed_count / elapsed
                        print(f"[Rank {rank}] 进度: {processed_count}/{len(my_range)} | Speed: {speed:.1f} img/s",
                              flush=True)

            except Exception as e:
                print(f"[Rank {rank}] Error processing {img_path}: {e}")
                continue

        # 处理剩下的尾巴
        flush_batch()

        total_time = time.time() - start_time
        print(f"[Rank {rank}] 任务完成! 处理: {processed_count}, 跳过: {skipped_count}, 耗时: {total_time:.1f}s",
              flush=True)

        # 等待所有卡完成
        dist.barrier()
        cleanup_ddp()

    except Exception as e:
        print(f"[Rank {rank}] 发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    # 1. 默认配置路径
    config_path = "./configs/config_stage1.yaml"
    if not os.path.exists(config_path):
        # 兼容一下可能的路径
        config_path = "../configs/config_stage1.yaml"

    print(f"[Main] 加载配置文件: {config_path}")
    cfg = load_config(config_path)

    # 2. 设置可见显卡 (物理 -> 逻辑映射)
    # 格式 "4,5,6,7" -> 程序内看到 cuda:0,1,2,3
    gpus = str(cfg.get("gpus", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    gpu_list = [g for g in gpus.split(",") if g.strip()]
    world_size = len(gpu_list)
    print(f"[Main] 启动多卡提取，使用物理GPU: {gpus} (Total: {world_size})")

    # 3. 准备数据 (在主进程读取，节省内存开销)
    json_path = cfg["llava_med_json"]
    print(f"[Main] 读取样本列表: {json_path}")

    if not os.path.exists(json_path):
        print(f"Error: JSON文件未找到 {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 调试模式：如果配置了 max_samples，截断数据
    max_samples = cfg.get("max_samples", None)
    if max_samples is not None:
        print(f"[Main] Debug模式: 仅使用前 {max_samples} 条数据")
        samples = samples[:max_samples]

    print(f"[Main] 总样本数: {len(samples)}，准备分发给 {world_size} 个进程...")

    # 4. 启动多进程
    mp.spawn(
        worker,
        args=(world_size, cfg, samples),
        nprocs=world_size,
        join=True
    )

    print("[Main] 所有任务执行完毕。")


if __name__ == "__main__":
    main()