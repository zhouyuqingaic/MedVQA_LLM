# engine/trainer_stage1.py
# -*- coding: utf-8 -*-
"""
创建引擎层 Trainer 核心逻辑
我们将原 trainer_stage1.py 中的 run_training 函数，并替换所有内部依赖为 utils 和 engine 层的函数。

职责： Stage 1 训练的核心逻辑执行器。
负责： DDP 初始化，模型/数据集构建，HF Trainer 组装和训练循环。
"""

import torch
from typing import Dict, Any

import os
import sys


from transformers import TrainingArguments, Trainer

# === 从 Utils 层导入依赖 ===
from utils.ddp import setup_ddp, cleanup_ddp,launch_ddp
from utils.config import load_config,get_gpus_and_world_size  # 理论上 launch_ddp 的调用者会加载 config
# from utils.model_utils import load_tokenizer # 现已移入 builder

# === 从 Data 层导入依赖 ===
from med_vqa_datasets.dataset_llava_med_alignment_500k_subset import build_llava_med_dataset
from med_vqa_datasets.collators import LlavaMedChatCollator

# === 从 Engine 层导入依赖 ===
from utils.builder import build_vision_llm


def run_training(rank: int, world_size: int, config: Dict[str, Any]):
    """
    每个 GPU / rank 上启动的训练函数 (由 utils.ddp.launch_ddp 调用)。
    """
    try:
        # ------------------ 1. 初始化 DDP (使用 utils/ddp) ------------------
        # setup_ddp 会自动处理 MASTER_ADDR/PORT 和 CUDA_VISIBLE_DEVICES
        setup_ddp(rank, world_size)
        current_device = torch.cuda.current_device()

        if rank == 0:
            print(f"[Stage1] 启动训练进程，Rank={rank}, World Size={world_size}")
            print(f"[Stage1] 使用模型: {config['model_path']}")
            print(f"[Stage1] 输出目录: {config['output_dir']}")

        # ------------------ 2. 读取关键配置 ------------------
        OUTPUT_DIR = config["output_dir"]
        MICRO_BATCH_SIZE = int(config.get("micro_batch_size", 4))
        GRAD_ACC_STEPS = int(config.get("gradient_accumulation_steps", 4))
        LEARNING_RATE = float(config.get("learning_rate", 2e-4))
        EPOCHS = int(config.get("epochs", 1))
        MAX_LENGTH = int(config.get("max_length", 1024))
        NUM_WORKERS = int(config.get("num_workers", 4))
        MAX_SAMPLES = config.get("max_samples", None)

        # bf16 配置 (从 config 中提取)
        use_bf16 = config.get("bf16", True)
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

        # ------------------ 3. 构建模型和 Tokenizer (使用 engine/builder) ------------------
        model, tokenizer, image_feat_dim = build_vision_llm(
            config=config,
            rank=rank,
            compute_dtype=compute_dtype,
        )

        # ------------------ 4. 准备 Dataset (使用 med_vqa_datasets) ------------------
        if rank == 0:
            print("[Stage1] 构建 LlavaMedAlignBiomedCLIPDataset ...")

        # build_llava_med_dataset 应该从 config 中读取 llava_med_json/image_feature_dir
        dataset = build_llava_med_dataset(config, split="train", verbose=True)

        full_len = len(dataset)
        if rank == 0:
            print(f"[Stage1][Dataset] 样本总数: {full_len}")
            if MAX_SAMPLES is not None:
                print(f"[Stage1][Dataset] 仅使用前 {MAX_SAMPLES} 条样本做训练（调试用）")

        # ------------------ 5. 构建自定义 Collator (使用 med_vqa_datasets/collators) ------------------
        data_collator = LlavaMedChatCollator(
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            image_token_pool=str(config.get("image_token_pool", "cls")),
        )

        # ------------------ 6. 组装 TrainingArguments ------------------
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=use_bf16,
            logging_steps=10,
            save_strategy="epoch",
            ddp_find_unused_parameters=False,
            report_to="none",
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            dataloader_num_workers=NUM_WORKERS,
            remove_unused_columns=False,  # 确保 'image_feat' 不被干掉
            local_rank=rank,
        )

        # ------------------ 7. 构建 Trainer 并开始训练 ------------------
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # 仅让 rank 0 做最终保存
        if rank == 0:
            trainer.save_model(OUTPUT_DIR)
            print(f"[Stage1] 训练完成，模型已保存至: {OUTPUT_DIR}")

        # ------------------ 8. 清理 DDP (使用 utils/ddp) ------------------
        cleanup_ddp()

    except Exception as e:
        print(f"[Stage1][Rank {rank}] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)


def main():
    """
    主入口：

    1. 读取 config_stage1.yaml
    2. 解析 GPU 列表
    3. 使用 launch_ddp 启动 run_training
    """
    # 1. 加载配置
    # 假设配置文件位于项目根目录下的 configs/ 文件夹
    config_path = "configs/config_stage1.yaml"
    if not os.path.exists(config_path):
        # 兼容一下原代码中使用的相对路径
        config_path = "../configs/config_stage1.yaml"

    config = load_config(config_path)

    # 2. 解析 GPU 列表和 World Size (使用 utils/config)
    gpus, world_size = get_gpus_and_world_size(config)

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"[Stage1 Main] 准备在以下 GPU 上启动训练: {gpus} (Total: {world_size})")

    # 3. 使用 launch_ddp 启动多进程训练 (使用 utils/ddp)
    launch_ddp(
        target_fn=run_training,
        world_size=world_size,
        config=config,
    )


if __name__ == "__main__":
    main()