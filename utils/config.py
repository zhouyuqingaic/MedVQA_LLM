# utils/config.py
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    统一的配置加载函数。
    从 YAML 文件中加载配置并返回字典。

    参数
    ----
    config_path : str
        配置文件路径，如 "./configs/config_stage1.yaml"

    返回
    ----
    cfg : dict
        解析后的配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_gpus_and_world_size(config: Dict[str, Any]) -> tuple[str, int]:
    """
    从配置中解析 GPU 列表和 World Size。
    """
    gpus = str(config.get("gpus", "0"))
    gpu_list = [g for g in gpus.split(",") if g.strip()]
    world_size = len(gpu_list)
    return gpus, world_size

# 示例：如果您需要一个统一的 main 入口来加载配置（可选）
if __name__ == "__main__":
    # 假设 config_stage1.yaml 在父目录的 configs 文件夹中
    cfg = load_config("../configs/config_stage1.yaml")
    print(cfg)