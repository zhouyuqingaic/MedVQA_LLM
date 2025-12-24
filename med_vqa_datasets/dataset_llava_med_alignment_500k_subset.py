# -*- coding: utf-8 -*-
"""
dataset_llava_med_alignment_500k_subset.py

用于 LLaVA-Med 对齐阶段 (Stage-1) 的 Dataset 定义：
- 从 config_stage1.yaml 读取所有路径与超参数
- 读取 llava_med_alignment_500k_subset.json
- 使用预先提取好的 BiomedCLIP 图像特征 (*.pt)
- 构建适用于 Qwen2.5-7B-Instruct 的对话样本 (chat 格式)

推荐使用方式：
    from transformers import AutoTokenizer
    from dataset_llava_med_alignment_500k_subset import (
        load_config,
        build_llava_med_dataset,
        make_llava_med_collate_fn,
    )

    cfg = load_config("./configs/config_stage1.yaml")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    dataset = build_llava_med_dataset(cfg, split="train")
    collate_fn = make_llava_med_collate_fn(tokenizer, cfg.get("max_length", 1024))

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("micro_batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    for batch in dataloader:
        # batch["input_ids"], batch["attention_mask"], batch["labels"], batch["image_feat"]
        ...

"""

import os
import json
from typing import List, Dict, Any, Optional

import yaml
import torch
from torch.utils.data import Dataset


# =========================================================
# 1. 配置读取工具
# =========================================================

def load_config(config_path: str) -> dict:
    """
    读取 YAML 配置文件并返回字典。

    参数
    ----
    config_path : str
        配置文件路径，比如 "./configs/config_stage1.yaml"

    返回
    ----
    cfg : dict
        解析后的配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =========================================================
# 2. LLaVA-Med 对齐数据集定义
# =========================================================

class LLaVAMedAlignmentDataset(Dataset):
    """
    LLaVA-Med 对齐数据集 (基于 llava_med_alignment_500k_subset.json)。

    特点：
    - 使用已经预提取好的 BiomedCLIP 图像特征 (.pt)
    - 保留原始对话结构 (human / gpt)，转换为 Qwen 习惯的 role ("user" / "assistant")
    - 可选地添加统一的 system prompt
    - 在 __getitem__ 中只做轻量工作：加载 feature + 封装 chat 结构
      真正的 tokenizer.apply_chat_template / padding 交给外面的 collate_fn 完成
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        verbose: bool = True,
    ):
        """
        参数
        ----
        cfg : dict
            从 config_stage1.yaml 读取的配置字典。
            需要的关键字段：
                - llava_med_json
                - llava_med_image_feature_dir
                - image_feat_dim
                - max_samples (可选)
                - system_prompt (可选)
        split : str, optional
            数据集划分，目前对齐数据只有一份，可以写 "train" 方便后续扩展。
        verbose : bool, optional
            是否打印一些统计信息。
        """
        super().__init__()
        self.cfg = cfg
        self.split = split

        # 从配置中读取路径
        json_path = cfg["llava_med_json"]
        feat_dir = cfg["llava_med_image_feature_dir"]
        self.image_feat_dim = int(cfg.get("image_feat_dim", 512))

        self.system_prompt: Optional[str] = cfg.get("system_prompt", None)
        self.max_samples: Optional[int] = cfg.get("max_samples", None)

        if verbose:
            print(f"[Dataset] 使用 JSON: {json_path}")
            print(f"[Dataset] 使用图像特征目录: {feat_dir}")
            if self.system_prompt is not None:
                print(f"[Dataset] system_prompt 已启用。")
            if self.max_samples is not None:
                print(f"[Dataset] 仅使用前 {self.max_samples} 条样本 (调试模式)。")

        # 读取 JSON 数据
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"llava_med_json 不存在: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        # 调试模式：截断
        if self.max_samples is not None:
            all_samples = all_samples[: self.max_samples]

        # 预过滤掉没有对应 .pt 的样本，避免 __getitem__ 中频繁报错
        self.samples: List[Dict[str, Any]] = []
        self.feature_paths: List[str] = []

        missing_feat_count = 0
        for sp in all_samples:
            sid = sp["id"]
            feat_path = os.path.join(feat_dir, f"{sid}.pt")
            if os.path.exists(feat_path):
                self.samples.append(sp)
                self.feature_paths.append(feat_path)
            else:
                missing_feat_count += 1

        if verbose:
            print(f"[Dataset] 总样本数(JSON 中): {len(all_samples)}")
            print(f"[Dataset] 有特征的样本数: {len(self.samples)}")
            print(f"[Dataset] 缺失特征的样本数: {missing_feat_count}")

        if len(self.samples) == 0:
            raise RuntimeError(
                "LLaVAMedAlignmentDataset 中没有可用样本："
                "请检查 llava_med_image_feature_dir 下是否已经生成对应的 .pt 文件。"
            )

        """
        关键点：你现在 pt 存的是 Tensor[T,768]（或 dict 里带 tokens/image_tokens/...），Stage-1 Dataset 统一输出 image_feat: Tensor[768]。
        """
        self.image_token_dim = int(cfg.get("image_token_dim", 768))
        self.image_feat_dim = int(cfg.get("image_feat_dim", self.image_token_dim))
        self.image_token_pool = str(cfg.get("image_token_pool", "cls")).lower()

    # ------------------------------
    # Dataset 标准接口
    # ------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_feature(self, feat_path: str) -> torch.Tensor:
        feat_obj = torch.load(feat_path, map_location="cpu")

        # dict：优先 tokens，再兼容旧 key
        if isinstance(feat_obj, dict):
            if "tokens" in feat_obj:
                feat = feat_obj["tokens"]
            elif "image_tokens" in feat_obj:
                feat = feat_obj["image_tokens"]
            elif "token_embeddings" in feat_obj:
                feat = feat_obj["token_embeddings"]
            elif "patch_tokens" in feat_obj:
                feat = feat_obj["patch_tokens"]
            elif "img_feat" in feat_obj:
                feat = feat_obj["img_feat"]
            elif "image_emb" in feat_obj:
                feat = feat_obj["image_emb"]
            else:
                raise ValueError(f"Unsupported dict keys={list(feat_obj.keys())} in {feat_path}")
        else:
            feat = feat_obj

        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat, dtype=torch.float32)
        feat = feat.float()

        # [1, T, C] -> [T, C]
        if feat.dim() == 3:
            if feat.size(0) != 1:
                raise ValueError(f"3D tokens should be [1,T,C], got {tuple(feat.shape)} ({feat_path})")
            feat = feat.squeeze(0)

        # 2D：tokens [T,C] or vector [1,C]
        if feat.dim() == 2:
            if feat.size(0) == 1 and feat.size(1) == self.image_feat_dim:
                vec = feat.squeeze(0)  # [C]
            else:
                if feat.size(-1) != self.image_token_dim:
                    raise ValueError(
                        f"tokens last-dim mismatch: expect {self.image_token_dim}, got {feat.size(-1)} ({feat_path})")
                if self.image_token_pool == "cls":
                    vec = feat[0]
                elif self.image_token_pool == "mean":
                    vec = feat.mean(dim=0)
                else:
                    raise ValueError(f"Unknown image_token_pool={self.image_token_pool!r}")
        elif feat.dim() == 1:
            vec = feat
        else:
            raise ValueError(f"Unsupported feature ndim={feat.dim()} for {feat_path}")

        if vec.numel() != self.image_feat_dim:
            raise ValueError(
                f"feature dim mismatch after pooling: expect {self.image_feat_dim}, got {vec.numel()} ({feat_path})")

        return vec

    def _build_chat_from_sample(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将原始 JSON 中的 "conversatons" 字段转换为 chat 结构：

        原始格式示例：
            "conversatons": [
              {
                "from": "human",
                "value": "Summarize the visual content of the image.\\n<image>"
              },
              {
                "from": "gpt",
                "value": "Chemical structures of EGFR inhibitors."
              }
            ]

        转为：
            [
                {"role": "system", "content": "..."} (可选)
                {"role": "user", "content": "... <image> ..."},
                {"role": "assistant", "content": "..."},
            ]

        注意：
        - JSON 里字段名是 "conversatons" (拼写少了一个 i)，要保持一致。
        - "from": "human"/"gpt" 转为 "user"/"assistant"。
        """
        conv_key_candidates = ["conversatons", "conversations"]
        conv_list = None
        for k in conv_key_candidates:
            if k in sample:
                conv_list = sample[k]
                break
        if conv_list is None:
            raise KeyError(
                f"样本中未找到对话字段，尝试 keys={conv_key_candidates}，"
                f"实际 keys={list(sample.keys())}"
            )

        chat: List[Dict[str, str]] = []

        # 可选：加入统一的 system prompt
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})

        # 逐条转换
        for turn in conv_list:
            src_role = turn.get("from", "")
            content = turn.get("value", "")

            # role 映射：human -> user, gpt -> assistant
            if src_role == "human":
                role = "user"
            elif src_role == "gpt":
                role = "assistant"
            else:
                # 兜底：未知 role 当成 user
                role = "user"

            chat.append({"role": role, "content": content})

        return chat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单条样本，包含：
        - id: str
        - image_feat: torch.FloatTensor, shape [image_feat_dim]
        - chat: List[{"role": ..., "content": ...}]
        - raw_sample: 原始 JSON 结构 (可选，用于调试)
        """
        sample = self.samples[idx]
        feat_path = self.feature_paths[idx]

        # 加载图像特征
        image_feat = self._load_image_feature(feat_path)

        # 构建对话 (chat) 结构
        chat = self._build_chat_from_sample(sample)

        return {
            "id": sample["id"],
            "image_feat": image_feat,  # [image_feat_dim]
            "chat": chat,              # List[Dict[str, str]]
            "raw_sample": sample,      # 调试用
        }


# =========================================================
# 3. Collate 函数：把若干样本打包成一个 batch
# =========================================================

def make_llava_med_collate_fn(tokenizer, max_length: int = 1024):
    """
    根据给定 tokenizer 构造一个适用于 LLaVAMedAlignmentDataset 的 collate_fn。

    主要功能：
    - 使用 Qwen 的 chat_template 将多轮对话展开为文本，统一 token 化
    - 做 padding / truncation，得到 input_ids / attention_mask
    - 构造 labels (目前简单地等于 input_ids，并将 padding 位置设为 -100)
      → 更精细的监督（只在 assistant 段计算 loss）可在此基础上做二次开发
    - 将 image_feat 堆叠成 [B, image_feat_dim]

    使用示例：
        collate_fn = make_llava_med_collate_fn(tokenizer, max_length=cfg["max_length"])
        dataloader = DataLoader(dataset, batch_size=..., collate_fn=collate_fn)
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 图像特征：简单拼成一个矩阵 [B, D]
        image_feats = torch.stack([item["image_feat"] for item in batch], dim=0)

        # 2. 利用 chat_template 构造文本
        #    Qwen2.5 官方 tokenizer 通常会带 apply_chat_template 方法
        texts: List[str] = []
        for item in batch:
            chat = item["chat"]
            # add_generation_prompt=False：训练时我们已经包含了 assistant 的回复
            text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # 3. 一次性 tokenizer 处理整个 batch，自动 padding + truncation
        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]          # [B, L]
        attention_mask = enc["attention_mask"]  # [B, L]

        # 4. 构造 labels
        #    最简单的方法：labels = input_ids（语言模型标准自回归训练）
        #    同时把 padding 的位置置为 -100，避免影响 loss 计算
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # 如果后续你想只在 assistant 段计算 loss，可以在这里进一步
        # 根据 chat 结构生成一个 mask，然后把非 assistant 位置设为 -100。

        # 5. 返回 batch 字典
        return {
            "input_ids": input_ids,             # [B, L]
            "attention_mask": attention_mask,   # [B, L]
            "labels": labels,                   # [B, L]
            "image_feat": image_feats,          # [B, D]
            "ids": [item["id"] for item in batch],
            "raw_samples": [item["raw_sample"] for item in batch],
        }

    return collate_fn


# =========================================================
# 4. 便捷构造函数：从 cfg 一步到 Dataset
# =========================================================

def build_llava_med_dataset(
    cfg: Dict[str, Any],
    split: str = "train",
    verbose: bool = True,
) -> LLaVAMedAlignmentDataset:
    """
    从配置字典快速构造 LLaVAMedAlignmentDataset。

    参数
    ----
    cfg : dict
        config_stage1.yaml 解析后的字典。
    split : str
        数据集划分，目前只有 "train"，预留扩展。
    verbose : bool
        是否打印统计信息。

    返回
    ----
    dataset : LLaVAMedAlignmentDataset
    """
    return LLaVAMedAlignmentDataset(cfg, split=split, verbose=verbose)


# =========================================================
# 5. 简单自测 / 调试入口
# =========================================================

if __name__ == "__main__":
    """
    python dataset_llava_med_alignment_500k_subset.py

    用于快速检查：
    - 配置文件能否正确读取
    - 数据集是否能正常构建
    - 单条样本的结构是否符合预期
    """
    default_cfg_path_candidates = [
        "../configs/config_stage1.yaml",
    ]

    cfg_path = None
    for p in default_cfg_path_candidates:
        if os.path.exists(p):
            cfg_path = p
            break

    if cfg_path is None:
        raise FileNotFoundError(
            "未找到 config_stage1.yaml，请检查路径。"
            f"尝试过: {default_cfg_path_candidates}"
        )

    print(f"[Debug] 使用配置文件: {cfg_path}")
    cfg = load_config(cfg_path)

    dataset = build_llava_med_dataset(cfg, split="train", verbose=True)
    print(f"[Debug] Dataset size: {len(dataset)}")

    for i in range(10):
        # 随便取一条看下结构
        sample = dataset[i]
        print("[Debug] Sample keys:", sample.keys())
        print("[Debug]   id:", sample["id"])
        print("[Debug]   image_feat shape:", sample["image_feat"].shape)
        print("[Debug]   chat[0]:", sample["chat"][0])
        if len(sample["chat"]) > 1:
            print("[Debug]   chat[1]:", sample["chat"][1])
        print('_-_'*10)
