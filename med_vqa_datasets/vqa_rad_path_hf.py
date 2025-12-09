# med_vqa_datasets/vqa_rad_path_hf.py
# -*- coding: utf-8 -*-
"""
HF 版本的 VQA-RAD / PathVQA 数据集封装
====================================

用途：
- 给 Stage-2 (MedVQA) 的 text-only / prefix / cross-attn baseline 提供统一的数据入口；
- 不强绑定 BiomedCLIP 或 Qwen，只负责：
  - 从 HuggingFace datasets 加载样本；
  - 把 image -> PIL / tensor；
  - 返回 question / answer 文本；
  - （可选）用 tokenizer 对 question / answer 做编码。

你可以在训练脚本里：
- text-only baseline：直接忽略 image / image_feat；
- prefix / cross-attn baseline：利用 image 做 BiomedCLIP 特征，再喂给 Vision-LLM。
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image


# =========================
# 通用工具
# =========================

def _ensure_rgb(image: Image.Image | Any) -> Image.Image:
    """防御性地把任意输入转成三通道 RGB PIL Image。"""
    if isinstance(image, Image.Image):
        img = image
    else:
        # HuggingFace 有时会给 numpy array
        img = Image.fromarray(image)
    return img.convert("RGB")


@dataclass
class HFTextFields:
    """一个小 dataclass，用来约定 question/answer 的字段名。"""
    image: str = "image"
    question: str = "question"
    answer: str = "answer"


# =========================
# 1. VQA-RAD
# =========================

class VQARADHFDataset(Dataset):
    """
    HuggingFace 版本 VQA-RAD Dataset 封装。

    每个样本返回：
    - image         : 经过 transform 的图像 (tensor 或 PIL, 取决于 transform)
    - question      : str
    - answer        : str
    - （可选）question_input_ids, question_attention_mask
    - （可选）answer_input_ids,   answer_attention_mask
    """

    def __init__(
        self,
        hf_split,
        image_transform: Optional[Callable] = None,
        tokenizer=None,
        max_q_len: int = 64,
        max_a_len: int = 32,
        text_fields: HFTextFields = HFTextFields(),
    ):
        """
        hf_split:
            load_dataset(...) 返回的某个 split，比如 dataset["train"] / dataset["test"]
        image_transform:
            对 PIL.Image 进行的预处理（Resize/ToTensor/Normalize...）
        tokenizer:
            任意 HuggingFace tokenizer；如果为 None，则不做文本编码，只返回原始字符串。
        """
        self.data = hf_split
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_q_len = int(max_q_len)
        self.max_a_len = int(max_a_len)
        self.fields = text_fields

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # 1) 图像
        img = _ensure_rgb(item[self.fields.image])
        if self.image_transform is not None:
            image = self.image_transform(img)
        else:
            image = img  # 交给下游自己处理

        # 2) 文本
        question = str(item[self.fields.question])
        answer = str(item[self.fields.answer])

        sample: Dict[str, Any] = {
            "image": image,
            "question": question,
            "answer": answer,
        }

        # 3) 可选：用 tokenizer 做编码（方便 text-only / prefix baseline 直接喂给 Qwen）
        if self.tokenizer is not None:
            q_enc = self.tokenizer(
                question,
                max_length=self.max_q_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            a_enc = self.tokenizer(
                answer,
                max_length=self.max_a_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            sample["question_input_ids"] = q_enc["input_ids"].squeeze(0)
            sample["question_attention_mask"] = q_enc["attention_mask"].squeeze(0)
            sample["answer_input_ids"] = a_enc["input_ids"].squeeze(0)
            sample["answer_attention_mask"] = a_enc["attention_mask"].squeeze(0)

        return sample


def build_vqa_rad_dataloaders(
    cache_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_transform: Optional[Callable] = None,
    tokenizer=None,
    max_q_len: int = 64,
    max_a_len: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建 VQA-RAD 的 train / test DataLoader。

    参数
    ----
    cache_dir:
        HuggingFace datasets 的缓存 / 下载目录。
    image_transform:
        图像预处理（如果你后面要接 BiomedCLIP，通常不需要在这里 Normalize，
        而是直接用 backbone.preprocess_image；这里可以只 Resize -> ToTensor）。

    返回
    ----
    train_loader, test_loader
    """
    hf_dataset = load_dataset(
        "flaviagiammarino/vqa-rad",
        cache_dir=cache_dir,
    )

    train_ds = VQARADHFDataset(
        hf_split=hf_dataset["train"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )
    test_ds = VQARADHFDataset(
        hf_split=hf_dataset["test"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=shuffle_train, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, test_loader


# =========================
# 2. PathVQA
# =========================

class PathVQAHFDataset(Dataset):
    """
    HuggingFace 版本 PathVQA Dataset 封装。

    返回字段与 VQARADHFDataset 相同，方便统一调用。
    """

    def __init__(
        self,
        hf_split,
        image_transform: Optional[Callable] = None,
        tokenizer=None,
        max_q_len: int = 64,
        max_a_len: int = 32,
        text_fields: HFTextFields = HFTextFields(),
    ):
        self.data = hf_split
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_q_len = int(max_q_len)
        self.max_a_len = int(max_a_len)
        self.fields = text_fields

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        img = _ensure_rgb(item[self.fields.image])
        if self.image_transform is not None:
            image = self.image_transform(img)
        else:
            image = img

        question = str(item[self.fields.question])
        answer = str(item[self.fields.answer])

        sample: Dict[str, Any] = {
            "image": image,
            "question": question,
            "answer": answer,
        }

        if self.tokenizer is not None:
            q_enc = self.tokenizer(
                question,
                max_length=self.max_q_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            a_enc = self.tokenizer(
                answer,
                max_length=self.max_a_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            sample["question_input_ids"] = q_enc["input_ids"].squeeze(0)
            sample["question_attention_mask"] = q_enc["attention_mask"].squeeze(0)
            sample["answer_input_ids"] = a_enc["input_ids"].squeeze(0)
            sample["answer_attention_mask"] = a_enc["attention_mask"].squeeze(0)

        return sample


def build_hf_vqa_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """
    使用 med_vqa_datasets.vqa_rad_path_hf 中的 VQARADHFDataset / PathVQAHFDataset
    来构建 HF 版本的 VQA 数据集。

    返回的数据样本包含：
        - image   : PIL.Image 或 tensor（我们在 text-only 模式下可以忽略）
        - question: str
        - answer  : str
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "vqa-rad":
        hf_name = "flaviagiammarino/vqa-rad"
        ds_cls = VQARADHFDataset
    elif dataset_name == "path-vqa":
        hf_name = "flaviagiammarino/path-vqa"
        ds_cls = PathVQAHFDataset
    else:
        raise ValueError(f"[build_hf_vqa_dataset] 未知的数据集名称: {dataset_name}")

    hf_dataset = load_dataset(hf_name, cache_dir=cache_dir)

    if split not in hf_dataset:
        raise ValueError(
            f"[build_hf_vqa_dataset] split='{split}' 不存在于 {hf_name} 数据集，可用 split: {list(hf_dataset.keys())}"
        )

    hf_split = hf_dataset[split]

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(hf_split))
        hf_split = hf_split.select(range(max_samples))

    # 这里不在 dataset 内部做 tokenizer / image_transform，全部交给下游（collator + BiomedCLIP）
    ds = ds_cls(
        hf_split=hf_split,
        image_transform=None,
        tokenizer=None,
    )
    return ds


def build_path_vqa_dataloaders(
    cache_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_transform: Optional[Callable] = None,
    tokenizer=None,
    max_q_len: int = 64,
    max_a_len: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 PathVQA 的 train / val / test DataLoader。
    """
    hf_dataset = load_dataset(
        "flaviagiammarino/path-vqa",
        cache_dir=cache_dir,
    )

    train_ds = PathVQAHFDataset(
        hf_split=hf_dataset["train"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )
    val_ds = PathVQAHFDataset(
        hf_split=hf_dataset["validation"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )
    test_ds = PathVQAHFDataset(
        hf_split=hf_dataset["test"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=shuffle_train, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader


# =========================
# 3. 简单自测
# =========================
if __name__ == "__main__":
    """
    直接运行本文件做一个 sanity check：
    - 下载 / 读取 HF 数据集
    - 构建 DataLoader
    - 打印一个 batch 的 shape 和例子 question/answer
    """
    from torchvision import transforms

    # 可选：你之前用到的代理，可以自行取消注释
    # os.environ["http_proxy"] = "http://10.109.70.128:7897"
    # os.environ["https_proxy"] = "http://10.109.70.128:7897"

    cache_root = "/home/yuqing/Datas"  # 按需修改
    vqa_rad_cache = os.path.join(cache_root, "vqa-rad")
    path_vqa_cache = os.path.join(cache_root, "path-vqa")
    os.makedirs(vqa_rad_cache, exist_ok=True)
    os.makedirs(path_vqa_cache, exist_ok=True)

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("==== 测试 VQA-RAD ====")
    tr_loader, te_loader = build_vqa_rad_dataloaders(
        cache_dir=vqa_rad_cache,
        batch_size=4,
        num_workers=2,
        image_transform=img_transform,
        tokenizer=None,
    )
    batch = next(iter(tr_loader))
    print("VQA-RAD image:", batch["image"].shape)
    print("VQA-RAD q:", batch["question"][0])
    print("VQA-RAD a:", batch["answer"][0])

    print("\n==== 测试 PathVQA ====")
    tr_loader, val_loader, te_loader = build_path_vqa_dataloaders(
        cache_dir=path_vqa_cache,
        batch_size=4,
        num_workers=2,
        image_transform=img_transform,
        tokenizer=None,
    )
    batch = next(iter(tr_loader))
    print("PathVQA image:", batch["image"].shape)
    print("PathVQA q:", batch["question"][0])
    print("PathVQA a:", batch["answer"][0])
