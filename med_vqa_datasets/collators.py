# med_vqa_datasets/collators.py
# -*- coding: utf-8 -*-
"""
职责： 集中处理各种 Dataset 的 Collate 函数。
"""

import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase

from backbones.biomedclip_backbone import BiomedCLIPBackbone


class LlavaMedChatCollator:
    """
    将 LLaVAMedAlignmentDataset 返回的样本列表 (batch) 转换成模型可直接训练的 batch。

    核心功能：
    - 使用 tokenizer.apply_chat_template 将多轮对话拼成一个长文本。
    - 对文本进行 padding & truncation。
    - 构造 labels (通常 labels = input_ids，padding 设为 -100)。
    - 堆叠 image_feat。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 取出 batch 中每个样本的 chat 列表 与 image_feat
        chats = [sample["chat"] for sample in batch]
        image_feats = [sample["image_feat"] for sample in batch]

        # 2. 使用 Qwen 自带的 chat_template 将多轮对话拼成一个长文本
        texts: List[str] = []
        for messages in chats:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 训练时一般不加生成提示
            )
            texts.append(text)

        # 3. 将 batch 的文本一起 tokenizer，做 padding & truncation
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,       # batch 内对齐长度
            truncation=True,    # 超长则截断
            return_tensors="pt",
            return_token_type_ids=False, #去掉type_ids
        )

        # 4. labels = input_ids（不掩码 user 部分）
        input_ids = tokenized["input_ids"]
        labels = input_ids.clone()
        # 将 padding 位置设为 -100
        labels[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = labels

        # 5. 堆叠 image_feat: list[Tensor(D_img)] -> Tensor(B, D_img)
        img_tensors = []
        for feat in image_feats:
            feat = torch.as_tensor(feat, dtype=torch.float32)
            if feat.dim() > 1:
                feat = feat.view(-1)  # e.g. [1,512] -> [512]
            img_tensors.append(feat)
        img_batch = torch.stack(img_tensors, dim=0)  # [B, D_img]

        tokenized["image_feat"] = img_batch


        return tokenized


# ===========================
# Data Collator：文本-only
# ===========================
@dataclass
class VQATextDataCollator:
    """
    文本-only 模式的 collator：

    - 使用 tokenizer.apply_chat_template 构造对话：
        [system] (可选)
        [user]      -> question (+ 提示当前看不到图像)
        [assistant] -> answer
    - 然后统一 padding / truncation
    - labels 对所有非 padding token 监督（后续你可以改成只监督 assistant 段）
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    system_prompt: Optional[str] = None
    add_image_hint: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations: List[str] = []

        for f in features:
            q = f["question"]
            a = f["answer"]

            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            user_content = q
            if self.add_image_hint:
                user_content = (
                    "You are a medical AI assistant.\n"
                    "You do NOT have access to the image, only the text question.\n\n"
                    f"Question: {q}"
                )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": a})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            conversations.append(text)

        enc = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ===========================
# Data Collator：多模态
# ===========================
@dataclass
class VQAMultimodalDataCollator:
    """
    多模态模式的 collator：

    - 使用 BiomedCLIPBackbone.preprocess + encode_image 将图像编码为 [B, D_img] 特征
    - 使用 tokenizer.apply_chat_template 构造对话：
        [system] (可选)
        [user]      -> question
        [assistant] -> answer
    - 返回：
        input_ids, attention_mask, labels, image_feat
      其中 image_feat 会被 HF Trainer 自动作为关键字参数传入 model.forward(image_feat=...)
    """

    tokenizer: PreTrainedTokenizerBase
    biomed_clip: BiomedCLIPBackbone
    max_length: int = 512
    system_prompt: Optional[str] = None

    def __post_init__(self):
        # BiomedCLIP 模型内部自己会管理 device，这里只保留一个标记
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        questions = [f["question"] for f in features]
        answers = [f["answer"] for f in features]

        # 1) 图像编码 -> BiomedCLIP 特征
        with torch.no_grad():
            # preprocess 返回单张图的 transform，这里逐张处理再 cat 成 batch
            pixel_list = [self.biomed_clip.preprocess(img).unsqueeze(0) for img in images]
            pixel_values = torch.cat(pixel_list, dim=0)  # [B, 3, H, W]
            pixel_values = pixel_values.to(self.device)

            image_feat = self.biomed_clip.encode_image(pixel_values)  # [B, D] 或 [B, 1, D]
            if image_feat.ndim == 3:
                # 兼容 [B, 1, D] 的情况
                image_feat = image_feat.squeeze(1)
            image_feat = image_feat.float().cpu()  # 先搬回 CPU，Trainer 会自动分发到各自设备

        # 2) 文本部分 -> chat 模板
        conversations: List[str] = []
        for q, a in zip(questions, answers):
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": f"Question: {q}"})
            messages.append({"role": "assistant", "content": a})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            conversations.append(text)

        enc = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_feat": image_feat,
        }
