# med_vqa_datasets/collators.py
# -*- coding: utf-8 -*-
"""
职责： 集中处理各种 Dataset 的 Collate 函数。
"""

import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase


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