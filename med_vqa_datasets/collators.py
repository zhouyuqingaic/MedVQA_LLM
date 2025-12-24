# med_vqa_datasets/collators.py
# -*- coding: utf-8 -*-
"""
职责： 集中处理各种 Dataset 的 Collate 函数。
"""


from PIL import Image
import os

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import hashlib
import torch


@dataclass
class LlavaMedChatCollator:
    tokenizer: Any
    max_length: int = 512
    image_token_pool: str = "cls"  # cls / mean

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts, full_texts = [], []
        feats = []

        for ex in examples:
            chat = ex["chat"]

            # ====== [新增] 防御式池化兜底：支持 [D] 或 [T,D] ======
            feat = ex["image_feat"]
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat)

            # 如果是 tokens：[T, D] -> [D]
            if feat.dim() == 2:
                pool = (self.image_token_pool or "cls").lower()

                # 兼容 [1, D]（有些数据会多一维）
                if feat.size(0) == 1:
                    feat = feat.squeeze(0)
                else:
                    if pool == "cls":
                        feat = feat[0]  # 默认 tokens[0] 是 CLS
                    elif pool == "mean":
                        # 注意：这里 mean 会包含 CLS；如果你想排除 CLS，用 feat[1:].mean(0)
                        feat = feat.mean(dim=0)
                    else:
                        raise ValueError(
                            f"Unknown image_token_pool={self.image_token_pool!r}, expected 'cls' or 'mean'.")
            # 如果是 [D]，保持不动
            elif feat.dim() == 1:
                pass
            else:
                raise ValueError(f"Unsupported image_feat ndim={feat.dim()} shape={tuple(feat.shape)}")

            # 最终必须是 [D]
            if feat.dim() != 1:
                raise ValueError(f"After pooling, expect 1D image_feat [D], got shape={tuple(feat.shape)}")

            feats.append(feat)
            # ====== [新增结束] ======

            # 找到最后一个 assistant（更稳）
            last_asst = None
            for i in range(len(chat) - 1, -1, -1):
                if chat[i].get("role") == "assistant":
                    last_asst = i
                    break

            if last_asst is None:
                # 没有 assistant：退化为全 mask（不建议出现）
                full_msgs = chat
                prompt_msgs = chat
            else:
                full_msgs = chat[: last_asst + 1]
                prompt_msgs = chat[:last_asst]  # 不含 assistant 内容

            full_texts.append(
                self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            )
            prompt_texts.append(
                self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            )

        full_enc = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        labels = full_enc["input_ids"].clone()
        labels[full_enc["attention_mask"] == 0] = -100

        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        for i, l in enumerate(prompt_lens.tolist()):
            l = min(int(l), labels.size(1))
            labels[i, :l] = -100

        full_enc["labels"] = labels
        full_enc["image_feat"] = torch.stack(feats, dim=0)  # (B, D)
        return full_enc


# ===========================
# Data Collator：文本-only
# ===========================
@dataclass
class VQATextDataCollator:
    tokenizer: Any
    max_length: int = 512
    system_prompt: str = ""
    add_image_hint: bool = False

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts = []
        full_texts = []

        for ex in examples:
            q = str(ex["question"])
            a = str(ex["answer"])

            sys = self.system_prompt or ""
            if self.add_image_hint:
                sys = (sys + "\nIn this training stage you will NOT see the image, only the text question.\n").strip()

            # full conversation (has assistant answer)
            full_msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            full_text = self.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)

            # prompt-only conversation (stops at assistant start)
            prompt_msgs = full_msgs[:-1]  # drop assistant answer
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        full_enc = self.tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        labels = full_enc["input_ids"].clone()
        # mask padding
        labels[full_enc["attention_mask"] == 0] = -100

        # mask prompt part (system+user+assistant_start)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)  # [B]
        for i, l in enumerate(prompt_lens.tolist()):
            l = min(int(l), labels.size(1))
            labels[i, :l] = -100

        full_enc["labels"] = labels
        return full_enc

# ===========================
# Data Collator：多模态
# ===========================
def image_sha1_key(img: Image.Image) -> str:
    """
    与 gen_vqa_rad_path_pt.py 完全一致的 key 策略：
    sha1(RGB像素 + size + "RGB"标记)

    ⚠️ 注意：不要对 img 做随机增强/裁剪，否则 key 会改变，查不到特征。
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.convert("RGB")
    raw = img.tobytes()

    h = hashlib.sha1()
    h.update(b"RGB")
    h.update(str(img.size).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


class PtImageFeatStore:
    """
    一个轻量、可复用的 pt 特征仓库（CPU 上）：
    - 只在初始化时 torch.load 一次
    - get_by_pil(img) -> Tensor[D]
    """
    def __init__(self, pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"[PtImageFeatStore] pt file not found: {pt_path}")

        payload = torch.load(pt_path, map_location="cpu")

        # 兼容两种保存格式：
        # 1) {"meta":..., "features":{key:tensor}, "failed":[...]}  (你当前脚本就是这种)
        # 2) {key:tensor}  (极简格式)
        if isinstance(payload, dict) and "features" in payload:
            self.meta = payload.get("meta", {})
            self.features = payload["features"]
        elif isinstance(payload, dict):
            self.meta = {}
            self.features = payload
        else:
            raise TypeError(f"[PtImageFeatStore] Unsupported pt payload type: {type(payload)}")

        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise RuntimeError(f"[PtImageFeatStore] Empty features in pt: {pt_path}")

        # 推断维度（用于 debug/兜底）
        any_feat = next(iter(self.features.values()))
        if not isinstance(any_feat, torch.Tensor):
            any_feat = torch.tensor(any_feat)
        self.feat_dim = int(any_feat.numel())

    def get_by_pil(self, img: Image.Image) -> torch.Tensor:
        key = image_sha1_key(img)
        if key not in self.features:
            raise KeyError(
                f"[PtImageFeatStore] Key not found in pt: {key}\n"
                f"Hint: make sure training-side key strategy is identical to pt generation."
            )
        feat = self.features[key]
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat)
        return feat


@dataclass
class VQAMultimodalPtDataCollator:
    """
    ✅ 推荐：Stage-2 multi 的离线特征版 collator
    - 不跑 BiomedCLIP
    - 不碰 CUDA
    - 只负责把 pt 里的 image_feat 拼进 batch

    参数
    ----
    text_collator:
        你现有的 VQATextDataCollator（负责 input_ids/labels/chat template）
    image_feat_pt_path:
        离线生成的 .pt 文件路径（config 顶层 vqa_rad_pt_output/path_vqa_pt_output）
    strict:
        True  -> 缺特征直接报错（建议）
        False -> 缺特征用 0 向量兜底（只用于 debug）
    """
    text_collator: Any
    image_feat_pt_path: str
    strict: bool = True

    def __post_init__(self):
        self.store = PtImageFeatStore(self.image_feat_pt_path)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) 文本部分：完全复用你现有逻辑
        batch = self.text_collator(examples)

        # 2) 图像特征：CPU 查表 + stack
        feats = []
        for ex in examples:
            try:
                img = ex["image"]  # 你的 vqa_rad_path_hf.py 返回 PIL.Image（image_transform=None）
                feat = self.store.get_by_pil(img)
            except Exception:
                if self.strict:
                    raise
                feat = torch.zeros(self.store.feat_dim, dtype=torch.float16)
            feats.append(feat)

        batch["image_feat"] = torch.stack(feats, dim=0)  # (B, D)
        return batch

