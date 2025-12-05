"""
VQA-RAD 数据管线（面向 BiomedCLIPBackbone 的简洁重构版）
==================================================
- 与 BiomedCLIPBackbone 严格对齐（processor/tokenizer 均从 backbone 获取）
- 清晰：数据流与错误提示友好；支持分层切分与调试子集
"""
from __future__ import annotations
import json, random, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from backbones.biomedclip_backbone import BiomedCLIPBackbone


def normalize_answer(text: str, lower: bool = True, strip_punct: bool = False) -> str:
    t = (text or "").strip()
    if lower: t = t.lower()
    if strip_punct:
        import string
        t = t.translate(str.maketrans("", "", string.punctuation)).strip()
        t = " ".join(t.split())
    return t


class AnswerEncoder:
    def __init__(self) -> None:
        self._cls2id: Dict[str, int] = {}
        self.classes_: List[str] = []

    def fit(self, answers: Iterable[str], sort: bool = True) -> "AnswerEncoder":
        uniq = list({a for a in (normalize_answer(a) for a in answers) if a})
        if sort: uniq.sort()
        self.classes_ = uniq
        self._cls2id = {c: i for i, c in enumerate(self.classes_)}
        return self

    def to_id(self, a: str) -> int:
        key = normalize_answer(a)
        if key not in self._cls2id:
            raise KeyError(f"未知答案：{a!r}（规范化为 {key!r} 不在 label space 中）")
        return self._cls2id[key]

    def to_text(self, idx: int) -> str:
        return self.classes_[idx]

    def __len__(self) -> int:
        return len(self.classes_)


def _load_vqa_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError("VQA-RAD JSON 格式错误：应为列表或包含 'questions' 的字典。")
    return data


def _extract_item(raw: Dict[str, Any]) -> Optional[Tuple[str, str, str, Optional[str]]]:
    image_file = str(raw.get("image_name") or raw.get("image") or "").strip()
    question = str(raw.get("question") or "").strip()
    answer = str(raw.get("answer") or "").strip()
    answer_type = raw.get("answer_type", raw.get("question_type"))
    answer_type = str(answer_type).strip().upper() if isinstance(answer_type, str) else answer_type
    if not image_file or not question or not answer:
        return None
    return image_file, question, answer, answer_type


def _filter_and_clean_records(records: List[Dict[str, Any]], image_dir: str, answer_type: str = "CLOSED",
                              drop_missing_file: bool = True, verbose: bool = True) -> List[Dict[str, Any]]:
    at = (answer_type or "ALL").upper()
    keep, bad = [], 0
    for r in records:
        parsed = _extract_item(r)
        if parsed is None:
            bad += 1; continue
        image_file, q, a, rtype = parsed
        if at != "ALL" and isinstance(rtype, str) and rtype.upper() != at:
            continue
        if drop_missing_file:
            path = Path(image_dir) / image_file
            if not path.is_file():
                bad += 1; continue
        keep.append({"image": image_file, "question": q, "answer": a, "answer_type": rtype})
    if verbose and bad > 0:
        warnings.warn(f"[Data] 跳过异常/缺失样本 {bad} 条（字段缺失或图像不存在）。")
    return keep


def build_processors_from_backbone(backbone: BiomedCLIPBackbone):
    """从 backbone 取图像预处理与 tokenizer，并适配为 Dataset 期望的接口。"""
    preprocess = getattr(backbone, "preprocess", None) or getattr(backbone, "preprocess_val", None)
    if preprocess is None:
        raise RuntimeError("backbone 未暴露 preprocess/preprocess_val。")
    tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("backbone 未暴露 tokenizer。")
    context_len = int(getattr(backbone, "context_length", 256))

    class _ImageProcessorAdapter:
        def __init__(self, transform): self.transform = transform
        def __call__(self, img, return_tensors="pt"):
            x = self.transform(img)
            return {"pixel_values": x.unsqueeze(0) if return_tensors == "pt" else x}

    class _TokenizerAdapter:
        def __init__(self, tok, ctx_len: int): self.tok, self.ctx_len = tok, ctx_len
        def __call__(self, text, return_tensors="pt", padding="max_length", truncation=True, max_length=None):
            import torch
            texts = [text] if isinstance(text, str) else list(text)
            max_len = max_length or self.ctx_len
            ids = self.tok(texts)
            if not torch.is_tensor(ids): ids = torch.tensor(ids, dtype=torch.long)
            L = ids.shape[-1]
            if L < max_len:
                pad = torch.zeros((ids.shape[0], max_len - L), dtype=torch.long)
                ids = torch.cat([ids, pad], dim=1)
            elif L > max_len:
                ids = ids[:, :max_len]
            attention_mask = (ids != 0).long()
            return {"input_ids": ids, "attention_mask": attention_mask}

    return _ImageProcessorAdapter(preprocess), _TokenizerAdapter(tokenizer, context_len)


@dataclass
class VQARADItem:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    meta: Dict[str, Any]


class VQARADDataset(Dataset):
    """VQA-RAD 数据集（与 BiomedCLIP 对齐；含增强接口）。"""
    def __init__(self,
                 rows: List[Dict[str, Any]], image_dir: str,
                 image_processor, tokenizer, label_encoder: AnswerEncoder,
                 max_text_len: int = 32,
                 image_aug: Optional = None, text_aug: Optional = None,
                 apply_augment: bool = False) -> None:
        super().__init__()
        self.rows = rows; self.image_dir = Path(image_dir)
        self.image_processor = image_processor; self.tokenizer = tokenizer
        self.label_encoder = label_encoder; self.max_text_len = int(max_text_len)
        self.image_aug = image_aug; self.text_aug = text_aug; self.apply_augment = bool(apply_augment)

    def __len__(self) -> int: return len(self.rows)

    def _load_image(self, image_file: str) -> Image.Image:
        return Image.open(self.image_dir / image_file).convert("RGB")

    def __getitem__(self, idx: int) -> VQARADItem:
        row = self.rows[idx]
        image_file, question, answer = row["image"], row["question"], row["answer"]

        img = self._load_image(image_file)
        if self.apply_augment and self.image_aug is not None:
            try: img = self.image_aug(img)
            except Exception: pass
        pv = self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0).to(torch.float32)

        txt = question
        if self.apply_augment and self.text_aug is not None:
            try: txt = self.text_aug(txt)
            except Exception: pass
        tok = self.tokenizer(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_text_len)
        input_ids = tok["input_ids"].squeeze(0).to(torch.long)
        attention_mask = tok["attention_mask"].squeeze(0).to(torch.long)

        y = self.label_encoder.to_id(answer)
        return VQARADItem(pixel_values=pv, input_ids=input_ids, attention_mask=attention_mask, label=y,
                          meta={"image": image_file, "question": question, "answer": answer})


def _stratified_split(rows: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42):
    rnd = random.Random(seed); bucket: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        bucket.setdefault(normalize_answer(r["answer"]), []).append(r)
    train, val = [], []
    for items in bucket.values():
        rnd.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        val.extend(items[:n_val]); train.extend(items[n_val:])
    rnd.shuffle(train); rnd.shuffle(val)
    return train, val


def _random_split(rows: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42):
    rnd = random.Random(seed); idx = list(range(len(rows))); rnd.shuffle(idx)
    n_val = max(1, int(round(len(rows) * val_ratio)))
    val_idx = set(idx[:n_val]); train, val = [], []
    for i, r in enumerate(rows): (val if i in val_idx else train).append(r)
    return train, val


def _build_answer_encoder(rows: List[Dict[str, Any]]) -> AnswerEncoder:
    return AnswerEncoder().fit((r["answer"] for r in rows), sort=True)


def _collate_fn(batch: List[VQARADItem]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([b.pixel_values for b in batch], dim=0)
    input_ids = torch.stack([b.input_ids for b in batch], dim=0)
    attention_mask = torch.stack([b.attention_mask for b in batch], dim=0)
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)

    # 新增：把 question / image 文件名带上，供分割器使用
    questions = [b.meta.get("question", "") for b in batch]
    image_files = [b.meta.get("image", "") for b in batch]

    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "questions": questions,
            "image_files": image_files,
            }


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    import random as pyrandom, numpy as np
    pyrandom.seed(worker_seed); np.random.seed(worker_seed)


def create_data_loaders_from_backbone(
    backbone: BiomedCLIPBackbone,
    json_path: str,
    image_dir: str,
    batch_size: int = 32,
    answer_type: str = "CLOSED",
    max_text_len: int = 32,
    val_ratio: float = 0.1,
    stratified: bool = True,
    debug_samples: int = 0,
    seed: int = 42,
    num_workers: int = 96,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    image_aug: Optional = None,
    text_aug: Optional = None,
    augment_in: str = "train",
):
    raw = _load_vqa_json(json_path)
    rows = _filter_and_clean_records(raw, image_dir=image_dir, answer_type=answer_type, drop_missing_file=True, verbose=True)
    if debug_samples and debug_samples > 0:
        rows = rows[:debug_samples]; warnings.warn(f"[Debug] 仅使用前 {debug_samples} 条样本进行调试。")

    if stratified: train_rows, val_rows = _stratified_split(rows, val_ratio=val_ratio, seed=seed)
    else:          train_rows, val_rows = _random_split(rows, val_ratio=val_ratio, seed=seed)

    image_processor, tokenizer = build_processors_from_backbone(backbone)

    label_encoder = _build_answer_encoder(rows)
    num_answers = len(label_encoder)

    aug_train = augment_in in ("train", "both"); aug_val = augment_in in ("val", "both")
    train_set = VQARADDataset(train_rows, image_dir, image_processor, tokenizer, label_encoder, max_text_len, image_aug, text_aug, aug_train)
    val_set   = VQARADDataset(val_rows,   image_dir, image_processor, tokenizer, label_encoder, max_text_len, image_aug, text_aug, aug_val)

    generator = torch.Generator(); generator.manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=(persistent_workers and num_workers > 0),
                  drop_last=False, collate_fn=_collate_fn, worker_init_fn=_seed_worker, generator=generator)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader   = DataLoader(val_set,   shuffle=False, **common)

    return train_loader, val_loader, num_answers, label_encoder