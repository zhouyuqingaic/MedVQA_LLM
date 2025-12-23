# biomedclip_backbone.py
# -*- coding: utf-8 -*-
"""
BiomedCLIPBackbone
==================

本文件是你项目里对 BiomedCLIP(OpenCLIP) 的统一封装，特点：

1) **只支持“本地目录加载”**：
   目录中必须包含：
   - open_clip_config.json
   - open_clip_pytorch_model.bin / open_clip_pytorch_model.pt（二选一）

2) **不额外叠加归一化 / LN**：
   encode_image / encode_text 输出就是 BiomedCLIP 原生输出（与 open_clip.encode_* 一致）

3) 统一暴露：
   - preprocess / preprocess_val : PIL -> tensor（含 mean/std）
   - tokenize(texts)            : list[str] -> {"input_ids","attention_mask"}
   - encode_image(pixel_values) : [B,3,H,W] -> [B, D] 全局向量（D=embed_dim, 通常 512）
   - encode_image_tokens(...)   : [B,3,H,W] -> [B, T, C] token 序列（CLS + patch tokens）
   - encode_text(...)           : [B,L] -> [B, D]

4) **关于 tokens 的维度（最重要）**
   - encode_image_tokens **始终返回“视觉塔 trunk 的原始 tokens”**，即 ViT 的 width 维度 C（例如 ViT-B/16@224 -> C=768）
   - 我们**刻意不在 backbone 内做 768->512 的 CLIP 投影（visual.proj）**

   设计意图（务必读）：
   - 你的 Stage-2 训练采用“离线 pt 缓存 tokens”，训练时只优化 Adapter/Resampler/LLM；
   - 一旦你把 768->512 的投影在离线阶段做完写进 pt，这个投影就被“写死”，无法作为可训练模块适配 VQA；
   - 因此我们在 backbone 侧只负责 **稳定抽取 raw tokens**，把任何投影/压缩/重采样全部交给训练侧可学习模块。

依赖：
- torch
- open_clip（你的环境里已安装）
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

import open_clip
from open_clip.factory import _MODEL_CONFIGS  # 注册本地模型配置


# ======================================================================
#                       小工具：临时改属性并恢复
# ======================================================================
@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any):
    """
    临时把 obj.attr 改成 value；退出 with 后恢复原值。

    如果 obj 没有这个属性，则不会修改，也不会报错。
    """
    if not hasattr(obj, attr):
        yield None
        return

    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield old
    finally:
        setattr(obj, attr, old)


def _split_pooled_and_tokens(out: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    把 open_clip 可能返回的多种结构统一解析成：
    - pooled: [B, D] 或 None
    - tokens: [B, T, C] 或 None

    open_clip 不同版本可能返回：
    - Tensor
    - (pooled, tokens)
    - dict 里带 tokens
    """
    pooled = None
    tokens = None

    if isinstance(out, dict):
        tokens = (
            out.get("tokens")
            or out.get("image_tokens")
            or out.get("token_embeddings")
            or out.get("patch_tokens")
            or out.get("all_tokens")
        )
        pooled = (
            out.get("pooled")
            or out.get("image_features")
            or out.get("image_embeds")
            or out.get("embeddings")
            or out.get("feat")
        )

    elif isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x) and x.dim() == 3:
                tokens = x
            elif torch.is_tensor(x) and x.dim() == 2:
                pooled = x

    elif torch.is_tensor(out):
        if out.dim() == 3:
            tokens = out
        elif out.dim() == 2:
            pooled = out

    # 单图情况下 tokens 可能是 [T, C]，补 batch 维
    if tokens is not None and tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)

    return pooled, tokens


# ======================================================================
#                              Backbone
# ======================================================================
class BiomedCLIPBackbone(nn.Module):
    """
    统一 BiomedCLIP backbone 封装（本地加载版）。
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        context_length: int = 256,
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ) -> None:
        super().__init__()

        self.model_dir = str(model_dir)
        self.device = torch.device(device)
        self.context_length = int(context_length)

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"[BiomedCLIPBackbone] 只支持“本地目录加载”，请传入包含权重与配置文件的目录：{self.model_dir}"
            )

        # 1) 必须文件检查
        cfg_path = os.path.join(self.model_dir, "open_clip_config.json")
        bin_path = os.path.join(self.model_dir, "open_clip_pytorch_model.bin")
        pt_path = os.path.join(self.model_dir, "open_clip_pytorch_model.pt")

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"[BiomedCLIPBackbone] 缺少配置文件：{cfg_path}")
        if not (os.path.isfile(bin_path) or os.path.isfile(pt_path)):
            raise FileNotFoundError(
                "[BiomedCLIPBackbone] 缺少权重文件：open_clip_pytorch_model.bin / open_clip_pytorch_model.pt 至少存在一个。"
            )

        # 2) 读取 open_clip_config.json 并注册到 open_clip 的本地模型配置表
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)

        model_cfg = config_json["model_cfg"]
        preprocess_cfg = config_json["preprocess_cfg"]

        # local model name 随便取，只要注册到 _MODEL_CONFIGS 即可
        local_model_name = "biomedclip_local_refactored"
        if local_model_name not in _MODEL_CONFIGS:
            _MODEL_CONFIGS[local_model_name] = model_cfg

        # 选择权重文件
        weight_file = bin_path if os.path.isfile(bin_path) else pt_path

        # open_clip.create_model_and_transforms 的 preprocess 参数前缀是 image_*
        image_kwargs = {f"image_{k}": v for k, v in preprocess_cfg.items()}

        print(f"[BiomedCLIPBackbone] Loading BiomedCLIP from: {self.model_dir}")
        print(f"[BiomedCLIPBackbone] Using weight file: {weight_file}")
        print(f"[BiomedCLIPBackbone] Target device: {self.device}")

        # 3) 构建模型与 transforms
        self.clip, _, self._image_preprocess = open_clip.create_model_and_transforms(
            model_name=local_model_name,
            pretrained=weight_file,
            device=self.device,
            **image_kwargs,
        )
        self._tokenizer = open_clip.get_tokenizer(local_model_name)

        # 4) 对上层暴露统一接口字段（兼容你项目中其他代码）
        self.preprocess = self._image_preprocess
        self.preprocess_val = self._image_preprocess
        self.tokenizer = self._tokenizer

        # embed_dim（CLIP 的对齐空间维度，通常是 512）
        embed_dim = int(model_cfg.get("embed_dim", 512))
        self.img_dim = embed_dim
        self.txt_dim = embed_dim

        # 5) 冻结视觉 / 文本（可选）
        if freeze_vision:
            for p in self.clip.visual.parameters():
                p.requires_grad = False
            print("[BiomedCLIPBackbone] Vision encoder is FROZEN.")
        else:
            print("[BiomedCLIPBackbone] Vision encoder is TRAINABLE.")

        if freeze_text:
            text_encoder = None
            if hasattr(self.clip, "text"):
                text_encoder = self.clip.text
            elif hasattr(self.clip, "transformer"):
                text_encoder = self.clip.transformer
            elif hasattr(self.clip, "bert"):
                text_encoder = self.clip.bert

            if text_encoder is None:
                print(
                    f"[BiomedCLIPBackbone] Warning: cannot find text encoder to freeze "
                    f"(no .text/.transformer/.bert)."
                )
            else:
                for p in text_encoder.parameters():
                    p.requires_grad = False
                print("[BiomedCLIPBackbone] Text encoder is FROZEN.")
        else:
            print("[BiomedCLIPBackbone] Text encoder is TRAINABLE.")

        print(
            f"[BiomedCLIPBackbone] Load success. "
            f"img_dim={self.img_dim}, txt_dim={self.txt_dim}, context_length={self.context_length}"
        )

    # ------------------------------------------------------------------
    #                           编码：全局（512）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        图像全局编码（BiomedCLIP 原生输出）：
        输入:
            pixel_values: [B, 3, H, W]（必须是 self.preprocess 的输出）
        输出:
            [B, img_dim]（通常 512）
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        return self.clip.encode_image(pixel_values)

    @torch.no_grad()
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        文本全局编码（BiomedCLIP 原生输出）：
        输入:
            input_ids: [B, L]（建议由 self.tokenize 生成）
        输出:
            [B, txt_dim]（通常 512）
        """
        _ = attention_mask  # open_clip 通常不用 attention_mask，这里只是为了接口统一
        input_ids = input_ids.to(self.device, non_blocking=True)
        return self.clip.encode_text(input_ids)

    # ------------------------------------------------------------------
    #                     编码：token 序列（raw width，例如 768）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image_tokens(
        self,
        pixel_values: torch.Tensor,
        include_cls: bool = True,
    ) -> torch.Tensor:
        """
        提取图像 token 序列（CLS + patch tokens），**始终返回 raw width（例如 768）**。

        重要说明：
        - 本函数只负责“稳定抽取 tokens”
        - 不做任何 768->512 的 CLIP 投影，不做降维、不做 resampler
        - 训练侧用可学习的 projector/resampler 把 tokens 映射到 LLM hidden，保证梯度可学习
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        visual = self.clip.visual

        tokens: Optional[torch.Tensor] = None

        # ------------------------------------------------------------
        # 0) 最常见情况：open_clip 的 TimmModel 把 timm ViT 放在 visual.trunk
        #    trunk.forward_features 返回 [B, 197, 768]（含 CLS）
        # ------------------------------------------------------------
        trunk = getattr(visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward_features"):
            try:
                out = trunk.forward_features(pixel_values)
                _, tokens = _split_pooled_and_tokens(out)
                if tokens is None and torch.is_tensor(out):
                    if out.dim() == 3:
                        tokens = out
                    elif out.dim() == 4:
                        tokens = out.flatten(2).transpose(1, 2)
            except Exception:
                tokens = None

        # ------------------------------------------------------------
        # 1) 尝试：visual.forward_features（有些实现直接暴露）
        # ------------------------------------------------------------
        if tokens is None and hasattr(visual, "forward_features"):
            out = visual.forward_features(pixel_values)
            _, tokens = _split_pooled_and_tokens(out)
            if tokens is None and torch.is_tensor(out):
                if out.dim() == 3:
                    tokens = out
                elif out.dim() == 4:
                    tokens = out.flatten(2).transpose(1, 2)

        # ------------------------------------------------------------
        # 2) 尝试：clip.encode_image(..., output_tokens/return_tokens=...)
        # ------------------------------------------------------------
        if tokens is None:
            for flag in ("output_tokens", "return_tokens", "return_all_tokens", "return_token_embeddings"):
                try:
                    out = self.clip.encode_image(pixel_values, **{flag: True})
                except TypeError:
                    continue
                _, tokens = _split_pooled_and_tokens(out)
                if tokens is not None:
                    break

        # ------------------------------------------------------------
        # 3) 尝试：visual.output_tokens=True（如果 visual 支持）
        # ------------------------------------------------------------
        if tokens is None and hasattr(visual, "output_tokens"):
            with _temporary_attr(visual, "output_tokens", True):
                out = visual(pixel_values)
            _, tokens = _split_pooled_and_tokens(out)

        # ------------------------------------------------------------
        # 仍失败：给出更具体的 debug 信息
        # ------------------------------------------------------------
        if tokens is None:
            raise RuntimeError(
                "[encode_image_tokens] Cannot extract tokens.\n"
                f"  visual_type={type(visual)}\n"
                f"  has_trunk={hasattr(visual, 'trunk')}\n"
                f"  has_forward_features={hasattr(visual, 'forward_features')}\n"
                f"  has_output_tokens={hasattr(visual, 'output_tokens')}\n"
                "说明：你当前 open_clip 版本/视觉塔实现不提供 token 输出。"
            )

        # 单图情况下 tokens 可能是 [T, C]，补 batch 维
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        if not include_cls:
            if tokens.size(1) <= 1:
                raise RuntimeError("tokens length <= 1, cannot drop CLS token.")
            tokens = tokens[:, 1:, :]

        return tokens

    # ------------------------------------------------------------------
    #                         forward：方便上层统一调用
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        默认前向：
            返回图像全局特征 + 文本全局特征 + 拼接全局向量

        注意：你的训练代码一般不会用这个 forward，
              更常用 encode_image / encode_image_tokens / encode_text。
        """
        img_feat = self.encode_image(pixel_values)
        txt_feat = self.encode_text(input_ids, attention_mask)
        global_feat = torch.cat([img_feat, txt_feat], dim=-1)
        return {"img_feat": img_feat, "txt_feat": txt_feat, "global_feat": global_feat}

    # ------------------------------------------------------------------
    #                       实用工具函数：保持兼容
    # ------------------------------------------------------------------
    def preprocess_image(self, pil_or_ndarray) -> torch.Tensor:
        """单张图像的 CLIP 预处理，返回 [3,H,W] tensor。"""
        return self._image_preprocess(pil_or_ndarray)

    def tokenize(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """
        把 batch 文本转成 input_ids / attention_mask（attention_mask 这里只是占位统一接口）。
        open_clip tokenizer 返回的是 tensor（不是 HF BatchEncoding）。
        """
        input_ids = self._tokenizer(texts, context_length=self.context_length)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ======================================================================
#                               自测
# ======================================================================
if __name__ == "__main__":
    """
    自测目标：
    1) 能加载本地 BiomedCLIP（需要你提供 --model_dir）
    2) encode_image 输出 [B,512]
    3) encode_image_tokens 输出 [B,T,width]（例如 [1,197,768]）

    运行示例：
      python backbones/biomedclip_backbone.py \
        --model_dir /home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
        --device cuda:0
    """
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    backbone = BiomedCLIPBackbone(
        model_dir=args.model_dir,
        device=args.device,
        context_length=256,
        freeze_vision=True,
        freeze_text=True,
    )
    backbone.eval()

    # 构造一张随机 PIL 图（不依赖 numpy）
    rand = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).cpu()
    pil = Image.frombytes("RGB", (224, 224), bytes(rand.flatten().tolist()))

    # 预处理 -> [1,3,224,224]
    x = backbone.preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        g = backbone.encode_image(x)
        t_cls = backbone.encode_image_tokens(x, include_cls=True)
        t_patch = backbone.encode_image_tokens(x, include_cls=False)

    print("[SelfTest] encode_image:", tuple(g.shape), g.dtype, g.device)
    print("[SelfTest] encode_image_tokens (cls):", tuple(t_cls.shape), t_cls.dtype, t_cls.device)
    print("[SelfTest] encode_image_tokens (patch-only):", tuple(t_patch.shape), t_patch.dtype, t_patch.device)

    # 推断 patch grid（如果是标准 ViT）
    if t_cls.dim() == 3 and t_cls.size(1) > 1:
        patch = t_cls.size(1) - 1
        grid = int((patch) ** 0.5)
        if grid * grid == patch:
            print(f"[SelfTest] patch_tokens={patch}, patch_grid={grid}x{grid}")