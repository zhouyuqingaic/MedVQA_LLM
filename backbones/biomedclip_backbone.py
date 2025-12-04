# -*- coding: utf-8 -*-
"""
BiomedCLIP BackBone（重构版，无额外归一化）
==========================================
特点：
- 只支持“本地目录加载”：目录中必须包含
    - open_clip_config.json
    - open_clip_pytorch_model.bin / open_clip_pytorch_model.pt（二选一）
- 使用 open_clip 官方推荐方式构建模型（参考 HuggingFace README 的 2.2 本地加载示例）
- 不再手动访问 visual.conv1 / conv1_1，避免 TimmModel 命名差异导致的 AttributeError
- **不添加任何额外 LayerNorm / 归一化**，encode_image / encode_text 的输出即为 BiomedCLIP 原生输出
- 暴露：
    - preprocess_image()    : PIL / ndarray -> tensor
    - tokenize()            : list[str] -> input_ids / attention_mask
    - encode_image()        : pixel_values -> 图像全局特征
    - encode_text()         : input_ids -> 文本全局特征
    - forward()             : 同时编码图像 + 文本，返回 dict
"""

import os
import json
from typing import Dict

import torch
import torch.nn as nn
import open_clip
from open_clip.factory import _MODEL_CONFIGS  # 注册本地模型配置


class BiomedCLIPBackbone(nn.Module):
    """
    统一的 BiomedCLIP backbone 封装：
    - 只做“全局图像 / 文本特征”的抽取，不再手写 ViT 细节
    - 通过 config 中的 embed_dim 自动确定输出维度
    - 不在输出上叠加任何额外归一化层，完全沿用 BiomedCLIP 原生输出
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

        # --------- 基本属性 ---------
        self.model_dir = str(model_dir)
        self.device = torch.device(device)
        self.context_length = int(context_length)

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"[BiomedCLIPBackbone] 只支持“本地目录加载”，请传入包含权重与配置文件的目录：{self.model_dir}"
            )

        # --------- 1. 检查必须文件 ---------
        cfg_path = os.path.join(self.model_dir, "open_clip_config.json")
        bin_path = os.path.join(self.model_dir, "open_clip_pytorch_model.bin")
        pt_path = os.path.join(self.model_dir, "open_clip_pytorch_model.pt")

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"[BiomedCLIPBackbone] 缺少配置文件：{cfg_path}")
        if not (os.path.isfile(bin_path) or os.path.isfile(pt_path)):
            raise FileNotFoundError(
                "[BiomedCLIPBackbone] 缺少权重文件："
                "open_clip_pytorch_model.bin / open_clip_pytorch_model.pt 至少存在一个。"
            )

        # --------- 2. 读取 config 并注册到 open_clip 的本地模型表 ---------
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)

        model_cfg = config_json["model_cfg"]
        preprocess_cfg = config_json["preprocess_cfg"]

        # 这个名字可以随便取，只要在 _MODEL_CONFIGS 里注册就行
        local_model_name = "biomedclip_local_refactored"
        if local_model_name not in _MODEL_CONFIGS:
            _MODEL_CONFIGS[local_model_name] = model_cfg

        # 选择实际使用的权重文件
        weight_file = bin_path if os.path.isfile(bin_path) else pt_path

        # 把 preprocess 的 image_* 参数交给 open_clip 来构建 transform
        image_kwargs = {f"image_{k}": v for k, v in preprocess_cfg.items()}

        print(f"[BiomedCLIPBackbone] Loading BiomedCLIP from: {self.model_dir}")
        print(f"[BiomedCLIPBackbone] Using weight file: {weight_file}")
        print(f"[BiomedCLIPBackbone] Target device: {self.device}")

        # --------- 3. 调用 open_clip 创建完整 CLIP 模型 + 预处理 ---------
        # 对齐官方 README 的本地加载方式（只是把路径改为你自己的目录）
        self.clip, _, self._image_preprocess = open_clip.create_model_and_transforms(
            model_name=local_model_name,
            pretrained=weight_file,
            device=self.device,
            **image_kwargs,
        )
        self._tokenizer = open_clip.get_tokenizer(local_model_name)

        # clip 已经在 device 上，这里不用再手动 .to(self.device)

        # --------- 4. 统一暴露的接口（对上层友好） ---------
        # 与你之前可运行版本保持一致的属性命名
        self.preprocess = self._image_preprocess
        self.preprocess_val = self._image_preprocess
        self.tokenizer = self._tokenizer

        # 从配置中读取 embed_dim，避免依赖内部命名（比如 visual.head.out_features 等）
        embed_dim = int(model_cfg.get("embed_dim", 512))
        self.img_dim = embed_dim
        self.txt_dim = embed_dim

        # 👉 不做任何额外归一化，保持 BiomedCLIP 原生输出
        # self.norm_img = nn.Identity()
        # self.norm_txt = nn.Identity()

        # --------- 5. 是否冻结视觉 / 文本编码器 ---------
        if freeze_vision:
            for p in self.clip.visual.parameters():
                p.requires_grad = False
            print("[BiomedCLIPBackbone] Vision encoder is FROZEN.")
        else:
            print("[BiomedCLIPBackbone] Vision encoder is TRAINABLE.")

        # if freeze_text:
        #     for p in self.clip.transformer.parameters():
        #         p.requires_grad = False
        #     print("[BiomedCLIPBackbone] Text encoder is FROZEN.")
        # else:
        #     print("[BiomedCLIPBackbone] Text encoder is TRAINABLE.")

        # 修改后的代码：自动检测属性名
        if freeze_text:
            # 尝试查找常见的文本编码器属性名
            if hasattr(self.clip, "text"):
                # BiomedCLIP / CustomTextCLIP 通常走这里
                text_encoder = self.clip.text
            elif hasattr(self.clip, "transformer"):
                # 标准 CLIP 通常走这里
                text_encoder = self.clip.transformer
            elif hasattr(self.clip, "bert"):
                text_encoder = self.clip.bert
            else:
                print(
                    f"Warning: [{self.__class__.__name__}] Could not find text encoder to freeze (no .text or .transformer).")
                text_encoder = None

            if text_encoder is not None:
                for p in text_encoder.parameters():
                    p.requires_grad = False
                print(f"[{self.__class__.__name__}] Text encoder frozen.")

        print(
            f"[BiomedCLIPBackbone] Load success. "
            f"img_dim={self.img_dim}, txt_dim={self.txt_dim}, context_length={self.context_length}"
        )

    # ------------------------------------------------------------------
    #   编码函数：对上层暴露的“干净接口”
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        图像编码（原生 BiomedCLIP 输出）：
        - 输入：pixel_values: [B, 3, H, W]，需先经过 preprocess (CLIP mean/std)
        - 输出：img_feat: [B, img_dim]
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        img_feat = self.clip.encode_image(pixel_values)  # open_clip 已经处理好视觉塔 & 投影
        # 不做额外 LayerNorm / 归一化，直接返回 BiomedCLIP 原始 embedding
        return img_feat

    @torch.no_grad()
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        文本编码（原生 BiomedCLIP 输出）：
        - 输入：input_ids: [B, L]，建议由 self.tokenize() 生成
        - 输出：txt_feat: [B, txt_dim]
        """
        # open_clip 的 text encoder 不使用 attention_mask，这里保持接口一致即可
        input_ids = input_ids.to(self.device, non_blocking=True)
        txt_feat = self.clip.encode_text(input_ids)
        # 不做额外 LayerNorm / 归一化
        return txt_feat

    # ------------------------------------------------------------------
    #   forward：同时编码 image + text，方便上层直接调用
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        默认前向：
        - 同时输出图像 & 文本的全局特征，以及一个简单的拼接全局向量

        返回：
            {
                "img_feat":   [B, img_dim],
                "txt_feat":   [B, txt_dim],
                "global_feat":[B, img_dim + txt_dim]
            }
        """
        img_feat = self.encode_image(pixel_values)
        txt_feat = self.encode_text(input_ids, attention_mask)

        # 你之前就是简单拼接，保持行为不变
        global_feat = torch.cat([img_feat, txt_feat], dim=-1)
        return {
            "img_feat": img_feat,
            "txt_feat": txt_feat,
            "global_feat": global_feat,
        }

    # ------------------------------------------------------------------
    #   实用工具函数：跟之前可运行版本保持一致
    # ------------------------------------------------------------------
    def preprocess_image(self, pil_or_ndarray) -> torch.Tensor:
        """
        对单张图片做 BiomedCLIP 标准预处理，返回 [3, H, W] 的 tensor。
        """
        return self._image_preprocess(pil_or_ndarray)

    def tokenize(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """
        把一个 batch 的文本转成 input_ids / attention_mask。
        - 注意：open_clip 的 tokenizer 返回的是 tensor，而不是 HF 的 BatchEncoding。
        """
        ids = self._tokenizer(texts, context_length=self.context_length)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


# ======================================================================
#                              自测脚本
# ======================================================================
if __name__ == "__main__":
    """
    简单自测：
    1. 从本地目录加载 BiomedCLIP
    2. 用随机 tensor 模拟一批图像 + 文本 id
    3. 打印输出的 shape，检查是否符合预期
    """

    # TODO: 按你自己的路径修改
    LOCAL_DIR = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    print(f"[TEST] Running BiomedCLIPBackbone test on device: {device}")

    backbone = BiomedCLIPBackbone(
        model_dir=LOCAL_DIR,
        device=device,
        context_length=256,
        freeze_vision=False,
        freeze_text=False,
    )

    # ---- 构造假数据：2 张随机图像 + 2 条伪文本 ----
    B = 2
    dummy_img = torch.randn(B, 3, 224, 224)  # 只是 shape 检查用
    dummy_texts = ["this is a dummy sentence", "another dummy text"]

    # tokenizer -> input_ids / attention_mask
    tok_out = backbone.tokenize(dummy_texts)
    input_ids = tok_out["input_ids"]
    attention_mask = tok_out["attention_mask"]

    # 把数据丢到同一 device
    dummy_img = dummy_img.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = backbone(
            pixel_values=dummy_img,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("\n[TEST] Output shapes:")
    print(f"  img_feat   : {tuple(out['img_feat'].shape)}")     # [B, img_dim]
    print(f"  txt_feat   : {tuple(out['txt_feat'].shape)}")     # [B, txt_dim]
    print(f"  global_feat: {tuple(out['global_feat'].shape)}") # [B, img_dim + txt_dim]

    print("\n[TEST] Done. BiomedCLIPBackbone is working.")

    #自测输出
    """
    /opt/anaconda3/condabin/conda run -n MoEBiomedVQA_LLM --no-capture-output python /home/yuqing/RemoteProjects/MedVQA_LLM_P02/backbones/biomedclip_backbone.py 
    /home/yuqing/.conda/envs/MoEBiomedVQA_LLM/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
    [TEST] Running BiomedCLIPBackbone test on device: cuda:4
    [BiomedCLIPBackbone] Loading BiomedCLIP from: /home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    [BiomedCLIPBackbone] Using weight file: /home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin
    [BiomedCLIPBackbone] Target device: cuda:4
    [BiomedCLIPBackbone] Vision encoder is TRAINABLE.
    [BiomedCLIPBackbone] Text encoder is TRAINABLE.
    [BiomedCLIPBackbone] Load success. img_dim=512, txt_dim=512, context_length=256

    [TEST] Output shapes:
      img_feat   : (2, 512)
      txt_feat   : (2, 512)
      global_feat: (2, 1024)

    [TEST] Done. BiomedCLIPBackbone is working.

    Process finished with exit code 0 
    """
