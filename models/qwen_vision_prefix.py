# med_vqa_trainer/qwen_vision_prefix.py
# -*- coding: utf-8 -*-
"""
QwenVisionPrefixModel

最小可用 Vision Adapter（Prefix Token 版）：
- 输入: 文本 input_ids / attention_mask / labels + 预提取的 image_feat
- image_feat: [B, D_img]，其中 D_img = config["image_feat_dim"]（通常为 512）
- 做法:
    1) prefix = Linear(D_img -> hidden_size) -> [B, hidden]
    2) prefix.unsqueeze(1) -> [B, 1, hidden]，拼到 input embeddings 前面
    3) attention_mask 前补一列 1
    4) labels 前补一列 -100（不参与 loss）
- 其它 forward 参数全部透传给底层 Qwen LLM
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn


class QwenVisionPrefixModel(nn.Module):
    """
    一个简单的 wrapper，把预提取的 image_feat 作为单个 prefix token
    注入到 Qwen 的输入 embedding 序列前面。

    使用方式：
        base_llm = AutoModelForCausalLM.from_pretrained(...)
        base_llm = prepare_model_for_kbit_training(base_llm)
        base_llm = get_peft_model(base_llm, lora_config)

        model = QwenVisionPrefixModel(
            llm=base_llm,
            image_feat_dim=512,
            prefix_dropout=0.0,
            use_image_feat=True,
        )

    Trainer 侧只需要在 batch 里多塞一个键 "image_feat" 即可。
    """

    def __init__(
        self,
        llm: nn.Module,
        image_feat_dim: int,
        prefix_dropout: float = 0.0,
        use_image_feat: bool = True,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.use_image_feat = bool(use_image_feat)

        # 保留 config，方便 HF Trainer / generation 等使用
        # （PeftModel / AutoModelForCausalLM 都有 .config）
        self.config = getattr(llm, "config", None)

        hidden_size = int(self.config.hidden_size)
        self.image_feat_dim = int(image_feat_dim)

        # image_feat:[B, D_img] -> proj:[B, hidden_size]
        self.vision_proj = nn.Linear(self.image_feat_dim, hidden_size, bias=True)

        # prefix token 上的 dropout（可选）
        if prefix_dropout > 0.0:
            self.prefix_dropout = nn.Dropout(prefix_dropout)
        else:
            self.prefix_dropout = nn.Identity()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Args
        ----
        input_ids        : (B, L)
        attention_mask   : (B, L)
        labels           : (B, L)，标准 causal LM 标签
        image_feat       : (B, D_img) 或 (B, 1, D_img)

        Returns
        -------
        outputs: dict or ModelOutput，与原始 Qwen 模型保持一致
        """

        # 1. 如果没开视觉，或者 batch 没给 image_feat，就退化为原始 LLM
        if (not self.use_image_feat) or (image_feat is None):
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # 2. 规范 image_feat 的形状: [B, D_img]
        if image_feat.dim() == 3:
            # e.g. [B, 1, D_img] -> [B, D_img]
            image_feat = image_feat.squeeze(1)
        assert image_feat.dim() == 2, f"image_feat 预期形状 [B, D_img]，但得到 {image_feat.shape}"

        # 3. 投影到 hidden_size，得到 prefix token: [B, 1, hidden]
        #    注意：此处不做归一化等，先保持最小实现
        prefix = self.vision_proj(image_feat)           # [B, hidden]
        prefix = prefix.unsqueeze(1)                    # [B, 1, hidden]
        prefix = self.prefix_dropout(prefix)            # [B, 1, hidden]

        # 4. 获取原始 token embedding：embedding(input_ids) -> [B, L, hidden]
        #    注意：这里必须用 inputs_embeds，而不能再传 input_ids，
        #          否则 prefix token 会被忽略。
        if input_ids is None:
            raise ValueError("QwenVisionPrefixModel 目前要求必须提供 input_ids。")

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, hidden]

        # 在序列维度前端拼接 prefix token
        inputs_embeds = torch.cat([prefix, input_embeds], dim=1)   # [B, L+1, hidden]

        # 5. attention_mask 前补一列 1（prefix 总是可见）
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, L+1]

        # 6. labels 前补一列 -100（prefix 不参与 loss）
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), 1),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)  # [B, L+1]

        # 7. 调用底层 LLM，使用 inputs_embeds 形式
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        兼容 HF Trainer 的接口:
        - Trainer 会在训练前调用 model.gradient_checkpointing_enable(...)
        - 这里直接转发给内部的 self.llm
        """
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                return self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            else:
                return self.llm.gradient_checkpointing_enable()
        return None  # 没有这个方法就什么都不做

    def gradient_checkpointing_disable(self):
        """
        对应的关闭接口，同样转发给内部 LLM。
        """
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            return self.llm.gradient_checkpointing_disable()
        return None