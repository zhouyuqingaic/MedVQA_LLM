# models/qwen_vision_adapter.py
# -*- coding: utf-8 -*-
"""
QwenWithVisionAdapter
=====================
一个简单、可读性强的 Cross-Attn 视觉适配器实现。

设计要点：
- 不侵入 Qwen / PEFT 的内部结构，只把 LLM 当成一个黑盒：
    1）先让 LLM 正常 forward，拿到 last_hidden_state
    2）用图像特征做 Cross-Attn，得到一个残差更新
    3）用 LLM 的 lm_head 计算 logits，自行计算 loss
- 图像侧通过可插拔的 Projector（linear / mlp / multihead / moe）把 BiomedCLIP 的全局特征
  映射为若干个“视觉 token”。
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from models.vision_projectors import build_projector


class QwenWithVisionAdapter(nn.Module):
    """
    Cross-Attn 版 Vision Adapter：
    - 通过若干视觉 token 对文本 token 做一次 Cross-Attn
    - 视觉 token 由 BiomedCLIP 全局特征经过 Projector 投影得到
    """

    def __init__(
        self,
        llm: nn.Module,
        image_feat_dim: int,
        num_image_tokens: int = 4,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.0,
        use_image_feat: bool = True,
        use_gate: bool = True,
        # ------- 新增：Projector 配置 -------
        projector_type: str = "mlp",               # ["linear", "mlp", "multihead", "moe"]
        mlp_hidden_dim: Optional[int] = None,
        multihead_inner_dim: Optional[int] = None,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.image_feat_dim = int(image_feat_dim)
        self.num_image_tokens = int(num_image_tokens)
        self.use_image_feat = bool(use_image_feat)

        hidden_size = getattr(llm, "config", None).hidden_size
        self.hidden_size = int(hidden_size)

        # Cross-Attn：使用文本 token 作为 Query，视觉 token 作为 Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
            batch_first=True,  # 方便使用 [B, L, H] 形状
        )

        self.use_gate = bool(use_gate)
        if self.use_gate:
            # 可学习 gate，控制视觉注入强度；初始为 0，相当于一开始几乎不注入
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)

        # ------- 使用可插拔 Projector -------
        self.projector_type = projector_type
        self.image_proj = build_projector(
            projector_type=projector_type,
            input_dim=self.image_feat_dim,
            hidden_size=self.hidden_size,
            num_tokens=self.num_image_tokens,
            mlp_hidden_dim=mlp_hidden_dim,
            multihead_inner_dim=multihead_inner_dim,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
        )

    # ------------------------------------------------------------------
    #   图像特征投影：BiomedCLIP feat -> 视觉 token
    # ------------------------------------------------------------------
    def _project_image_tokens(
        self,
        image_feat: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        将 [B, image_feat_dim] 的图像特征投影为 [B, num_image_tokens, hidden_size]，
        并自动对齐 dtype / device。

        如果 image_feat 为 None 或 use_image_feat=False，则返回 None。
        """
        if (image_feat is None) or (not self.use_image_feat):
            return None

        # 对齐 device / dtype（这里顺便处理你之前提到的 dtype 不一致问题）
        device = hidden_states.device
        dtype = hidden_states.dtype

        img = torch.as_tensor(image_feat, device=device, dtype=dtype)
        if img.dim() > 2:
            # 兼容 [B, 1, D] / [B, D, 1] 等情况
            img = img.view(img.size(0), -1)

        if img.size(-1) != self.image_feat_dim:
            raise ValueError(
                f"[QwenWithVisionAdapter] image_feat dim mismatch: expect {self.image_feat_dim}, got {img.size(-1)}"
            )

        # Projector: [B, image_feat_dim] -> [B, hidden_size * num_image_tokens]
        proj = self.image_proj(img)
        B = proj.size(0)
        tokens = proj.view(B, self.num_image_tokens, self.hidden_size)  # [B, M, H]
        return tokens

    # ------------------------------------------------------------------
    #   前向：LLM + Cross-Attn + lm_head
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        参数
        ----
        input_ids : [B, L]
        attention_mask : [B, L]，可为 None
        image_feat : [B, image_feat_dim]，BiomedCLIP 图像全局特征
        labels : [B, L]，使用 -100 作为 ignore_index

        返回
        ----
        dict(loss=..., logits=...)
        """
        # 1) 先让 LLM 正常跑一遍，获取文本隐藏状态
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            labels=None,   # loss 我们在外面算
            **kwargs,
        )
        # 兼容不同 Output 类型：
        # - 有些模型返回 last_hidden_state
        # - CausalLMOutputWithPast 只在 hidden_states 里给出各层隐藏状态
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state  # [B, L, H]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # hidden_states 是 tuple(layer0, layer1, ..., last_layer)
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        else:
            raise RuntimeError(
                "[QwenWithVisionAdapter] 模型输出中既没有 last_hidden_state 也没有 hidden_states，"
                "请检查底层 llm 的返回类型。"
            )



        # 2) 由图像特征生成视觉 token，并做一次 Cross-Attn
        img_tokens = self._project_image_tokens(image_feat, hidden_states)  # [B, M, H] or None
        if img_tokens is not None:
            attn_output, _ = self.cross_attn(
                query=hidden_states,   # [B, L, H]
                key=img_tokens,        # [B, M, H]
                value=img_tokens,
                need_weights=False,
            )
            if self.gate is not None:
                gate = torch.tanh(self.gate)  # 限制在 [-1, 1]
                hidden_states = hidden_states + gate * attn_output
            else:
                hidden_states = hidden_states + attn_output

        # 3) 使用 LLM 的 lm_head 计算 logits
        if not hasattr(self.llm, "lm_head"):
            raise AttributeError(
                "[QwenWithVisionAdapter] llm 缺少 lm_head 属性，"
                "请确认传入的是 *ForCausalLM 模型而不是 bare Model。"
            )
        logits = self.llm.lm_head(hidden_states)  # [B, L, vocab_size]

        # 4) 计算 loss（如果提供了 labels）
        # inside QwenWithVisionAdapter.forward, after logits computed
        loss = None
        if labels is not None:
            # Causal LM shift: predict token t+1 at position t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            vocab_size = shift_logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
            )
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    #   转发 gradient checkpointing 的开关给底层 llm
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        llm = getattr(self, "llm", None)
        if llm is None:
            return

        # 新版 transformers 会传 gradient_checkpointing_kwargs
        if hasattr(llm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                try:
                    llm.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                except TypeError:
                    # 兼容旧版，只接受 no-arg 的情况
                    llm.gradient_checkpointing_enable()
            else:
                llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        llm = getattr(self, "llm", None)
        if llm is None:
            return

        if hasattr(llm, "gradient_checkpointing_disable"):
            llm.gradient_checkpointing_disable()

    @property
    def config(self):
        # 让外面访问 model.config 时拿到底层 Qwen 的 config
        return getattr(self.llm, "config", None)


# =========================================================
# 自测：只检查形状，不依赖真实 Qwen 权重
# =========================================================
if __name__ == "__main__":
    class DummyConfig:
        def __init__(self, hidden_size: int, vocab_size: int) -> None:
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size

    class DummyLLM(nn.Module):
        """
        一个极简的假 LLM：
        - input_ids 通过 embedding -> transformer (这里用 Linear 代替) -> hidden_states
        - lm_head 再映射到 vocab
        只用于检查 QwenWithVisionAdapter 的形状是否正确。
        """
        def __init__(self, hidden_size: int = 64, vocab_size: int = 100) -> None:
            super().__init__()
            self.config = DummyConfig(hidden_size=hidden_size, vocab_size=vocab_size)
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.ff = nn.Linear(hidden_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            emb = self.embedding(input_ids)      # [B, L, H]
            hidden = torch.tanh(self.ff(emb))    # [B, L, H]

            class Output:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            if return_dict:
                return Output(last_hidden_state=hidden)
            return (hidden,)

    torch.manual_seed(0)
    B, L = 2, 16
    image_feat_dim = 512
    num_image_tokens = 4
    hidden_size = 64
    vocab_size = 100

    dummy_llm = DummyLLM(hidden_size=hidden_size, vocab_size=vocab_size)

    adapter = QwenWithVisionAdapter(
        llm=dummy_llm,
        image_feat_dim=image_feat_dim,
        num_image_tokens=num_image_tokens,
        cross_attn_heads=4,
        cross_attn_dropout=0.0,
        use_image_feat=True,
        use_gate=True,
        projector_type="mlp",
        moe_num_experts=4,
        moe_top_k=2,
    )

    input_ids = torch.randint(0, vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    image_feat = torch.randn(B, image_feat_dim)
    labels = torch.randint(0, vocab_size, (B, L))

    out = adapter(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_feat=image_feat,
        labels=labels,
    )

    print("[TEST] logits shape:", tuple(out["logits"].shape))
    print("[TEST] loss:", float(out["loss"]))
    print("[TEST] Done. QwenWithVisionAdapter seems to work with dummy LLM.")
