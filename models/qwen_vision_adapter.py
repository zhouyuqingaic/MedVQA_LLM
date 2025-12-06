# models/qwen_vision_adapter.py
# -*- coding: utf-8 -*-
"""
QwenWithVisionAdapter + VisionCrossAttnAdapterLayer

这个文件实现了一个「最小可用版本」的 Cross-Attn Vision Adapter，用于把
预提取的图像特征 (image_feat) 融合到 Qwen 文本 LLM 中。

核心设计：
=========
- 保持已有的 QLoRA / Peft Qwen 基座不变（作为 `llm` 传入）；
- 额外挂一个视觉分支：
    1) 将 image_feat: [B, D_img] 投影成若干个 image tokens: [B, M, H]
    2) 使用一个轻量的 Cross-Attn Adapter，让文本 hidden_states 作为 Query，
       image tokens 作为 Key/Value，做一次融合；
    3) 将融合后的 hidden_states 交给 lm_head 预测下一个 token。

注意：
=====
- 这是一个「浅层」的 Adapter：我们只在 LLM 最后一层 hidden_states 上做一次 cross-attn，
  没有深入每一层 decoder block。优点是实现简单、易于集成，适合作为第一版实验。
- 为了保持接口简单，在训练 (labels 不为 None) 时，我们自己计算 loss，
  并构造一个 CausalLMOutputWithPast；在推理 / generate 时，建议仍使用 prefix 版，
  或在后续版本中扩展 generate 对 vision 的支持。
"""

from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    # 仅用于类型提示与构造标准输出结构；真正运行时需安装 transformers
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # transformers 未安装时，给个兜底占位类型，便于静态检查
    PreTrainedModel = nn.Module  # type: ignore

    class CausalLMOutputWithPast(dict):  # type: ignore
        """
        简单兜底：当 transformers 不可用时，用一个 dict 充当输出结构，
        避免 import 错误影响其他模块的导入。
        """
        def __getattr__(self, item):
            return self[item]


class VisionCrossAttnAdapterLayer(nn.Module):
    """
    轻量级 Cross-Attn Adapter:
    -------------------------
    - 输入:
        hidden_states : [B, T, H]  文本 token 对应的隐藏状态
        image_tokens  : [B, M, H]  从图像特征投影得到的一小串视觉 token
    - 输出:
        out : [B, T, H]  融合视觉信息后的隐藏状态

    实现细节:
    - 使用 nn.MultiheadAttention 实现一次 cross-attention:
        Q = hidden_states, K = V = image_tokens
    - 使用残差连接 + LayerNorm:
        out = LayerNorm(hidden_states + gate * attn_out)
    - 使用可学习 gate (sigmoid 标量)，控制视觉信息注入的强度。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 使用 batch_first=True，这样输入输出都是 [B, T, H]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_size)

        if use_gate:
            # 将 gate 初始化为一个较小的负值，开始时视觉影响很弱，训练更稳定
            self.gate = nn.Parameter(torch.tensor([-2.0]))
        else:
            self.register_parameter("gate", None)

    def forward(self, hidden_states: Tensor, image_tokens: Tensor) -> Tensor:
        """
        参数
        ----
        hidden_states : Tensor, [B, T, H]
            文本序列的隐藏状态。
        image_tokens : Tensor, [B, M, H]
            视觉 token 序列。

        返回
        ----
        out : Tensor, [B, T, H]
            融合了视觉信息的文本隐藏状态。
        """
        if image_tokens is None:
            # 没有视觉 token 时，直接返回原始 hidden_states
            return hidden_states

        # Q = 文本，K = V = 图像 token
        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )  # [B, T, H]

        if self.gate is not None:
            # 使用 sigmoid 将 gate 压到 (0,1)，控制视觉信息注入强度
            gate = torch.sigmoid(self.gate)
            attn_out = gate * attn_out

        out = hidden_states + attn_out
        out = self.ln(out)
        return out


class QwenWithVisionAdapter(nn.Module):
    """
    QwenWithVisionAdapter
    =====================
    该类将一个已经构建好的 Qwen CausalLM (通常是 QLoRA + PeftModel) 包装为
    一个带 Cross-Attn Vision Adapter 的多模态模型。

    用法示例 (在 builder.py 中类似如下)：
    ---------------------------------------
    base_model = build_qlora_base_model(...)
    vision_model = QwenWithVisionAdapter(
        llm=base_model,
        image_feat_dim=512,
        num_image_tokens=4,
        cross_attn_heads=8,
        cross_attn_dropout=0.1,
        use_image_feat=True,
        use_gate=True,
    )

    关键功能：
    ---------
    - 接收 batch 中的 "image_feat" 字段 (形状 [B, D_img])；
    - 将 image_feat 投影为若干个 image tokens: [B, M, H]；
    - 将 LLM 最后一层 hidden_states 与 image tokens 做一次 cross-attn 融合；
    - 使用融合后的 hidden_states 通过 lm_head 计算 logits，并手动计算 Causal LM loss。
    """

    def __init__(
        self,
        llm: PreTrainedModel,
        image_feat_dim: int,
        num_image_tokens: int = 4,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.0,
        use_image_feat: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()

        self.llm = llm  # 已经应用 QLoRA 的 QwenForCausalLM (PeftModel)
        self.config = getattr(llm, "config", None)

        if self.config is None or not hasattr(self.config, "hidden_size"):
            raise ValueError(
                "llm.config.hidden_size 未找到，请确认传入的是 HuggingFace CausalLM 模型。"
            )

        hidden_size = int(self.config.hidden_size)
        self.image_feat_dim = int(image_feat_dim)
        self.num_image_tokens = int(num_image_tokens)
        self.use_image_feat = bool(use_image_feat)

        # 用一个线性层把 image_feat: [B, D_img] 映射到 M 个 token 的拼接向量
        # 形状：[B, M * H]，再 reshape 成 [B, M, H]
        self.image_proj = nn.Linear(self.image_feat_dim, hidden_size * self.num_image_tokens)

        # Cross-Attn Adapter
        self.vision_adapter = VisionCrossAttnAdapterLayer(
            hidden_size=hidden_size,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
            use_gate=use_gate,
        )

    # ------------------------------------------------------------------
    # 可训练参数打印函数：配合 utils.model_utils.print_trainable_parameters 使用
    # ------------------------------------------------------------------
    def print_trainable_parameters(self):
        """
        打印 LoRA (文本侧) 与 Vision Adapter (视觉侧) 的可训练参数统计。

        - 如果 self.llm 自身已经是 PeftModel，会先调用其自带的
          print_trainable_parameters()，打印 LoRA_txt。
        - 然后再单独统计 Vision Adapter (image_proj + vision_adapter) 的参数量。
        """
        # 1) 文本侧 (通常是 QLoRA 的 peft_model)
        if hasattr(self.llm, "print_trainable_parameters"):
            print("====== [QwenWithVisionAdapter] 文本侧 LoRA / 可训练参数 ======")
            self.llm.print_trainable_parameters()

        # 2) 视觉侧 Adapter 参数统计
        vision_params = list(self.image_proj.parameters()) + list(self.vision_adapter.parameters())
        num_vision = sum(p.numel() for p in vision_params if p.requires_grad)
        total_vision = sum(p.numel() for p in vision_params)

        print("====== [QwenWithVisionAdapter] 视觉侧 Vision Adapter 参数 ======")
        print(
            f"vision trainable params: {num_vision:,} || "
            f"vision all params: {total_vision:,} || "
            f"trainable%: {100 * num_vision / max(total_vision, 1):.4f}"
        )

    # ------------------------------------------------------------------
    # 前向传播：训练时使用 Cross-Attn Adapter 融合视觉；推理时可选择退化为纯文本
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        image_feat: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        参数
        ----
        input_ids : LongTensor, [B, T]
            文本 token id。
        attention_mask : LongTensor, [B, T], 可选
            1 表示有效 token，0 表示 padding。
        labels : LongTensor, [B, T], 可选
            训练标签，通常与 input_ids 形状一致，padding 位置为 -100。
        image_feat : FloatTensor, [B, D_img], 可选
            预提取的图像特征（例如 BiomedCLIP 512 维 embedding）。

        说明
        ----
        - 当 labels 不为 None（训练 / eval）且 image_feat 存在且 use_image_feat=True 时，
          会启用 Vision Adapter，将图像特征融合进最后的 hidden_states。
        - 当 labels 为 None（通常是 generate 调用）时，为了保持行为简单可靠，
          当前版本直接退化为纯文本 LLM 的 forward（不做视觉融合）。
          你可以在后续版本里根据需要扩展 generate 对 vision 的支持。
        """
        past_key_values = kwargs.get("past_key_values", None)

        # 1) 几种情况直接退化为纯文本 LLM：
        #    - 不启用视觉
        #    - 没有图像特征
        #    - 使用 past_key_values（通常是 generate 的增量解码阶段）
        #    - 没有 labels（即推理阶段）
        if (
            (not self.use_image_feat)
            or (image_feat is None)
            or (past_key_values is not None)
            or (labels is None)
        ):
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # ------------------ 2. 以下是训练 / eval 阶段，启用 Vision Adapter ------------------
        if input_ids is None:
            raise ValueError("当使用 Vision Adapter 时，input_ids 不能为空。")

        # 规范 image_feat 形状到 [B, D_img]
        if image_feat.dim() == 1:
            image_feat = image_feat.unsqueeze(0)
        elif image_feat.dim() == 3:
            b, t, d = image_feat.shape
            if t != 1:
                raise ValueError(
                    f"[QwenWithVisionAdapter] image_feat 形状为 [B, {t}, {d}]，"
                    f"目前仅支持 t==1 的情况。"
                )
            image_feat = image_feat.view(b, d)

        if image_feat.dim() != 2:
            raise ValueError(
                f"[QwenWithVisionAdapter] image_feat 维度必须为 2，当前为 {image_feat.dim()}。"
            )

        bsz = image_feat.size(0)
        if attention_mask is None:
            # 若未显式传入 mask，则按 pad_token_id 自动创建
            pad_token_id = getattr(self.llm.config, "pad_token_id", -100)
            attention_mask = (input_ids != pad_token_id).long()

        # 2.1 先调用底层 LLM，获取最后一层 hidden_states
        #     注意：这里不传 labels，由我们自己计算 loss。
        base_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            labels=None,
            **{k: v for k, v in kwargs.items() if k != "past_key_values"},
        )

        # hidden_states: [B, T, H]
        if not hasattr(base_outputs, "hidden_states") or base_outputs.hidden_states is None:
            raise RuntimeError(
                "[QwenWithVisionAdapter] LLM 输出中没有 hidden_states，"
                "请确认 llm.forward(..., output_hidden_states=True) 是否生效。"
            )

        hidden_states = base_outputs.hidden_states[-1]  # [B, T, H]
        hidden_size = hidden_states.size(-1)

        # 2.2 将 image_feat 投影成 [B, M, H] 作为 image tokens
        img_tokens_flat = self.image_proj(image_feat)  # [B, M*H]
        img_tokens = img_tokens_flat.view(bsz, self.num_image_tokens, hidden_size)  # [B, M, H]

        # 2.3 使用 Cross-Attn Adapter 融合视觉信息
        fused_hidden = self.vision_adapter(hidden_states, img_tokens)  # [B, T, H]

        # 2.4 使用 LLM 的 lm_head 计算 logits
        if not hasattr(self.llm, "lm_head"):
            raise RuntimeError(
                "[QwenWithVisionAdapter] self.llm 没有 lm_head 属性，"
                "请确认传入的是 AutoModelForCausalLM 类型模型。"
            )

        logits = self.llm.lm_head(fused_hidden)  # [B, T, V]

        # 2.5 手动计算 Causal LM Loss (与 HF CausalLMLoss 对齐)
        #     标准做法：对 logits 向右平移一位，对 labels 向左平移一位。
        shift_logits = logits[..., :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()       # [B, T-1]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(T-1)), V]
            shift_labels.view(-1),                         # [(B*(T-1))]
            ignore_index=-100,
        )

        # 构造一个与 CausalLMOutputWithPast 接口兼容的输出
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
        return output


# -----------------------------------------------------------------------------
# 简单自测：在 __main__ 中构造一个假 LLM，验证形状与前向逻辑是否无误。
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    class DummyConfig:
        def __init__(self, hidden_size: int = 64, vocab_size: int = 1000, pad_token_id: int = 0):
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id

    class DummyLLM(nn.Module):
        """
        一个极简的 LLM，占位用于本文件的自测。
        - embed_tokens: 将 input_ids 映射到 hidden_states
        - lm_head: 将 hidden_states 映射到 logits
        - forward: 返回一个带有 hidden_states 的简单对象，模拟 HF 的输出结构。
        """
        def __init__(self, config: DummyConfig):
            super().__init__()
            self.config = config
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            self.ln = nn.LayerNorm(config.hidden_size)

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            output_hidden_states: bool = False,
            use_cache: bool = False,
            labels: Optional[Tensor] = None,
            **kwargs,
        ):
            # 简单地将 input_ids 映射到 embedding 并做一次 LayerNorm
            hidden = self.embed_tokens(input_ids)  # [B, T, H]
            hidden = self.ln(hidden)

            logits = self.lm_head(hidden)

            class Output:
                # 模拟 transformers 的输出结构
                def __init__(self, logits, hidden_states):
                    self.logits = logits
                    self.hidden_states = hidden_states
                    self.attentions = None

            if output_hidden_states:
                return Output(
                    logits=logits,
                    hidden_states=(hidden,),  # 只有一层，放在 tuple 里
                )
            else:
                return Output(
                    logits=logits,
                    hidden_states=None,
                )

    # ----------------------- 开始自测 -----------------------
    print("[SelfTest] 开始测试 QwenWithVisionAdapter + VisionCrossAttnAdapterLayer ...")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 构建 Dummy LLM 与 Vision Adapter 包装
    dummy_cfg = DummyConfig(hidden_size=32, vocab_size=128, pad_token_id=0)
    dummy_llm = DummyLLM(dummy_cfg).to(device)

    vision_model = QwenWithVisionAdapter(
        llm=dummy_llm,
        image_feat_dim=16,
        num_image_tokens=4,
        cross_attn_heads=4,
        cross_attn_dropout=0.0,
        use_image_feat=True,
        use_gate=True,
    ).to(device)

    # 2) 构造一批假数据
    batch_size = 2
    seq_len = 10
    image_dim = 16

    input_ids = torch.randint(0, dummy_cfg.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # labels 与 input_ids 相同，简单起见；padding 用 -100 以对齐 HF 逻辑
    labels = input_ids.clone()
    labels[input_ids == dummy_cfg.pad_token_id] = -100

    image_feat = torch.randn(batch_size, image_dim, device=device)

    # 3) 前向一遍，检查 loss 与 logits 形状
    outputs = vision_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_feat=image_feat,
    )

    print(f"[SelfTest] logits 形状: {tuple(outputs.logits.shape)}")
    print(f"[SelfTest] loss: {float(outputs.loss):.4f}")

    # 4) 反向传播看是否正常
    outputs.loss.backward()
    print("[SelfTest] 反向传播成功，梯度正常。")

    # 5) 打印可训练参数统计
    vision_model.print_trainable_parameters()

    print("[SelfTest] QwenWithVisionAdapter 自测完成 ✅")
