# models/qwen_vision_prefix.py
# -*- coding: utf-8 -*-
"""
QwenVisionPrefixModel (显式 global token + (N-1) 个 slot token 版)

核心思路：
- 冻结/微调好的 Qwen LLM 作为基础 `llm`；
- 额外挂一个「视觉前缀模块」：
    - 输入：预提取的图像特征 image_feat，形状 [B, D_img]（例如 BiomedCLIP 的 512 维特征）
    - 生成：
        - 1 个显式的 global token
        - (N-1) 个 slot token
    - 组合得到 [B, N_prefix, hidden_size] 的 prefix token 序列
    - 在 embedding 维度上拼接到文本 token 前面
    - attention_mask / labels 相应在前面补一段前缀的位置

本文件实现的是：
- 单一视觉向量 -> 1 个 global token + (N-1) 个 slot token 的多 prefix 版本
- 兼容你现有的 Qwen + LoRA 训练流程和 HF Trainer
- 保证代码结构清晰、注释详细，并附带一个 __main__ 下的自测 Demo
"""

from __future__ import annotations

from typing import Optional, Any, Dict

import torch
import torch.nn as nn
from torch import Tensor


class QwenVisionPrefixModel(nn.Module):
    """
    一个轻量的「视觉前缀」包装器，将任意 Causal LLM (如 Qwen) 扩展为 Vision-Language 模型。

    参数
    ----
    llm : nn.Module
        已加载好的语言模型（通常是 Qwen + LoRA / QLoRA），
        要求至少包含：
            - `config.hidden_size`
            - `get_input_embeddings()` / `set_input_embeddings()`
            - `forward`(input_ids=..., inputs_embeds=..., attention_mask=..., labels=..., **kwargs)
    image_feat_dim : int
        图像特征维度 D_img（例如 BiomedCLIP 的 512）。
    prefix_dropout : float
        前缀 token 的 dropout 概率，用于正则化。
    use_image_feat : bool
        是否在 forward 中实际使用 image_feat。如果为 False，则退化为纯文本 LLM。
    num_prefix_tokens : int, 默认 1
        使用多少个视觉 prefix token。
        - num_prefix_tokens=1：
            只有 1 个 global token，无 slot token；
        - num_prefix_tokens>1：
            第 0 个 token 是 global，其余 (N-1) 个是 slot token。
    """

    def __init__(
        self,
        llm: nn.Module,
        image_feat_dim: int,
        prefix_dropout: float = 0.0,
        use_image_feat: bool = True,
        num_prefix_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.llm = llm
        if not hasattr(self.llm, "config") or not hasattr(self.llm.config, "hidden_size"):
            raise ValueError(
                "QwenVisionPrefixModel 期望底层 llm 拥有 `config.hidden_size` 属性，"
                "请确认传入的是 HuggingFace 风格的 CausalLM 模型。"
            )

        hidden_size = self.llm.config.hidden_size

        # 视觉前缀相关配置
        self.image_feat_dim: int = int(image_feat_dim)
        self.num_prefix_tokens: int = int(num_prefix_tokens)
        self.use_image_feat: bool = bool(use_image_feat)

        if self.num_prefix_tokens <= 0:
            raise ValueError("num_prefix_tokens 必须是正整数。")

        # 显式 global token：将 [B, D_img] -> [B, hidden]
        self.global_proj = nn.Linear(self.image_feat_dim, hidden_size)

        # slot token：当 num_prefix_tokens > 1 时，使用一层线性层生成 (N-1) 个 slot
        # [B, D_img] -> [B, (N-1)*hidden] -> [B, N-1, hidden]
        if self.num_prefix_tokens > 1:
            self.slot_proj = nn.Linear(
                self.image_feat_dim,
                hidden_size * (self.num_prefix_tokens - 1),
            )
        else:
            self.slot_proj = None  # 不需要 slot token

        self.prefix_dropout = nn.Dropout(prefix_dropout)

        # 方便外部直接访问 config（例如 Trainer / 其它工具有时会用到）
        # 这里直接透传底层 llm 的 config
        self.config = getattr(self.llm, "config", None)

    # ------------------------------------------------------------------
    # 核心前向：在文本 embedding 前拼接 [global_token] + [slot_tokens]
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        image_feat: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        """
        参数
        ----
        input_ids : LongTensor of shape (B, L)
            文本 token id。
        attention_mask : LongTensor of shape (B, L), 可选
            标准的 attention mask，1 表示可见，0 表示 padding。
            若为 None，则内部会根据 input_ids 自动创建全 1 mask。
        labels : LongTensor of shape (B, L), 可选
            语言模型训练的标签，通常与 input_ids 形状一致，
            使用 -100 作为 ignore_index。
        image_feat : FloatTensor of shape (B, D_img), 可选
            预提取的图像特征。如果 use_image_feat=False 或 image_feat=None，
            则退化为纯文本 LLM。
        **kwargs :
            其余参数全部透传给底层 llm（例如 past_key_values、position_ids 等）。

        返回
        ----
        直接返回底层 llm 的输出对象（通常包含 loss / logits 等）。
        """

        # HF generate() 在第二步以后会传入 past_key_values；
        # 对于这种情况，说明 prefix 已经在第一步被编码进 KV cache，
        # 之后的 step 不要再重复拼接视觉前缀，直接走纯文本路径即可。
        past_key_values = kwargs.get("past_key_values", None)

        # 不使用视觉信息 / 没有图像特征 / 已经进入增量生成阶段：
        # -> 直接调用底层 LLM 的 forward，不做任何改动。
        if (not self.use_image_feat) or (image_feat is None) or (past_key_values is not None):
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # ===========================
        # 以下：真正执行「global + slot tokens」逻辑
        # ===========================
        if input_ids is None:
            raise ValueError("QwenVisionPrefixModel 目前要求在训练阶段必须提供 input_ids。")

        # 1. 规范 image_feat 形状：期望为 [B, D_img]
        if image_feat.dim() == 1:
            # 单样本 [D_img] -> [1, D_img]
            image_feat = image_feat.unsqueeze(0)
        elif image_feat.dim() == 3:
            # 有些情况可能是 [B, 1, D_img]，这里压掉中间那一维
            b, t, d = image_feat.shape
            if t != 1:
                raise ValueError(f"期望 image_feat 形状为 [B, D_img] 或 [B, 1, D_img]，但收到 {tuple(image_feat.shape)}")
            image_feat = image_feat.view(b, d)

        if image_feat.dim() != 2:
            raise ValueError(f"image_feat 期望维度为 [B, D_img]，但收到 {tuple(image_feat.shape)}")

        if image_feat.size(0) != input_ids.size(0):
            raise ValueError(
                f"image_feat batch 大小 ({image_feat.size(0)}) 与 input_ids batch 大小 ({input_ids.size(0)}) 不一致。"
            )

        bsz = image_feat.size(0)
        hidden_size = self.llm.config.hidden_size

        # 2. 生成 global token: [B, D_img] -> [B, hidden] -> [B, 1, hidden]
        global_token = self.global_proj(image_feat)           # [B, H]
        global_token = global_token.view(bsz, 1, hidden_size) # [B, 1, H]

        # 3. 生成 slot tokens（如果需要）：[B, D_img] -> [B, (N-1)*H] -> [B, N-1, H]
        if self.num_prefix_tokens > 1:
            if self.slot_proj is None:
                raise RuntimeError("num_prefix_tokens > 1，但 slot_proj 尚未初始化。")
            slot_tokens = self.slot_proj(image_feat)  # [B, (N-1)*H]
            slot_tokens = slot_tokens.view(bsz, self.num_prefix_tokens - 1, hidden_size)  # [B, N-1, H]
            # 组合 prefix: [ global_token, slot_tokens ]
            prefix = torch.cat([global_token, slot_tokens], dim=1)  # [B, N_prefix, H]
        else:
            # 只有 1 个 global token，无 slot token
            prefix = global_token  # [B, 1, H]

        # 4. dropout + dtype 对齐
        prefix = self.prefix_dropout(prefix)

        input_embed_layer = self.llm.get_input_embeddings()
        embed_dtype = input_embed_layer.weight.dtype
        prefix = prefix.to(dtype=embed_dtype)

        # 5. 计算原始文本 token 的 embedding：[B, L, hidden]
        input_embeds = input_embed_layer(input_ids)
        # 6. 在序列前面拼接视觉 prefix token -> [B, N_prefix + L, hidden]
        inputs_embeds = torch.cat([prefix, input_embeds], dim=1)

        # 7. 构造新的 attention_mask
        if attention_mask is None:
            # 如果没有传 mask，就假设所有 token 均有效，构造全 1 mask
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        prefix_mask = torch.ones(
            (attention_mask.size(0), self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )  # [B, N_prefix]
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, N_prefix + L]

        # 8. 构造新的 labels：前缀部分全部填 -100（不参与 loss）
        new_labels = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels 形状 {tuple(labels.shape)} 与 input_ids 形状 {tuple(input_ids.shape)} 不一致。"
                )

            ignore_prefix = torch.full(
                (labels.size(0), self.num_prefix_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )  # [B, N_prefix]
            new_labels = torch.cat([ignore_prefix, labels], dim=1)  # [B, N_prefix + L]

        # 9. 调用底层 LLM：此时不再传 input_ids，而是传 inputs_embeds
        #    其余参数（如 past_key_values、position_ids 等）全部透传。
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

        return outputs

    # ------------------------------------------------------------------
    # 一些便利函数：直接转发给内部 LLM，方便和 HF Trainer 等工具配合
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        """
        方便外部访问 / 修改 embedding 层（直接转发给内部 LLM）。
        """
        if hasattr(self.llm, "get_input_embeddings"):
            return self.llm.get_input_embeddings()
        raise AttributeError("内部 llm 不支持 get_input_embeddings()")

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """
        方便外部替换 embedding 层（直接转发给内部 LLM）。
        """
        if hasattr(self.llm, "set_input_embeddings"):
            return self.llm.set_input_embeddings(new_embeddings)
        raise AttributeError("内部 llm 不支持 set_input_embeddings()")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        """
        HF Trainer 在开启 gradient_checkpointing 时，会调用 model.gradient_checkpointing_enable()。
        这里直接转发给内部 LLM，保证兼容性。
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


# =============================================================================
# 自测代码：不依赖真实 Qwen / HF 模型，跑一个最小 demo
# =============================================================================
if __name__ == "__main__":
    """
    这个自测主要验证：
    1. 显式 global + slot token 逻辑在形状上是自洽的；
    2. labels / attention_mask 的对齐没有问题；
    3. 可以正常反向传播。
    """
    print("[SelfTest] 开始 QwenVisionPrefixModel (global + slot tokens) 版本的自测...")

    class DummyConfig:
        def __init__(self, hidden_size: int, vocab_size: int):
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.pad_token_id = 0

    class DummyLLM(nn.Module):
        """
        一个极简的 CausalLM，用来本地自测，不依赖 transformers。
        - embedding: vocab -> hidden
        - lm_head: hidden -> vocab
        - forward: 支持 input_ids / inputs_embeds + labels
        """

        def __init__(self, hidden_size: int = 32, vocab_size: int = 100):
            super().__init__()
            self.config = DummyConfig(hidden_size=hidden_size, vocab_size=vocab_size)
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def get_input_embeddings(self):
            return self.embed

        def set_input_embeddings(self, new_embeddings: nn.Embedding):
            self.embed = new_embeddings

        def forward(
            self,
            input_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            **kwargs: Any,
        ):
            # 简单的 LM：对每个位置做一个线性分类
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("DummyLLM: 必须提供 input_ids 或 inputs_embeds 之一。")
                x = self.embed(input_ids)
            else:
                x = inputs_embeds

            logits = self.lm_head(x)  # [B, L, V]
            loss = None
            if labels is not None:
                vocab_size = logits.size(-1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, vocab_size),  # [B*L, V]
                    labels.view(-1),              # [B*L]
                )

            # 返回一个简单的对象，模拟 transformers 的输出结构
            return type("DummyOutput", (), {"loss": loss, "logits": logits})

    # 1. 构造一个 Dummy LLM 和 VisionPrefix 包装器
    dummy_llm = DummyLLM(hidden_size=32, vocab_size=100)
    model = QwenVisionPrefixModel(
        llm=dummy_llm,
        image_feat_dim=8,
        prefix_dropout=0.1,
        use_image_feat=True,
        num_prefix_tokens=4,  # 1 global + 3 slot tokens
    )

    # 2. 构造一批假数据：input_ids, labels, image_feat
    batch_size = 2
    seq_len = 5
    img_dim = 8

    input_ids = torch.randint(low=0, high=dummy_llm.config.vocab_size, size=(batch_size, seq_len))
    labels = input_ids.clone()
    image_feat = torch.randn(batch_size, img_dim)

    # 3. 前向 & 反向一遍，检查形状与 loss
    outputs = model(
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
        image_feat=image_feat,
    )

    print(f"[SelfTest] 输出 logits 形状: {tuple(outputs.logits.shape)}")
    print(f"[SelfTest] loss: {float(outputs.loss):.4f}")

    outputs.loss.backward()
    print("[SelfTest] 反向传播成功，参数梯度正常。")
    print("[SelfTest] QwenVisionPrefixModel (global + slot tokens) 自测完成 ✅")
