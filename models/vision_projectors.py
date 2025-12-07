# models/vision_projectors.py
# -*- coding: utf-8 -*-
"""
一组可插拔的视觉投影模块 (Projector)：
- LinearProjector : 单层线性层，对应你现在的实现 (baseline)
- MLPProjector    : Linear -> GELU -> Linear，对齐 LLaVA-1.5 的常用做法
- MultiHeadProjector : 为每个 image token 单独建一个小 MLP（更细粒度）
- MoEProjector    : 简单的 Mixture-of-Experts Projector，用于“专病专治”

所有 Projector 的输入 / 输出约定：
    输入:  x  [B, input_dim]，例如 BiomedCLIP 的全局特征 [B, 512]
    输出:  y  [B, hidden_size * num_tokens]
        - 在 QwenWithVisionAdapter 里通常会再 reshape:
              y.view(B, num_tokens, hidden_size)
"""

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjector(nn.Module):
    """
    最基础的线性投影： y = Wx + b
    这一版等价于你现在在 QwenWithVisionAdapter 里用的 nn.Linear。
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim] -> [B, output_dim]
        return self.proj(x)


class MLPProjector(nn.Module):
    """
    MLP 版 Projector，对齐 LLaVA-1.5：
        Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)
    通常 hidden_dim 可以等于 input_dim，或略微放大。
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim  # 一个简单但效果不错的默认设置
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim] -> [B, output_dim]
        return self.net(x)


class MultiHeadProjector(nn.Module):
    """
    为每个“图像 token”单独建一个小 MLP：
        - Projector 数量 = num_tokens
        - 每个 Projector 输出一个 hidden_size 维的 token 向量
    最终会 flatten 成 [B, hidden_size * num_tokens]，与其余 Projector 对齐。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_tokens: int = 4,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_tokens = int(num_tokens)
        if inner_dim is None:
            inner_dim = input_dim

        # 为每个 token 建一个小 MLP: Linear -> GELU -> Linear
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, hidden_size),
                )
                for _ in range(self.num_tokens)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        return: [B, hidden_size * num_tokens]
        """
        B = x.size(0)
        token_list = []
        for mlp in self.projectors:
            token = mlp(x)          # [B, hidden_size]
            token_list.append(token.unsqueeze(1))  # [B, 1, hidden_size]

        tokens = torch.cat(token_list, dim=1)      # [B, num_tokens, hidden_size]
        return tokens.reshape(B, self.num_tokens * self.hidden_size)


class MoEProjector(nn.Module):
    """
    简单版 Mixture-of-Experts Projector（MoE）：
        - 有 num_experts 个专家，每个专家是一个 Linear: input_dim -> output_dim
        - 通过 Router (Linear: input_dim -> num_experts) 决定每个样本用哪些专家
        - 采用 Top-K Routing：每个样本只激活 k 个专家，计算量可控

    说明：
    - 这里实现的是“参数稀疏、计算部分稀疏”的轻量 MoE，足以覆盖医学多模态场景。
    - 为了可读性，我们采用 PyTorch 的 for-loop 实现；在 num_experts = 4、
      batch_size 较小的场景里完全够用。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        k: int = 2,
    ) -> None:
        super().__init__()
        assert num_experts >= 1, "num_experts 必须 >= 1"
        assert 1 <= k <= num_experts, "k 必须在 [1, num_experts] 之间"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = int(num_experts)
        self.k = int(k)

        # Router: input_dim -> num_experts
        self.router = nn.Linear(input_dim, num_experts)

        # Experts: 每个专家一个独立的线性层
        # 为了方便使用 F.linear，这里权重形状是 [num_experts, output_dim, input_dim]
        self.expert_weight = nn.Parameter(
            torch.empty(self.num_experts, output_dim, input_dim)
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts, output_dim)
        )

        # 使用 xavier 初始化专家权重
        nn.init.xavier_uniform_(self.expert_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        return: [B, output_dim]
        """
        B, D = x.shape
        assert D == self.input_dim, f"MoEProjector 输入维度不匹配: 期望 {self.input_dim}, 实际 {D}"

        # 1. Router 计算每个样本在各个专家上的 logit： [B, num_experts]
        router_logits = self.router(x)

        # 2. Top-K 选择：selected_experts: [B, k], routing_weights: [B, k]
        routing_logits, selected_experts = torch.topk(router_logits, self.k, dim=-1)
        routing_weights = F.softmax(routing_logits, dim=-1)  # 归一化成权重

        # 3. 按专家聚合计算：最终输出 [B, output_dim]
        final = x.new_zeros(B, self.output_dim)

        # 对每个 expert 单独处理，便于理解与调试
        for expert_id in range(self.num_experts):
            # mask: [B, k] -> 当前 expert 被选中的位置为 True
            mask = (selected_experts == expert_id)  # bool

            if not mask.any():
                continue  # 这个 expert 在当前 batch 中没有被选中

            # 每个样本在这个 expert 上的总权重（通常只有一个位置为非零）
            # shape: [B]
            weight_per_sample = (routing_weights * mask.float()).sum(dim=-1)

            # 找到真正需要 this expert 的样本索引
            active_idx = torch.nonzero(weight_per_sample > 0, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                continue

            x_e = x[active_idx]  # [B_e, input_dim]

            # 线性变换：F.linear 的参数形状是 (out_features, in_features)
            w = self.expert_weight[expert_id]  # [output_dim, input_dim]
            b = self.expert_bias[expert_id]    # [output_dim]
            out_e = F.linear(x_e, w, b)        # [B_e, output_dim]

            # 对输出加权并写回 final
            final[active_idx] += out_e * weight_per_sample[active_idx].unsqueeze(-1)

        return final


# 一个简单的工厂函数：根据字符串创建 Projector
def build_projector(
    projector_type: Literal["linear", "mlp", "multihead", "moe"],
    input_dim: int,
    hidden_size: int,
    num_tokens: int,
    mlp_hidden_dim: Optional[int] = None,
    multihead_inner_dim: Optional[int] = None,
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
) -> nn.Module:
    """
    根据 projector_type 构造对应的 Projector。

    返回的模块统一满足：
        forward(x: [B, input_dim]) -> [B, hidden_size * num_tokens]
    """
    projector_type = projector_type.lower()
    output_dim = hidden_size * num_tokens

    if projector_type == "linear":
        return LinearProjector(input_dim, output_dim)
    elif projector_type == "mlp":
        return MLPProjector(input_dim, output_dim, hidden_dim=mlp_hidden_dim)
    elif projector_type == "multihead":
        return MultiHeadProjector(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            inner_dim=multihead_inner_dim,
        )
    elif projector_type == "moe":
        return MoEProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=moe_num_experts,
            k=moe_top_k,
        )
    else:
        raise ValueError(f"未知的 projector_type: {projector_type!r}，应为 ['linear', 'mlp', 'multihead', 'moe'] 之一。")


# =========================================================
# 自测脚本：python models/vision_projectors.py
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    input_dim = 512
    hidden_size = 4096
    num_tokens = 4

    x = torch.randn(B, input_dim)

    for proj_type in ["linear", "mlp", "multihead", "moe"]:
        print(f"\n[TEST] projector_type = {proj_type}")

        projector = build_projector(
            projector_type=proj_type,
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            mlp_hidden_dim=None,
            multihead_inner_dim=None,
            moe_num_experts=4,
            moe_top_k=2,
        )

        y = projector(x)
        print("  input shape :", tuple(x.shape))
        print("  output shape:", tuple(y.shape))

        # 进一步检查是否可以 reshape 成 [B, num_tokens, hidden_size]
        y_tokens = y.view(B, num_tokens, hidden_size)
        print("  tokens shape:", tuple(y_tokens.shape))

    print("\n[TEST] 所有 Projector 形状检查通过。")
