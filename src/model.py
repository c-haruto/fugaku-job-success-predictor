"""ジョブ成功/失敗を二値分類する多層パーセプトロン（MLP）の定義。"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config


class JobSuccessMLP(nn.Module):
    """
    シンプルな全結合ニューラルネットワーク。
    出力はロジット（シグモイド適用前の値）。損失関数側でシグモイドを含む
    BCEWithLogitsLoss を使うことで数値的に安定させる。
    """

    def __init__(self, input_dim: int, hidden_dims=config.HIDDEN_DIMS, dropout: float = config.DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
