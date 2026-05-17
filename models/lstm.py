"""LSTM 网络模块。

实现两种使用模式：
1. 误差补偿模式（LSTMErrorCompensator）：
   输入特征 = 拼接的 [P(k-i), e(k-i)]（位置+速度+滤波误差，12维）
   输出     = 下一时刻的理论误差补偿值 ê(k+1)（6维）
   用于 EKF-LSTM 融合。

2. 独立预测模式（LSTMTrajectoryPredictor）：
   输入特征 = 历史位置-速度序列 P(k-i)（6维）
   输出     = 下一时刻的位置-速度 P̂(k+1)（6维）
   用作单一 LSTM 基线对比。

训练参数：Adam(lr=0.01), epochs=60, batch_size=16, RMSE 损失。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(nn.Module):
    """基础 LSTM 网络：LSTM 堆叠层 + 全连接输出层。

    论文未公开 hidden_size/num_layers 等具体值，
    本复现取 hidden_size=64, num_layers=2 作为合理工程值，并使用 tanh 作为输出激活。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 6,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """整体向量 RMSE：sqrt(mean((pred - target)²))。

    与论文式(3.33)一致：先对所有元素求 MSE，再开方。
    """
    return torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)


def train_lstm(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 60,
    lr: float = 0.01,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = False,
) -> list[float]:
    """通用 LSTM 训练循环。

    Args:
        model: LSTM 网络
        X_train: shape (M, seq_len, input_size)
        Y_train: shape (M, output_size)
    Returns:
        每个 epoch 的训练 loss 列表。
    """
    model.to(device)
    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    Y_tensor = torch.from_numpy(Y_train.astype(np.float32))
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = rmse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.6f}")
    return history


@torch.no_grad()
def lstm_predict(
    model: nn.Module, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """前向预测：X shape (M, seq_len, input_size) → (M, output_size)。"""
    model.eval()
    model.to(device)
    x = torch.from_numpy(X.astype(np.float32)).to(device)
    out = model(x).cpu().numpy()
    return out
