"""数据预处理：min-max 归一化与反归一化。

关键约束：min/max 必须仅由训练集（前 40%）统计得到，
然后将同一组参数应用于测试集，避免数据泄露。
"""

from __future__ import annotations

import numpy as np


class MinMaxScaler:
    """逐特征 min-max 归一化器，将每一列映射到 [0, 1]。"""

    def __init__(self, eps: float = 1e-8):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """根据训练数据 X (n_samples, n_features) 计算 min/max。"""
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "请先调用 fit"
        denom = (self.max_ - self.min_)
        denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "请先调用 fit"
        return X_scaled * (self.max_ - self.min_) + self.min_


def make_seq_dataset(
    features: np.ndarray, targets: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """将单步特征/目标重塑为 LSTM 训练所需的滑动窗口。

    features: shape (N, F)
    targets:  shape (N, O)
    返回：
        X_seq: shape (M, seq_len, F)
        Y_seq: shape (M, O)，其中 Y_seq[i] = targets[i + seq_len]，即预测窗口的下一时刻
    """
    N = features.shape[0]
    M = N - seq_len
    if M <= 0:
        raise ValueError(f"样本数 {N} 不足以构造长度 {seq_len} 的滑动窗口")
    X_seq = np.zeros((M, seq_len, features.shape[1]), dtype=np.float64)
    Y_seq = np.zeros((M, targets.shape[1]), dtype=np.float64)
    for i in range(M):
        X_seq[i] = features[i : i + seq_len]
        Y_seq[i] = targets[i + seq_len]
    return X_seq, Y_seq
