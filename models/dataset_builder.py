"""LSTM 误差补偿数据集构造器。

将原 ekf_lstm.py 中的全局函数 _build_lstm_dataset 封装为
DatasetBuilder 类，实现单一职责、输入验证与可独立测试。
"""

from __future__ import annotations

import numpy as np


class DatasetBuilder:
    """LSTM 数据集构造器，负责构造误差补偿模型的训练/测试样本。

    输入特征：第 k-seq_len+1 ... k 步的 [P, e]（拼接），形状 (M, seq_len, 12)
    标签    ：第 k+1 步的 e，形状 (M, 6)
    """

    def __init__(self, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len 必须 >= 1，当前值: {seq_len}")
        self.seq_len = seq_len

    def build(
        self,
        P_seq: np.ndarray,
        e_seq: np.ndarray,
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """构造 LSTM 训练样本。

        Args:
            P_seq: 协方差序列，形状 (N, 6)
            e_seq: 误差序列，形状 (N, 6)
            start: 滑动窗口在 P_seq 中的起始索引
            end: 滑动窗口在 P_seq 中的结束索引（半开区间）

        Returns:
            X: 特征序列，形状 (M, seq_len, 12)
            Y: 标签序列，形状 (M, 6)
        """
        self._validate_inputs(P_seq, e_seq, start, end)

        samples_X: list[np.ndarray] = []
        samples_Y: list[np.ndarray] = []
        for k in range(start + self.seq_len - 1, end - 1):
            feat = np.concatenate(
                [
                    P_seq[k - self.seq_len + 1 : k + 1],
                    e_seq[k - self.seq_len + 1 : k + 1],
                ],
                axis=1,
            )
            samples_X.append(feat)
            samples_Y.append(e_seq[k + 1])
        X = np.stack(samples_X, axis=0)
        Y = np.stack(samples_Y, axis=0)
        return X, Y

    def _validate_inputs(
        self,
        P_seq: np.ndarray,
        e_seq: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        if P_seq.shape[0] != e_seq.shape[0]:
            raise ValueError(
                f"P_seq 和 e_seq 的长度必须相同，"
                f"当前 P_seq={P_seq.shape[0]}，e_seq={e_seq.shape[0]}"
            )
        if start < 0:
            raise ValueError(f"start 必须 >= 0，当前值: {start}")
        if end < start + self.seq_len + 1:
            raise ValueError(
                f"end 必须 >= start + seq_len + 1 "
                f"({start + self.seq_len + 1})，当前值: {end}"
            )
        if end > P_seq.shape[0]:
            raise ValueError(
                f"end 不能超过序列长度 ({P_seq.shape[0]})，当前值: {end}"
            )
