"""EKF-LSTM 融合模型。

实现完整的训练 + 在线预测流程：
1. 对整段轨迹运行一次 EKF，得到滤波估计序列 X_filt 与滤波误差 e。
2. 取前 40% 步 作为训练集，构造 LSTM 输入：
       特征 = 时刻 k-2/k-1/k 的 [P, e]（12维）
       标签 = 时刻 k+1 的滤波误差 e (6维)
3. 训练 LSTM，使其学习"基于历史 P 和 e 预测下一步滤波误差"的映射。
4. 在测试集上滚动预测：用上一轮 LSTM 输出的 ê(k+1) 与 k+1 时刻的 EKF 估计相加，
   得到融合预测 X_final(k+1)。

时序对齐：LSTM 在时刻 k 预测 ê(k+1)，缓存；
        在时刻 k+1 EKF 更新完毕后取出缓存与 X̂(k+1|k+1) 融合。
"""

from __future__ import annotations

import numpy as np
import torch

from config.params import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    LSTM_SEQ_LEN,
    PV_DIM,
    TRAIN_RATIO,
)
from data.preprocessor import MinMaxScaler
from models.dataset_builder import DatasetBuilder
from models.ekf_service import EKFService
from models.lstm import LSTMNet, lstm_predict, train_lstm


class EKFLSTM:
    """EKF + LSTM 误差补偿融合预测器。

    使用方式：
        ekflstm = EKFLSTM(model="CV", state_dim=6, T=0.05, sigma_w2=0.35, sigma_v2=0.05)
        result = ekflstm.run(Z_obs, X_true, X0)
        result["X_final"]  # 融合预测序列
    """

    def __init__(
        self,
        model: str,
        state_dim: int,
        T: float,
        sigma_w2: float,
        sigma_v2: float,
        seq_len: int = LSTM_SEQ_LEN,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        epochs: int = LSTM_EPOCHS,
        lr: float = LSTM_LR,
        batch_size: int = LSTM_BATCH_SIZE,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.state_dim = state_dim
        self.T = T
        self.sigma_w2 = sigma_w2
        self.sigma_v2 = sigma_v2
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.ekf_service = EKFService(model, state_dim, T, sigma_w2, sigma_v2)
        self.dataset_builder = DatasetBuilder(seq_len)

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.net: LSTMNet | None = None

    def run(
        self,
        Z_obs: np.ndarray,
        X_true: np.ndarray,
        X0: np.ndarray,
        train_ratio: float = TRAIN_RATIO,
        verbose: bool = False,
    ) -> dict:
        """端到端运行：EKF → 构造数据 → 训练 LSTM → 融合预测。

        Returns:
            dict 包含 X_filt（EKF 估计）、X_final（融合预测）、e_filt（滤波误差）、
            train_history（loss 曲线）等关键中间产物。
        """
        self._setup_random_seed()

        X_filt, e_filt = self.ekf_service.run(Z_obs, X0, X_true)
        assert e_filt is not None

        train_data = self._prepare_training_data(X_filt, e_filt, train_ratio, verbose)
        history = self._train_model(train_data, verbose)
        X_final, Y_pred = self._predict_and_fuse(X_filt, e_filt)

        return self._build_result(X_filt, X_final, e_filt, Y_pred, history, train_data["n_train"])

    def _setup_random_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _prepare_training_data(
        self,
        X_filt: np.ndarray,
        e_filt: np.ndarray,
        train_ratio: float,
        verbose: bool,
    ) -> dict:
        N = X_filt.shape[0]
        n_train = int(N * train_ratio)
        P_seq = X_filt[:, :PV_DIM]

        X_train, Y_train = self.dataset_builder.build(P_seq, e_filt, 0, n_train)

        if verbose:
            print(
                f"  [LSTM 训练数据] X={X_train.shape}, Y={Y_train.shape}"
                f"  (前 {n_train} 步用于训练)"
            )

        X_flat = X_train.reshape(-1, X_train.shape[-1])
        self.x_scaler.fit(X_flat)
        self.y_scaler.fit(Y_train)

        X_train_norm = self.x_scaler.transform(X_flat).reshape(X_train.shape)
        Y_train_norm = self.y_scaler.transform(Y_train)

        return {
            "X_train": X_train_norm,
            "Y_train": Y_train_norm,
            "n_train": n_train,
        }

    def _train_model(self, train_data: dict, verbose: bool) -> list[float]:
        self.net = LSTMNet(
            input_size=train_data["X_train"].shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=PV_DIM,
        )
        return train_lstm(
            self.net,
            train_data["X_train"],
            train_data["Y_train"],
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self.device,
            verbose=verbose,
        )

    def _predict_and_fuse(
        self, X_filt: np.ndarray, e_filt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        N = X_filt.shape[0]
        P_seq = X_filt[:, :PV_DIM]

        X_full, _ = self.dataset_builder.build(P_seq, e_filt, 0, N)
        X_full_norm = self.x_scaler.transform(
            X_full.reshape(-1, X_full.shape[-1])
        ).reshape(X_full.shape)
        Y_pred_norm = lstm_predict(self.net, X_full_norm, device=self.device)
        Y_pred = self.y_scaler.inverse_transform(Y_pred_norm)

        X_final = X_filt.copy()
        for idx in range(Y_pred.shape[0]):
            target_k = idx + self.seq_len
            if target_k >= N:
                break
            X_final[target_k, :PV_DIM] = X_filt[target_k, :PV_DIM] + Y_pred[idx]

        return X_final, Y_pred

    def _build_result(
        self,
        X_filt: np.ndarray,
        X_final: np.ndarray,
        e_filt: np.ndarray,
        lstm_pred: np.ndarray,
        train_history: list[float],
        n_train: int,
    ) -> dict:
        return {
            "X_filt": X_filt,
            "X_final": X_final,
            "e_filt": e_filt,
            "lstm_pred": lstm_pred,
            "n_train": n_train,
            "train_history": train_history,
            "valid_start": self.seq_len,
        }
