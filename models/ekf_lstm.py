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
    EKF_P0_SCALE,
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
from models.ekf import EKF
from models.lstm import LSTMNet, lstm_predict, train_lstm
from models.motion_models import (
    make_f_and_jacobian,
    radar_H,
    radar_h,
)


def _build_P0(model: str, state_dim: int) -> np.ndarray:
    """根据模型类型构造合理的初始协方差对角线。

    - CV：所有通道使用统一大初值，允许 EKF 快速收敛；
    - CA：位置/速度通道使用大初值（100），加速度通道使用小初值（0.01）。
      因为采用"加速度恒定"的工程化 Γ_ca，加速度被视为已知的运动学参数；
    - CT：位置/速度通道使用大初值（100），角速度通道使用小初值（0.01）。
      角速度被视为"匀转速运动学参数"，先验近似已知；若给定大初始方差，
      更新步会通过位置-角速度的非零互相关项把雷达观测残差错误分配到角速度
      通道，使滤波器发散。
    """
    P0 = np.full(state_dim, EKF_P0_SCALE, dtype=np.float64)
    if model.upper() in ("CA", "CT"):
        P0[6:9] = 0.01
    return P0


def run_ekf(
    model: str,
    Z_obs: np.ndarray,
    X0: np.ndarray,
    state_dim: int,
    T: float,
    sigma_w2: float,
    sigma_v2: float,
    X_true: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """对整段观测序列运行 EKF，返回滤波状态序列与滤波误差序列。

    Args:
        model: CV/CA/CT
        Z_obs: shape (N, 3) 观测序列
        X0: 初始状态
        X_true: 若提供则同步返回滤波误差 e(k) = X_true[k][:6] - X̂(k|k)[:6]
    """
    N = Z_obs.shape[0]
    P0 = _build_P0(model, state_dim)
    ekf = EKF(state_dim, 3, sigma_w2, sigma_v2, X0, P0_scale=P0)
    f_func, jac_func, Gamma = make_f_and_jacobian(model, T)

    X_filt = np.zeros((N, state_dim), dtype=np.float64)
    X_filt[0] = ekf.state

    for k in range(1, N):
        ekf.predict(f_func, Gamma, Phi=jac_func)
        H_k = radar_H(ekf.state, state_dim)
        ekf.update(Z_obs[k], radar_h, H_k)
        X_filt[k] = ekf.state

    e_filt: np.ndarray | None = None
    if X_true is not None:
        e_filt = X_true[:, :PV_DIM] - X_filt[:, :PV_DIM]
    return X_filt, e_filt


def _build_lstm_dataset(
    P_seq: np.ndarray,
    e_seq: np.ndarray,
    seq_len: int,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造误差补偿 LSTM 的训练样本。

    输入特征：第 k-seq_len+1 ... k 步的 [P, e]（拼接），形状 (M, seq_len, 12)
    标签    ：第 k+1 步的 e，形状 (M, 6)

    start/end 指定滑动窗口在 P_seq 中的索引范围（半开区间），
    要求 end >= start + seq_len + 1。
    """
    samples_X = []
    samples_Y = []
    for k in range(start + seq_len - 1, end - 1):
        feat = np.concatenate(
            [P_seq[k - seq_len + 1 : k + 1], e_seq[k - seq_len + 1 : k + 1]],
            axis=1,
        )
        samples_X.append(feat)
        samples_Y.append(e_seq[k + 1])
    X = np.stack(samples_X, axis=0)
    Y = np.stack(samples_Y, axis=0)
    return X, Y


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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        N = Z_obs.shape[0]
        n_train = int(N * train_ratio)

        X_filt, e_filt = run_ekf(
            self.model,
            Z_obs,
            X0,
            self.state_dim,
            self.T,
            self.sigma_w2,
            self.sigma_v2,
            X_true=X_true,
        )
        assert e_filt is not None
        P_seq = X_filt[:, :PV_DIM]

        X_train, Y_train = _build_lstm_dataset(
            P_seq, e_filt, self.seq_len, start=0, end=n_train
        )
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

        self.net = LSTMNet(
            input_size=X_train.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=PV_DIM,
        )
        history = train_lstm(
            self.net,
            X_train_norm,
            Y_train_norm,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self.device,
            verbose=verbose,
        )

        X_full, _ = _build_lstm_dataset(
            P_seq, e_filt, self.seq_len, start=0, end=N
        )
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
            X_final[target_k, :PV_DIM] = (
                X_filt[target_k, :PV_DIM] + Y_pred[idx]
            )

        return {
            "X_filt": X_filt,
            "X_final": X_final,
            "e_filt": e_filt,
            "lstm_pred": Y_pred,
            "n_train": n_train,
            "train_history": history,
            "valid_start": self.seq_len,
        }
