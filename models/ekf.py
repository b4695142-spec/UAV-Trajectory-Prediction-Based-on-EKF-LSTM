"""扩展卡尔曼滤波器（EKF）。

支持 CV / CA / CT 三种运动模型：
- CV/CA：状态转移函数为线性，雅可比矩阵 Φ 等于固定矩阵 F；
- CT：状态转移函数非线性，雅可比矩阵需由调用方提供（推荐复步差分计算）。

协方差更新采用 Joseph 形式：
    P(k|k) = (I - K·H)·P(k|k-1)·(I - K·H)ᵀ + K·R·Kᵀ
以保证对称性和正定性。

过程噪声协方差自动按 Q = σ_w²·Γ·Γᵀ 构造，与论文表述一致。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class EKF:
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        sigma_w2: float,
        sigma_v2: float,
        X0: np.ndarray,
        P0_scale: float | np.ndarray = 100.0,
    ) -> None:
        """
        Args:
            P0_scale: 标量（构造 P0 = scale·I）或长度 state_dim 的对角线向量。
                对 CT 模型，建议角速度通道使用较小值（如 0.01），因为 ω 是匀转速
                模型的先验参数而非高度不确定的状态；否则第一次更新时会把残差错误
                分配到角速度通道，导致滤波发散。
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.sigma_w2 = float(sigma_w2)
        self.sigma_v2 = float(sigma_v2)

        self.X = X0.astype(np.float64).copy()
        if np.isscalar(P0_scale):
            self.P = np.eye(state_dim, dtype=np.float64) * float(P0_scale)
        else:
            P0_arr = np.asarray(P0_scale, dtype=np.float64)
            if P0_arr.ndim == 1:
                self.P = np.diag(P0_arr)
            else:
                self.P = P0_arr.astype(np.float64).copy()

        self.R = np.eye(obs_dim, dtype=np.float64) * self.sigma_v2

    def predict(
        self,
        f_func: Callable[[np.ndarray], np.ndarray],
        Gamma: np.ndarray,
        Phi: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """预测步。

        Args:
            f_func: 状态转移函数 f(X)。CV/CA 可传 lambda X: F @ X；CT 传 ct_f。
            Gamma: 噪声驱动矩阵（state_dim × 3）。
            Phi: 雅可比矩阵 ∂f/∂X̂。可以是 ndarray（CV/CA）或函数（CT 在当前 X 处求解）。
        """
        if Phi is None:
            raise ValueError("EKF.predict 需要传入 Phi（CV/CA 直接传 F，CT 传 jacobian 函数）")

        if callable(Phi):
            Phi_mat = Phi(self.X)
        else:
            Phi_mat = Phi

        Q = self.sigma_w2 * (Gamma @ Gamma.T)

        self.X = f_func(self.X)
        self.P = Phi_mat @ self.P @ Phi_mat.T + Q

    def update(
        self,
        Z: np.ndarray,
        h_func: Callable[[np.ndarray], np.ndarray],
        H: np.ndarray,
    ) -> None:
        """更新步。

        Joseph 形式协方差更新保证对称正定。
        """
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        z_pred = h_func(self.X)
        innov = Z - z_pred
        if innov.shape[0] >= 3:
            innov[2] = np.arctan2(np.sin(innov[2]), np.cos(innov[2]))

        self.X = self.X + K @ innov

        I = np.eye(self.state_dim, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    @property
    def state(self) -> np.ndarray:
        return self.X.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
