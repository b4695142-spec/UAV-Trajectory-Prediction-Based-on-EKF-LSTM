"""EKF 运行服务。

封装 EKF 滤波流程与初始协方差配置，将原 ekf_lstm.py 中的
全局函数 run_ekf / _build_P0 统一收归到 EKFService 类中，
实现单一职责与可独立测试。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.params import EKF_P0_SCALE, PV_DIM
from models.ekf import EKF
from models.motion_models import make_f_and_jacobian, radar_H, radar_h


class EKFService:
    """EKF 运行服务，封装滤波流程与初始协方差构造。"""

    def __init__(
        self,
        model: str,
        state_dim: int,
        T: float,
        sigma_w2: float,
        sigma_v2: float,
    ) -> None:
        self.model = model
        self.state_dim = state_dim
        self.T = T
        self.sigma_w2 = sigma_w2
        self.sigma_v2 = sigma_v2

    def build_initial_covariance(self) -> np.ndarray:
        """根据模型类型构造合理的初始协方差对角线。

        - CV：所有通道使用统一大初值，允许 EKF 快速收敛；
        - CA：位置/速度通道使用大初值（100），加速度通道使用小初值（0.01），
          因为采用"加速度恒定"的工程化 Γ_ca，加速度被视为已知的运动学参数；
        - CT：位置/速度通道使用大初值（100），角速度通道使用小初值（0.01），
          角速度被视为"匀转速运动学参数"，先验近似已知；若给定大初始方差，
          更新步会通过位置-角速度的非零互相关项把雷达观测残差错误分配到角速度
          通道，使滤波器发散。
        """
        P0 = np.full(self.state_dim, EKF_P0_SCALE, dtype=np.float64)
        if self.model.upper() in ("CA", "CT"):
            P0[6:9] = 0.01
        return P0

    def run(
        self,
        Z_obs: np.ndarray,
        X0: np.ndarray,
        X_true: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """对整段观测序列运行 EKF，返回滤波状态序列与滤波误差序列。

        Args:
            Z_obs: shape (N, 3) 观测序列
            X0: 初始状态
            X_true: 若提供则同步返回滤波误差 e(k) = X_true[k][:6] - X̂(k|k)[:6]
        """
        N = Z_obs.shape[0]
        P0 = self.build_initial_covariance()
        ekf = EKF(
            self.state_dim, 3, self.sigma_w2, self.sigma_v2, X0, P0_scale=P0
        )
        f_func, jac_func, Gamma = make_f_and_jacobian(self.model, self.T)

        X_filt = np.zeros((N, self.state_dim), dtype=np.float64)
        X_filt[0] = ekf.state

        for k in range(1, N):
            ekf.predict(f_func, Gamma, Phi=jac_func)
            H_k = radar_H(ekf.state, self.state_dim)
            ekf.update(Z_obs[k], radar_h, H_k)
            X_filt[k] = ekf.state

        e_filt: Optional[np.ndarray] = None
        if X_true is not None:
            e_filt = X_true[:, :PV_DIM] - X_filt[:, :PV_DIM]
        return X_filt, e_filt
