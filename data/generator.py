"""轨迹数据生成模块。

依据运动模型递推无人机的真实状态序列 X_true，
并按雷达观测方程 Z(k) = h(X(k)) + V(k) 生成带噪声的观测序列。

生成过程严格遵循实施方案 §3.2 模块6 的参数表，
其中 CT 模型每步需要从当前状态重新计算 F_ct。
"""

from __future__ import annotations

import numpy as np

from models.motion_models import (
    ca_F,
    ca_Gamma,
    ct_F,
    ct_Gamma,
    cv_F,
    cv_Gamma,
    radar_h,
)


def generate_trajectory(
    model: str,
    X0: np.ndarray,
    T: float,
    N: int,
    sigma_w2: float,
    sigma_v2: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成给定运动模型下的真实状态序列与观测序列。

    Args:
        model: "CV" / "CA" / "CT"
        X0: 初始状态向量
        T: 探测间隔（s）
        N: 总采样步数
        sigma_w2: 系统噪声方差
        sigma_v2: 观测噪声方差
        seed: 随机数种子（保证可复现）

    Returns:
        X_true: shape (N, state_dim)，真实状态序列
        Z_obs:  shape (N, 3)，雷达观测序列 (r, θ, φ)
    """
    rng = np.random.default_rng(seed)
    name = model.upper()
    state_dim = X0.shape[0]

    if name == "CV":
        F = cv_F(T)
        Gamma = cv_Gamma(T)
    elif name == "CA":
        F = ca_F(T)
        Gamma = ca_Gamma(T)
    elif name == "CT":
        F = None
        Gamma = ct_Gamma(T)
    else:
        raise ValueError(f"未知模型: {model}")

    X_true = np.zeros((N, state_dim), dtype=np.float64)
    Z_obs = np.zeros((N, 3), dtype=np.float64)

    X = X0.astype(np.float64).copy()
    X_true[0] = X
    Z_obs[0] = radar_h(X) + rng.normal(0.0, np.sqrt(sigma_v2), size=3)

    sigma_w = np.sqrt(sigma_w2)

    for k in range(1, N):
        W = rng.normal(0.0, sigma_w, size=3)
        if name == "CT":
            F_ct = ct_F(X, T)
            X = F_ct @ X + Gamma @ W
        else:
            X = F @ X + Gamma @ W
        X_true[k] = X
        V = rng.normal(0.0, np.sqrt(sigma_v2), size=3)
        Z_obs[k] = radar_h(X) + V

    return X_true, Z_obs


def init_state_from_obs(Z1: np.ndarray, state_dim: int) -> np.ndarray:
    """根据首次雷达观测 Z(1)=(r,θ,φ) 反推初始状态向量。

    位置：x = r·sin(θ)·cos(φ), y = r·sin(θ)·sin(φ), z = r·cos(θ)
    速度/加速度/角速度分量初始化为 0。
    """
    r, theta, phi = Z1[0], Z1[1], Z1[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    X0 = np.zeros(state_dim, dtype=np.float64)
    X0[0] = x
    X0[1] = y
    X0[2] = z
    return X0
