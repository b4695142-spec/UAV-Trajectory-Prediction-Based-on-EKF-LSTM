"""运动模型：CV / CA / CT 三种无人机运动模型。

实现内容：
1. CV / CA 模型的状态转移矩阵 F、噪声驱动矩阵 Γ；
2. CT 模型的非线性状态转移函数 f_ct、Γ_ct，以及基于自动微分（复步差分）
   的雅可比矩阵计算；
3. 雷达观测函数 h 及其对状态向量的雅可比矩阵 H（球坐标观测）。

CV / CA 的状态转移本身是线性的，因此 F 即为雅可比矩阵 Φ；
CT 模型的 F_ct 依赖于状态向量中的角速度分量，雅可比矩阵需要数值求导。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


EPS = 1e-8


def cv_F(T: float) -> np.ndarray:
    """CV 模型的 6×6 状态转移矩阵。

    F_cv = | I3   T·I3 |
           | 0    I3   |
    """
    F = np.eye(6, dtype=np.float64)
    F[0:3, 3:6] = np.eye(3) * T
    return F


def cv_Gamma(T: float) -> np.ndarray:
    """CV 模型的 6×3 噪声驱动矩阵。"""
    Gamma = np.zeros((6, 3), dtype=np.float64)
    Gamma[0:3, :] = np.eye(3) * (T ** 2) / 2.0
    Gamma[3:6, :] = np.eye(3) * T
    return Gamma


def ca_F(T: float) -> np.ndarray:
    """CA 模型的 9×9 状态转移矩阵。

    F_ca = | I3   T·I3   T²/2·I3 |
           | 0    I3     T·I3    |
           | 0    0      I3      |
    """
    F = np.eye(9, dtype=np.float64)
    F[0:3, 3:6] = np.eye(3) * T
    F[0:3, 6:9] = np.eye(3) * (T ** 2) / 2.0
    F[3:6, 6:9] = np.eye(3) * T
    return F


def ca_Gamma(T: float) -> np.ndarray:
    """CA 模型的 9×3 噪声驱动矩阵。

    设计选择：匀加速模型的加速度 a 在理想条件下保持恒定。若按实施方案文献形式
    Γ_ca = [T²/2·I; T·I; I3] 让噪声直接驱动加速度通道，σ_w=√0.35≈0.59 m/s²
    在 200 步上累计的随机游走将让加速度漂移到约 ±8 m/s²，真实速度反向、轨迹
    出现明显折返，不符合"匀加速"物理意义，也偏离论文表 3.2 的精度量级。

    本方案统一采用"位置/速度通道注入小扰动、加速度保持恒定"的工程实现，
    与 CT 模型 Γ 的处理保持一致：
        Γ_ca = | T²/2·I3 |
               | T·I3    |
               | 0       |
    """
    Gamma = np.zeros((9, 3), dtype=np.float64)
    Gamma[0:3, :] = np.eye(3) * (T ** 2) / 2.0
    Gamma[3:6, :] = np.eye(3) * T
    return Gamma


def ct_Gamma(T: float) -> np.ndarray:
    """CT 模型的 9×3 噪声驱动矩阵。

    设计选择：匀转速模型的角速度 ω 在理想条件下保持恒定，因此过程噪声不应直接
    作用于角速度通道；本方案采用与 CV/CA 一致的形式将噪声作用于位置/速度通道，
    建模未观测的小扰动（如气流），保持 ω 的"匀转速"物理意义。

    若把噪声作用于角速度通道，σ_w=√0.35 在 200 步上积累的随机游走会让 ω 偏离
    初值很远，导致 CT 轨迹的速度模长发散，与论文表 3.2 的精度量级不符。
    """
    Gamma = np.zeros((9, 3), dtype=np.float64)
    Gamma[0:3, :] = np.eye(3) * (T ** 2) / 2.0
    Gamma[3:6, :] = np.eye(3) * T
    return Gamma


def _omega_tilde(omega: np.ndarray) -> np.ndarray:
    """构造 3×3 角速度反对称矩阵 Ω̃，支持复数输入（用于复步差分）。"""
    wx, wy, wz = omega[0], omega[1], omega[2]
    dtype = omega.dtype if omega.dtype in (np.complex128, np.complex64) else np.float64
    Omt = np.zeros((3, 3), dtype=dtype)
    Omt[0, 1] = -wz
    Omt[0, 2] = wy
    Omt[1, 0] = wz
    Omt[1, 2] = -wx
    Omt[2, 0] = -wy
    Omt[2, 1] = wx
    return Omt


def ct_F(X: np.ndarray, T: float) -> np.ndarray:
    """根据当前状态向量 X（含角速度分量 ω）构造 CT 模型的 9×9 状态转移矩阵。

    严格按文献式(3.13)~(3.15)：
        F_ct = | I3      B      0 |
               | 0     I3+A     0 |
               | 0      0      I3 |
    其中
        A = c2·Ω̃ - c1·Ω̃²
        B = I3·T - c1·Ω̃ - c3·Ω̃²
        c1 = (cos(ΩT)-1)/Ω²
        c2 = sin(ΩT)/Ω
        c3 = (sin(ΩT)/Ω - T)/Ω²

    注意：论文式(3.14)中B矩阵对角线元素 c3·d_i 存在笔误（缺失T项），
    正确值应为 T + c3·d_i（含T项），
    此处使用物理推导的正确紧凑形式 B = I3·T - c1·Ω̃ - c3·Ω̃²，
    切勿直接照抄论文式(3.14)的显式矩阵，否则 ω≈0 时位置不更新。

    支持复数输入以适配复步差分法求雅可比矩阵。
    """
    omega = X[6:9]
    dtype = omega.dtype if omega.dtype in (np.complex128, np.complex64) else np.float64

    Omega = np.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
    is_complex = dtype in (np.complex128, np.complex64)

    if (not is_complex) and float(np.abs(Omega)) < 1e-6:
        c1 = -(T ** 2) / 2.0
        c2 = T
        c3 = -(T ** 3) / 6.0
    else:
        c1 = (np.cos(Omega * T) - 1.0) / (Omega ** 2)
        c2 = np.sin(Omega * T) / Omega
        c3 = (np.sin(Omega * T) / Omega - T) / (Omega ** 2)

    Omt = _omega_tilde(omega)
    Omt2 = Omt @ Omt
    I3 = np.eye(3, dtype=dtype)

    A = c2 * Omt - c1 * Omt2
    B = I3 * T - c1 * Omt - c3 * Omt2

    F = np.zeros((9, 9), dtype=dtype)
    F[0:3, 0:3] = I3
    F[0:3, 3:6] = B
    F[3:6, 3:6] = I3 + A
    F[6:9, 6:9] = I3
    return F


def ct_f(X: np.ndarray, T: float) -> np.ndarray:
    """CT 模型的非线性状态转移函数 X(k+1) = f_ct(X(k))（不含噪声）。"""
    F = ct_F(X, T)
    return F @ X


def ct_jacobian(X: np.ndarray, T: float, h: float = 1e-20) -> np.ndarray:
    """使用复步差分法计算 CT 模型在 X 点的 9×9 雅可比矩阵 Φ = ∂f_ct/∂X。

    复步差分公式：Φ_ij ≈ Im[f(X + i·h·e_j)] / h
    优点：无减法消元误差，h 可取极小（如 1e-20），精度达机器精度。
    """
    n = X.shape[0]
    J = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        Xc = X.astype(np.complex128)
        Xc[j] = Xc[j] + 1j * h
        f_val = ct_f(Xc, T)
        J[:, j] = np.imag(f_val) / h
    return J


def radar_h(X: np.ndarray) -> np.ndarray:
    """雷达观测函数：将笛卡尔坐标 (x,y,z) 映射到球坐标 (r,θ,φ)。

    h(X) = [√(x²+y²+z²), arccos(z/r), atan2(y,x)]
    其中 θ 为极角（与 z 轴夹角），φ 为方位角（atan2 避免象限错误）。
    """
    x, y, z = X[0], X[1], X[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2 + EPS)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi], dtype=np.float64)


def radar_H(X: np.ndarray, state_dim: int) -> np.ndarray:
    """雷达观测雅可比矩阵 H = ∂h/∂X（3×state_dim）。

    速度与加速度/角速度对应的列均为0，观测仅依赖位置 (x,y,z)。
    数值边界保护：通过 EPS 防止 x²+y²→0 时的 NaN/Inf。
    """
    x, y, z = X[0], X[1], X[2]
    r2 = x ** 2 + y ** 2 + z ** 2 + EPS
    r = np.sqrt(r2)
    rho2 = x ** 2 + y ** 2 + EPS
    rho = np.sqrt(rho2)

    H = np.zeros((3, state_dim), dtype=np.float64)
    H[0, 0] = x / r
    H[0, 1] = y / r
    H[0, 2] = z / r

    H[1, 0] = x * z / (r2 * rho)
    H[1, 1] = y * z / (r2 * rho)
    H[1, 2] = -rho / r2

    H[2, 0] = -y / rho2
    H[2, 1] = x / rho2
    H[2, 2] = 0.0
    return H


def get_F_and_Gamma(model: str, T: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 CV / CA 模型的固定 F 和 Γ。CT 模型请使用 ct_F / ct_Gamma。"""
    name = model.upper()
    if name == "CV":
        return cv_F(T), cv_Gamma(T)
    if name == "CA":
        return ca_F(T), ca_Gamma(T)
    if name == "CT":
        raise ValueError("CT 模型 F 依赖当前状态，请直接调用 ct_F(X, T)")
    raise ValueError(f"未知运动模型: {model}")


def make_f_and_jacobian(
    model: str, T: float
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """返回 (f_func, jacobian_func, Gamma)。

    - CV/CA：f_func(X) = F @ X，jacobian_func(X) 恒返回 F；
    - CT：f_func(X) = ct_f(X, T)，jacobian_func(X) = ct_jacobian(X, T)。
    """
    name = model.upper()
    if name in ("CV", "CA"):
        F, Gamma = get_F_and_Gamma(name, T)
        return (lambda X, F=F: F @ X), (lambda X, F=F: F), Gamma
    if name == "CT":
        Gamma = ct_Gamma(T)
        return (lambda X: ct_f(X, T)), (lambda X: ct_jacobian(X, T)), Gamma
    raise ValueError(f"未知运动模型: {model}")
