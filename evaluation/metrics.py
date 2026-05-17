"""评估指标：距离误差、速度误差、单步运行时长统计。

关键约束（实施方案 §3.2 模块7）：
- 距离误差只对位置子向量 P=[x,y,z] 求范数；
- 速度误差只对速度子向量 V=[vx,vy,vz] 求范数；
- 不能对完整状态向量求范数，否则会混淆量纲。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorStats:
    """位置 + 速度的统计指标。"""

    pos_mean: float
    pos_max: float
    vel_mean: float
    vel_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pos_mean": self.pos_mean,
            "pos_max": self.pos_max,
            "vel_mean": self.vel_mean,
            "vel_max": self.vel_max,
        }


def position_errors(X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """逐步位置误差 ‖P_pred - P_true‖（米），形状 (N,)。"""
    diff = X_pred[:, 0:3] - X_true[:, 0:3]
    return np.linalg.norm(diff, axis=1)


def velocity_errors(X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """逐步速度误差 ‖V_pred - V_true‖（m/s），形状 (N,)。"""
    diff = X_pred[:, 3:6] - X_true[:, 3:6]
    return np.linalg.norm(diff, axis=1)


def compute_error_stats(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> ErrorStats:
    """计算位置/速度的平均误差与最大误差。

    可通过 start/end 参数排除融合初始边界（前 seq_len 步）。
    """
    if end is None:
        end = X_true.shape[0]
    pos_err = position_errors(X_true[start:end], X_pred[start:end])
    vel_err = velocity_errors(X_true[start:end], X_pred[start:end])
    return ErrorStats(
        pos_mean=float(np.mean(pos_err)),
        pos_max=float(np.max(pos_err)),
        vel_mean=float(np.mean(vel_err)),
        vel_max=float(np.max(vel_err)),
    )


def format_stats_table(rows: list[tuple[str, str, ErrorStats]]) -> str:
    """格式化精度对比表。

    rows: [(model_name, algorithm_name, stats), ...]
    """
    header = (
        f"{'Model':<6} {'Algorithm':<10} "
        f"{'PosMean(m)':>12} {'PosMax(m)':>12} "
        f"{'VelMean(m/s)':>14} {'VelMax(m/s)':>14}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for model, alg, st in rows:
        lines.append(
            f"{model:<6} {alg:<10} "
            f"{st.pos_mean:>12.4f} {st.pos_max:>12.4f} "
            f"{st.vel_mean:>14.4f} {st.vel_max:>14.4f}"
        )
    return "\n".join(lines)
