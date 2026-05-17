"""可视化模块：3D 轨迹、距离误差曲线、速度误差曲线、单步时长柱状图、
2D 三视图投影。

绘图函数支持 `show` 参数：
- show=False（默认）：保存 PNG 后关闭 figure；
- show=True：保存 PNG 后保留 figure，可由 main.py 最后统一 `plt.show()`
  弹出，matplotlib 原生窗口支持鼠标拖拽旋转 3D 视角。

设计要点：
- 轨迹对比图展示完整 N 步轨迹（真实 + 各算法预测），
  便于看清起点、终点、整体形状。
- 3D 视角通过 ax.view_init(elev, azim) 调整为"原点靠近观察者"的方位。
- 2D 三视图（XY / XZ / YZ）的坐标系原点固定在左下角，
  且坐标轴范围统一基于数据集合包围盒+10%边距，避免原点被推到角落。
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册3D投影)


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


VIEW_ELEV = 22.0
VIEW_AZIM = -135.0

PALETTE = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_set_window_title(fig, title: str) -> None:
    """安全设置窗口标题。TkAgg 后端的窗口管理器使用 DejaVu Sans 字体，
    包含中文时会产生大量字体缺失警告，因此窗口标题统一使用英文/ASCII，
    图内绘图的中文不受影响（由 matplotlib.rcParams 字体设置）。"""
    try:
        manager = fig.canvas.manager
        if manager is not None:
            manager.set_window_title(title)
    except Exception:
        pass


def _set_3d_axes(ax, X_min: np.ndarray, X_max: np.ndarray, margin: float = 0.05) -> None:
    """统一 3D 坐标范围，含小幅边距；调整视角让原点靠近观察者。"""
    span = X_max - X_min
    span = np.where(span < 1e-6, 1.0, span)
    pad = span * margin
    lo = X_min - pad
    hi = X_max + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def _compute_bounds(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """计算多条轨迹的整体包围盒 (min, max) over the first 3 columns."""
    stacked = np.vstack([a[:, :3] for a in arrays])
    return stacked.min(axis=0), stacked.max(axis=0)


def plot_3d_trajectory(
    X_true: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    save_path: str,
    start: int = 0,
    show: bool = False,
) -> None:
    """绘制真实轨迹与各算法预测轨迹的 3D 对比图。

    完整展示 [start, N) 区间的轨迹，并标注起点（绿色）、终点（红色）；
    视角设置为 elev=22°, azim=-135°（左前下方观察，原点靠近观察者）。
    show=True 时保存后保留 figure，供 main.py 末尾 plt.show() 弹出。
    """
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        X_true[start:, 0],
        X_true[start:, 1],
        X_true[start:, 2],
        label="真实轨迹",
        color="black",
        linewidth=2.0,
    )
    ax.scatter(
        X_true[start, 0], X_true[start, 1], X_true[start, 2],
        color="green", s=50, marker="o", label="起点", zorder=5,
    )
    ax.scatter(
        X_true[-1, 0], X_true[-1, 1], X_true[-1, 2],
        color="red", s=50, marker="^", label="终点", zorder=5,
    )

    for (name, pred), color in zip(preds.items(), PALETTE):
        ax.plot(
            pred[start:, 0], pred[start:, 1], pred[start:, 2],
            label=name, color=color, linewidth=1.3, alpha=0.85,
        )

    arrays = [X_true[start:]] + [p[start:] for p in preds.values()]
    lo, hi = _compute_bounds(arrays)
    _set_3d_axes(ax, lo, hi)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    _safe_set_window_title(fig, os.path.basename(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if not show:
        plt.close(fig)


def plot_2d_projections(
    X_true: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    save_path: str,
    start: int = 0,
) -> None:
    """绘制 XY / XZ / YZ 三视图。

    每个子图均以 (a_min, b_min) 为左下角、(a_max, b_max) 为右上角，
    保证原点（数据最小值附近）靠近观察者（即图的左下方），不会偏居一角。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plane_specs = [
        ("XY", 0, 1, "X (m)", "Y (m)"),
        ("XZ", 0, 2, "X (m)", "Z (m)"),
        ("YZ", 1, 2, "Y (m)", "Z (m)"),
    ]

    for ax, (plane, ix, iy, xlabel, ylabel) in zip(axes, plane_specs):
        ax.plot(
            X_true[start:, ix], X_true[start:, iy],
            label="真实轨迹", color="black", linewidth=2.0,
        )
        ax.scatter(
            X_true[start, ix], X_true[start, iy],
            color="green", s=40, marker="o", label="起点", zorder=5,
        )
        ax.scatter(
            X_true[-1, ix], X_true[-1, iy],
            color="red", s=40, marker="^", label="终点", zorder=5,
        )
        for (name, pred), color in zip(preds.items(), PALETTE):
            ax.plot(
                pred[start:, ix], pred[start:, iy],
                label=name, color=color, linewidth=1.1, alpha=0.85,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{plane} 平面投影")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("auto")
        if plane == "XY":
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_curve(
    errors: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    save_path: str,
    start: int = 0,
) -> None:
    """绘制逐步误差曲线（位置或速度）。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, err), color in zip(errors.items(), PALETTE):
        steps = np.arange(start, start + err[start:].shape[0])
        ax.plot(steps, err[start:], label=name, color=color, linewidth=1.3)
    ax.set_xlabel("采样步")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_runtime_bar(
    runtimes: dict[str, float],
    title: str,
    save_path: str,
) -> None:
    """绘制各算法单步运行时长对比柱状图（ms 单位更直观）。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(runtimes.keys())
    values_ms = [v * 1000.0 for v in runtimes.values()]
    bars = ax.bar(names, values_ms, color=["tab:blue", "tab:orange", "tab:red"])
    ax.set_ylabel("单步运行时长 (ms)")
    ax.set_title(title)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1, label="50ms (0.05s) 阈值")
    for bar, v in zip(bars, values_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_true_trajectories(
    trajectories: dict[str, np.ndarray],
    save_path: str,
    show: bool = False,
) -> None:
    """三种运动模型下的真实轨迹一览图（3 列），统一视角与起止点标注。

    show=True 时保留 figure，供 main.py 末尾 plt.show() 弹出。
    """
    fig = plt.figure(figsize=(15, 5))
    for i, (name, X) in enumerate(trajectories.items(), start=1):
        ax = fig.add_subplot(1, len(trajectories), i, projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:blue", linewidth=1.6)
        ax.scatter(X[0, 0], X[0, 1], X[0, 2], color="green", s=40, marker="o", label="起点")
        ax.scatter(X[-1, 0], X[-1, 1], X[-1, 2], color="red", s=40, marker="^", label="终点")
        lo = X[:, :3].min(axis=0)
        hi = X[:, :3].max(axis=0)
        _set_3d_axes(ax, lo, hi)
        ax.set_title(f"{name} 模型真实轨迹")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left", fontsize=8)
    _safe_set_window_title(fig, "true_trajectories")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if not show:
        plt.close(fig)


