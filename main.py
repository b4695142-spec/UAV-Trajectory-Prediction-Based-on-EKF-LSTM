"""项目主入口：一键运行三种运动模型 × 三种算法的完整对比实验。

执行流程：
1. 生成 CV/CA/CT 三种模型下的真实轨迹与观测序列；
2. 输出三种模型的真实轨迹一览图；
3. 依次运行三种算法并产出对比图与精度表；
4. 在 results/ 目录下生成所有图表与最终精度对比表；
5. 弹出 4 个 matplotlib 3D 交互窗口（真实轨迹一览 + CV/CA/CT 算法对比），
   鼠标拖拽即可旋转视角，滚轮缩放，关闭所有窗口后程序退出。

运行：
    python main.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from config.params import RESULTS_DIR, SEED, get_config
from data.generator import generate_trajectory
from evaluation.metrics import format_stats_table
from evaluation.visualization import ensure_dir, plot_true_trajectories
from experiments.run_comparison import run_all


def main() -> None:
    ensure_dir(RESULTS_DIR)

    trajectories = {}
    for name in ("CV", "CA", "CT"):
        cfg = get_config(name)
        X_true, _ = generate_trajectory(
            name,
            cfg["X0"],
            cfg["T"],
            cfg["N"],
            cfg["sigma_w2"],
            cfg["sigma_v2"],
            seed=SEED,
        )
        trajectories[name] = X_true
    plot_true_trajectories(
        trajectories,
        os.path.join(RESULTS_DIR, "true_trajectories.png"),
        show=True,
    )
    print(f"已保存三种模型的真实轨迹图到 {RESULTS_DIR}/true_trajectories.png")

    summaries = run_all(verbose=True, show=True)

    print("\n========== 最终精度对比 ==========")
    rows = []
    for s in summaries:
        for alg in ("EKF", "LSTM", "EKF-LSTM"):
            rows.append((s["model"], alg, s["stats"][alg]))
    table_str = format_stats_table(rows)
    print(table_str)

    table_path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"\n精度对比表已保存到 {table_path}")

    print("\n========== 单步耗时汇总 (ms) ==========")
    for s in summaries:
        rt = s["runtimes"]
        print(
            f"  {s['model']:<3}  EKF={rt['EKF']*1000:.3f}  "
            f"LSTM={rt['LSTM']*1000:.3f}  EKF-LSTM={rt['EKF-LSTM']*1000:.3f}"
        )

    print(
        "\n弹出 3D 交互窗口（4 个）：鼠标拖拽旋转，滚轮缩放，"
        "关闭所有窗口后程序退出。"
    )
    plt.show()


if __name__ == "__main__":
    main()
