"""CV 模型单独实验入口。运行完成后自动弹出 3D 交互窗口。"""

from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.run_comparison import run_model_experiment


def main() -> None:
    run_model_experiment("CV", verbose=True, show=True)
    plt.show()


if __name__ == "__main__":
    main()
