# 基于 EKF-LSTM 的无人机轨迹预测与路径规划

本项目复现论文《基于无人机轨迹欺骗的轨迹预测与路径规划研究》中的 **EKF-LSTM 融合算法**，
覆盖 CV / CA / CT 三种运动模型以及 EKF / LSTM / EKF-LSTM 三种算法的对比实验。

## 项目结构

```
.
├── config/                   # 全局参数
│   └── params.py
├── data/                     # 轨迹生成 + 数据预处理
│   ├── generator.py
│   └── preprocessor.py
├── models/                   # 核心算法
│   ├── motion_models.py      # CV/CA/CT 状态方程、雷达观测、雅可比
│   ├── ekf.py                # EKF（Joseph 形式协方差更新）
│   ├── lstm.py               # LSTM 网络与训练循环
│   └── ekf_lstm.py           # EKF-LSTM 融合模型
├── evaluation/               # 指标与可视化
│   ├── metrics.py
│   └── visualization.py
├── experiments/              # 实验入口
│   ├── run_cv.py
│   ├── run_ca.py
│   ├── run_ct.py
│   └── run_comparison.py
├── results/                  # 运行后产生的图表与精度表
├── main.py                   # 一键运行入口
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 一键运行

```bash
python main.py
```

执行后将在 `results/` 下生成：

- `true_trajectories.png`：三种模型的真实轨迹一览
- `CV/CA/CT/*.png`：各模型的 3D 轨迹对比、2D 三视图、距离误差、速度误差、单步时长柱状图
- `summary_table.txt`：精度对比表

并自动**弹出 4 个 matplotlib 3D 交互窗口**（真实轨迹一览 + CV/CA/CT 算法对比），
**鼠标拖拽即可旋转视角，滚轮缩放**，关闭所有窗口后程序退出。

## 单独运行某个模型

```bash
python -m experiments.run_cv
python -m experiments.run_ca
python -m experiments.run_ct
```

## 关键实现要点

1. **CT 模型状态转移矩阵**：使用物理推导的紧凑形式 `B = I3·T - c1·Ω̃ - c3·Ω̃²`，
   避免论文式(3.14)中 B 矩阵对角线缺失 T 项的笔误。
2. **CT 模型雅可比矩阵**：采用复步差分法（`Im[f(X+ih)]/h`）数值求导，
   精度达机器精度。
3. **EKF 协方差更新**：采用 Joseph 形式，保证 P 的正定性与对称性。
4. **观测方程**：方位角 φ 使用 `atan2(y, x)`，并在残差中折叠到 (-π, π] 区间。
5. **LSTM 输入**：所有模型统一取 `[P, e]`（位置+速度+滤波误差，12 维），
   `seq_len=3`；标签为下一时刻的滤波误差（6 维）。
6. **数据划分**：仅用训练集前 40% 拟合 min-max 归一化参数，避免数据泄露。
7. **融合时序对齐**：LSTM 在 k 时刻预测 ê(k+1)，缓存到 k+1 时刻与
   EKF 输出 X̂(k+1|k+1) 相加。
8. **评估指标**：距离误差仅对位置子向量求范数，速度误差仅对速度子向量求范数，
   避免量纲混淆。

## 验证基准（论文表 3.2）

| 模型 | 算法     | 距离平均误差 | 距离最大误差 | 速度平均误差 | 速度最大误差 |
| -- | ------ | ------ | ------ | ------ | ------ |
| CV | EKF-LSTM | 0.083m | 0.202m |   -    |   -    |
| CA | EKF-LSTM | 0.293m | 0.639m | 0.067m/s | 0.138m/s |
| CT | EKF-LSTM | 0.583m | 1.186m | 0.117m/s | 0.260m/s |
