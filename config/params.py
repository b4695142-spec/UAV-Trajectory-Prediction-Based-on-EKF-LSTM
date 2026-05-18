"""全局参数配置

定义 CV/CA/CT 三种运动模型的初始状态、噪声方差、采样间隔等关键超参数，
以及 EKF、LSTM 训练相关的通用配置。所有实验脚本都从这里读取参数，
确保不同实验之间参数一致、可一键修改。
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────── 通用 ───────────────────────────

SEED = 42
# 随机数种子（文献术语：随机种子 / Random Seed）
# 用途：控制轨迹生成、EKF 滤波、LSTM 训练中所有随机过程，保证实验可复现
# 修改建议：一般无需修改；若需做多次独立实验取平均，可遍历不同种子值

# ─────────────────────────── 仿真参数 ───────────────────────────

T = 0.05
# 采样间隔（文献术语：采样周期 / 采样间隔 / Sampling Interval）
# 符号：T 或 Δt，单位：秒
# 含义：雷达两次相邻观测之间的时间差，是运动模型状态转移矩阵 F 和
#       噪声驱动矩阵 Γ 的核心参数
# 用途：CV 模型中 F[0:3, 3:6] = T·I₃，Γ[0:3] = T²/2·I₃，Γ[3:6] = T·I₃；
#       CA 模型中包含 T、T²/2 项；CT 模型中出现在 sin(ΩT)、cos(ΩT) 中
# 修改建议：增大 T → 轨迹更稀疏、预测更困难；减小 T → 轨迹更密集、
#           仿真时长缩短（总时长 = N_STEPS × T）。典型取值 0.01~1.0 s

N_STEPS = 200
# 总采样步数（文献术语：采样步数 / 仿真步数 / Number of Time Steps）
# 符号：N 或 K
# 含义：轨迹的总离散时间步数，仿真总时长 = N_STEPS × T = 200 × 0.05 = 10 s
# 用途：决定生成的真实状态序列 X_true 和观测序列 Z_obs 的长度
# 修改建议：增大 N_STEPS → 仿真时长更长、LSTM 训练样本更多；
#           减小 N_STEPS → 训练样本不足可能导致欠拟合。典型取值 100~1000

TRAIN_RATIO = 0.4
# 训练集比例（文献术语：训练集划分比例 / Training Split Ratio）
# 含义：前 TRAIN_RATIO 比例的采样步用于 LSTM 训练，剩余用于测试
#       训练集大小 = int(N_STEPS × TRAIN_RATIO) = 80 步
# 用途：LSTM 仅在训练集上拟合 EKF 滤波误差模式，测试集评估泛化能力；
#       MinMaxScaler 也仅在训练集上 fit，避免数据泄露
# 修改建议：增大 → LSTM 训练更充分但测试区间缩短；减小 → 测试区间更长
#           但训练可能不充分。典型取值 0.3~0.8

LSTM_SEQ_LEN = 3
# LSTM 输入序列长度（文献术语：滑动窗口长度 / 时间窗口 / 滑动窗口步长 / Sliding Window Length）
# 符号：L 或 w
# 含义：LSTM 使用过去 LSTM_SEQ_LEN 个时间步的历史数据作为输入窗口，
#       每个样本形状为 (seq_len, feature_dim)
# 用途：DatasetBuilder 以 seq_len 为窗口大小构建滑动窗口样本；
#       融合预测时 target_k = idx + seq_len 确定预测目标时刻
# 修改建议：增大 → LSTM 能捕获更长时序依赖，但有效样本数减少
#           （有效样本 = n_train - seq_len）；减小 → 样本更多但时序信息不足
#           典型取值 3~10，需与 N_STEPS 和 TRAIN_RATIO 协调

# ─────────────────────────── 噪声参数 ───────────────────────────

SIGMA_W2 = 0.35
# 过程噪声方差（文献术语：过程噪声方差 / 系统噪声方差 / Process Noise Variance）
# 符号：σ_w² 或 q
# 含义：建模无人机运动中未建模动态（如气流扰动、机动加速度）的强度
# 用途：轨迹生成时 W ~ N(0, σ_w²)，通过噪声驱动矩阵 Γ 注入状态递推
#       X(k+1) = F·X(k) + Γ·W(k)；EKF 中构造过程噪声协方差矩阵
#       Q = σ_w²·(Γ·Γᵀ)，在预测步 P = F·P·Fᵀ + Q 中使协方差膨胀
# 修改建议：增大 → 轨迹更不规则、EKF 预测不确定性更大、滤波更依赖观测；
#           减小 → 轨迹更平滑、EKF 更信任模型预测。典型取值 0.01~1.0

SIGMA_V2 = 0.05
# 观测噪声方差（文献术语：观测噪声方差 / 测量噪声方差 / Measurement Noise Variance）
# 符号：σ_v² 或 r
# 含义：雷达在球坐标 (r, θ, φ) 观测中的测量误差强度
# 用途：轨迹生成时 V ~ N(0, σ_v²)，叠加到雷达观测值 Z(k) = h(X(k)) + V(k)；
#       EKF 中构造观测噪声协方差矩阵 R = σ_v²·I₃，在更新步
#       S = H·P·Hᵀ + R 和 K = P·Hᵀ·S⁻¹ 中影响卡尔曼增益大小
# 修改建议：增大 → 观测更不可靠、EKF 更信任预测、卡尔曼增益减小；
#           减小 → 观测更可信、EKF 更依赖观测、滤波跟踪更紧密
#           典型取值 0.001~0.5，一般 σ_v² < σ_w²（观测比过程更精确）

# ─────────────────────────── EKF 参数 ───────────────────────────

EKF_P0_SCALE = 100.0
# EKF 初始协方差缩放因子（文献术语：初始状态协方差 / 初始估计误差协方差 / Initial Error Covariance）
# 符号：P₀ 或 P(0|0)
# 含义：EKF 初始状态协方差矩阵 P₀ 的对角线值，反映对初始状态的不确定程度
#       大值 → 初始不确定性高 → EKF 在前几步快速收敛到观测
# 用途：ekf_service.py 中 build_initial_covariance() 构造 P₀：
#       - CV 模型：P₀ = diag(100, 100, 100, 100, 100, 100)
#       - CA/CT 模型：位置速度通道 P₀[0:6] = 100，加速度/角速度通道 P₀[6:9] = 0.01
#       （CA 中加速度、CT 中角速度被视为已知运动学参数，先验近似确定，用小方差）
# 修改建议：增大 → EKF 初始更不信任先验状态，更快收敛但前几步估计波动大；
#           减小 → EKF 更信任初始状态，收敛慢但起步平稳
#           典型取值 10~1000；注意 CA/CT 模型中加速度/角速度通道固定为 0.01，
#           过大可能导致滤波发散

# ─────────────────────────── LSTM 超参数 ───────────────────────────

LSTM_HIDDEN_SIZE = 64
# LSTM 隐藏层维度（文献术语：隐藏层节点数 / 隐藏状态维度 / Hidden Size）
# 符号：d_h 或 n_hidden
# 含义：LSTM 每层的隐藏状态向量维度，直接决定模型容量（参数量）
# 用途：nn.LSTM(hidden_size=64) 和全连接层 fc = Linear(64, output_size)
# 修改建议：增大 → 模型表达能力更强但可能过拟合、训练更慢；
#           减小 → 模型更轻量但可能欠拟合。典型取值 32~256

LSTM_NUM_LAYERS = 2
# LSTM 堆叠层数（文献术语：LSTM 层数 / 堆叠层数 / Number of LSTM Layers）
# 符号：n_layers 或 L
# 含义：LSTM 网络的纵向堆叠深度，多层可提取更高层抽象特征
# 用途：nn.LSTM(num_layers=2)
# 修改建议：增大 → 可学习更复杂时序模式但梯度消失风险增大、训练更慢；
#           减小 → 模型更简单。典型取值 1~3，一般不超过 4 层

LSTM_EPOCHS = 60
# LSTM 训练轮数（文献术语：训练轮数 / 迭代次数 / Number of Epochs）
# 符号：E 或 n_epochs
# 含义：LSTM 在训练集上完整遍历的最大次数
# 用途：train_lstm() 中 for epoch in range(epochs) 控制训练循环
# 修改建议：增大 → 训练更充分但可能过拟合；减小 → 可能欠拟合
#           典型取值 30~200，建议配合早停（Early Stopping）使用

LSTM_LR = 0.01
# LSTM 学习率（文献术语：学习率 / Learning Rate）
# 符号：η 或 lr 或 α
# 含义：Adam 优化器的步长因子，控制每次参数更新的幅度
# 用途：optimizer = Adam(model.parameters(), lr=0.01)
# 修改建议：增大 → 收敛更快但可能震荡或发散；减小 → 收敛更稳定但更慢
#           典型取值 1e-4~1e-2，Adam 常用 1e-3；若训练 loss 震荡可降低至 1e-3

LSTM_BATCH_SIZE = 16
# LSTM 批大小（文献术语：批大小 / 小批量大小 / Batch Size）
# 符号：B 或 b
# 含义：每次梯度更新使用的样本数，训练集约 80 个样本，batch_size=16 → 每 epoch 约 5 个 batch
# 用途：DataLoader(dataset, batch_size=16, shuffle=True)
# 修改建议：增大 → 梯度估计更稳定但内存占用更高、泛化可能变差；
#           减小 → 梯度噪声更大（有正则化效果）但训练更慢
#           典型取值 8~64，需与训练集大小协调

# ─────────────────────────── 维度常量 ───────────────────────────

POS_DIM = 3
# 位置维度（文献术语：位置状态维度 / Position Dimension）
# 含义：三维空间位置 (x, y, z) 的维度

VEL_DIM = 3
# 速度维度（文献术语：速度状态维度 / Velocity Dimension）
# 含义：三维空间速度 (vx, vy, vz) 的维度

PV_DIM = POS_DIM + VEL_DIM
# 位置-速度联合维度（文献术语：位置速度状态维度 / Position-Velocity Dimension）
# 符号：n_pv
# 含义：EKF-LSTM 融合预测中核心操作的维度，LSTM 输出维度 = PV_DIM = 6
# 用途：提取 EKF 滤波结果的位置-速度子序列 X_filt[:, :PV_DIM]；
#       融合预测 X_final = X_filt + Y_pred（维度均为 PV_DIM）

# ─────────────────────────── 运动模型配置 ───────────────────────────

CV_CONFIG = {
    "name": "CV",
    # CV 模型（文献术语：恒速模型 / 匀速模型 / Constant Velocity Model）
    # 状态向量 X = [x, y, z, vx, vy, vz]ᵀ，假设目标匀速直线运动
    # 状态维度 = 6（位置 3 + 速度 3）
    "state_dim": 6,
    "X0": np.array([0.0, 0.0, 0.0, 5.0, 3.0, 5.0], dtype=np.float64),
    # 初始状态（文献术语：初始状态向量 / Initial State Vector）
    # 符号：X₀ 或 X(0)
    # [x₀, y₀, z₀, vx₀, vy₀, vz₀] = [0, 0, 0, 5, 3, 5] m/s
    # 修改建议：位置初始值影响轨迹起点，速度初始值影响运动方向和快慢
    "T": T,
    "N": N_STEPS,
    "sigma_w2": SIGMA_W2,
    "sigma_v2": SIGMA_V2,
}

CA_CONFIG = {
    "name": "CA",
    # CA 模型（文献术语：恒加速模型 / 匀加速模型 / Constant Acceleration Model）
    # 状态向量 X = [x, y, z, vx, vy, vz, ax, ay, az]ᵀ，假设目标匀加速运动
    # 状态维度 = 9（位置 3 + 速度 3 + 加速度 3）
    "state_dim": 9,
    "X0": np.array(
        [0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 1.0, 0.5, 1.5], dtype=np.float64
    ),
    # 初始状态向量
    # [x₀, y₀, z₀, vx₀, vy₀, vz₀, ax₀, ay₀, az₀]
    # = [0, 0, 0, 1, 2, 4, 1, 0.5, 1.5] (m, m/s, m/s²)
    # 修改建议：加速度初始值决定机动强度，过大会导致轨迹快速偏离
    "T": T,
    "N": N_STEPS,
    "sigma_w2": SIGMA_W2,
    "sigma_v2": SIGMA_V2,
}

CT_CONFIG = {
    "name": "CT",
    # CT 模型（文献术语：恒定转弯率模型 / 协调转弯模型 / Coordinated Turn Model / Constant Turn Rate Model）
    # 状态向量 X = [x, y, z, vx, vy, vz, ωx, ωy, ωz]ᵀ，假设目标匀速转弯
    # 状态维度 = 9（位置 3 + 速度 3 + 角速度 3）
    "state_dim": 9,
    "X0": np.array(
        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.1, 0.2, 0.3], dtype=np.float64
    ),
    # 初始状态向量
    # [x₀, y₀, z₀, vx₀, vy₀, vz₀, ωx₀, ωy₀, ωz₀]
    # = [0, 0, 0, 10, 10, 10, 0.1, 0.2, 0.3] (m, m/s, rad/s)
    # 修改建议：角速度初始值决定转弯速率，单位 rad/s；
    #           0.1 rad/s ≈ 5.7°/s 为中等转弯率，1.0 rad/s ≈ 57°/s 为急转弯
    "T": T,
    "N": N_STEPS,
    "sigma_w2": SIGMA_W2,
    "sigma_v2": SIGMA_V2,
}


def get_config(model_name: str) -> dict:
    """根据模型名（CV/CA/CT）返回对应的配置字典。"""
    name = model_name.upper()
    if name == "CV":
        return CV_CONFIG
    if name == "CA":
        return CA_CONFIG
    if name == "CT":
        return CT_CONFIG
    raise ValueError(f"未知运动模型: {model_name}")


RESULTS_DIR = "results"
# 实验结果输出目录（文献术语：无特定术语，工程配置项）
# 含义：保存轨迹图、误差曲线、精度对比表等实验结果的根目录
# 各模型子目录如 results/CV/、results/CA/、results/CT/
