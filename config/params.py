"""全局参数配置

定义 CV/CA/CT 三种运动模型的初始状态、噪声方差、采样间隔等关键超参数，
以及 EKF、LSTM 训练相关的通用配置。所有实验脚本都从这里读取参数，
确保不同实验之间参数一致、可一键修改。
"""

from __future__ import annotations

import numpy as np


SEED = 42

T = 0.05
N_STEPS = 200
TRAIN_RATIO = 0.4
LSTM_SEQ_LEN = 3

SIGMA_W2 = 0.35
SIGMA_V2 = 0.05

EKF_P0_SCALE = 100.0

LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 60
LSTM_LR = 0.01
LSTM_BATCH_SIZE = 16

POS_DIM = 3
VEL_DIM = 3
PV_DIM = POS_DIM + VEL_DIM


CV_CONFIG = {
    "name": "CV",
    "state_dim": 6,
    "X0": np.array([0.0, 0.0, 0.0, 5.0, 3.0, 5.0], dtype=np.float64),
    "T": T,
    "N": N_STEPS,
    "sigma_w2": SIGMA_W2,
    "sigma_v2": SIGMA_V2,
}

CA_CONFIG = {
    "name": "CA",
    "state_dim": 9,
    "X0": np.array(
        [0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 1.0, 0.5, 1.5], dtype=np.float64
    ),
    "T": T,
    "N": N_STEPS,
    "sigma_w2": SIGMA_W2,
    "sigma_v2": SIGMA_V2,
}

CT_CONFIG = {
    "name": "CT",
    "state_dim": 9,
    "X0": np.array(
        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.1, 0.2, 0.3], dtype=np.float64
    ),
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
