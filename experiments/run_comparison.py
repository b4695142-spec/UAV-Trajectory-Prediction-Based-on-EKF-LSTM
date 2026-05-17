"""三种算法（EKF / LSTM / EKF-LSTM）在三种运动模型下的对比实验。

每种模型独立完成：
1. 生成真实轨迹与观测序列；
2. 运行 EKF；
3. 训练并运行单一 LSTM 基线（轨迹端到端预测）；
4. 运行 EKF-LSTM 融合预测；
5. 计算误差指标、单步耗时；
6. 输出轨迹图、误差曲线、单步时长柱状图。
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from config.params import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    LSTM_SEQ_LEN,
    PV_DIM,
    RESULTS_DIR,
    SEED,
    TRAIN_RATIO,
    get_config,
)
from data.generator import generate_trajectory
from data.preprocessor import MinMaxScaler, make_seq_dataset
from evaluation.metrics import (
    compute_error_stats,
    format_stats_table,
    position_errors,
    velocity_errors,
)
from evaluation.visualization import (
    ensure_dir,
    plot_2d_projections,
    plot_3d_trajectory,
    plot_error_curve,
    plot_runtime_bar,
)
from models.ekf import EKF
from models.ekf_lstm import EKFLSTM
from models.ekf_service import EKFService
from models.lstm import LSTMNet, lstm_predict, train_lstm
from models.motion_models import (
    make_f_and_jacobian,
    radar_H,
    radar_h,
)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_lstm_baseline(
    X_filt: np.ndarray,
    train_end: int,
    seq_len: int = LSTM_SEQ_LEN,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    epochs: int = LSTM_EPOCHS,
    lr: float = LSTM_LR,
    batch_size: int = LSTM_BATCH_SIZE,
    verbose: bool = False,
) -> np.ndarray:
    """单一 LSTM 基线（独立预测模式）。

    输入：历史 seq_len 步的位置-速度 P(k-i)；
    输出：下一时刻相对当前时刻的位置-速度增量 ΔP = P(k+1) - P(k)。

    使用增量预测而非绝对位置预测的原因：
    无人机在测试集中的位置会显著超出训练集范围（如 CV 模型从 0~5m 推进到 5~50m），
    若直接预测绝对位置，归一化后的测试集输入会越界，导致模型外推性能崩溃。
    预测增量则使训练集和测试集的分布更稳定，与论文中 LSTM 基线 0.277m 的精度水平
    更一致。最终预测 P̂(k+1) = P(k) + ΔP̂。

    输入特征来自 EKF 滤波后的位置-速度（保证三种算法在相同的观测预处理下对比）。
    """
    P_seq = X_filt[:, :PV_DIM]

    train_features = P_seq[:train_end]
    train_targets = np.diff(P_seq, axis=0)[: train_end - 1]
    train_features_for_seq = train_features[:-1]
    train_X, train_Y = make_seq_dataset(
        train_features_for_seq, train_targets, seq_len
    )

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    train_flat = train_X.reshape(-1, PV_DIM)
    x_scaler.fit(train_flat)
    y_scaler.fit(train_Y)

    train_X_norm = x_scaler.transform(train_flat).reshape(train_X.shape)
    train_Y_norm = y_scaler.transform(train_Y)

    net = LSTMNet(
        input_size=PV_DIM,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=PV_DIM,
    )
    train_lstm(
        net,
        train_X_norm,
        train_Y_norm,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        verbose=verbose,
    )

    full_features = P_seq[:-1]
    full_targets = np.diff(P_seq, axis=0)
    full_X, _ = make_seq_dataset(full_features, full_targets, seq_len)
    full_X_norm = x_scaler.transform(full_X.reshape(-1, PV_DIM)).reshape(full_X.shape)
    Y_pred_norm = lstm_predict(net, full_X_norm)
    Y_pred_delta = y_scaler.inverse_transform(Y_pred_norm)

    X_pred = X_filt.copy()
    for idx in range(Y_pred_delta.shape[0]):
        anchor_k = idx + seq_len - 1
        target_k = anchor_k + 1
        if target_k >= X_pred.shape[0]:
            break
        X_pred[target_k, :PV_DIM] = P_seq[anchor_k] + Y_pred_delta[idx]
    return X_pred


def measure_ekf_runtime(
    model: str,
    Z_obs: np.ndarray,
    X0: np.ndarray,
    state_dim: int,
    T: float,
    sigma_w2: float,
    sigma_v2: float,
    repeats: int = 3,
) -> float:
    """测量 EKF 平均单步运行时长（秒）。"""
    N = Z_obs.shape[0]
    f_func, jac_func, Gamma = make_f_and_jacobian(model, T)

    durations: list[float] = []
    ekf_svc = EKFService(model, state_dim, T, sigma_w2, sigma_v2)
    P0 = ekf_svc.build_initial_covariance()
    for _ in range(repeats):
        ekf = EKF(state_dim, 3, sigma_w2, sigma_v2, X0, P0_scale=P0)
        t_start = time.perf_counter()
        for k in range(1, N):
            ekf.predict(f_func, Gamma, Phi=jac_func)
            H_k = radar_H(ekf.state, state_dim)
            ekf.update(Z_obs[k], radar_h, H_k)
        t_end = time.perf_counter()
        durations.append((t_end - t_start) / max(N - 1, 1))
    return float(np.mean(durations))


def measure_lstm_runtime(
    net: LSTMNet,
    seq_input_sample: np.ndarray,
    repeats: int = 200,
) -> float:
    """测量 LSTM 网络单次前向预测的平均耗时（秒）。"""
    x = torch.from_numpy(seq_input_sample.astype(np.float32)).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = net(x)
        t_start = time.perf_counter()
        for _ in range(repeats):
            _ = net(x)
        t_end = time.perf_counter()
    return float((t_end - t_start) / repeats)


def run_model_experiment(
    model_name: str,
    results_dir: str = RESULTS_DIR,
    seed: int = SEED,
    verbose: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """完成单个运动模型下的三算法对比实验。"""
    set_global_seed(seed)
    cfg = get_config(model_name)

    X0 = cfg["X0"]
    state_dim = cfg["state_dim"]
    T = cfg["T"]
    N = cfg["N"]
    sigma_w2 = cfg["sigma_w2"]
    sigma_v2 = cfg["sigma_v2"]
    n_train = int(N * TRAIN_RATIO)

    if verbose:
        print(f"\n========== {model_name} 模型实验 ==========")
        print(f"  状态维度={state_dim}  采样步={N}  T={T}s  训练集={n_train}步")

    X_true, Z_obs = generate_trajectory(
        model_name, X0, T, N, sigma_w2, sigma_v2, seed=seed
    )

    if verbose:
        print("  [1/3] 运行 EKF ...")
    ekf_svc = EKFService(model_name, state_dim, T, sigma_w2, sigma_v2)
    X_filt, _ = ekf_svc.run(Z_obs, X0, X_true)

    if verbose:
        print("  [2/3] 训练并运行单一 LSTM 基线 ...")
    X_lstm = run_lstm_baseline(X_filt, train_end=n_train, verbose=False)

    if verbose:
        print("  [3/3] 训练并运行 EKF-LSTM 融合 ...")
    ekflstm = EKFLSTM(
        model=model_name,
        state_dim=state_dim,
        T=T,
        sigma_w2=sigma_w2,
        sigma_v2=sigma_v2,
        seed=seed,
    )
    ekflstm_out = ekflstm.run(Z_obs, X_true, X0, train_ratio=TRAIN_RATIO, verbose=False)
    X_final = ekflstm_out["X_final"]
    valid_start = ekflstm_out["valid_start"]

    eval_start = max(valid_start, n_train)
    stats_ekf = compute_error_stats(X_true, X_filt, start=eval_start)
    stats_lstm = compute_error_stats(X_true, X_lstm, start=eval_start)
    stats_fused = compute_error_stats(X_true, X_final, start=eval_start)

    if verbose:
        rows = [
            (model_name, "EKF", stats_ekf),
            (model_name, "LSTM", stats_lstm),
            (model_name, "EKF-LSTM", stats_fused),
        ]
        print("\n" + format_stats_table(rows))

    ekf_time = measure_ekf_runtime(
        model_name, Z_obs, X0, state_dim, T, sigma_w2, sigma_v2, repeats=2
    )
    sample_input = np.zeros((LSTM_SEQ_LEN, ekflstm.net.lstm.input_size), dtype=np.float32)
    lstm_time = measure_lstm_runtime(ekflstm.net, sample_input, repeats=100)
    ekflstm_time = ekf_time + lstm_time

    runtimes = {
        "EKF": ekf_time,
        "LSTM": lstm_time,
        "EKF-LSTM": ekflstm_time,
    }
    if verbose:
        print(
            f"  单步耗时 (ms): EKF={ekf_time*1000:.3f}  "
            f"LSTM={lstm_time*1000:.3f}  EKF-LSTM={ekflstm_time*1000:.3f}"
        )

    ensure_dir(results_dir)
    model_dir = os.path.join(results_dir, model_name)
    ensure_dir(model_dir)

    preds = {
        "EKF": X_filt,
        "LSTM": X_lstm,
        "EKF-LSTM": X_final,
    }
    plot_3d_trajectory(
        X_true,
        preds,
        title=f"{model_name} 模型轨迹对比",
        save_path=os.path.join(model_dir, f"{model_name}_trajectory.png"),
        start=0,
        show=show,
    )
    plot_2d_projections(
        X_true,
        preds,
        title=f"{model_name} 模型轨迹 2D 投影",
        save_path=os.path.join(model_dir, f"{model_name}_trajectory_2d.png"),
        start=0,
    )

    pos_errs = {
        "EKF": position_errors(X_true, X_filt),
        "LSTM": position_errors(X_true, X_lstm),
        "EKF-LSTM": position_errors(X_true, X_final),
    }
    plot_error_curve(
        pos_errs,
        title=f"{model_name} 模型距离误差",
        ylabel="距离误差 (m)",
        save_path=os.path.join(model_dir, f"{model_name}_pos_error.png"),
        start=eval_start,
    )

    if model_name in ("CA", "CT"):
        vel_errs = {
            "EKF": velocity_errors(X_true, X_filt),
            "LSTM": velocity_errors(X_true, X_lstm),
            "EKF-LSTM": velocity_errors(X_true, X_final),
        }
        plot_error_curve(
            vel_errs,
            title=f"{model_name} 模型速度误差",
            ylabel="速度误差 (m/s)",
            save_path=os.path.join(model_dir, f"{model_name}_vel_error.png"),
            start=eval_start,
        )

    plot_runtime_bar(
        runtimes,
        title=f"{model_name} 模型单步运行时长对比",
        save_path=os.path.join(model_dir, f"{model_name}_runtime.png"),
    )

    return {
        "model": model_name,
        "X_true": X_true,
        "X_filt": X_filt,
        "X_lstm": X_lstm,
        "X_final": X_final,
        "stats": {
            "EKF": stats_ekf,
            "LSTM": stats_lstm,
            "EKF-LSTM": stats_fused,
        },
        "runtimes": runtimes,
        "eval_start": eval_start,
    }


def run_all(verbose: bool = True, show: bool = False) -> list[dict[str, Any]]:
    """运行 CV / CA / CT 三种模型的对比实验。"""
    ensure_dir(RESULTS_DIR)
    summaries: list[dict[str, Any]] = []
    for model_name in ("CV", "CA", "CT"):
        summary = run_model_experiment(model_name, verbose=verbose, show=show)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    run_all(verbose=True)
