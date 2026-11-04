"""Experimentos e gráficos do filtro de Kalman."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from kalman_filter import KalmanFilter

OUT = Path(__file__).parent / "results"


def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def simulate(measurement_variance, assumed_variance=None, seed=23, n=120):
    rng = np.random.default_rng(seed)
    dt = .5
    F = np.array([[1, dt], [0, 1.]])
    H = np.array([[1., 0.]])
    Q = np.array([[.002, 0], [0, .01]])
    R_true = np.array([[measurement_variance]])
    R_assumed = np.array([[assumed_variance if assumed_variance is not None else measurement_variance]])
    states = np.zeros((n, 2)); measurements = np.zeros(n)
    states[0] = [0, 1.2]
    for k in range(1, n):
        states[k] = F @ states[k - 1] + rng.multivariate_normal([0, 0], Q)
    measurements = states[:, 0] + rng.normal(0, np.sqrt(measurement_variance), n)
    filt = KalmanFilter(F, H, Q, R_assumed, [0, 0], np.diag([2., 2.]))
    estimates = np.zeros_like(states); variances = np.zeros(n)
    for k, z in enumerate(measurements):
        filt.predict() if k else None
        estimates[k], covariance, _, _ = filt.update([z])
        variances[k] = covariance[0, 0]
    return states, measurements, estimates, variances


def run(seed=23):
    OUT.mkdir(exist_ok=True)
    truth, measurements, estimates, variances = simulate(4.0, seed=seed)
    metrics = {
        "raw_position_rmse": rmse(truth[:, 0], measurements),
        "filtered_position_rmse": rmse(truth[:, 0], estimates[:, 0]),
        "raw_position_mae": mae(truth[:, 0], measurements),
        "filtered_position_mae": mae(truth[:, 0], estimates[:, 0]),
    }
    time = np.arange(len(truth)) * .5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, truth[:, 0], label="verdade", lw=2)
    ax.scatter(time, measurements, s=12, alpha=.35, label="sensor")
    ax.plot(time, estimates[:, 0], label="Kalman", color="#d1495b")
    ax.fill_between(time, estimates[:, 0] - 2*np.sqrt(variances),
                    estimates[:, 0] + 2*np.sqrt(variances), color="#d1495b", alpha=.15, label="95% aprox.")
    ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Posição (unid.)")
    ax.set_title("Estimação de posição e incerteza")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "state_estimation.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, measurements - truth[:, 0], label="erro bruto", alpha=.6)
    ax.plot(time, estimates[:, 0] - truth[:, 0], label="erro filtrado")
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Erro de posição"); ax.set_title("Erro antes e depois da filtragem")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "estimation_error.png", dpi=300); plt.close(fig)

    levels = [0.25, 1.0, 4.0, 16.0]
    comparison = {}
    for level in levels:
        truth_l, meas_l, est_l, _ = simulate(level, seed=seed)
        comparison[str(level)] = {"raw_rmse": rmse(truth_l[:, 0], meas_l),
                                  "filtered_rmse": rmse(truth_l[:, 0], est_l[:, 0])}
    metrics["noise_comparison"] = comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levels, [comparison[str(v)]["raw_rmse"] for v in levels], "o-", label="sensor")
    ax.plot(levels, [comparison[str(v)]["filtered_rmse"] for v in levels], "o-", label="Kalman")
    ax.set_xscale("log"); ax.set_xlabel("Variância R"); ax.set_ylabel("RMSE posição")
    ax.set_title("Influência do ruído de medição"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "noise_comparison.png", dpi=300); plt.close(fig)
    mismatch_truth, mismatch_meas, mismatch_est, _ = simulate(4.0, assumed_variance=.25, seed=seed)
    metrics["mismatched_assumption_rmse"] = rmse(mismatch_truth[:, 0], mismatch_est[:, 0])
    with (OUT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
