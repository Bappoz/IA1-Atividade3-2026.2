"""Projeto 3 completo: Filtro de Kalman e experimento de estimação industrial."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"


class KalmanFilter:
    """Filtro de Kalman linear com atualização de Joseph."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H = map(lambda value: np.asarray(value, dtype=float), (F, H))
        self.Q, self.R = map(lambda value: np.asarray(value, dtype=float), (Q, R))
        self.x, self.P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
        state_size = self.F.shape[0] if self.F.ndim == 2 else 0
        measurement_size = self.H.shape[0] if self.H.ndim == 2 else 0
        if (self.F.shape != (state_size, state_size) or self.H.shape != (measurement_size, state_size)
                or self.Q.shape != (state_size, state_size) or self.R.shape != (measurement_size, measurement_size)
                or self.x.shape != (state_size,) or self.P.shape != (state_size, state_size)):
            raise ValueError("Dimensões de F/H incompatíveis.")
        if (not all(np.all(np.isfinite(value)) for value in (self.F, self.H, self.Q, self.R, self.x, self.P))
                or not np.allclose(self.Q, self.Q.T) or not np.allclose(self.R, self.R.T)
                or not np.allclose(self.P, self.P.T)):
            raise ValueError("Parâmetros devem ser finitos e covariâncias simétricas.")
        if (np.any(np.linalg.eigvalsh(self.Q) < -1e-10)
                or np.any(np.linalg.eigvalsh(self.R) < -1e-10)
                or np.any(np.linalg.eigvalsh(self.P) < -1e-10)):
            raise ValueError("Covariâncias devem ser semidefinidas positivas.")

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        innovation = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        residual = np.eye(self.P.shape[0]) - K @ self.H
        self.x = self.x + K @ innovation
        self.P = residual @ self.P @ residual.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        return self.x.copy(), self.P.copy(), innovation, K


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def simulate(measurement_variance, assumed_variance=None, seed=23, n=120):
    rng, dt = np.random.default_rng(seed), .5
    F, H = np.array([[1, dt], [0, 1.]]), np.array([[1., 0.]])
    Q = np.array([[.002, 0], [0, .01]])
    states = np.zeros((n, 2)); states[0] = [0, 1.2]
    for k in range(1, n):
        states[k] = F @ states[k - 1] + rng.multivariate_normal([0, 0], Q)
    measurements = states[:, 0] + rng.normal(0, np.sqrt(measurement_variance), n)
    variance = measurement_variance if assumed_variance is None else assumed_variance
    filt = KalmanFilter(F, H, Q, [[variance]], [0, 0], np.diag([2., 2.]))
    estimates, variances = np.zeros_like(states), np.zeros(n)
    for k, measurement in enumerate(measurements):
        if k:
            filt.predict()
        estimates[k], covariance, _, _ = filt.update([measurement])
        variances[k] = covariance[0, 0]
    return states, measurements, estimates, variances


def run(seed=23):
    """Simula a trajetória, executa o filtro e salva métricas e figuras."""
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
    ax.set_title("Estimação de posição e incerteza"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "state_estimation.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, measurements - truth[:, 0], label="erro bruto", alpha=.6)
    ax.plot(time, estimates[:, 0] - truth[:, 0], label="erro filtrado")
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Erro de posição"); ax.set_title("Erro antes e depois da filtragem")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "estimation_error.png", dpi=300); plt.close(fig)
    levels, comparison = [0.25, 1.0, 4.0, 16.0], {}
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
    mismatch_truth, _, mismatch_est, _ = simulate(4.0, assumed_variance=.25, seed=seed)
    metrics["mismatched_assumption_rmse"] = rmse(mismatch_truth[:, 0], mismatch_est[:, 0])
    with (OUT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
