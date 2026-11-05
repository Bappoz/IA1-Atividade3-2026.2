"""Filtro de Kalman linear com atualização de Joseph."""
import numpy as np


class KalmanFilter:
    """Modelo x_k = F x_{k-1} + w e z_k = H x_k + v."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H = map(lambda value: np.asarray(value, dtype=float), (F, H))
        self.Q, self.R = map(lambda value: np.asarray(value, dtype=float), (Q, R))
        self.x, self.P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
        state_size = self.F.shape[0] if self.F.ndim == 2 else 0
        measurement_size = self.H.shape[0] if self.H.ndim == 2 else 0
        if (self.F.shape != (state_size, state_size)
                or self.H.shape != (measurement_size, state_size)
                or self.Q.shape != (state_size, state_size)
                or self.R.shape != (measurement_size, measurement_size)
                or self.x.shape != (state_size,)
                or self.P.shape != (state_size, state_size)):
            raise ValueError("Dimensões de F/H incompatíveis.")
        if (not all(np.all(np.isfinite(value)) for value in (self.F, self.H, self.Q, self.R, self.x, self.P))
                or not np.allclose(self.Q, self.Q.T)
                or not np.allclose(self.R, self.R.T)
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
        innovation = np.asarray(z) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        identity = np.eye(self.P.shape[0])
        residual = identity - K @ self.H
        self.x = self.x + K @ innovation
        self.P = residual @ self.P @ residual.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        return self.x.copy(), self.P.copy(), innovation, K
