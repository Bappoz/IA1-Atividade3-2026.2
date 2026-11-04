"""Filtro de Kalman linear com atualização de Joseph."""
import numpy as np


class KalmanFilter:
    """Modelo x_k = F x_{k-1} + w e z_k = H x_k + v."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H = map(np.asarray, (F, H))
        self.Q, self.R = map(np.asarray, (Q, R))
        self.x, self.P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
        if self.F.shape[0] != self.F.shape[1] or self.H.shape[1] != self.F.shape[0]:
            raise ValueError("Dimensões de F/H incompatíveis.")

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        innovation = np.asarray(z) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        identity = np.eye(self.P.shape[0])
        residual = identity - K @ self.H
        self.x = self.x + K @ innovation
        self.P = residual @ self.P @ residual.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        return self.x.copy(), self.P.copy(), innovation, K
