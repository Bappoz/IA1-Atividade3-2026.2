"""HMM discreto com Forward escalonado, Viterbi e Baum-Welch."""
from __future__ import annotations
import numpy as np


class DiscreteHMM:
    """Modelo HMM com matriz de transição A e emissão B."""

    def __init__(self, transition, emission, initial):
        self.transition = np.asarray(transition, dtype=float)
        self.emission = np.asarray(emission, dtype=float)
        self.initial = np.asarray(initial, dtype=float)
        self._validate()

    def _validate(self):
        if self.transition.shape[0] != self.transition.shape[1]:
            raise ValueError("A matriz de transição deve ser quadrada.")
        if self.emission.shape[0] != self.transition.shape[0]:
            raise ValueError("Número de estados inconsistente.")
        if not np.allclose(self.transition.sum(1), 1) or not np.allclose(self.emission.sum(1), 1):
            raise ValueError("Linhas de A e B devem somar 1.")
        if not np.isclose(self.initial.sum(), 1):
            raise ValueError("Distribuição inicial inválida.")

    def sample(self, length, rng):
        states, observations = [], []
        state = rng.choice(len(self.initial), p=self.initial)
        for _ in range(length):
            states.append(state)
            observations.append(rng.choice(self.emission.shape[1], p=self.emission[state]))
            state = rng.choice(len(self.initial), p=self.transition[state])
        return np.array(states), np.array(observations)

    def forward(self, observations):
        observations = np.asarray(observations, dtype=int)
        alpha = np.zeros((len(observations), len(self.initial)))
        scales = np.zeros(len(observations))
        alpha[0] = self.initial * self.emission[:, observations[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]
        for t in range(1, len(observations)):
            alpha[t] = (alpha[t - 1] @ self.transition) * self.emission[:, observations[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= max(scales[t], np.finfo(float).tiny)
        return alpha, scales

    def log_likelihood(self, observations):
        _, scales = self.forward(observations)
        return float(np.log(scales).sum())

    def viterbi(self, observations):
        observations = np.asarray(observations, dtype=int)
        n = len(self.initial)
        tiny = np.finfo(float).tiny
        log_a = np.log(np.maximum(self.transition, tiny))
        log_b = np.log(np.maximum(self.emission, tiny))
        log_pi = np.log(np.maximum(self.initial, tiny))
        delta = np.empty((len(observations), n))
        psi = np.zeros((len(observations), n), dtype=int)
        delta[0] = log_pi + log_b[:, observations[0]]
        for t in range(1, len(observations)):
            scores = delta[t - 1][:, None] + log_a
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores[psi[t], np.arange(n)] + log_b[:, observations[t]]
        path = np.zeros(len(observations), dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(len(observations) - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1].max())

    def baum_welch(self, sequences, iterations=30, tolerance=1e-5):
        """Estima A, B e pi por EM; retorna log-verossimilhanças."""
        history = []
        for _ in range(iterations):
            a_num = np.zeros_like(self.transition); a_den = np.zeros(len(self.initial))
            b_num = np.zeros_like(self.emission); b_den = np.zeros(len(self.initial))
            pi_acc = np.zeros(len(self.initial)); total_ll = 0
            for obs in sequences:
                alpha, scales = self.forward(obs)
                total_ll += np.log(scales).sum()
                beta = np.zeros_like(alpha); beta[-1] = 1
                for t in range(len(obs) - 2, -1, -1):
                    beta[t] = self.transition @ (self.emission[:, obs[t + 1]] * beta[t + 1])
                    beta[t] /= max(scales[t + 1], np.finfo(float).tiny)
                gamma = alpha * beta
                gamma /= gamma.sum(1, keepdims=True)
                pi_acc += gamma[0]
                for t in range(len(obs) - 1):
                    xi = alpha[t][:, None] * self.transition * self.emission[:, obs[t + 1]][None, :] * beta[t + 1]
                    xi /= max(xi.sum(), np.finfo(float).tiny)
                    a_num += xi; a_den += gamma[t]
                for t, symbol in enumerate(obs):
                    b_num[:, symbol] += gamma[t]; b_den += gamma[t]
            self.initial = pi_acc / pi_acc.sum()
            self.transition = a_num / np.maximum(a_den[:, None], 1e-12)
            self.emission = b_num / np.maximum(b_den[:, None], 1e-12)
            history.append(float(total_ll))
            if len(history) > 1 and abs(history[-1] - history[-2]) < tolerance:
                break
        return history


def build_hmm() -> DiscreteHMM:
    transition = [[.82, .12, .05, .01, 0.0], [.08, .72, .15, .04, .01],
                  [.02, .10, .70, .15, .03], [.01, .03, .12, .66, .18],
                  [.01, .01, .02, .10, .86]]
    emission = [[.70, .20, .07, .02, .01, 0.0], [.10, .55, .25, .08, .02, 0.0],
                [.03, .15, .55, .20, .06, .01], [.01, .04, .15, .55, .22, .03],
                [0.0, .01, .04, .15, .50, .30]]
    return DiscreteHMM(transition, emission, [.80, .10, .06, .03, .01])
