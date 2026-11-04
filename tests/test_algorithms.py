import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.extend([str(ROOT / "projeto_1_redes_bayesianas"), str(ROOT / "projeto_2_hmm"), str(ROOT / "projeto_3_kalman")])
from bayesian_network import build_industrial_network
from hmm import build_hmm
from kalman_filter import KalmanFilter


def test_bayesian_posterior_is_probability():
    net = build_industrial_network()
    value = net.posterior("mechanical_failure", {"high_temperature": True})
    assert 0 < value < 1


def test_hmm_forward_and_viterbi():
    model = build_hmm()
    observations = np.array([0, 1, 2, 3, 4])
    alpha, scales = model.forward(observations)
    path, score = model.viterbi(observations)
    assert alpha.shape == (5, 5)
    assert np.allclose(alpha.sum(1), 1)
    assert len(path) == 5 and np.isfinite(score) and np.all(scales > 0)


def test_kalman_covariance_stays_symmetric():
    f = KalmanFilter([[1, 1], [0, 1]], [[1, 0]], np.eye(2) * .01,
                     [[1]], [0, 1], np.eye(2))
    f.predict()
    _, covariance, _, _ = f.update([1.])
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-10)
