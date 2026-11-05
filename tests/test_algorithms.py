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


def test_bayesian_rejects_incomplete_cpt():
    with np.testing.assert_raises(ValueError):
        from bayesian_network import BayesianNetwork
        BayesianNetwork({"x": ()}, {"x": {}})


def test_hmm_forward_and_viterbi():
    model = build_hmm()
    observations = np.array([0, 1, 2, 3, 4])
    alpha, scales = model.forward(observations)
    path, score = model.viterbi(observations)
    assert alpha.shape == (5, 5)
    assert np.allclose(alpha.sum(1), 1)
    assert len(path) == 5 and np.isfinite(score) and np.all(scales > 0)


def test_hmm_rejects_empty_and_impossible_sequences():
    model = build_hmm()
    with np.testing.assert_raises(ValueError):
        model.forward([])
    with np.testing.assert_raises(ValueError):
        model.viterbi([99])


def test_hmm_viterbi_respects_zero_probability_transitions():
    from hmm import DiscreteHMM
    model = DiscreteHMM([[0, 1], [1, 0]], [[1, 0], [0, 1]], [1, 0])
    path, _ = model.viterbi([0, 1, 0, 1])
    assert np.array_equal(path, [0, 1, 0, 1])


def test_kalman_covariance_stays_symmetric():
    f = KalmanFilter([[1, 1], [0, 1]], [[1, 0]], np.eye(2) * .01,
                     [[1]], [0, 1], np.eye(2))
    f.predict()
    _, covariance, _, _ = f.update([1.])
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-10)


def test_kalman_rejects_inconsistent_dimensions():
    with np.testing.assert_raises(ValueError):
        KalmanFilter([[1, 0]], [[1, 0]], np.eye(2), [[1]], [0, 0], np.eye(2))
