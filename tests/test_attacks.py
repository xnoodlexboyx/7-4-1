import sys
import numpy as np
import pytest
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF
from attacks import MLAttacker

@pytest.fixture
def crp_data():
    rng = np.random.default_rng(789)
    n_stages = 32
    n_train = 1000
    n_test = 200
    challenges = rng.integers(0, 2, size=(n_train + n_test, n_stages))
    puf = ArbiterPUF(n_stages, seed=321)
    responses = puf.eval(challenges)
    return challenges[:n_train], responses[:n_train], challenges[n_train:], responses[n_train:]

def test_mlattacker_accuracy(crp_data):
    X_train, y_train, X_test, y_test = crp_data
    attacker = MLAttacker(n_stages=32)
    attacker.train(X_train, y_train)
    acc = attacker.accuracy(X_test, y_test)
    assert acc >= 0.97, f"Expected accuracy >= 0.97, got {acc}" 