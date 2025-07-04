import sys
import numpy as np
import pytest
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF

@pytest.fixture
def arbiter_puf():
    return ArbiterPUF(n_stages=8, seed=42)

@pytest.fixture
def challenges():
    rng = np.random.default_rng(123)
    return rng.integers(0, 2, size=(5, 8))

def test_eval_shape_and_values(arbiter_puf, challenges):
    responses = arbiter_puf.eval(challenges)
    assert responses.shape == (5,)
    assert set(np.unique(responses)).issubset({-1, 1})

def test_deterministic_with_seed():
    rng = np.random.default_rng(123)
    chals = rng.integers(0, 2, size=(5, 8))
    puf1 = ArbiterPUF(8, seed=99)
    puf2 = ArbiterPUF(8, seed=99)
    assert np.all(puf1.eval(chals) == puf2.eval(chals)) 