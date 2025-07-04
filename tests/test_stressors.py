import sys
import numpy as np
import pytest
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF
from stressors import apply_temperature
from analysis import bit_error_rate

@pytest.fixture
def arbiter_puf():
    return ArbiterPUF(n_stages=32, seed=123)

@pytest.fixture
def challenges():
    rng = np.random.default_rng(456)
    return rng.integers(0, 2, size=(200, 32))  # Increased to 200 challenges

def test_temperature_stressor_ber(arbiter_puf, challenges):
    # Increase k_T and sigma_noise for robustness
    puf_25 = apply_temperature(arbiter_puf, T_current=25, k_T=0.002, sigma_noise=0.05)
    puf_100 = apply_temperature(arbiter_puf, T_current=100, k_T=0.002, sigma_noise=0.05)
    resp_25 = puf_25.eval(challenges)
    resp_100 = puf_100.eval(challenges)
    ber = bit_error_rate(resp_25, resp_100)
    assert ber > 0, f"BER should be > 0, got {ber}" 