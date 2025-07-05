import numpy as np
import pytest
from ppet.analysis import simulate_ecc

@pytest.fixture
def golden_noisy_pair():
    golden = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    noisy = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])  # HDs: 1, 1
    return golden, noisy

def test_ecc_fail_all_if_t0(golden_noisy_pair):
    golden, noisy = golden_noisy_pair
    fail_rate = simulate_ecc(noisy, golden, t=0)
    assert np.isclose(fail_rate, 1.0)

def test_ecc_succeed_all_if_tmax(golden_noisy_pair):
    golden, noisy = golden_noisy_pair
    tmax = golden.shape[1]
    fail_rate = simulate_ecc(noisy, golden, t=tmax)
    assert np.isclose(fail_rate, 0.0) 