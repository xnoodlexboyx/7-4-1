import sys
import numpy as np
import pytest
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF

@pytest.fixture
def challenges():
    rng = np.random.default_rng(123)
    return rng.integers(0, 2, size=(5, 8))

# === Arbiter PUF Tests ===
@pytest.fixture
def arbiter_puf():
    return ArbiterPUF(n_stages=8, seed=42)

def test_arbiter_eval_shape_and_values(arbiter_puf, challenges):
    responses = arbiter_puf.eval(challenges)
    assert responses.shape == (5,)
    assert set(np.unique(responses)).issubset({-1, 1})

def test_arbiter_deterministic_with_seed():
    rng = np.random.default_rng(123)
    chals = rng.integers(0, 2, size=(5, 8))
    puf1 = ArbiterPUF(8, seed=99)
    puf2 = ArbiterPUF(8, seed=99)
    assert np.all(puf1.eval(chals) == puf2.eval(chals))

def test_arbiter_serialization(arbiter_puf, challenges):
    json_data = arbiter_puf.to_json()
    restored = ArbiterPUF.from_json(json_data)
    original_responses = arbiter_puf.eval(challenges)
    restored_responses = restored.eval(challenges)
    assert np.array_equal(original_responses, restored_responses)

# === SRAM PUF Tests ===
@pytest.fixture
def sram_puf():
    return SRAMPUF(n_cells=8, seed=42, radiation_hardening=0.95, low_power_mode=True)

def test_sram_eval_shape_and_values(sram_puf, challenges):
    responses = sram_puf.eval(challenges)
    assert responses.shape == (5,)
    assert set(np.unique(responses)).issubset({-1, 1})

def test_sram_defense_parameters(sram_puf):
    assert sram_puf.radiation_hardening == 0.95
    assert sram_puf.low_power_mode == True
    assert hasattr(sram_puf, 'vth_variations')
    assert hasattr(sram_puf, 'noise_sensitivity')

def test_sram_deterministic_with_seed():
    chals = np.random.default_rng(123).integers(0, 2, size=(5, 8))
    puf1 = SRAMPUF(8, seed=99, radiation_hardening=0.9)
    puf2 = SRAMPUF(8, seed=99, radiation_hardening=0.9)
    assert np.all(puf1.eval(chals) == puf2.eval(chals))

def test_sram_serialization(sram_puf, challenges):
    json_data = sram_puf.to_json()
    restored = SRAMPUF.from_json(json_data)
    original_responses = sram_puf.eval(challenges)
    restored_responses = restored.eval(challenges)
    assert np.array_equal(original_responses, restored_responses)

# === Ring Oscillator PUF Tests ===
@pytest.fixture
def ro_puf():
    return RingOscillatorPUF(n_rings=8, seed=42, emi_resistance=0.9, freq_stability=0.8)

def test_ro_eval_shape_and_values(ro_puf, challenges):
    responses = ro_puf.eval(challenges)
    assert responses.shape == (5,)
    assert set(np.unique(responses)).issubset({-1, 1})

def test_ro_defense_parameters(ro_puf):
    assert ro_puf.emi_resistance == 0.9
    assert ro_puf.freq_stability == 0.8
    assert hasattr(ro_puf, 'base_frequencies')
    assert hasattr(ro_puf, 'emi_susceptibility')
    # Check frequency range is reasonable (around 100MHz)
    assert np.all(ro_puf.base_frequencies > 50)
    assert np.all(ro_puf.base_frequencies < 150)

def test_ro_deterministic_with_seed():
    chals = np.random.default_rng(123).integers(0, 2, size=(5, 8))
    puf1 = RingOscillatorPUF(8, seed=99, emi_resistance=0.85)
    puf2 = RingOscillatorPUF(8, seed=99, emi_resistance=0.85)
    assert np.all(puf1.eval(chals) == puf2.eval(chals))

def test_ro_serialization(ro_puf, challenges):
    json_data = ro_puf.to_json()
    restored = RingOscillatorPUF.from_json(json_data)
    original_responses = ro_puf.eval(challenges)
    restored_responses = restored.eval(challenges)
    assert np.array_equal(original_responses, restored_responses)

# === Butterfly PUF Tests ===
@pytest.fixture
def butterfly_puf():
    return ButterflyPUF(n_butterflies=8, seed=42, metastability_time=1.0, crosstalk_resistance=0.85)

def test_butterfly_eval_shape_and_values(butterfly_puf, challenges):
    responses = butterfly_puf.eval(challenges)
    assert responses.shape == (5,)
    assert set(np.unique(responses)).issubset({-1, 1})

def test_butterfly_defense_parameters(butterfly_puf):
    assert butterfly_puf.metastability_time == 1.0
    assert butterfly_puf.crosstalk_resistance == 0.85
    assert hasattr(butterfly_puf, 'latch_imbalances')
    assert hasattr(butterfly_puf, 'settling_times')
    assert hasattr(butterfly_puf, 'crosstalk_factors')
    # Check settling times are positive
    assert np.all(butterfly_puf.settling_times > 0)

def test_butterfly_deterministic_with_seed():
    chals = np.random.default_rng(123).integers(0, 2, size=(5, 8))
    puf1 = ButterflyPUF(8, seed=99, crosstalk_resistance=0.8)
    puf2 = ButterflyPUF(8, seed=99, crosstalk_resistance=0.8)
    assert np.all(puf1.eval(chals) == puf2.eval(chals))

def test_butterfly_serialization(butterfly_puf, challenges):
    json_data = butterfly_puf.to_json()
    restored = ButterflyPUF.from_json(json_data)
    original_responses = butterfly_puf.eval(challenges)
    restored_responses = restored.eval(challenges)
    assert np.array_equal(original_responses, restored_responses)

# === Cross-PUF Tests ===
def test_different_puf_types_different_responses():
    """Test that different PUF types give different responses to same challenges"""
    challenges = np.random.default_rng(123).integers(0, 2, size=(10, 8))
    seed = 42
    
    arbiter = ArbiterPUF(8, seed=seed)
    sram = SRAMPUF(8, seed=seed)
    ro = RingOscillatorPUF(8, seed=seed)
    butterfly = ButterflyPUF(8, seed=seed)
    
    arbiter_resp = arbiter.eval(challenges)
    sram_resp = sram.eval(challenges)
    ro_resp = ro.eval(challenges)
    butterfly_resp = butterfly.eval(challenges)
    
    # Different PUF types should generally give different responses
    # (though not guaranteed to be different for every challenge)
    total_same = 0
    total_same += np.sum(arbiter_resp == sram_resp)
    total_same += np.sum(arbiter_resp == ro_resp)
    total_same += np.sum(arbiter_resp == butterfly_resp)
    total_same += np.sum(sram_resp == ro_resp)
    total_same += np.sum(sram_resp == butterfly_resp)
    total_same += np.sum(ro_resp == butterfly_resp)
    
    # Should not have all responses identical across all PUF types
    assert total_same < 6 * len(challenges)  # Some variation expected

def test_challenge_length_mismatch():
    """Test that wrong challenge length raises assertion error"""
    puf = ArbiterPUF(8, seed=42)
    wrong_challenges = np.random.randint(0, 2, size=(5, 10))  # Wrong length
    
    with pytest.raises(AssertionError, match="Challenge length mismatch"):
        puf.eval(wrong_challenges)

def test_single_challenge_eval():
    """Test evaluation with single challenge (1D array)"""
    puf = ArbiterPUF(8, seed=42)
    single_challenge = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    response = puf.eval(single_challenge)
    assert response.shape == (1,)
    assert response[0] in [-1, 1] 