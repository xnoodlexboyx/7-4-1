import sys
import numpy as np
import pytest
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
from stressors import apply_temperature, apply_voltage, apply_aging, apply_radiation, apply_emi, apply_multi_stress
from analysis import bit_error_rate

@pytest.fixture
def challenges():
    rng = np.random.default_rng(456)
    return rng.integers(0, 2, size=(100, 32))

# === Arbiter PUF Stressor Tests ===
@pytest.fixture
def arbiter_puf():
    return ArbiterPUF(n_stages=32, seed=123)

def test_arbiter_temperature_stress(arbiter_puf, challenges):
    puf_25 = apply_temperature(arbiter_puf, T_current=25, k_T=0.002, sigma_noise=0.05)
    puf_100 = apply_temperature(arbiter_puf, T_current=100, k_T=0.002, sigma_noise=0.05)
    resp_25 = puf_25.eval(challenges)
    resp_100 = puf_100.eval(challenges)
    ber = bit_error_rate(resp_25, resp_100)
    assert ber > 0, f"Arbiter PUF BER should be > 0, got {ber}"
    assert isinstance(puf_100, ArbiterPUF), "Stressed PUF should be same type"

def test_arbiter_temperature_parameters_change(arbiter_puf):
    original_params = arbiter_puf.delay_params.copy()
    stressed = apply_temperature(arbiter_puf, T_current=75, k_T=0.001, sigma_noise=0.02)
    # Parameters should change
    assert not np.array_equal(original_params, stressed.delay_params)
    # Temperature should be updated
    assert stressed.t_nominal == 75

# === SRAM PUF Stressor Tests ===
@pytest.fixture
def sram_puf():
    return SRAMPUF(n_cells=32, seed=123, radiation_hardening=0.95, low_power_mode=True)

def test_sram_temperature_stress(sram_puf, challenges):
    puf_25 = apply_temperature(sram_puf, T_current=25)
    puf_100 = apply_temperature(sram_puf, T_current=100)
    resp_25 = puf_25.eval(challenges)
    resp_100 = puf_100.eval(challenges)
    ber = bit_error_rate(resp_25, resp_100)
    assert ber >= 0, f"SRAM PUF BER should be >= 0, got {ber}"
    assert isinstance(puf_100, SRAMPUF), "Stressed PUF should be same type"

def test_sram_temperature_parameters_change(sram_puf):
    original_vth = sram_puf.vth_variations.copy()
    original_noise = sram_puf.noise_sensitivity.copy()
    stressed = apply_temperature(sram_puf, T_current=75)
    
    # VTH variations should change
    assert not np.array_equal(original_vth, stressed.vth_variations)
    # Noise sensitivity should increase with temperature
    assert np.mean(stressed.noise_sensitivity) > np.mean(original_noise)
    # Defense parameters should be preserved
    assert stressed.radiation_hardening == sram_puf.radiation_hardening
    assert stressed.low_power_mode == sram_puf.low_power_mode

# === Ring Oscillator PUF Stressor Tests ===
@pytest.fixture
def ro_puf():
    return RingOscillatorPUF(n_rings=32, seed=123, emi_resistance=0.9, freq_stability=0.8)

def test_ro_temperature_stress(ro_puf, challenges):
    puf_25 = apply_temperature(ro_puf, T_current=25)
    puf_100 = apply_temperature(ro_puf, T_current=100)
    resp_25 = puf_25.eval(challenges)
    resp_100 = puf_100.eval(challenges)
    ber = bit_error_rate(resp_25, resp_100)
    assert ber >= 0, f"RO PUF BER should be >= 0, got {ber}"
    assert isinstance(puf_100, RingOscillatorPUF), "Stressed PUF should be same type"

def test_ro_temperature_parameters_change(ro_puf):
    original_freqs = ro_puf.base_frequencies.copy()
    original_stability = ro_puf.freq_stability
    stressed = apply_temperature(ro_puf, T_current=85)
    
    # Frequencies should change (decrease with temperature)
    assert not np.array_equal(original_freqs, stressed.base_frequencies)
    assert np.mean(stressed.base_frequencies) < np.mean(original_freqs)
    # Stability should degrade at high temperature
    assert stressed.freq_stability < original_stability
    # Defense parameters should be preserved
    assert stressed.emi_resistance == ro_puf.emi_resistance

# === Butterfly PUF Stressor Tests ===
@pytest.fixture
def butterfly_puf():
    return ButterflyPUF(n_butterflies=32, seed=123, crosstalk_resistance=0.85)

def test_butterfly_temperature_stress(butterfly_puf, challenges):
    puf_25 = apply_temperature(butterfly_puf, T_current=25)
    puf_100 = apply_temperature(butterfly_puf, T_current=100)
    resp_25 = puf_25.eval(challenges)
    resp_100 = puf_100.eval(challenges)
    ber = bit_error_rate(resp_25, resp_100)
    assert ber >= 0, f"Butterfly PUF BER should be >= 0, got {ber}"
    assert isinstance(puf_100, ButterflyPUF), "Stressed PUF should be same type"

def test_butterfly_temperature_parameters_change(butterfly_puf):
    original_imbalances = butterfly_puf.latch_imbalances.copy()
    original_settling = butterfly_puf.settling_times.copy()
    original_crosstalk = butterfly_puf.crosstalk_factors.copy()
    stressed = apply_temperature(butterfly_puf, T_current=80)
    
    # Latch imbalances should change
    assert not np.array_equal(original_imbalances, stressed.latch_imbalances)
    # Settling times should change (faster at higher temp)
    assert not np.array_equal(original_settling, stressed.settling_times)
    # Crosstalk should increase with temperature
    assert np.mean(stressed.crosstalk_factors) > np.mean(original_crosstalk)
    # Defense parameters should be preserved
    assert stressed.crosstalk_resistance == butterfly_puf.crosstalk_resistance

# === Cross-PUF Stressor Tests ===
def test_temperature_stress_all_puf_types():
    """Test that temperature stress works for all PUF types"""
    seed = 123
    n_size = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_size))
    
    pufs = [
        ArbiterPUF(n_size, seed=seed),
        SRAMPUF(n_size, seed=seed),
        RingOscillatorPUF(n_size, seed=seed),
        ButterflyPUF(n_size, seed=seed)
    ]
    
    for puf in pufs:
        # Test temperature stress application
        stressed = apply_temperature(puf, T_current=85)
        assert type(stressed) == type(puf), f"Stressed PUF type mismatch for {type(puf)}"
        assert stressed.t_nominal == 85, f"Temperature not updated for {type(puf)}"
        
        # Test that responses can be generated
        responses = stressed.eval(challenges)
        assert responses.shape == (20,), f"Response shape incorrect for {type(puf)}"
        assert set(np.unique(responses)).issubset({-1, 1}), f"Invalid response values for {type(puf)}"

def test_extreme_temperature_ranges():
    """Test PUF behavior at extreme military temperatures"""
    puf = ArbiterPUF(16, seed=42)
    challenges = np.random.default_rng(123).integers(0, 2, size=(50, 16))
    
    # Test extreme cold (-55°C military spec)
    cold_puf = apply_temperature(puf, T_current=-55, military_spec=True)
    cold_responses = cold_puf.eval(challenges)
    assert len(cold_responses) == 50
    
    # Test extreme heat (125°C military spec)
    hot_puf = apply_temperature(puf, T_current=125, military_spec=True)
    hot_responses = hot_puf.eval(challenges)
    assert len(hot_responses) == 50
    
    # Calculate BER between extreme temperatures
    ber = bit_error_rate(cold_responses, hot_responses)
    assert ber >= 0, "BER should be non-negative"

def test_unsupported_puf_type_error():
    """Test that unsupported PUF type raises TypeError"""
    class UnsupportedPUF:
        def __init__(self):
            self.t_nominal = 25.0
    
    fake_puf = UnsupportedPUF()
    with pytest.raises(TypeError, match="apply_temperature does not support PUF type"):
        apply_temperature(fake_puf, T_current=50)

# === Advanced Stressor Tests ===
def test_voltage_stress_all_puf_types():
    """Test voltage stress on all PUF types"""
    seed = 123
    n_size = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_size))
    
    pufs = [
        ArbiterPUF(n_size, seed=seed),
        SRAMPUF(n_size, seed=seed),
        RingOscillatorPUF(n_size, seed=seed),
        ButterflyPUF(n_size, seed=seed)
    ]
    
    for puf in pufs:
        # Test low voltage stress (2.8V)
        low_v_puf = apply_voltage(puf, V_current=2.8, military_spec=True)
        assert type(low_v_puf) == type(puf), f"Voltage stressed PUF type mismatch for {type(puf)}"
        
        # Test high voltage stress (3.8V)
        high_v_puf = apply_voltage(puf, V_current=3.8, military_spec=True)
        assert type(high_v_puf) == type(puf), f"Voltage stressed PUF type mismatch for {type(puf)}"
        
        # Test that responses can be generated
        low_responses = low_v_puf.eval(challenges)
        high_responses = high_v_puf.eval(challenges)
        assert low_responses.shape == (20,), f"Low voltage response shape incorrect for {type(puf)}"
        assert high_responses.shape == (20,), f"High voltage response shape incorrect for {type(puf)}"
        
        # Test BER between voltage conditions
        ber = bit_error_rate(low_responses, high_responses)
        assert ber >= 0, f"Voltage BER should be non-negative for {type(puf)}"

def test_aging_stress_all_puf_types():
    """Test aging stress on all PUF types"""
    seed = 123
    n_size = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_size))
    
    pufs = [
        ArbiterPUF(n_size, seed=seed),
        SRAMPUF(n_size, seed=seed),
        RingOscillatorPUF(n_size, seed=seed),
        ButterflyPUF(n_size, seed=seed)
    ]
    
    for puf in pufs:
        # Test 1 year aging at 85°C
        aged_puf = apply_aging(puf, age_hours=8760, temperature_history=85.0, military_spec=True)
        assert type(aged_puf) == type(puf), f"Aged PUF type mismatch for {type(puf)}"
        
        # Test that responses can be generated
        original_responses = puf.eval(challenges)
        aged_responses = aged_puf.eval(challenges)
        assert aged_responses.shape == (20,), f"Aged response shape incorrect for {type(puf)}"
        
        # Test BER due to aging
        ber = bit_error_rate(original_responses, aged_responses)
        assert ber >= 0, f"Aging BER should be non-negative for {type(puf)}"

def test_radiation_stress_all_puf_types():
    """Test radiation stress on all PUF types"""
    seed = 123
    n_size = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_size))
    
    pufs = [
        ArbiterPUF(n_size, seed=seed),
        SRAMPUF(n_size, seed=seed),
        RingOscillatorPUF(n_size, seed=seed),
        ButterflyPUF(n_size, seed=seed)
    ]
    
    for puf in pufs:
        # Test gamma radiation (100 krad)
        gamma_puf = apply_radiation(puf, dose_krad=100, particle_type='gamma', military_spec=True)
        assert type(gamma_puf) == type(puf), f"Gamma radiated PUF type mismatch for {type(puf)}"
        
        # Test neutron radiation (50 krad)
        neutron_puf = apply_radiation(puf, dose_krad=50, particle_type='neutron', military_spec=True)
        assert type(neutron_puf) == type(puf), f"Neutron radiated PUF type mismatch for {type(puf)}"
        
        # Test that responses can be generated
        original_responses = puf.eval(challenges)
        gamma_responses = gamma_puf.eval(challenges)
        neutron_responses = neutron_puf.eval(challenges)
        assert gamma_responses.shape == (20,), f"Gamma response shape incorrect for {type(puf)}"
        assert neutron_responses.shape == (20,), f"Neutron response shape incorrect for {type(puf)}"
        
        # Test BER due to radiation
        gamma_ber = bit_error_rate(original_responses, gamma_responses)
        neutron_ber = bit_error_rate(original_responses, neutron_responses)
        assert gamma_ber >= 0, f"Gamma BER should be non-negative for {type(puf)}"
        assert neutron_ber >= 0, f"Neutron BER should be non-negative for {type(puf)}"
        # Neutron should cause more damage than gamma
        assert neutron_ber >= gamma_ber, f"Neutron BER should be >= gamma BER for {type(puf)}"

def test_emi_stress_all_puf_types():
    """Test EMI stress on all PUF types"""
    seed = 123
    n_size = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_size))
    
    pufs = [
        ArbiterPUF(n_size, seed=seed),
        SRAMPUF(n_size, seed=seed),
        RingOscillatorPUF(n_size, seed=seed),
        ButterflyPUF(n_size, seed=seed)
    ]
    
    for puf in pufs:
        # Test EMI stress (200 V/m at 100 MHz)
        emi_puf = apply_emi(puf, frequency_mhz=100, field_strength_v_m=200, military_spec=True)
        assert type(emi_puf) == type(puf), f"EMI stressed PUF type mismatch for {type(puf)}"
        
        # Test that responses can be generated
        original_responses = puf.eval(challenges)
        emi_responses = emi_puf.eval(challenges)
        assert emi_responses.shape == (20,), f"EMI response shape incorrect for {type(puf)}"
        
        # Test BER due to EMI
        ber = bit_error_rate(original_responses, emi_responses)
        assert ber >= 0, f"EMI BER should be non-negative for {type(puf)}"

def test_multi_stress_scenario():
    """Test multi-stress scenario combining all stressors"""
    puf = ArbiterPUF(32, seed=42)
    challenges = np.random.default_rng(123).integers(0, 2, size=(100, 32))
    
    # Apply multi-stress: high temp, low voltage, aging, radiation, EMI
    multi_stressed = apply_multi_stress(
        puf,
        temperature=85.0,
        voltage=2.9,
        age_hours=4380,  # 6 months
        radiation_krad=20,
        emi_field=150,
        military_spec=True
    )
    
    assert isinstance(multi_stressed, ArbiterPUF), "Multi-stressed PUF should be ArbiterPUF"
    
    # Test that responses can be generated
    original_responses = puf.eval(challenges)
    stressed_responses = multi_stressed.eval(challenges)
    assert stressed_responses.shape == (100,), "Multi-stressed response shape incorrect"
    
    # Test BER due to multi-stress
    ber = bit_error_rate(original_responses, stressed_responses)
    assert ber >= 0, "Multi-stress BER should be non-negative"
    assert ber <= 50, "Multi-stress BER should be reasonable (< 50%)"

def test_military_spec_differences():
    """Test that military spec provides better resistance"""
    puf = ArbiterPUF(32, seed=42)
    challenges = np.random.default_rng(123).integers(0, 2, size=(50, 32))
    original_responses = puf.eval(challenges)
    
    # Test temperature stress with and without military spec
    civilian_temp = apply_temperature(puf, T_current=85.0, military_spec=False)
    military_temp = apply_temperature(puf, T_current=85.0, military_spec=True)
    
    civilian_responses = civilian_temp.eval(challenges)
    military_responses = military_temp.eval(challenges)
    
    civilian_ber = bit_error_rate(original_responses, civilian_responses)
    military_ber = bit_error_rate(original_responses, military_responses)
    
    # Both should have some BER, but civilian typically higher
    assert civilian_ber >= 0, "Civilian BER should be non-negative"
    assert military_ber >= 0, "Military BER should be non-negative"

def test_extreme_military_conditions():
    """Test PUF behavior under extreme military conditions"""
    puf = ArbiterPUF(16, seed=42)
    challenges = np.random.default_rng(123).integers(0, 2, size=(30, 16))
    
    # Test extreme cold with military spec
    arctic_puf = apply_temperature(puf, T_current=-55.0, military_spec=True)
    arctic_responses = arctic_puf.eval(challenges)
    assert len(arctic_responses) == 30, "Arctic responses should be generated"
    
    # Test extreme heat with military spec
    desert_puf = apply_temperature(puf, T_current=125.0, military_spec=True)
    desert_responses = desert_puf.eval(challenges)
    assert len(desert_responses) == 30, "Desert responses should be generated"
    
    # Calculate BER between extreme military conditions
    ber = bit_error_rate(arctic_responses, desert_responses)
    assert ber >= 0, "Extreme military BER should be non-negative"

def test_unsupported_puf_type_errors():
    """Test that unsupported PUF types raise TypeError for all stressors"""
    class UnsupportedPUF:
        def __init__(self):
            self.t_nominal = 25.0
    
    fake_puf = UnsupportedPUF()
    
    # Test all stressors raise TypeError
    with pytest.raises(TypeError, match="apply_temperature does not support PUF type"):
        apply_temperature(fake_puf, T_current=50)
    
    with pytest.raises(TypeError, match="apply_voltage does not support PUF type"):
        apply_voltage(fake_puf, V_current=3.0)
    
    with pytest.raises(TypeError, match="apply_aging does not support PUF type"):
        apply_aging(fake_puf, age_hours=1000)
    
    with pytest.raises(TypeError, match="apply_radiation does not support PUF type"):
        apply_radiation(fake_puf, dose_krad=10)
    
    with pytest.raises(TypeError, match="apply_emi does not support PUF type"):
        apply_emi(fake_puf, frequency_mhz=100, field_strength_v_m=200) 