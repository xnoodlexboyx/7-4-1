#!/usr/bin/env python3
"""
Validation script for new PUF implementations.
Tests all PUF types and stressor functionality without requiring pytest.
"""

import sys
import numpy as np
sys.path.insert(0, "ppet-thesis")

from ppet.puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
from ppet.stressors import apply_temperature
from ppet.analysis import bit_error_rate

def test_puf_basic_functionality():
    """Test basic functionality of all PUF types"""
    print("=== Testing Basic PUF Functionality ===")
    
    n_stages = 16
    n_challenges = 10
    rng = np.random.default_rng(123)
    challenges = rng.integers(0, 2, size=(n_challenges, n_stages))
    
    # Test all PUF types
    pufs = [
        ("Arbiter", ArbiterPUF(n_stages, seed=42)),
        ("SRAM", SRAMPUF(n_stages, seed=42, radiation_hardening=0.95, low_power_mode=True)),
        ("RingOsc", RingOscillatorPUF(n_stages, seed=42, emi_resistance=0.9, freq_stability=0.8)),
        ("Butterfly", ButterflyPUF(n_stages, seed=42, metastability_time=1.0, crosstalk_resistance=0.85))
    ]
    
    for puf_name, puf in pufs:
        print(f"\n--- Testing {puf_name} PUF ---")
        
        # Test evaluation
        responses = puf.eval(challenges)
        assert responses.shape == (n_challenges,), f"{puf_name}: Wrong response shape"
        assert set(np.unique(responses)).issubset({-1, 1}), f"{puf_name}: Invalid response values"
        print(f"✓ {puf_name} evaluation works, responses: {responses}")
        
        # Test deterministic behavior
        responses2 = puf.eval(challenges)
        assert np.array_equal(responses, responses2), f"{puf_name}: Non-deterministic behavior"
        print(f"✓ {puf_name} deterministic behavior confirmed")
        
        # Test serialization
        json_data = puf.to_json()
        restored = puf.__class__.from_json(json_data)
        restored_responses = restored.eval(challenges)
        assert np.array_equal(responses, restored_responses), f"{puf_name}: Serialization failed"
        print(f"✓ {puf_name} serialization works")

def test_defense_parameters():
    """Test defense-specific parameters for each PUF type"""
    print("\n=== Testing Defense Parameters ===")
    
    # SRAM PUF defense parameters
    sram = SRAMPUF(16, seed=42, radiation_hardening=0.9, low_power_mode=True)
    assert sram.radiation_hardening == 0.9
    assert sram.low_power_mode == True
    print("✓ SRAM PUF defense parameters: radiation_hardening=0.9, low_power_mode=True")
    
    # Ring Oscillator PUF defense parameters
    ro = RingOscillatorPUF(16, seed=42, emi_resistance=0.85, freq_stability=0.75)
    assert ro.emi_resistance == 0.85
    assert ro.freq_stability == 0.75
    assert np.all(ro.base_frequencies > 50), "Frequencies should be reasonable"
    print("✓ Ring Oscillator PUF defense parameters: emi_resistance=0.85, freq_stability=0.75")
    
    # Butterfly PUF defense parameters
    butterfly = ButterflyPUF(16, seed=42, metastability_time=1.5, crosstalk_resistance=0.8)
    assert butterfly.metastability_time == 1.5
    assert butterfly.crosstalk_resistance == 0.8
    assert np.all(butterfly.settling_times > 0), "Settling times should be positive"
    print("✓ Butterfly PUF defense parameters: metastability_time=1.5, crosstalk_resistance=0.8")

def test_temperature_stressors():
    """Test temperature stressors on all PUF types"""
    print("\n=== Testing Temperature Stressors ===")
    
    n_stages = 16
    challenges = np.random.default_rng(456).integers(0, 2, size=(20, n_stages))
    
    pufs = [
        ("Arbiter", ArbiterPUF(n_stages, seed=123)),
        ("SRAM", SRAMPUF(n_stages, seed=123)),
        ("RingOsc", RingOscillatorPUF(n_stages, seed=123)),
        ("Butterfly", ButterflyPUF(n_stages, seed=123))
    ]
    
    for puf_name, puf in pufs:
        print(f"\n--- Testing {puf_name} Temperature Stress ---")
        
        # Apply temperature stress
        stressed = apply_temperature(puf, T_current=85, k_T=0.001, sigma_noise=0.02)
        assert type(stressed) == type(puf), f"{puf_name}: Type mismatch after stress"
        assert stressed.t_nominal == 85, f"{puf_name}: Temperature not updated"
        
        # Test responses
        orig_responses = puf.eval(challenges)
        stress_responses = stressed.eval(challenges)
        ber = bit_error_rate(orig_responses, stress_responses)
        
        print(f"✓ {puf_name} temperature stress applied, BER: {ber:.1f}%")
        assert ber >= 0, f"{puf_name}: Invalid BER"

def test_extreme_conditions():
    """Test PUFs under extreme military conditions"""
    print("\n=== Testing Extreme Military Conditions ===")
    
    puf = ArbiterPUF(16, seed=42)
    challenges = np.random.default_rng(789).integers(0, 2, size=(30, 16))
    
    # Test extreme temperatures
    cold_puf = apply_temperature(puf, T_current=-40)  # Arctic conditions
    hot_puf = apply_temperature(puf, T_current=125)   # Desert/engine bay conditions
    
    cold_responses = cold_puf.eval(challenges)
    hot_responses = hot_puf.eval(challenges)
    extreme_ber = bit_error_rate(cold_responses, hot_responses)
    
    print(f"✓ Extreme temperature range (-40°C to 125°C) BER: {extreme_ber:.1f}%")
    assert len(cold_responses) == 30
    assert len(hot_responses) == 30

def test_cross_puf_diversity():
    """Test that different PUF types produce diverse responses"""
    print("\n=== Testing Cross-PUF Diversity ===")
    
    seed = 42
    n_stages = 16
    challenges = np.random.default_rng(555).integers(0, 2, size=(50, n_stages))
    
    # Create all PUF types with same seed
    arbiter = ArbiterPUF(n_stages, seed=seed)
    sram = SRAMPUF(n_stages, seed=seed)
    ro = RingOscillatorPUF(n_stages, seed=seed)
    butterfly = ButterflyPUF(n_stages, seed=seed)
    
    # Get responses
    arbiter_resp = arbiter.eval(challenges)
    sram_resp = sram.eval(challenges)
    ro_resp = ro.eval(challenges)
    butterfly_resp = butterfly.eval(challenges)
    
    # Calculate pairwise agreements
    agreements = []
    pairs = [
        ("Arbiter vs SRAM", arbiter_resp, sram_resp),
        ("Arbiter vs RingOsc", arbiter_resp, ro_resp), 
        ("Arbiter vs Butterfly", arbiter_resp, butterfly_resp),
        ("SRAM vs RingOsc", sram_resp, ro_resp),
        ("SRAM vs Butterfly", sram_resp, butterfly_resp),
        ("RingOsc vs Butterfly", ro_resp, butterfly_resp)
    ]
    
    for pair_name, resp1, resp2 in pairs:
        agreement = np.mean(resp1 == resp2) * 100
        agreements.append(agreement)
        print(f"✓ {pair_name} agreement: {agreement:.1f}%")
    
    # Should have diversity (not all identical)
    avg_agreement = np.mean(agreements)
    print(f"✓ Average cross-PUF agreement: {avg_agreement:.1f}% (diversity confirmed)")
    assert avg_agreement < 95, "PUF types should show diversity"

def main():
    """Run all validation tests"""
    print("Defense-Oriented PUF Implementation Validation")
    print("=" * 50)
    
    try:
        test_puf_basic_functionality()
        test_defense_parameters()
        test_temperature_stressors()
        test_extreme_conditions()
        test_cross_puf_diversity()
        
        print("\n" + "=" * 50)
        print("🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("\nImplemented PUF types:")
        print("• Arbiter PUF - Linear additive delay model")
        print("• SRAM PUF - Threshold voltage variations with radiation hardening")
        print("• Ring Oscillator PUF - Frequency variations with EMI resistance")
        print("• Butterfly PUF - Metastability resolution with crosstalk resistance")
        print("\nDefense-specific features:")
        print("• Temperature stress modeling for all PUF types")
        print("• Military temperature range testing (-40°C to 125°C)")
        print("• Radiation hardening parameters for SRAM PUFs")
        print("• EMI resistance modeling for Ring Oscillator PUFs")
        print("• Low-power operation modes")
        print("• Cross-talk resistance for dense military electronics")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())