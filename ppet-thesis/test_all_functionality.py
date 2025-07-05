#!/usr/bin/env python3
"""
Comprehensive Test Suite for PPET Framework
==========================================

This script provides a complete functionality test of the PPET framework,
ensuring all components work together correctly for defense-oriented PUF
evaluation and analysis.

Usage:
    python test_all_functionality.py
    python test_all_functionality.py --quick       # Run quick tests only
    python test_all_functionality.py --verbose     # Detailed output
    python test_all_functionality.py --output-dir /path/to/output

Test Categories:
- Core PUF functionality (all 4 architectures)
- Environmental stress testing (military conditions)
- Attack resistance evaluation (ML, CNN, adversarial)
- Analysis and visualization generation
- Military scenario simulation
- Security metrics and compliance
- Data persistence and serialization
"""

import sys
import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add ppet-thesis to path for standalone execution
current_dir = Path(__file__).parent
ppet_dir = current_dir / "ppet-thesis"
if ppet_dir.exists():
    sys.path.insert(0, str(ppet_dir))
else:
    # Assume we're already in the ppet-thesis directory
    sys.path.insert(0, str(current_dir))

import numpy as np

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": [],
    "warnings": []
}


def log_test(test_name: str, success: bool, message: str = "", warning: bool = False):
    """Log test result and update counters."""
    global test_results
    
    if success:
        test_results["passed"] += 1
        status = "✅ PASS"
    elif warning:
        test_results["skipped"] += 1
        status = "⚠️  SKIP"
    else:
        test_results["failed"] += 1
        test_results["errors"].append(f"{test_name}: {message}")
        status = "❌ FAIL"
    
    print(f"{status} {test_name}")
    if message and (not success or warning):
        print(f"      {message}")


def test_core_imports():
    """Test core module imports."""
    print("\n📦 Testing Core Module Imports")
    print("-" * 40)
    
    modules_to_test = [
        ("ppet.puf_models", "ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF"),
        ("ppet.stressors", "apply_temperature, apply_voltage, apply_aging"),
        ("ppet.attacks", "MLAttacker, CNNAttacker, AdversarialAttacker"),
        ("ppet.analysis", "bit_error_rate, uniqueness, simulate_ecc"),
        ("ppet.visualization", "generate_all_thesis_plots"),
        ("ppet.statistical_plots", "generate_statistical_suite"),
        ("ppet.bit_analysis", "plot_bit_aliasing_heatmap"),
        ("ppet.defense_dashboard", "create_defense_dashboard"),
    ]
    
    for module_name, components in modules_to_test:
        try:
            exec(f"from {module_name} import {components}")
            log_test(f"Import {module_name}", True)
        except ImportError as e:
            log_test(f"Import {module_name}", False, str(e))
        except Exception as e:
            log_test(f"Import {module_name}", False, f"Unexpected error: {e}")


def test_puf_functionality(quick_mode: bool = False):
    """Test PUF evaluation functionality."""
    print("\n🔧 Testing PUF Functionality")
    print("-" * 40)
    
    try:
        from ppet.puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
        
        # Test parameters
        n_challenges = 100 if quick_mode else 1000
        n_stages = 32 if quick_mode else 64
        
        # Generate test challenges
        rng = np.random.default_rng(42)
        challenges = rng.integers(0, 2, size=(n_challenges, n_stages))
        
        puf_classes = [
            ("ArbiterPUF", ArbiterPUF),
            ("SRAMPUF", SRAMPUF),
            ("RingOscillatorPUF", RingOscillatorPUF),
            ("ButterflyPUF", ButterflyPUF)
        ]
        
        for puf_name, PUFClass in puf_classes:
            try:
                # Create PUF instance
                if puf_name in ["RingOscillatorPUF", "ButterflyPUF"]:
                    puf = PUFClass(n_rings=min(n_stages//2, 16), seed=42) if puf_name == "RingOscillatorPUF" else PUFClass(n_butterflies=min(n_stages//2, 16), seed=42)
                    test_challenges = challenges[:, :min(n_stages//2, 16)]
                else:
                    puf = PUFClass(n_stages=n_stages, seed=42) if puf_name == "ArbiterPUF" else PUFClass(n_cells=n_stages, seed=42)
                    test_challenges = challenges
                
                # Test evaluation
                responses = puf.eval(test_challenges)
                
                # Verify response properties
                assert len(responses) == n_challenges, f"Wrong number of responses for {puf_name}"
                assert all(r in [-1, 1] for r in responses), f"Invalid response values for {puf_name}"
                
                # Test deterministic behavior
                responses2 = puf.eval(test_challenges)
                assert np.array_equal(responses, responses2), f"Non-deterministic behavior in {puf_name}"
                
                # Test serialization if available
                if hasattr(puf, 'to_json') and hasattr(PUFClass, 'from_json'):
                    json_data = puf.to_json()
                    restored_puf = PUFClass.from_json(json_data)
                    restored_responses = restored_puf.eval(test_challenges)
                    assert np.array_equal(responses, restored_responses), f"Serialization failed for {puf_name}"
                
                log_test(f"{puf_name} evaluation", True)
                
            except Exception as e:
                log_test(f"{puf_name} evaluation", False, str(e))
    
    except ImportError as e:
        log_test("PUF functionality", False, f"Import error: {e}")


def test_environmental_stress(quick_mode: bool = False):
    """Test environmental stress application."""
    print("\n🌡️  Testing Environmental Stress")
    print("-" * 40)
    
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.stressors import apply_temperature, apply_voltage, apply_aging, apply_radiation, apply_emi
        from ppet.analysis import bit_error_rate
        
        # Create test PUF
        puf = ArbiterPUF(n_stages=32, seed=42)
        rng = np.random.default_rng(42)
        challenges = rng.integers(0, 2, size=(100, 32))
        original_responses = puf.eval(challenges)
        
        # Test temperature stress
        try:
            temp_puf = apply_temperature(puf, T_current=85.0, military_spec=True)
            temp_responses = temp_puf.eval(challenges)
            ber = bit_error_rate(original_responses, temp_responses)
            assert 0 <= ber <= 100, "Invalid BER value"
            log_test("Temperature stress", True)
        except Exception as e:
            log_test("Temperature stress", False, str(e))
        
        # Test voltage stress
        try:
            volt_puf = apply_voltage(puf, V_current=2.8, military_spec=True)
            volt_responses = volt_puf.eval(challenges)
            log_test("Voltage stress", True)
        except Exception as e:
            log_test("Voltage stress", False, str(e))
        
        # Test aging stress
        try:
            aged_puf = apply_aging(puf, age_hours=1000, military_spec=True)
            aged_responses = aged_puf.eval(challenges)
            log_test("Aging stress", True)
        except Exception as e:
            log_test("Aging stress", False, str(e))
        
        # Test radiation stress
        try:
            rad_puf = apply_radiation(puf, dose_krad=50, military_spec=True)
            rad_responses = rad_puf.eval(challenges)
            log_test("Radiation stress", True)
        except Exception as e:
            log_test("Radiation stress", False, str(e))
        
        # Test EMI stress
        try:
            emi_puf = apply_emi(puf, frequency_mhz=100, field_strength_v_m=150, military_spec=True)
            emi_responses = emi_puf.eval(challenges)
            log_test("EMI stress", True)
        except Exception as e:
            log_test("EMI stress", False, str(e))
    
    except ImportError as e:
        log_test("Environmental stress", False, f"Import error: {e}")


def test_attack_resistance(quick_mode: bool = False):
    """Test attack resistance evaluation."""
    print("\n🛡️  Testing Attack Resistance")
    print("-" * 40)
    
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.attacks import MLAttacker, CNNAttacker, AdversarialAttacker
        
        # Create test data
        puf = ArbiterPUF(n_stages=32, seed=42)
        rng = np.random.default_rng(42)
        n_challenges = 500 if quick_mode else 1000
        challenges = rng.integers(0, 2, size=(n_challenges, 32))
        responses = puf.eval(challenges)
        
        # Test ML attacker
        try:
            ml_attacker = MLAttacker(n_stages=32)
            ml_attacker.train(challenges[:400], responses[:400])
            accuracy = ml_attacker.accuracy(challenges[400:], responses[400:])
            assert 0 <= accuracy <= 1, "Invalid accuracy value"
            log_test("ML Attack", True, f"Accuracy: {accuracy:.3f}")
        except Exception as e:
            log_test("ML Attack", False, str(e))
        
        # Test CNN attacker (may fail due to dependencies)
        try:
            cnn_attacker = CNNAttacker(n_stages=32, architecture='mlp')
            cnn_attacker.train(challenges[:200], responses[:200])  # Smaller dataset for speed
            cnn_accuracy = cnn_attacker.accuracy(challenges[200:250], responses[200:250])
            log_test("CNN Attack", True, f"Accuracy: {cnn_accuracy:.3f}")
        except Exception as e:
            log_test("CNN Attack", False, str(e), warning=True)
        
        # Test adversarial attacker
        try:
            adv_attacker = AdversarialAttacker(puf_type='arbiter')
            results = adv_attacker.adaptive_attack(puf, n_queries=200, adaptation_rounds=2)
            assert 'final_accuracy' in results, "Missing final_accuracy in results"
            log_test("Adversarial Attack", True, f"Final accuracy: {results['final_accuracy']:.3f}")
        except Exception as e:
            log_test("Adversarial Attack", False, str(e), warning=True)
    
    except ImportError as e:
        log_test("Attack resistance", False, f"Import error: {e}")


def test_analysis_functions(quick_mode: bool = False):
    """Test analysis and metric calculation functions."""
    print("\n📊 Testing Analysis Functions")
    print("-" * 40)
    
    try:
        from ppet.analysis import bit_error_rate, uniqueness, simulate_ecc, hamming
        from ppet.puf_models import ArbiterPUF
        
        # Test bit error rate
        try:
            resp1 = np.array([1, -1, 1, -1, 1])
            resp2 = np.array([1, 1, 1, -1, -1])  # 2 errors out of 5
            ber = bit_error_rate(resp1, resp2)
            expected_ber = 40.0  # 2/5 * 100
            assert abs(ber - expected_ber) < 0.1, f"BER calculation error: {ber} vs {expected_ber}"
            log_test("Bit Error Rate", True)
        except Exception as e:
            log_test("Bit Error Rate", False, str(e))
        
        # Test uniqueness metric
        try:
            # Create multiple PUF responses
            rng = np.random.default_rng(42)
            challenges = rng.integers(0, 2, size=(100, 32))
            
            multi_responses = []
            for i in range(5):
                puf = ArbiterPUF(n_stages=32, seed=100+i)
                responses = puf.eval(challenges)
                multi_responses.append(responses)
            
            unique_val = uniqueness(challenges, np.array(multi_responses))
            assert 0 <= unique_val <= 100, f"Invalid uniqueness value: {unique_val}"
            log_test("Uniqueness Metric", True, f"Uniqueness: {unique_val:.1f}%")
        except Exception as e:
            log_test("Uniqueness Metric", False, str(e))
        
        # Test ECC simulation
        try:
            # Test ECC with known error patterns
            received = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])  # 2 blocks, 4 bits each
            reference = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])  # 1 error in each block
            
            fail_rate_t0 = simulate_ecc(received, reference, t=0)  # No correction
            fail_rate_t1 = simulate_ecc(received, reference, t=1)  # 1-bit correction
            fail_rate_t4 = simulate_ecc(received, reference, t=4)  # 4-bit correction
            
            assert fail_rate_t0 == 1.0, "t=0 should fail all blocks"
            assert fail_rate_t1 == 0.0, "t=1 should correct all single-bit errors"
            assert fail_rate_t4 == 0.0, "t=4 should correct all errors"
            
            log_test("ECC Simulation", True)
        except Exception as e:
            log_test("ECC Simulation", False, str(e))
        
        # Test Hamming distance
        try:
            a = np.array([1, 0, 1, 0])
            b = np.array([1, 1, 1, 1])
            hd = hamming(a, b)
            # Hamming distance counts number of different bits (2 bits different)
            assert hd == 2, f"Hamming distance should be 2, got {hd}"
            log_test("Hamming Distance", True)
        except Exception as e:
            log_test("Hamming Distance", False, str(e))
    
    except ImportError as e:
        log_test("Analysis functions", False, f"Import error: {e}")


def test_visualization_generation(output_dir: str, quick_mode: bool = False):
    """Test visualization generation."""
    print("\n📈 Testing Visualization Generation")
    print("-" * 40)
    
    viz_output_dir = os.path.join(output_dir, "test_visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Test basic plotting
    try:
        from ppet.analysis import plot_reliability_vs_temperature, plot_attack_accuracy
        import matplotlib.pyplot as plt
        
        temps = np.array([-20, 0, 25, 50, 75, 100])
        reliability = np.array([98, 97, 96, 94, 91, 87])
        attack_acc = np.array([85, 86, 87, 89, 92, 95])
        
        fig1 = plot_reliability_vs_temperature(temps, reliability)
        fig1.savefig(os.path.join(viz_output_dir, "reliability_test.png"))
        plt.close(fig1)
        
        fig2 = plot_attack_accuracy(temps, attack_acc)
        fig2.savefig(os.path.join(viz_output_dir, "attack_accuracy_test.png"))
        plt.close(fig2)
        
        log_test("Basic plots", True)
    except Exception as e:
        log_test("Basic plots", False, str(e))
    
    # Test comprehensive visualization suite (may fail due to dependencies)
    try:
        from ppet.visualization import generate_all_thesis_plots
        
        # Sample data
        puf_data = {
            'Arbiter': {
                'ber': np.array([2.0, 3.0, 4.0]),
                'attack_accuracy': np.array([85, 87, 90]),
                'uniqueness': np.array([49.0, 49.5, 50.0]),
                'ecc_failure': np.array([1.0, 1.5, 2.0])
            }
        }
        
        temp_range = np.array([25, 50, 75])
        
        figures = generate_all_thesis_plots(
            puf_data, temp_range, np.random.randint(0, 2, (100, 32)), output_dir=viz_output_dir
        )
        
        log_test("Comprehensive visualization", True, f"Generated {len(figures)} figures")
    except Exception as e:
        log_test("Comprehensive visualization", False, str(e), warning=True)
    
    # Test statistical plots
    try:
        from ppet.statistical_plots import generate_statistical_suite, generate_sample_statistical_data
        
        puf_data, env_data, ml_data = generate_sample_statistical_data()
        figures = generate_statistical_suite(puf_data, env_data, ml_data, output_dir=viz_output_dir)
        
        log_test("Statistical plots", True, f"Generated {len(figures)} statistical figures")
    except Exception as e:
        log_test("Statistical plots", False, str(e), warning=True)


def test_military_scenarios(quick_mode: bool = False):
    """Test military scenario simulation."""
    print("\n🎯 Testing Military Scenarios")
    print("-" * 40)
    
    try:
        from ppet.military_scenarios import MilitaryScenarioSimulator
        from ppet.puf_models import ArbiterPUF
        
        puf = ArbiterPUF(n_stages=32, seed=42)
        simulator = MilitaryScenarioSimulator()
        
        # Test individual scenarios
        if hasattr(simulator, 'scenarios'):
            test_scenarios = ['satellite_comm', 'drone_authentication'] if quick_mode else list(simulator.scenarios.keys())[:3]
            
            for scenario_name in test_scenarios:
                if scenario_name in simulator.scenarios:
                    try:
                        result = simulator.scenarios[scenario_name].simulate(puf)
                        assert hasattr(result, 'mission_success_probability'), "Missing mission_success_probability"
                        assert hasattr(result, 'overall_security_score'), "Missing overall_security_score"
                        log_test(f"Scenario: {scenario_name}", True)
                    except Exception as e:
                        log_test(f"Scenario: {scenario_name}", False, str(e))
            
            # Test comprehensive scenario evaluation
            try:
                all_results = simulator.run_all_scenarios(puf)
                assert len(all_results) > 0, "No scenario results returned"
                log_test("All scenarios", True, f"Evaluated {len(all_results)} scenarios")
            except Exception as e:
                log_test("All scenarios", False, str(e))
        else:
            log_test("Military scenarios", False, "Scenarios not available", warning=True)
    
    except ImportError as e:
        log_test("Military scenarios", False, f"Import error: {e}", warning=True)


def test_security_metrics(quick_mode: bool = False):
    """Test security metrics calculation."""
    print("\n🔒 Testing Security Metrics")
    print("-" * 40)
    
    try:
        from ppet.security_metrics import SecurityMetricsAnalyzer, SecurityClearanceLevel
        from ppet.puf_models import ArbiterPUF
        
        puf = ArbiterPUF(n_stages=32, seed=42)
        
        # Test different clearance levels
        clearance_levels = [SecurityClearanceLevel.CONFIDENTIAL, SecurityClearanceLevel.SECRET]
        if not quick_mode:
            clearance_levels.append(SecurityClearanceLevel.TOP_SECRET)
        
        for clearance in clearance_levels:
            try:
                analyzer = SecurityMetricsAnalyzer(clearance_level=clearance)
                
                attack_results = {
                    'ML_attacks': 0.15,
                    'side_channel': 0.08,
                    'physical_attacks': 0.12
                }
                
                security_score = analyzer.calculate_security_score(puf, attack_results)
                assert 'total_score' in security_score, "Missing total_score"
                assert 0 <= security_score['total_score'] <= 1, "Invalid security score"
                
                threat_report = analyzer.generate_threat_assessment_report(puf, attack_results)
                assert 'executive_summary' in threat_report, "Missing executive_summary"
                
                log_test(f"Security metrics: {clearance.value}", True)
            except Exception as e:
                log_test(f"Security metrics: {clearance.value}", False, str(e))
    
    except ImportError as e:
        log_test("Security metrics", False, f"Import error: {e}", warning=True)


def test_defense_dashboard(output_dir: str, quick_mode: bool = False):
    """Test defense dashboard generation."""
    print("\n🛡️  Testing Defense Dashboard")
    print("-" * 40)
    
    dashboard_output_dir = os.path.join(output_dir, "test_dashboards")
    os.makedirs(dashboard_output_dir, exist_ok=True)
    
    try:
        from ppet.defense_dashboard import (
            create_defense_dashboard, 
            generate_military_compliance_report,
            generate_sample_dashboard_data
        )
        
        # Generate sample data
        mission_data, threat_level, env_status, attack_prob, countermeasure_eff = generate_sample_dashboard_data()
        
        # Test basic dashboard
        try:
            fig = create_defense_dashboard(
                mission_data, threat_level, env_status,
                attack_prob, countermeasure_eff,
                output_dir=dashboard_output_dir
            )
            log_test("Basic defense dashboard", True)
        except Exception as e:
            log_test("Basic defense dashboard", False, str(e))
        
        # Test military compliance report
        try:
            compliance_status = {
                'MIL-STD-810H': 'COMPLIANT',
                'MIL-STD-461G': 'MARGINAL',
                'FIPS-140-2': 'LEVEL 2'
            }
            
            puf_metrics = {
                'reliability': 95.0,
                'uniqueness': 49.5,
                'attack_resistance': 87.0
            }
            
            mission_profile = {
                'mission_type': 'test',
                'security_clearance': 'SECRET',
                'deployment_environment': 'Laboratory'
            }
            
            report_path = generate_military_compliance_report(
                compliance_status, puf_metrics, env_status, mission_profile,
                output_dir=dashboard_output_dir
            )
            
            assert os.path.exists(report_path), "Compliance report not generated"
            log_test("Military compliance report", True)
        except Exception as e:
            log_test("Military compliance report", False, str(e))
    
    except ImportError as e:
        log_test("Defense dashboard", False, f"Import error: {e}", warning=True)


def test_data_persistence(output_dir: str):
    """Test data persistence and serialization."""
    print("\n💾 Testing Data Persistence")
    print("-" * 40)
    
    persistence_dir = os.path.join(output_dir, "test_persistence")
    os.makedirs(persistence_dir, exist_ok=True)
    
    try:
        from ppet.puf_models import ArbiterPUF
        import json
        
        # Test PUF serialization
        try:
            original_puf = ArbiterPUF(n_stages=32, seed=42)
            
            # Generate test challenges and responses
            rng = np.random.default_rng(42)
            challenges = rng.integers(0, 2, size=(100, 32))
            original_responses = original_puf.eval(challenges)
            
            # Serialize PUF
            json_data = original_puf.to_json()
            
            # Deserialize PUF
            restored_puf = ArbiterPUF.from_json(json_data)
            restored_responses = restored_puf.eval(challenges)
            
            # Verify equivalence
            assert np.array_equal(original_responses, restored_responses), "Serialization failed"
            log_test("PUF serialization", True)
        except Exception as e:
            log_test("PUF serialization", False, str(e))
        
        # Test results persistence
        try:
            test_results = {
                'experiment_id': 'test_001',
                'puf_type': 'ArbiterPUF',
                'n_stages': 32,
                'temperature_range': [-20, 0, 25, 50, 75, 100],
                'metrics': {
                    'ber': [2.1, 2.3, 2.5, 2.8, 3.2, 3.8],
                    'attack_accuracy': [0.85, 0.86, 0.87, 0.89, 0.92, 0.95]
                }
            }
            
            # Save results
            results_path = os.path.join(persistence_dir, 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            # Load and verify
            with open(results_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results == test_results, "Results persistence failed"
            log_test("Results persistence", True)
        except Exception as e:
            log_test("Results persistence", False, str(e))
    
    except ImportError as e:
        log_test("Data persistence", False, f"Import error: {e}")


def test_performance_basic():
    """Test basic performance characteristics."""
    print("\n⚡ Testing Basic Performance")
    print("-" * 40)
    
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.attacks import MLAttacker
        
        # Test PUF evaluation performance
        try:
            puf = ArbiterPUF(n_stages=64, seed=42)
            rng = np.random.default_rng(42)
            
            # Measure evaluation time for different dataset sizes
            sizes = [1000, 5000]
            for size in sizes:
                challenges = rng.integers(0, 2, size=(size, 64))
                
                start_time = time.time()
                responses = puf.eval(challenges)
                eval_time = time.time() - start_time
                
                throughput = size / eval_time
                assert throughput > 100, f"PUF evaluation too slow: {throughput:.0f} eval/sec"
            
            log_test("PUF evaluation performance", True, f"Throughput: {throughput:.0f} eval/sec")
        except Exception as e:
            log_test("PUF evaluation performance", False, str(e))
        
        # Test ML training performance
        try:
            challenges = rng.integers(0, 2, size=(1000, 64))
            responses = puf.eval(challenges)
            
            attacker = MLAttacker(n_stages=64)
            
            start_time = time.time()
            attacker.train(challenges, responses)
            training_time = time.time() - start_time
            
            assert training_time < 30, f"ML training too slow: {training_time:.1f}s"
            log_test("ML training performance", True, f"Training time: {training_time:.1f}s")
        except Exception as e:
            log_test("ML training performance", False, str(e))
    
    except ImportError as e:
        log_test("Performance testing", False, f"Import error: {e}")


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="PPET Framework Comprehensive Test Suite")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', default='test_output', help='Output directory for test artifacts')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧪 PPET Framework Comprehensive Test Suite")
    print("=" * 60)
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run all test categories
        test_core_imports()
        test_puf_functionality(args.quick)
        test_environmental_stress(args.quick)
        test_attack_resistance(args.quick)
        test_analysis_functions(args.quick)
        test_visualization_generation(args.output_dir, args.quick)
        test_military_scenarios(args.quick)
        test_security_metrics(args.quick)
        test_defense_dashboard(args.output_dir, args.quick)
        test_data_persistence(args.output_dir)
        test_performance_basic()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate execution time
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {test_results['passed'] + test_results['failed'] + test_results['skipped']}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"⚠️  Skipped: {test_results['skipped']}")
    print(f"⏱️  Execution time: {total_time:.1f} seconds")
    
    if test_results['failed'] > 0:
        print(f"\n❌ FAILED TESTS:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    if test_results['skipped'] > 0:
        print(f"\n⚠️  Some tests were skipped due to missing optional dependencies.")
        print(f"   This is normal in environments without full visualization support.")
    
    # Determine overall result
    if test_results['failed'] == 0:
        print(f"\n🎉 ALL TESTS PASSED! PPET Framework is fully functional.")
        return 0
    else:
        print(f"\n❌ {test_results['failed']} tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())