#!/usr/bin/env python3
"""
PPET Framework Validation Script
===============================

This script provides comprehensive validation of the complete PPET framework,
ensuring all components are properly integrated and functioning correctly for
defense-oriented PUF analysis and evaluation.

The validation process includes:
1. Framework integrity checks
2. Core functionality validation
3. Military scenario compliance
4. Performance benchmarking
5. Security assessment validation
6. Documentation and API compliance

Usage:
    python validate_complete_framework.py
    python validate_complete_framework.py --comprehensive    # Full validation
    python validate_complete_framework.py --benchmark        # Include performance benchmarks
    python validate_complete_framework.py --output-dir /path/to/output

Exit codes:
    0: All validations passed
    1: Critical failures detected
    2: Non-critical issues found
"""

import sys
import os
import time
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import tempfile

# Add ppet-thesis to path for standalone execution
current_dir = Path(__file__).parent
ppet_dir = current_dir / "ppet-thesis"
if ppet_dir.exists():
    sys.path.insert(0, str(ppet_dir))
else:
    sys.path.insert(0, str(current_dir))

import numpy as np

# Validation results tracking
validation_results = {
    "framework_integrity": {"passed": 0, "failed": 0, "issues": []},
    "core_functionality": {"passed": 0, "failed": 0, "issues": []},
    "military_compliance": {"passed": 0, "failed": 0, "issues": []},
    "performance": {"passed": 0, "failed": 0, "issues": []},
    "security": {"passed": 0, "failed": 0, "issues": []},
    "documentation": {"passed": 0, "failed": 0, "issues": []},
    "overall_score": 0.0,
    "deployment_ready": False,
    "critical_issues": [],
    "recommendations": []
}


def log_validation(category: str, test_name: str, success: bool, message: str = "", critical: bool = False):
    """Log validation result and update tracking."""
    global validation_results
    
    if success:
        validation_results[category]["passed"] += 1
        status = "✅ PASS"
    else:
        validation_results[category]["failed"] += 1
        validation_results[category]["issues"].append(f"{test_name}: {message}")
        if critical:
            validation_results["critical_issues"].append(f"{category}.{test_name}: {message}")
        status = "❌ FAIL" if not critical else "🚨 CRITICAL"
    
    print(f"{status} {test_name}")
    if message and not success:
        print(f"      {message}")


def validate_framework_integrity():
    """Validate framework structure and imports."""
    print("\n📦 Validating Framework Integrity")
    print("-" * 50)
    
    # Check directory structure
    required_dirs = [
        "ppet-thesis/ppet",
        "ppet-thesis/tests", 
        "ppet-thesis/scripts"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            log_validation("framework_integrity", f"Directory: {dir_path}", True)
        else:
            log_validation("framework_integrity", f"Directory: {dir_path}", False, "Missing required directory", critical=True)
    
    # Check core module files
    core_modules = [
        "ppet-thesis/ppet/__init__.py",
        "ppet-thesis/ppet/puf_models.py",
        "ppet-thesis/ppet/stressors.py",
        "ppet-thesis/ppet/attacks.py",
        "ppet-thesis/ppet/analysis.py"
    ]
    
    for module_path in core_modules:
        if os.path.exists(module_path):
            log_validation("framework_integrity", f"Module: {os.path.basename(module_path)}", True)
        else:
            log_validation("framework_integrity", f"Module: {os.path.basename(module_path)}", False, "Missing core module", critical=True)
    
    # Check advanced modules (non-critical)
    advanced_modules = [
        "ppet-thesis/ppet/visualization.py",
        "ppet-thesis/ppet/statistical_plots.py",
        "ppet-thesis/ppet/bit_analysis.py",
        "ppet-thesis/ppet/defense_dashboard.py",
        "ppet-thesis/ppet/military_scenarios.py",
        "ppet-thesis/ppet/security_metrics.py",
        "ppet-thesis/ppet/side_channel.py",
        "ppet-thesis/ppet/physical_attacks.py"
    ]
    
    for module_path in advanced_modules:
        if os.path.exists(module_path):
            log_validation("framework_integrity", f"Advanced: {os.path.basename(module_path)}", True)
        else:
            log_validation("framework_integrity", f"Advanced: {os.path.basename(module_path)}", False, "Missing advanced module")
    
    # Test imports
    critical_imports = [
        ("ppet.puf_models", ["ArbiterPUF", "SRAMPUF", "RingOscillatorPUF", "ButterflyPUF"]),
        ("ppet.stressors", ["apply_temperature", "apply_voltage"]),
        ("ppet.attacks", ["MLAttacker"]),
        ("ppet.analysis", ["bit_error_rate", "simulate_ecc"])
    ]
    
    for module_name, components in critical_imports:
        try:
            module = __import__(module_name, fromlist=components)
            for component in components:
                if hasattr(module, component):
                    log_validation("framework_integrity", f"Import: {module_name}.{component}", True)
                else:
                    log_validation("framework_integrity", f"Import: {module_name}.{component}", False, "Component not found", critical=True)
        except ImportError as e:
            log_validation("framework_integrity", f"Import: {module_name}", False, str(e), critical=True)


def validate_core_functionality():
    """Validate core PUF functionality."""
    print("\n🔧 Validating Core Functionality")
    print("-" * 50)
    
    try:
        from ppet.puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
        from ppet.stressors import apply_temperature
        from ppet.attacks import MLAttacker
        from ppet.analysis import bit_error_rate, simulate_ecc
        
        # Test each PUF type
        rng = np.random.default_rng(42)
        test_challenges = rng.integers(0, 2, size=(100, 32))
        
        puf_classes = [
            ("ArbiterPUF", ArbiterPUF, 32),
            ("SRAMPUF", SRAMPUF, 32),
            ("RingOscillatorPUF", RingOscillatorPUF, 16),
            ("ButterflyPUF", ButterflyPUF, 16)
        ]
        
        for puf_name, PUFClass, n_stages in puf_classes:
            try:
                # Create PUF
                if puf_name == "SRAMPUF":
                    puf = PUFClass(n_cells=n_stages, seed=42)
                elif puf_name == "RingOscillatorPUF":
                    puf = PUFClass(n_rings=n_stages, seed=42)
                elif puf_name == "ButterflyPUF":
                    puf = PUFClass(n_butterflies=n_stages, seed=42)
                else:
                    puf = PUFClass(n_stages=n_stages, seed=42)
                
                # Test evaluation
                challenges = test_challenges[:, :n_stages]
                responses = puf.eval(challenges)
                
                # Validate responses
                if len(responses) == 100 and all(r in [-1, 1] for r in responses):
                    log_validation("core_functionality", f"{puf_name} evaluation", True)
                else:
                    log_validation("core_functionality", f"{puf_name} evaluation", False, "Invalid responses", critical=True)
                
                # Test deterministic behavior
                responses2 = puf.eval(challenges)
                if np.array_equal(responses, responses2):
                    log_validation("core_functionality", f"{puf_name} determinism", True)
                else:
                    log_validation("core_functionality", f"{puf_name} determinism", False, "Non-deterministic", critical=True)
                
            except Exception as e:
                log_validation("core_functionality", f"{puf_name} functionality", False, str(e), critical=True)
        
        # Test environmental stress
        try:
            puf = ArbiterPUF(n_stages=32, seed=42)
            original_responses = puf.eval(test_challenges)
            
            stressed_puf = apply_temperature(puf, T_current=75.0)
            stressed_responses = stressed_puf.eval(test_challenges)
            
            ber = bit_error_rate(original_responses, stressed_responses)
            if 0 <= ber <= 100:
                log_validation("core_functionality", "Environmental stress", True)
            else:
                log_validation("core_functionality", "Environmental stress", False, f"Invalid BER: {ber}")
                
        except Exception as e:
            log_validation("core_functionality", "Environmental stress", False, str(e))
        
        # Test ML attacks
        try:
            attacker = MLAttacker(n_stages=32)
            attacker.train(test_challenges, original_responses)
            accuracy = attacker.accuracy(test_challenges, original_responses)
            
            if 0 <= accuracy <= 1:
                log_validation("core_functionality", "ML attacks", True)
            else:
                log_validation("core_functionality", "ML attacks", False, f"Invalid accuracy: {accuracy}")
                
        except Exception as e:
            log_validation("core_functionality", "ML attacks", False, str(e))
        
        # Test ECC simulation
        try:
            # Simple ECC test
            received = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
            reference = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])
            
            fail_rate = simulate_ecc(received, reference, t=1)
            if 0 <= fail_rate <= 1:
                log_validation("core_functionality", "ECC simulation", True)
            else:
                log_validation("core_functionality", "ECC simulation", False, f"Invalid fail rate: {fail_rate}")
                
        except Exception as e:
            log_validation("core_functionality", "ECC simulation", False, str(e))
            
    except ImportError as e:
        log_validation("core_functionality", "Core imports", False, str(e), critical=True)


def validate_military_compliance():
    """Validate military compliance features."""
    print("\n🎯 Validating Military Compliance")
    print("-" * 50)
    
    # Test military temperature ranges
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.stressors import apply_temperature
        
        puf = ArbiterPUF(n_stages=32, seed=42)
        rng = np.random.default_rng(42)
        challenges = rng.integers(0, 2, size=(50, 32))
        
        # Test military temperature spec (-55°C to 125°C)
        military_temps = [-55, -20, 0, 25, 85, 125]
        
        for temp in military_temps:
            try:
                stressed_puf = apply_temperature(puf, T_current=temp, military_spec=True)
                responses = stressed_puf.eval(challenges)
                
                if len(responses) == 50 and all(r in [-1, 1] for r in responses):
                    log_validation("military_compliance", f"Military temp {temp}°C", True)
                else:
                    log_validation("military_compliance", f"Military temp {temp}°C", False, "Invalid responses")
                    
            except Exception as e:
                log_validation("military_compliance", f"Military temp {temp}°C", False, str(e))
    
    except ImportError as e:
        log_validation("military_compliance", "Military temperature", False, str(e))
    
    # Test military scenarios (if available)
    try:
        from ppet.military_scenarios import MilitaryScenarioSimulator
        
        simulator = MilitaryScenarioSimulator()
        puf = ArbiterPUF(n_stages=32, seed=42)
        
        if hasattr(simulator, 'scenarios'):
            key_scenarios = ['satellite_comm', 'drone_authentication', 'battlefield_iot']
            
            for scenario_name in key_scenarios:
                if scenario_name in simulator.scenarios:
                    try:
                        result = simulator.scenarios[scenario_name].simulate(puf)
                        if hasattr(result, 'mission_success_probability'):
                            log_validation("military_compliance", f"Scenario: {scenario_name}", True)
                        else:
                            log_validation("military_compliance", f"Scenario: {scenario_name}", False, "Invalid result format")
                    except Exception as e:
                        log_validation("military_compliance", f"Scenario: {scenario_name}", False, str(e))
        else:
            log_validation("military_compliance", "Military scenarios", False, "Scenarios not available")
            
    except ImportError as e:
        log_validation("military_compliance", "Military scenarios", False, "Module not available")
    
    # Test security clearance levels
    try:
        from ppet.security_metrics import SecurityMetricsAnalyzer, SecurityClearanceLevel
        
        clearance_levels = [SecurityClearanceLevel.CONFIDENTIAL, SecurityClearanceLevel.SECRET, SecurityClearanceLevel.TOP_SECRET]
        
        for clearance in clearance_levels:
            try:
                analyzer = SecurityMetricsAnalyzer(clearance_level=clearance)
                log_validation("military_compliance", f"Clearance: {clearance.value}", True)
            except Exception as e:
                log_validation("military_compliance", f"Clearance: {clearance.value}", False, str(e))
                
    except ImportError as e:
        log_validation("military_compliance", "Security clearances", False, "Module not available")


def validate_performance_benchmarks():
    """Validate performance characteristics."""
    print("\n⚡ Validating Performance Benchmarks")
    print("-" * 50)
    
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.attacks import MLAttacker
        
        # PUF evaluation performance
        try:
            puf = ArbiterPUF(n_stages=64, seed=42)
            rng = np.random.default_rng(42)
            
            # Test with different dataset sizes
            sizes = [1000, 5000]
            
            for size in sizes:
                challenges = rng.integers(0, 2, size=(size, 64))
                
                start_time = time.time()
                responses = puf.eval(challenges)
                eval_time = time.time() - start_time
                
                throughput = size / eval_time
                
                # Require minimum 100 evaluations per second
                if throughput >= 100:
                    log_validation("performance", f"PUF eval {size} challenges", True, f"{throughput:.0f} eval/sec")
                else:
                    log_validation("performance", f"PUF eval {size} challenges", False, f"Too slow: {throughput:.0f} eval/sec")
        
        except Exception as e:
            log_validation("performance", "PUF evaluation performance", False, str(e))
        
        # ML training performance
        try:
            challenges = rng.integers(0, 2, size=(1000, 64))
            responses = puf.eval(challenges)
            
            attacker = MLAttacker(n_stages=64)
            
            start_time = time.time()
            attacker.train(challenges, responses)
            training_time = time.time() - start_time
            
            # Require training to complete within 30 seconds
            if training_time <= 30:
                log_validation("performance", "ML training performance", True, f"{training_time:.1f}s")
            else:
                log_validation("performance", "ML training performance", False, f"Too slow: {training_time:.1f}s")
                
        except Exception as e:
            log_validation("performance", "ML training performance", False, str(e))
        
        # Memory usage validation (basic check)
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create larger dataset
            large_challenges = rng.integers(0, 2, size=(10000, 64))
            large_responses = puf.eval(large_challenges)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Require memory increase to be reasonable (< 200MB)
            if memory_increase < 200:
                log_validation("performance", "Memory usage", True, f"{memory_increase:.1f}MB increase")
            else:
                log_validation("performance", "Memory usage", False, f"High memory usage: {memory_increase:.1f}MB")
                
        except ImportError:
            log_validation("performance", "Memory usage", False, "psutil not available")
        except Exception as e:
            log_validation("performance", "Memory usage", False, str(e))
            
    except ImportError as e:
        log_validation("performance", "Performance benchmarks", False, str(e))


def validate_security_features():
    """Validate security-specific features."""
    print("\n🔒 Validating Security Features")
    print("-" * 50)
    
    # Test attack resistance validation
    try:
        from ppet.puf_models import ArbiterPUF
        from ppet.attacks import MLAttacker
        
        puf = ArbiterPUF(n_stages=64, seed=42)
        rng = np.random.default_rng(42)
        challenges = rng.integers(0, 2, size=(1000, 64))
        responses = puf.eval(challenges)
        
        # Test defense evaluation
        try:
            attacker = MLAttacker(n_stages=64)
            defense_results = attacker.defense_evaluation(puf)
            
            required_keys = ['attack_accuracy', 'defense_effectiveness', 'defense_rating']
            if all(key in defense_results for key in required_keys):
                log_validation("security", "Defense evaluation", True)
            else:
                log_validation("security", "Defense evaluation", False, "Missing required keys")
                
        except Exception as e:
            log_validation("security", "Defense evaluation", False, str(e))
        
        # Test attack complexity analysis
        try:
            complexity_results = attacker.attack_complexity_analysis(puf, [100, 200, 400])
            
            if 'sample_sizes' in complexity_results and 'accuracies' in complexity_results:
                log_validation("security", "Attack complexity analysis", True)
            else:
                log_validation("security", "Attack complexity analysis", False, "Invalid results format")
                
        except Exception as e:
            log_validation("security", "Attack complexity analysis", False, str(e))
    
    except ImportError as e:
        log_validation("security", "Security features", False, str(e))
    
    # Test side-channel resistance (if available)
    try:
        from ppet.side_channel import MultiChannelAttacker
        
        sc_attacker = MultiChannelAttacker()
        log_validation("security", "Side-channel module", True)
        
    except ImportError:
        log_validation("security", "Side-channel module", False, "Module not available")
    except Exception as e:
        log_validation("security", "Side-channel module", False, str(e))
    
    # Test physical attack resistance (if available)
    try:
        from ppet.physical_attacks import ComprehensivePhysicalAttacker, AttackComplexity
        
        physical_attacker = ComprehensivePhysicalAttacker(AttackComplexity.MEDIUM)
        log_validation("security", "Physical attacks module", True)
        
    except ImportError:
        log_validation("security", "Physical attacks module", False, "Module not available")
    except Exception as e:
        log_validation("security", "Physical attacks module", False, str(e))


def validate_documentation_api():
    """Validate documentation and API compliance."""
    print("\n📚 Validating Documentation & API")
    print("-" * 50)
    
    # Check for key documentation files
    doc_files = [
        "CLAUDE.md",
        "ppet-thesis/requirements.txt",
        "ppet-thesis/setup.py"
    ]
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            log_validation("documentation", f"File: {doc_file}", True)
        else:
            log_validation("documentation", f"File: {doc_file}", False, "Missing documentation file")
    
    # Test API consistency
    try:
        from ppet.puf_models import ArbiterPUF
        
        # Check that all PUFs have consistent API
        puf = ArbiterPUF(n_stages=32, seed=42)
        
        # Test required methods
        required_methods = ['eval', 'to_json']
        for method in required_methods:
            if hasattr(puf, method):
                log_validation("documentation", f"API method: {method}", True)
            else:
                log_validation("documentation", f"API method: {method}", False, "Missing required method")
        
        # Test serialization API
        try:
            json_data = puf.to_json()
            restored = ArbiterPUF.from_json(json_data)
            log_validation("documentation", "Serialization API", True)
        except Exception as e:
            log_validation("documentation", "Serialization API", False, str(e))
            
    except ImportError as e:
        log_validation("documentation", "API consistency", False, str(e))
    
    # Check test coverage
    test_files = [
        "ppet-thesis/tests/test_puf_models.py",
        "ppet-thesis/tests/test_stressors.py",
        "ppet-thesis/tests/test_attacks.py",
        "ppet-thesis/tests/test_analysis.py",
        "ppet-thesis/tests/test_integration.py",
        "ppet-thesis/tests/test_performance.py"
    ]
    
    existing_tests = sum(1 for test_file in test_files if os.path.exists(test_file))
    coverage_percentage = (existing_tests / len(test_files)) * 100
    
    if coverage_percentage >= 80:
        log_validation("documentation", "Test coverage", True, f"{coverage_percentage:.0f}% coverage")
    else:
        log_validation("documentation", "Test coverage", False, f"Low coverage: {coverage_percentage:.0f}%")


def generate_validation_report(output_dir: str) -> str:
    """Generate comprehensive validation report."""
    
    # Calculate overall scores
    total_passed = sum(cat["passed"] for cat in validation_results.values() if isinstance(cat, dict) and "passed" in cat)
    total_failed = sum(cat["failed"] for cat in validation_results.values() if isinstance(cat, dict) and "failed" in cat)
    total_tests = total_passed + total_failed
    
    if total_tests > 0:
        validation_results["overall_score"] = (total_passed / total_tests) * 100
    else:
        validation_results["overall_score"] = 0.0
    
    # Determine deployment readiness
    critical_failures = len(validation_results["critical_issues"])
    overall_score = validation_results["overall_score"]
    
    validation_results["deployment_ready"] = (critical_failures == 0 and overall_score >= 80)
    
    # Generate recommendations
    recommendations = []
    
    if critical_failures > 0:
        recommendations.append("CRITICAL: Address critical failures before deployment")
    
    if overall_score < 80:
        recommendations.append("Improve test coverage and fix failing validations")
    
    if validation_results["performance"]["failed"] > 0:
        recommendations.append("Optimize performance bottlenecks")
    
    if validation_results["military_compliance"]["failed"] > 0:
        recommendations.append("Enhance military compliance features")
    
    if not recommendations:
        recommendations.append("Framework is ready for deployment")
    
    validation_results["recommendations"] = recommendations
    
    # Save detailed report
    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return report_path


def run_external_tests(comprehensive: bool = False) -> bool:
    """Run external test suites."""
    print("\n🧪 Running External Test Suites")
    print("-" * 50)
    
    test_commands = [
        ("test_all_functionality.py", ["python", "test_all_functionality.py", "--quick" if not comprehensive else ""]),
    ]
    
    if comprehensive:
        # Add pytest if available
        if os.path.exists("ppet-thesis/tests"):
            test_commands.append(("pytest", ["python", "-m", "pytest", "ppet-thesis/tests/", "-v"]))
    
    all_passed = True
    
    for test_name, command in test_commands:
        try:
            print(f"Running {test_name}...")
            # Filter out empty strings from command
            clean_command = [cmd for cmd in command if cmd.strip()]
            result = subprocess.run(clean_command, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                log_validation("framework_integrity", f"External test: {test_name}", True)
            else:
                log_validation("framework_integrity", f"External test: {test_name}", False, f"Exit code: {result.returncode}")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            log_validation("framework_integrity", f"External test: {test_name}", False, "Test timeout")
            all_passed = False
        except FileNotFoundError:
            log_validation("framework_integrity", f"External test: {test_name}", False, "Test file not found")
            all_passed = False
        except Exception as e:
            log_validation("framework_integrity", f"External test: {test_name}", False, str(e))
            all_passed = False
    
    return all_passed


def main():
    """Main validation execution function."""
    parser = argparse.ArgumentParser(description="PPET Framework Comprehensive Validation")
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive validation including all tests')
    parser.add_argument('--benchmark', action='store_true', help='Include performance benchmarks')
    parser.add_argument('--output-dir', default='validation_output', help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 PPET Framework Comprehensive Validation")
    print("=" * 70)
    print(f"Comprehensive mode: {'ON' if args.comprehensive else 'OFF'}")
    print(f"Performance benchmarks: {'ON' if args.benchmark else 'OFF'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run all validation categories
        validate_framework_integrity()
        validate_core_functionality()
        validate_military_compliance()
        
        if args.benchmark:
            validate_performance_benchmarks()
        
        validate_security_features()
        validate_documentation_api()
        
        if args.comprehensive:
            run_external_tests(comprehensive=True)
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        return 2
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate execution time
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    report_path = generate_validation_report(args.output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🏁 VALIDATION SUMMARY")
    print("=" * 70)
    
    for category, results in validation_results.items():
        if isinstance(results, dict) and "passed" in results:
            total = results["passed"] + results["failed"]
            if total > 0:
                percentage = (results["passed"] / total) * 100
                print(f"{category.replace('_', ' ').title():20}: {results['passed']:3}/{total:3} ({percentage:5.1f}%)")
    
    print(f"\nOverall Score: {validation_results['overall_score']:.1f}%")
    print(f"Execution Time: {total_time:.1f} seconds")
    print(f"Deployment Ready: {'YES' if validation_results['deployment_ready'] else 'NO'}")
    
    if validation_results["critical_issues"]:
        print(f"\n🚨 CRITICAL ISSUES ({len(validation_results['critical_issues'])}):")
        for issue in validation_results["critical_issues"]:
            print(f"   - {issue}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in validation_results["recommendations"]:
        print(f"   - {rec}")
    
    print(f"\n📄 Detailed report: {report_path}")
    
    # Determine exit code
    if validation_results["critical_issues"]:
        print(f"\n❌ VALIDATION FAILED: Critical issues must be resolved")
        return 1
    elif validation_results["overall_score"] < 80:
        print(f"\n⚠️  VALIDATION INCOMPLETE: Score below 80%")
        return 2
    else:
        print(f"\n✅ VALIDATION PASSED: Framework ready for deployment")
        return 0


if __name__ == "__main__":
    exit(main())