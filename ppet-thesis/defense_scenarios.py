"""
Defense-Oriented Attack Scenarios for PPET
==========================================

This module implements comprehensive attack scenarios tailored for military
and national security applications. It combines ML attacks, side-channel analysis,
and physical attacks to model realistic threat scenarios.

Defense Applications:
- Satellite communication PUF security evaluation
- Drone authentication system vulnerability assessment  
- Battlefield IoT device security testing
- Supply chain hardware integrity verification
- Critical infrastructure protection analysis

Threat Models:
- Nation-state adversaries with advanced capabilities
- Military adversaries with battlefield access
- Supply chain attackers with insider access
- Sophisticated criminal organizations
- Academic researchers with advanced tools
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from puf_models import BasePUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
from attacks import MLAttacker, CNNAttacker, AdversarialAttacker


class ThreatActor(Enum):
    """Classification of threat actors for defense scenarios."""
    SCRIPT_KIDDIE = "script_kiddie"
    CRIMINAL_ORG = "criminal_organization"
    ACADEMIC = "academic_researcher"
    CORPORATE_SPY = "corporate_espionage"
    TERRORIST = "terrorist_organization"
    MILITARY = "military_adversary"
    NATION_STATE = "nation_state"


class OperationalEnvironment(Enum):
    """Operational environment classifications."""
    LABORATORY = "controlled_laboratory"
    OFFICE = "office_environment"
    FIELD = "field_deployment"
    BATTLEFIELD = "hostile_battlefield"
    SPACE = "space_environment"
    UNDERWATER = "underwater_operation"


@dataclass
class DefenseScenario:
    """Container for defense scenario configuration."""
    scenario_name: str
    threat_actor: ThreatActor
    environment: OperationalEnvironment
    time_constraint_hours: float
    budget_constraint_usd: float
    access_level: str  # 'remote', 'physical', 'insider'
    detection_tolerance: float  # 0.0 to 1.0
    success_threshold: float  # Required attack success rate
    allowed_damage: str  # 'none', 'minimal', 'moderate', 'severe'


@dataclass
class ScenarioResult:
    """Results from defense scenario evaluation."""
    scenario: DefenseScenario
    attack_success: bool
    overall_success_rate: float
    time_required_hours: float
    cost_incurred_usd: float
    damage_caused: str
    detection_risk: float
    extracted_information: Dict[str, Any]
    countermeasures_triggered: List[str]
    threat_assessment: str


# Mock classes for testing - in real implementation these would import from other modules
class MockAttackResult:
    def __init__(self, success_prob=0.3, cost=10000, secrets=None):
        self.success_probability = success_prob
        self.cost_estimate_usd = cost
        self.detection_probability = 0.1
        self.extracted_secrets = secrets or {}
        self.damage_level = 'minimal'


class MockAttacker:
    def __init__(self, attack_type='generic'):
        self.attack_type = attack_type
    
    def collect_traces(self, puf, n_traces=100):
        return np.random.random((n_traces, 1000))
    
    def analyze_em_leakage(self, traces):
        return {
            'attack_success': False,
            'leakage_detected': False,
            'snr_db': 3.0
        }
    
    def perform_dpa_attack(self, traces):
        return {'leakage_detected': False, 'attack_success': False}
    
    def perform_cpa_attack(self, traces):
        return {'attack_success': False, 'correlation_peak': 0.1}
    
    def hardware_trojan_insertion(self, puf, trojan_type):
        return MockAttackResult(success_prob=0.2, cost=100000)
    
    def inject_voltage_fault(self, puf, challenge, voltage, duration):
        return MockAttackResult(success_prob=0.3, cost=5000)
    
    def comprehensive_physical_attack(self, puf):
        return {
            'imaging': MockAttackResult(success_prob=0.4, cost=50000),
            'probing': MockAttackResult(success_prob=0.2, cost=25000)
        }
    
    def comprehensive_attack(self, puf, n_traces=1000):
        return {
            'combined_attack': {
                'success_rate': 0.3,
                'attack_success': False
            }
        }
    
    def circuit_imaging_attack(self, puf):
        return MockAttackResult(success_prob=0.5, cost=500000)


# Mock attack classes for scenarios
EMAnalysisAttacker = lambda **kwargs: MockAttacker('em')
PowerAnalysisAttacker = lambda attack_type, **kwargs: MockAttacker('power')
FaultInjectionAttacker = lambda fault_type, **kwargs: MockAttacker('fault')
SupplyChainAttacker = lambda target: MockAttacker('supply')
ComprehensivePhysicalAttacker = lambda complexity: MockAttacker('physical')
MultiChannelAttacker = lambda: MockAttacker('multi_channel')
ReverseEngineeringAttacker = lambda complexity: MockAttacker('reverse')
AttackComplexity = type('AttackComplexity', (), {'MEDIUM': 'medium', 'EXTREME': 'extreme'})


class SatelliteCommScenario:
    """Satellite communication PUF security evaluation scenario."""
    
    def __init__(self):
        self.scenario_config = DefenseScenario(
            scenario_name="Satellite Communication Security",
            threat_actor=ThreatActor.NATION_STATE,
            environment=OperationalEnvironment.SPACE,
            time_constraint_hours=720,  # 30 days
            budget_constraint_usd=10000000,  # $10M
            access_level='remote',
            detection_tolerance=0.1,
            success_threshold=0.9,
            allowed_damage='none'
        )
    
    def evaluate_puf_security(self, puf: BasePUF) -> ScenarioResult:
        """Evaluate PUF security in satellite communication context."""
        start_time = time.time()
        total_cost = 0
        attack_results = {}
        extracted_info = {}
        
        # Phase 1: Remote signal intelligence
        em_attacker = EMAnalysisAttacker(frequency_range=(1e6, 6e9), distance_m=1000)
        em_result = em_attacker.analyze_em_leakage(np.random.random((100, 1000)))
        attack_results['remote_em'] = em_result
        total_cost += 500000
        
        # Phase 2: ML attacks
        adversarial = AdversarialAttacker(puf_type='satellite')
        adaptive_result = adversarial.adaptive_attack(puf, n_queries=50000, adaptation_rounds=10)
        attack_results['adaptive_ml'] = adaptive_result
        total_cost += 1000000
        
        # Phase 3: Supply chain analysis
        supply_attacker = SupplyChainAttacker('manufacturer')
        trojan_result = supply_attacker.hardware_trojan_insertion(puf, 'passive')
        attack_results['supply_chain'] = trojan_result
        total_cost += trojan_result.cost_estimate_usd
        
        # Calculate results
        elapsed_time = (time.time() - start_time) / 3600 + 720  # Add realistic timeline
        
        ml_success = adaptive_result['final_accuracy'] > self.scenario_config.success_threshold
        em_success = em_result['attack_success']
        supply_success = trojan_result.success_probability > 0.5
        
        overall_success = ml_success or em_success or supply_success
        success_rate = sum([ml_success, em_success, supply_success]) / 3.0
        
        detection_risk = max(0.05, 0.1 if ml_success else 0.0, 
                           trojan_result.detection_probability if supply_success else 0.0)
        
        if success_rate >= 0.8:
            threat_level = "CRITICAL - Satellite communication compromise possible"
        elif success_rate >= 0.5:
            threat_level = "HIGH - Partial satellite system vulnerability"
        elif success_rate >= 0.3:
            threat_level = "MEDIUM - Limited attack vectors successful"
        else:
            threat_level = "LOW - Satellite PUF security adequate"
        
        return ScenarioResult(
            scenario=self.scenario_config,
            attack_success=overall_success,
            overall_success_rate=success_rate,
            time_required_hours=elapsed_time,
            cost_incurred_usd=total_cost,
            damage_caused='none',
            detection_risk=detection_risk,
            extracted_information=extracted_info,
            countermeasures_triggered=[],
            threat_assessment=threat_level
        )


class DroneAuthScenario:
    """Drone authentication system vulnerability assessment scenario."""
    
    def __init__(self):
        self.scenario_config = DefenseScenario(
            scenario_name="Drone Swarm Authentication",
            threat_actor=ThreatActor.MILITARY,
            environment=OperationalEnvironment.BATTLEFIELD,
            time_constraint_hours=12,
            budget_constraint_usd=500000,
            access_level='physical',
            detection_tolerance=0.3,
            success_threshold=0.7,
            allowed_damage='moderate'
        )
    
    def evaluate_puf_security(self, puf: BasePUF) -> ScenarioResult:
        """Evaluate PUF security in drone authentication context."""
        start_time = time.time()
        total_cost = 0
        attack_results = {}
        extracted_info = {}
        damage_level = 'none'
        
        # Phase 1: Battlefield EM interception
        em_attacker = EMAnalysisAttacker(frequency_range=(100e6, 2e9), distance_m=50)
        em_result = em_attacker.analyze_em_leakage(np.random.random((200, 1000)))
        attack_results['battlefield_em'] = em_result
        total_cost += 50000
        
        # Phase 2: Physical analysis
        physical_attacker = ComprehensivePhysicalAttacker(AttackComplexity.MEDIUM)
        physical_results = physical_attacker.comprehensive_physical_attack(puf)
        attack_results['physical_analysis'] = physical_results
        
        for attack_name, result in physical_results.items():
            total_cost += min(result.cost_estimate_usd, 100000)
            if result.success_probability > 0.5 and result.damage_level in ['moderate', 'severe']:
                damage_level = 'moderate'
        
        # Phase 3: Rapid ML attack
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        challenges = rng.integers(0, 2, size=(2000, n_stages))
        responses = puf.eval(challenges)
        
        ml_attacker = MLAttacker(n_stages)
        ml_attacker.train(challenges, responses)
        
        test_challenges = rng.integers(0, 2, size=(500, n_stages))
        test_responses = puf.eval(test_challenges)
        ml_accuracy = ml_attacker.accuracy(test_challenges, test_responses)
        
        attack_results['rapid_ml'] = {'accuracy': ml_accuracy}
        total_cost += 10000
        
        # Calculate results
        elapsed_time = (time.time() - start_time) / 3600 + 8
        
        if elapsed_time > self.scenario_config.time_constraint_hours:
            success_rate = 0.2
            overall_success = False
        else:
            em_success = em_result['attack_success']
            physical_success = any(r.success_probability > 0.5 for r in physical_results.values())
            ml_success = ml_accuracy > self.scenario_config.success_threshold
            
            overall_success = em_success or physical_success or ml_success
            success_rate = sum([em_success, physical_success, ml_success]) / 3.0
        
        detection_risk = 0.4
        
        if success_rate >= 0.8:
            threat_level = "CRITICAL - Drone swarm authentication compromised"
        elif success_rate >= 0.5:
            threat_level = "HIGH - Drone authentication partially vulnerable"
        elif success_rate >= 0.3:
            threat_level = "MEDIUM - Some attack vectors successful"
        else:
            threat_level = "LOW - Drone PUF security adequate for battlefield"
        
        return ScenarioResult(
            scenario=self.scenario_config,
            attack_success=overall_success,
            overall_success_rate=success_rate,
            time_required_hours=elapsed_time,
            cost_incurred_usd=total_cost,
            damage_caused=damage_level,
            detection_risk=detection_risk,
            extracted_information=extracted_info,
            countermeasures_triggered=[],
            threat_assessment=threat_level
        )


class IoTFieldScenario:
    """Battlefield IoT device security testing scenario."""
    
    def __init__(self):
        self.scenario_config = DefenseScenario(
            scenario_name="Battlefield IoT Security",
            threat_actor=ThreatActor.TERRORIST,
            environment=OperationalEnvironment.FIELD,
            time_constraint_hours=48,
            budget_constraint_usd=100000,
            access_level='physical',
            detection_tolerance=0.5,
            success_threshold=0.6,
            allowed_damage='severe'
        )
    
    def evaluate_puf_security(self, puf: BasePUF) -> ScenarioResult:
        """Evaluate PUF security in IoT field deployment context."""
        start_time = time.time()
        total_cost = 0
        attack_results = {}
        extracted_info = {}
        
        # Phase 1: Fault injection
        fault_attacker = FaultInjectionAttacker('voltage', precision=0.6)
        fault_results = []
        
        for voltage in [2.0, 3.5, 5.0]:
            for duration in [1e-6, 1e-5, 1e-4]:
                fault_result = fault_attacker.inject_voltage_fault(puf, None, voltage, duration)
                fault_results.append(fault_result)
                total_cost += min(fault_result.cost_estimate_usd / 10, 5000)
        
        attack_results['fault_injection'] = fault_results
        successful_faults = [r for r in fault_results if r.success_probability > 0.5]
        
        # Phase 2: Side-channel analysis
        power_attacker = PowerAnalysisAttacker('dpa', noise_level=0.2)
        dpa_result = power_attacker.perform_dpa_attack(np.random.random((500, 1000)))
        cpa_result = power_attacker.perform_cpa_attack(np.random.random((500, 1000)))
        
        attack_results['power_analysis'] = {'dpa': dpa_result, 'cpa': cpa_result}
        total_cost += 25000
        
        # Phase 3: Basic ML attack
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        challenges = rng.integers(0, 2, size=(1000, n_stages))
        responses = puf.eval(challenges)
        
        ml_attacker = MLAttacker(n_stages)
        ml_attacker.train(challenges, responses)
        
        test_challenges = rng.integers(0, 2, size=(200, n_stages))
        test_responses = puf.eval(test_challenges)
        ml_accuracy = ml_attacker.accuracy(test_challenges, test_responses)
        
        attack_results['field_ml'] = {'accuracy': ml_accuracy}
        total_cost += 5000
        
        # Calculate results
        elapsed_time = (time.time() - start_time) / 3600 + 36
        
        fault_success = len(successful_faults) > 0
        power_success = dpa_result['leakage_detected'] or cpa_result['attack_success']
        ml_success = ml_accuracy > self.scenario_config.success_threshold
        
        overall_success = fault_success or power_success or ml_success
        success_rate = sum([fault_success, power_success, ml_success]) / 3.0
        
        damage_level = 'moderate' if fault_success else 'minimal'
        detection_risk = 0.6
        
        if success_rate >= 0.8:
            threat_level = "CRITICAL - IoT network security compromised"
        elif success_rate >= 0.5:
            threat_level = "HIGH - IoT devices vulnerable to field attacks"
        elif success_rate >= 0.3:
            threat_level = "MEDIUM - Partial IoT security vulnerabilities"
        else:
            threat_level = "LOW - IoT PUF security adequate for field deployment"
        
        return ScenarioResult(
            scenario=self.scenario_config,
            attack_success=overall_success,
            overall_success_rate=success_rate,
            time_required_hours=elapsed_time,
            cost_incurred_usd=total_cost,
            damage_caused=damage_level,
            detection_risk=detection_risk,
            extracted_information=extracted_info,
            countermeasures_triggered=[],
            threat_assessment=threat_level
        )


class SupplyChainScenario:
    """Supply chain hardware integrity verification scenario."""
    
    def __init__(self):
        self.scenario_config = DefenseScenario(
            scenario_name="Supply Chain Integrity",
            threat_actor=ThreatActor.NATION_STATE,
            environment=OperationalEnvironment.LABORATORY,
            time_constraint_hours=2000,
            budget_constraint_usd=50000000,
            access_level='insider',
            detection_tolerance=0.05,
            success_threshold=0.95,
            allowed_damage='none'
        )
    
    def evaluate_puf_security(self, puf: BasePUF) -> ScenarioResult:
        """Evaluate PUF security in supply chain context."""
        start_time = time.time()
        total_cost = 0
        attack_results = {}
        extracted_info = {}
        
        # Phase 1: Supply chain infiltration
        supply_attacker = SupplyChainAttacker('manufacturer')
        trojan_results = []
        for trojan_type in ['passive', 'active', 'hybrid']:
            trojan_result = supply_attacker.hardware_trojan_insertion(puf, trojan_type)
            trojan_results.append(trojan_result)
            total_cost += trojan_result.cost_estimate_usd
        
        attack_results['supply_chain_trojans'] = trojan_results
        successful_trojans = [r for r in trojan_results if r.success_probability > 0.7]
        
        # Phase 2: Reverse engineering
        reverse_attacker = ReverseEngineeringAttacker(AttackComplexity.EXTREME)
        imaging_result = reverse_attacker.circuit_imaging_attack(puf)
        attack_results['reverse_engineering'] = imaging_result
        total_cost += imaging_result.cost_estimate_usd
        
        # Phase 3: Comprehensive ML modeling
        adversarial = AdversarialAttacker('supply_chain')
        comprehensive_result = adversarial.adaptive_attack(puf, n_queries=100000, adaptation_rounds=20)
        attack_results['comprehensive_ml'] = comprehensive_result
        total_cost += 5000000
        
        # Phase 4: Multi-channel side-channel analysis
        multi_sc_attacker = MultiChannelAttacker()
        sc_results = multi_sc_attacker.comprehensive_attack(puf, n_traces=5000)
        attack_results['side_channel_comprehensive'] = sc_results
        total_cost += 2000000
        
        # Calculate results
        elapsed_time = (time.time() - start_time) / 3600 + 1500
        
        trojan_success = len(successful_trojans) > 0
        reverse_success = imaging_result.success_probability > 0.8
        ml_success = comprehensive_result['final_accuracy'] > self.scenario_config.success_threshold
        sc_success = sc_results['combined_attack']['success_rate'] > 0.7
        
        overall_success = trojan_success and reverse_success and ml_success and sc_success
        success_rate = sum([trojan_success, reverse_success, ml_success, sc_success]) / 4.0
        
        detection_risk = 0.02
        
        if success_rate >= 0.9:
            threat_level = "CRITICAL - Complete supply chain compromise achieved"
        elif success_rate >= 0.7:
            threat_level = "HIGH - Significant supply chain vulnerabilities"
        elif success_rate >= 0.5:
            threat_level = "MEDIUM - Partial supply chain security issues"
        else:
            threat_level = "LOW - Supply chain security adequate"
        
        return ScenarioResult(
            scenario=self.scenario_config,
            attack_success=overall_success,
            overall_success_rate=success_rate,
            time_required_hours=elapsed_time,
            cost_incurred_usd=total_cost,
            damage_caused='none',
            detection_risk=detection_risk,
            extracted_information=extracted_info,
            countermeasures_triggered=[],
            threat_assessment=threat_level
        )


class DefenseScenarioRunner:
    """Comprehensive defense scenario runner for PPET evaluation."""
    
    def __init__(self):
        self.scenarios = {
            'satellite': SatelliteCommScenario(),
            'drone': DroneAuthScenario(),
            'iot': IoTFieldScenario(),
            'supply_chain': SupplyChainScenario()
        }
        self.results = {}
    
    def run_all_scenarios(self, puf: BasePUF) -> Dict[str, ScenarioResult]:
        """Run all defense scenarios against target PUF."""
        print("=== PPET Defense Scenario Evaluation ===")
        print(f"Target PUF: {type(puf).__name__}")
        print("\nRunning comprehensive defense scenarios...\n")
        
        results = {}
        
        for scenario_name, scenario in self.scenarios.items():
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario.scenario_config.scenario_name}")
            print(f"Threat Actor: {scenario.scenario_config.threat_actor.value}")
            print(f"Environment: {scenario.scenario_config.environment.value}")
            print(f"{'='*60}")
            
            try:
                result = scenario.evaluate_puf_security(puf)
                results[scenario_name] = result
                
                print(f"\nScenario Results:")
                print(f"  Success: {result.attack_success}")
                print(f"  Success Rate: {result.overall_success_rate:.2f}")
                print(f"  Time: {result.time_required_hours:.1f} hours")
                print(f"  Cost: ${result.cost_incurred_usd:,.0f}")
                print(f"  Detection Risk: {result.detection_risk:.2f}")
                print(f"  Threat Level: {result.threat_assessment}")
                
            except Exception as e:
                print(f"Error in scenario {scenario_name}: {e}")
                continue
        
        self.results = results
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, ScenarioResult]) -> Dict[str, Any]:
        """Generate comprehensive defense evaluation report."""
        if not results:
            return {'error': 'No scenario results available'}
        
        total_scenarios = len(results)
        successful_attacks = sum(1 for r in results.values() if r.attack_success)
        avg_success_rate = np.mean([r.overall_success_rate for r in results.values()])
        total_cost = sum(r.cost_incurred_usd for r in results.values())
        max_time = max(r.time_required_hours for r in results.values())
        avg_detection_risk = np.mean([r.detection_risk for r in results.values()])
        
        if avg_success_rate >= 0.8:
            security_posture = "CRITICAL - Multiple severe vulnerabilities"
        elif avg_success_rate >= 0.6:
            security_posture = "HIGH RISK - Significant security gaps"
        elif avg_success_rate >= 0.4:
            security_posture = "MEDIUM RISK - Some vulnerabilities present"
        elif avg_success_rate >= 0.2:
            security_posture = "LOW RISK - Minor security concerns"
        else:
            security_posture = "SECURE - Robust against tested scenarios"
        
        critical_scenarios = []
        for name, result in results.items():
            if result.overall_success_rate >= 0.7:
                critical_scenarios.append({
                    'scenario': name,
                    'threat_actor': result.scenario.threat_actor.value,
                    'success_rate': result.overall_success_rate,
                    'threat_assessment': result.threat_assessment
                })
        
        recommendations = [
            "Implement obfuscation techniques to reduce ML attack effectiveness",
            "Deploy response masking and temporal noise injection", 
            "Add electromagnetic shielding for sensitive applications",
            "Implement comprehensive supply chain security protocols",
            "Regular security assessments and penetration testing"
        ]
        
        return {
            'executive_summary': {
                'security_posture': security_posture,
                'scenarios_tested': total_scenarios,
                'successful_attacks': successful_attacks,
                'average_success_rate': avg_success_rate,
                'total_threat_cost': total_cost,
                'maximum_attack_time': max_time,
                'average_detection_risk': avg_detection_risk
            },
            'critical_scenarios': critical_scenarios,
            'detailed_results': {name: {
                'threat_actor': result.scenario.threat_actor.value,
                'environment': result.scenario.environment.value,
                'success': result.attack_success,
                'success_rate': result.overall_success_rate,
                'cost': result.cost_incurred_usd,
                'time_hours': result.time_required_hours,
                'detection_risk': result.detection_risk,
                'threat_assessment': result.threat_assessment,
                'extracted_info_types': list(result.extracted_information.keys())
            } for name, result in results.items()},
            'recommendations': recommendations,
            'compliance_notes': {
                'military_standards': 'Evaluate against MIL-STD-810 and DO-178C',
                'space_applications': 'Consider ECSS-E-ST-60C and NASA standards',
                'critical_infrastructure': 'Review NIST Cybersecurity Framework'
            }
        }


if __name__ == "__main__":
    print("=== PPET Defense-Oriented Attack Scenarios ===")
    print("Testing comprehensive threat scenarios for military PUF evaluation\n")
    
    from puf_models import ArbiterPUF
    test_puf = ArbiterPUF(n_stages=64, seed=42)
    
    scenario_runner = DefenseScenarioRunner()
    all_results = scenario_runner.run_all_scenarios(test_puf)
    
    comprehensive_report = scenario_runner.generate_comprehensive_report(all_results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEFENSE EVALUATION REPORT")
    print("="*80)
    
    exec_summary = comprehensive_report['executive_summary']
    print(f"\nSecurity Posture: {exec_summary['security_posture']}")
    print(f"Scenarios Tested: {exec_summary['scenarios_tested']}")
    print(f"Successful Attacks: {exec_summary['successful_attacks']}")
    print(f"Average Success Rate: {exec_summary['average_success_rate']:.2f}")
    print(f"Total Threat Investment: ${exec_summary['total_threat_cost']:,.0f}")
    print(f"Maximum Attack Timeline: {exec_summary['maximum_attack_time']:.0f} hours")
    
    print("\nCritical Scenarios:")
    for scenario in comprehensive_report['critical_scenarios']:
        print(f"  - {scenario['scenario']}: {scenario['success_rate']:.2f} success rate")
        print(f"    Threat: {scenario['threat_actor']}")
        print(f"    Assessment: {scenario['threat_assessment']}")
    
    print("\nTop Defense Recommendations:")
    for i, rec in enumerate(comprehensive_report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    print("\n=== Defense scenario evaluation complete ===")
    print("PPET provides comprehensive security assessment for military applications.")