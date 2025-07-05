"""
Security Metrics Framework for PPET
===================================

This module implements comprehensive security metrics and analysis framework
for defense-oriented PUF evaluation including:

- Military-grade security scoring
- Threat assessment and risk analysis
- Compliance evaluation frameworks
- Performance benchmarking
- Operational readiness assessment

Defense Applications:
- Quantify PUF security for military procurement
- Generate compliance reports for defense standards
- Provide operational readiness assessments
- Support security certification processes
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
from .puf_models import BasePUF
from .analysis import bit_error_rate, uniqueness

# Try to import optional packages
try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SecurityClearanceLevel(Enum):
    """Security clearance classifications."""
    UNCLASSIFIED = "unclassified"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class MilitaryStandard(Enum):
    """Military and defense standards."""
    MIL_STD_810 = "MIL-STD-810"  # Environmental Engineering
    MIL_STD_461 = "MIL-STD-461"  # EMI Requirements
    FIPS_140_2 = "FIPS-140-2"    # Cryptographic Modules
    COMMON_CRITERIA = "Common-Criteria"  # IT Security Evaluation
    DO_178C = "DO-178C"          # Airborne Software
    NATO_STANAG = "NATO-STANAG"  # NATO Standards


class ThreatModel(Enum):
    """Threat model classifications."""
    ACADEMIC = "academic_researcher"
    CRIMINAL = "criminal_organization"
    TERRORIST = "terrorist_group"
    MILITARY = "military_adversary"
    NATION_STATE = "nation_state_actor"


@dataclass
class SecurityMetric:
    """Container for individual security metrics."""
    name: str
    value: float
    weight: float
    threshold: float
    unit: str
    description: str
    compliance_status: str


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment results."""
    threat_model: ThreatModel
    attack_vectors: Dict[str, float]
    success_probability: float
    cost_estimate_usd: float
    time_estimate_hours: float
    detection_probability: float
    mitigation_effectiveness: float
    risk_level: str


@dataclass
class ComplianceReport:
    """Military compliance assessment report."""
    standard: MilitaryStandard
    compliance_score: float
    status: str
    requirements_met: List[str]
    requirements_failed: List[str]
    recommendations: List[str]
    certification_ready: bool


class SecurityMetricsAnalyzer:
    """
    Comprehensive security metrics analyzer for military PUF evaluation.
    
    Provides quantitative security assessment using military-grade metrics
    and compliance frameworks.
    """
    
    def __init__(self, clearance_level=SecurityClearanceLevel.SECRET):
        """
        Initialize security metrics analyzer.
        
        Parameters
        ----------
        clearance_level : SecurityClearanceLevel
            Required security clearance level
        """
        self.clearance_level = clearance_level
        self.military_weights = self._initialize_military_weights()
        self.compliance_thresholds = self._initialize_compliance_thresholds()
        self.threat_models = self._initialize_threat_models()
    
    def _initialize_military_weights(self) -> Dict[str, float]:
        """Initialize military security metric weights."""
        return {
            'reliability': 0.25,      # Operational reliability
            'uniqueness': 0.20,       # Device uniqueness
            'attack_resistance': 0.30, # Security against attacks
            'environmental_stability': 0.15, # Environmental robustness
            'availability': 0.10      # System availability
        }
    
    def _initialize_compliance_thresholds(self) -> Dict[MilitaryStandard, Dict[str, float]]:
        """Initialize compliance thresholds for military standards."""
        return {
            MilitaryStandard.FIPS_140_2: {
                'level_1': 0.7,
                'level_2': 0.8,
                'level_3': 0.9,
                'level_4': 0.95
            },
            MilitaryStandard.MIL_STD_810: {
                'temperature': 0.9,
                'humidity': 0.85,
                'vibration': 0.9,
                'shock': 0.95
            },
            MilitaryStandard.MIL_STD_461: {
                'emi_resistance': 0.8,
                'emi_emission': 0.9
            },
            MilitaryStandard.COMMON_CRITERIA: {
                'eal_1': 0.6,
                'eal_2': 0.7,
                'eal_3': 0.8,
                'eal_4': 0.85,
                'eal_5': 0.9,
                'eal_6': 0.95,
                'eal_7': 0.98
            }
        }
    
    def _initialize_threat_models(self) -> Dict[ThreatModel, Dict[str, Any]]:
        """Initialize threat model characteristics."""
        return {
            ThreatModel.ACADEMIC: {
                'budget_usd': 100000,
                'time_months': 12,
                'expertise_level': 0.8,
                'equipment_access': 0.7,
                'motivation': 'research'
            },
            ThreatModel.CRIMINAL: {
                'budget_usd': 500000,
                'time_months': 6,
                'expertise_level': 0.6,
                'equipment_access': 0.5,
                'motivation': 'financial'
            },
            ThreatModel.TERRORIST: {
                'budget_usd': 1000000,
                'time_months': 18,
                'expertise_level': 0.7,
                'equipment_access': 0.6,
                'motivation': 'disruption'
            },
            ThreatModel.MILITARY: {
                'budget_usd': 10000000,
                'time_months': 24,
                'expertise_level': 0.9,
                'equipment_access': 0.9,
                'motivation': 'intelligence'
            },
            ThreatModel.NATION_STATE: {
                'budget_usd': 100000000,
                'time_months': 60,
                'expertise_level': 0.95,
                'equipment_access': 0.98,
                'motivation': 'strategic'
            }
        }
    
    def calculate_security_score(self, puf: BasePUF, attack_results: Dict[str, Any],
                               environmental_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive military security score.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        attack_results : Dict[str, Any]
            Results from attack analysis
        environmental_results : Dict[str, Any]
            Results from environmental testing
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive security assessment
        """
        print("Calculating military security metrics...")
        
        # Generate test data if not provided
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        
        # Basic reliability assessment
        test_challenges = rng.integers(0, 2, size=(1000, n_stages))
        responses1 = puf.eval(test_challenges)
        responses2 = puf.eval(test_challenges)  # Repeat evaluation
        
        reliability_score = np.mean(responses1 == responses2)
        
        # Uniqueness assessment
        if isinstance(puf, type(puf)):
            # Create multiple instances for uniqueness testing
            puf_instances = []
            for i in range(5):
                similar_puf = type(puf)(n_stages, seed=100 + i)
                puf_instances.append(similar_puf)
            
            uniqueness_responses = []
            for instance in puf_instances:
                resp = instance.eval(test_challenges[:100])  # Use subset for efficiency
                uniqueness_responses.append(resp)
            
            uniqueness_score = uniqueness(test_challenges[:100], np.array(uniqueness_responses))
        else:
            uniqueness_score = 0.5  # Default if can't evaluate
        
        # Extract attack resistance scores
        attack_resistance_score = self._calculate_attack_resistance(attack_results)
        
        # Extract environmental stability score  
        environmental_score = self._calculate_environmental_stability(environmental_results)
        
        # Calculate availability (simplified)
        availability_score = min(reliability_score, environmental_score)
        
        # Compile individual metrics
        metrics = {
            'reliability': SecurityMetric(
                name='Reliability',
                value=reliability_score,
                weight=self.military_weights['reliability'],
                threshold=0.95,
                unit='percentage',
                description='Operational reliability under normal conditions',
                compliance_status='PASS' if reliability_score >= 0.95 else 'FAIL'
            ),
            'uniqueness': SecurityMetric(
                name='Uniqueness',
                value=uniqueness_score,
                weight=self.military_weights['uniqueness'],
                threshold=0.45,
                unit='percentage',
                description='Inter-device uniqueness',
                compliance_status='PASS' if 0.45 <= uniqueness_score <= 0.55 else 'FAIL'
            ),
            'attack_resistance': SecurityMetric(
                name='Attack Resistance',
                value=attack_resistance_score,
                weight=self.military_weights['attack_resistance'],
                threshold=0.8,
                unit='percentage',
                description='Resistance against known attack vectors',
                compliance_status='PASS' if attack_resistance_score >= 0.8 else 'FAIL'
            ),
            'environmental_stability': SecurityMetric(
                name='Environmental Stability',
                value=environmental_score,
                weight=self.military_weights['environmental_stability'],
                threshold=0.9,
                unit='percentage',
                description='Stability under environmental stress',
                compliance_status='PASS' if environmental_score >= 0.9 else 'FAIL'
            ),
            'availability': SecurityMetric(
                name='Availability',
                value=availability_score,
                weight=self.military_weights['availability'],
                threshold=0.99,
                unit='percentage',
                description='System availability and uptime',
                compliance_status='PASS' if availability_score >= 0.99 else 'FAIL'
            )
        }
        
        # Calculate weighted total score
        total_score = sum(metric.value * metric.weight for metric in metrics.values())
        
        # Determine military grade
        if total_score >= 0.95:
            military_grade = 'A+'
            deployment_status = 'APPROVED'
            clearance_recommendation = SecurityClearanceLevel.TOP_SECRET
        elif total_score >= 0.9:
            military_grade = 'A'
            deployment_status = 'APPROVED'
            clearance_recommendation = SecurityClearanceLevel.SECRET
        elif total_score >= 0.85:
            military_grade = 'A-'
            deployment_status = 'APPROVED_WITH_CONDITIONS'
            clearance_recommendation = SecurityClearanceLevel.SECRET
        elif total_score >= 0.8:
            military_grade = 'B+'
            deployment_status = 'LIMITED_APPROVAL'
            clearance_recommendation = SecurityClearanceLevel.CONFIDENTIAL
        elif total_score >= 0.75:
            military_grade = 'B'
            deployment_status = 'CONDITIONAL_APPROVAL'
            clearance_recommendation = SecurityClearanceLevel.CONFIDENTIAL
        elif total_score >= 0.7:
            military_grade = 'B-'
            deployment_status = 'RESTRICTED_USE'
            clearance_recommendation = SecurityClearanceLevel.UNCLASSIFIED
        elif total_score >= 0.6:
            military_grade = 'C'
            deployment_status = 'NOT_RECOMMENDED'
            clearance_recommendation = SecurityClearanceLevel.UNCLASSIFIED
        else:
            military_grade = 'F'
            deployment_status = 'REJECTED'
            clearance_recommendation = SecurityClearanceLevel.UNCLASSIFIED
        
        return {
            'total_score': total_score,
            'military_grade': military_grade,
            'deployment_status': deployment_status,
            'clearance_recommendation': clearance_recommendation,
            'individual_metrics': metrics,
            'passing_metrics': sum(1 for m in metrics.values() if m.compliance_status == 'PASS'),
            'total_metrics': len(metrics),
            'assessment_timestamp': time.time()
        }
    
    def _calculate_attack_resistance(self, attack_results: Dict[str, Any]) -> float:
        """Calculate overall attack resistance score."""
        if not attack_results:
            return 0.5  # Default neutral score
        
        # Extract resistance scores from different attack types
        resistances = []
        
        # ML attack resistance
        if 'ml_resistance' in attack_results:
            resistances.append(attack_results['ml_resistance'])
        elif 'ml_accuracy' in attack_results:
            resistances.append(1.0 - attack_results['ml_accuracy'])
        
        # Side-channel resistance
        if 'side_channel_resistance' in attack_results:
            resistances.append(attack_results['side_channel_resistance'])
        elif 'side_channel_success' in attack_results:
            resistances.append(1.0 - attack_results['side_channel_success'])
        
        # Physical attack resistance
        if 'physical_resistance' in attack_results:
            resistances.append(attack_results['physical_resistance'])
        elif 'physical_success' in attack_results:
            resistances.append(1.0 - attack_results['physical_success'])
        
        # If no specific resistance scores, try to infer from attack results
        if not resistances:
            # Look for any attack success indicators
            for key, value in attack_results.items():
                if 'accuracy' in key.lower() or 'success' in key.lower():
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        resistances.append(1.0 - value)
        
        if resistances:
            return np.mean(resistances)
        else:
            return 0.75  # Conservative default
    
    def _calculate_environmental_stability(self, environmental_results: Dict[str, Any]) -> float:
        """Calculate environmental stability score."""
        if not environmental_results:
            return 0.8  # Default conservative estimate
        
        # Extract stability metrics
        stabilities = []
        
        # Temperature stability
        if 'temperature_resilience' in environmental_results:
            stabilities.append(environmental_results['temperature_resilience'])
        elif 'temperature_ber' in environmental_results:
            stabilities.append(1.0 - environmental_results['temperature_ber'] / 10.0)  # Assume 10% max BER
        
        # Other environmental factors
        environmental_factors = [
            'humidity_resistance', 'vibration_tolerance', 'shock_resistance',
            'emi_resistance', 'radiation_resistance', 'aging_resistance'
        ]
        
        for factor in environmental_factors:
            if factor in environmental_results:
                stabilities.append(environmental_results[factor])
        
        if stabilities:
            return np.mean(stabilities)
        else:
            return 0.85  # Conservative default
    
    def generate_threat_assessment_report(self, puf: BasePUF, 
                                        threat_model: ThreatModel = ThreatModel.MILITARY) -> ThreatAssessment:
        """
        Generate comprehensive threat assessment for specific threat model.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        threat_model : ThreatModel
            Threat actor to assess against
            
        Returns
        -------
        ThreatAssessment
            Comprehensive threat assessment
        """
        print(f"Generating threat assessment for {threat_model.value}...")
        
        threat_characteristics = self.threat_models[threat_model]
        
        # Estimate attack vector success probabilities based on threat capabilities
        attack_vectors = {
            'ml_attacks': self._estimate_ml_attack_success(threat_characteristics),
            'side_channel_attacks': self._estimate_side_channel_success(threat_characteristics),
            'physical_attacks': self._estimate_physical_attack_success(threat_characteristics),
            'supply_chain_attacks': self._estimate_supply_chain_success(threat_characteristics),
            'social_engineering': self._estimate_social_engineering_success(threat_characteristics)
        }
        
        # Calculate overall success probability
        # Use maximum success rate (adversary will use best attack)
        success_probability = max(attack_vectors.values())
        
        # Estimate cost and time based on threat model
        cost_estimate = threat_characteristics['budget_usd'] * 0.1  # Use 10% of budget
        time_estimate = threat_characteristics['time_months'] * 30 * 24 * 0.1  # 10% of time in hours
        
        # Detection probability based on threat sophistication
        expertise = threat_characteristics['expertise_level']
        detection_probability = max(0.1, 1.0 - expertise)  # Higher expertise = lower detection
        
        # Mitigation effectiveness
        mitigation_effectiveness = self._calculate_mitigation_effectiveness(attack_vectors)
        
        # Risk level assessment
        if success_probability >= 0.8:
            risk_level = "CRITICAL"
        elif success_probability >= 0.6:
            risk_level = "HIGH"
        elif success_probability >= 0.4:
            risk_level = "MEDIUM"
        elif success_probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return ThreatAssessment(
            threat_model=threat_model,
            attack_vectors=attack_vectors,
            success_probability=success_probability,
            cost_estimate_usd=cost_estimate,
            time_estimate_hours=time_estimate,
            detection_probability=detection_probability,
            mitigation_effectiveness=mitigation_effectiveness,
            risk_level=risk_level
        )
    
    def _estimate_ml_attack_success(self, threat_characteristics: Dict[str, Any]) -> float:
        """Estimate ML attack success probability for threat actor."""
        expertise = threat_characteristics['expertise_level']
        budget = threat_characteristics['budget_usd']
        time_months = threat_characteristics['time_months']
        
        # ML attacks scale with computational resources and expertise
        base_success = 0.3  # Baseline ML attack success
        expertise_bonus = expertise * 0.4
        resource_bonus = min(0.3, np.log10(budget / 10000) * 0.1)  # Logarithmic scaling
        time_bonus = min(0.2, time_months / 60)  # Up to 5 years
        
        return min(0.95, base_success + expertise_bonus + resource_bonus + time_bonus)
    
    def _estimate_side_channel_success(self, threat_characteristics: Dict[str, Any]) -> float:
        """Estimate side-channel attack success probability."""
        expertise = threat_characteristics['expertise_level']
        equipment_access = threat_characteristics['equipment_access']
        
        # Side-channel attacks require specialized equipment and expertise
        base_success = 0.2
        expertise_bonus = expertise * 0.5
        equipment_bonus = equipment_access * 0.3
        
        return min(0.9, base_success + expertise_bonus + equipment_bonus)
    
    def _estimate_physical_attack_success(self, threat_characteristics: Dict[str, Any]) -> float:
        """Estimate physical attack success probability."""
        budget = threat_characteristics['budget_usd']
        expertise = threat_characteristics['expertise_level']
        equipment_access = threat_characteristics['equipment_access']
        
        # Physical attacks require significant resources and expertise
        base_success = 0.1
        budget_bonus = min(0.4, np.log10(budget / 100000) * 0.15)
        expertise_bonus = expertise * 0.3
        equipment_bonus = equipment_access * 0.2
        
        return min(0.85, base_success + budget_bonus + expertise_bonus + equipment_bonus)
    
    def _estimate_supply_chain_success(self, threat_characteristics: Dict[str, Any]) -> float:
        """Estimate supply chain attack success probability."""
        budget = threat_characteristics['budget_usd']
        time_months = threat_characteristics['time_months']
        motivation = threat_characteristics['motivation']
        
        # Supply chain attacks require long-term commitment and resources
        base_success = 0.05
        budget_bonus = min(0.3, np.log10(budget / 1000000) * 0.1)
        time_bonus = min(0.4, time_months / 36)  # Up to 3 years
        
        # Motivation affects supply chain attack likelihood
        motivation_multiplier = {
            'research': 0.5,
            'financial': 1.0,
            'disruption': 1.2,
            'intelligence': 1.5,
            'strategic': 2.0
        }.get(motivation, 1.0)
        
        return min(0.7, (base_success + budget_bonus + time_bonus) * motivation_multiplier)
    
    def _estimate_social_engineering_success(self, threat_characteristics: Dict[str, Any]) -> float:
        """Estimate social engineering attack success probability."""
        expertise = threat_characteristics['expertise_level']
        motivation = threat_characteristics['motivation']
        
        # Social engineering depends on human factors
        base_success = 0.3  # Humans are often the weakest link
        expertise_bonus = expertise * 0.2
        
        motivation_bonus = {
            'research': 0.0,
            'financial': 0.1,
            'disruption': 0.2,
            'intelligence': 0.3,
            'strategic': 0.4
        }.get(motivation, 0.1)
        
        return min(0.8, base_success + expertise_bonus + motivation_bonus)
    
    def _calculate_mitigation_effectiveness(self, attack_vectors: Dict[str, float]) -> float:
        """Calculate effectiveness of potential mitigations."""
        # Simplified mitigation effectiveness calculation
        max_attack_success = max(attack_vectors.values())
        
        # Assume mitigations can reduce attack success by 50-80%
        if max_attack_success >= 0.8:
            mitigation_effectiveness = 0.6  # 60% reduction possible
        elif max_attack_success >= 0.6:
            mitigation_effectiveness = 0.7  # 70% reduction possible
        elif max_attack_success >= 0.4:
            mitigation_effectiveness = 0.8  # 80% reduction possible
        else:
            mitigation_effectiveness = 0.9  # 90% reduction possible
        
        return mitigation_effectiveness
    
    def evaluate_compliance(self, puf: BasePUF, security_score: Dict[str, Any],
                          standards: List[MilitaryStandard] = None) -> Dict[MilitaryStandard, ComplianceReport]:
        """
        Evaluate compliance against military and defense standards.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        security_score : Dict[str, Any]
            Security assessment results
        standards : List[MilitaryStandard], optional
            Standards to evaluate against
            
        Returns
        -------
        Dict[MilitaryStandard, ComplianceReport]
            Compliance reports for each standard
        """
        if standards is None:
            standards = [MilitaryStandard.FIPS_140_2, MilitaryStandard.MIL_STD_810, 
                        MilitaryStandard.COMMON_CRITERIA]
        
        compliance_reports = {}
        
        for standard in standards:
            print(f"Evaluating compliance with {standard.value}...")
            
            if standard == MilitaryStandard.FIPS_140_2:
                report = self._evaluate_fips_140_2_compliance(security_score)
            elif standard == MilitaryStandard.MIL_STD_810:
                report = self._evaluate_mil_std_810_compliance(security_score)
            elif standard == MilitaryStandard.COMMON_CRITERIA:
                report = self._evaluate_common_criteria_compliance(security_score)
            else:
                # Generic compliance evaluation
                report = self._evaluate_generic_compliance(standard, security_score)
            
            compliance_reports[standard] = report
        
        return compliance_reports
    
    def _evaluate_fips_140_2_compliance(self, security_score: Dict[str, Any]) -> ComplianceReport:
        """Evaluate FIPS 140-2 compliance."""
        total_score = security_score['total_score']
        thresholds = self.compliance_thresholds[MilitaryStandard.FIPS_140_2]
        
        requirements_met = []
        requirements_failed = []
        
        # Security level determination
        if total_score >= thresholds['level_4']:
            level = 4
            status = "LEVEL_4_COMPLIANT"
            requirements_met.extend([
                "Cryptographic module security",
                "Tamper detection and response",
                "Environmental failure protection",
                "Side-channel attack protection"
            ])
        elif total_score >= thresholds['level_3']:
            level = 3
            status = "LEVEL_3_COMPLIANT"
            requirements_met.extend([
                "Cryptographic module security",
                "Tamper detection",
                "Environmental failure protection"
            ])
            requirements_failed.append("Advanced side-channel protection")
        elif total_score >= thresholds['level_2']:
            level = 2
            status = "LEVEL_2_COMPLIANT"
            requirements_met.extend([
                "Cryptographic module security",
                "Basic tamper evidence"
            ])
            requirements_failed.extend([
                "Tamper detection and response",
                "Environmental failure protection"
            ])
        elif total_score >= thresholds['level_1']:
            level = 1
            status = "LEVEL_1_COMPLIANT"
            requirements_met.append("Basic cryptographic module security")
            requirements_failed.extend([
                "Tamper evidence",
                "Environmental protection",
                "Side-channel protection"
            ])
        else:
            level = 0
            status = "NON_COMPLIANT"
            requirements_failed.extend([
                "Basic cryptographic module security",
                "Tamper evidence",
                "Environmental protection"
            ])
        
        recommendations = []
        if level < 2:
            recommendations.append("Implement tamper detection mechanisms")
        if level < 3:
            recommendations.append("Add environmental failure protection")
        if level < 4:
            recommendations.append("Deploy advanced side-channel countermeasures")
        
        return ComplianceReport(
            standard=MilitaryStandard.FIPS_140_2,
            compliance_score=total_score,
            status=status,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            certification_ready=(level >= 2)
        )
    
    def _evaluate_mil_std_810_compliance(self, security_score: Dict[str, Any]) -> ComplianceReport:
        """Evaluate MIL-STD-810 compliance."""
        # Environmental stability metrics
        env_metrics = security_score['individual_metrics'].get('environmental_stability')
        env_score = env_metrics.value if env_metrics else 0.8
        
        thresholds = self.compliance_thresholds[MilitaryStandard.MIL_STD_810]
        
        requirements_met = []
        requirements_failed = []
        
        # Temperature testing
        if env_score >= thresholds['temperature']:
            requirements_met.append("Temperature testing (Method 501)")
        else:
            requirements_failed.append("Temperature testing (Method 501)")
        
        # Humidity testing
        if env_score >= thresholds['humidity']:
            requirements_met.append("Humidity testing (Method 507)")
        else:
            requirements_failed.append("Humidity testing (Method 507)")
        
        # Vibration testing
        if env_score >= thresholds['vibration']:
            requirements_met.append("Vibration testing (Method 514)")
        else:
            requirements_failed.append("Vibration testing (Method 514)")
        
        # Shock testing
        if env_score >= thresholds['shock']:
            requirements_met.append("Shock testing (Method 516)")
        else:
            requirements_failed.append("Shock testing (Method 516)")
        
        compliance_percentage = len(requirements_met) / (len(requirements_met) + len(requirements_failed))
        
        if compliance_percentage >= 0.9:
            status = "FULLY_COMPLIANT"
        elif compliance_percentage >= 0.75:
            status = "SUBSTANTIALLY_COMPLIANT"
        elif compliance_percentage >= 0.5:
            status = "PARTIALLY_COMPLIANT"
        else:
            status = "NON_COMPLIANT"
        
        recommendations = []
        if "Temperature testing" in requirements_failed:
            recommendations.append("Conduct extended temperature range testing")
        if "Humidity testing" in requirements_failed:
            recommendations.append("Perform humidity resistance evaluation")
        if "Vibration testing" in requirements_failed:
            recommendations.append("Execute vibration tolerance testing")
        if "Shock testing" in requirements_failed:
            recommendations.append("Implement shock resistance testing")
        
        return ComplianceReport(
            standard=MilitaryStandard.MIL_STD_810,
            compliance_score=compliance_percentage,
            status=status,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            certification_ready=(compliance_percentage >= 0.75)
        )
    
    def _evaluate_common_criteria_compliance(self, security_score: Dict[str, Any]) -> ComplianceReport:
        """Evaluate Common Criteria compliance."""
        total_score = security_score['total_score']
        thresholds = self.compliance_thresholds[MilitaryStandard.COMMON_CRITERIA]
        
        # Determine Evaluation Assurance Level (EAL)
        if total_score >= thresholds['eal_7']:
            eal_level = 7
            status = "EAL7_COMPLIANT"
        elif total_score >= thresholds['eal_6']:
            eal_level = 6
            status = "EAL6_COMPLIANT"
        elif total_score >= thresholds['eal_5']:
            eal_level = 5
            status = "EAL5_COMPLIANT"
        elif total_score >= thresholds['eal_4']:
            eal_level = 4
            status = "EAL4_COMPLIANT"
        elif total_score >= thresholds['eal_3']:
            eal_level = 3
            status = "EAL3_COMPLIANT"
        elif total_score >= thresholds['eal_2']:
            eal_level = 2
            status = "EAL2_COMPLIANT"
        elif total_score >= thresholds['eal_1']:
            eal_level = 1
            status = "EAL1_COMPLIANT"
        else:
            eal_level = 0
            status = "NON_COMPLIANT"
        
        requirements_met = []
        requirements_failed = []
        
        # Requirements based on EAL level
        eal_requirements = {
            1: ["Functional testing", "Security target evaluation"],
            2: ["Structural testing", "Vulnerability assessment"],
            3: ["Development environment controls", "Systematic testing"],
            4: ["Design review", "Independent testing"],
            5: ["Semiformal design verification", "Penetration testing"],
            6: ["Formal design verification", "Systematic vulnerability analysis"],
            7: ["Formal top-level specification", "Comprehensive vulnerability analysis"]
        }
        
        for level in range(1, eal_level + 1):
            requirements_met.extend(eal_requirements.get(level, []))
        
        for level in range(eal_level + 1, 8):
            requirements_failed.extend(eal_requirements.get(level, []))
        
        recommendations = []
        if eal_level < 4:
            recommendations.append("Enhance security testing and documentation")
        if eal_level < 5:
            recommendations.append("Implement formal verification processes")
        if eal_level < 6:
            recommendations.append("Deploy comprehensive vulnerability analysis")
        
        return ComplianceReport(
            standard=MilitaryStandard.COMMON_CRITERIA,
            compliance_score=total_score,
            status=status,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            certification_ready=(eal_level >= 3)
        )
    
    def _evaluate_generic_compliance(self, standard: MilitaryStandard, 
                                   security_score: Dict[str, Any]) -> ComplianceReport:
        """Generic compliance evaluation for unsupported standards."""
        total_score = security_score['total_score']
        
        if total_score >= 0.9:
            status = "LIKELY_COMPLIANT"
        elif total_score >= 0.8:
            status = "POTENTIALLY_COMPLIANT"
        else:
            status = "UNLIKELY_COMPLIANT"
        
        return ComplianceReport(
            standard=standard,
            compliance_score=total_score,
            status=status,
            requirements_met=["General security requirements"],
            requirements_failed=["Specific standard requirements (not evaluated)"],
            recommendations=[f"Conduct specific {standard.value} evaluation"],
            certification_ready=False
        )
    
    def generate_comprehensive_report(self, puf: BasePUF, 
                                    attack_results: Dict[str, Any] = None,
                                    environmental_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive security and compliance report.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        attack_results : Dict[str, Any], optional
            Attack analysis results
        environmental_results : Dict[str, Any], optional
            Environmental testing results
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive security report
        """
        print("Generating comprehensive security assessment report...")
        
        # Use default empty results if not provided
        if attack_results is None:
            attack_results = {}
        if environmental_results is None:
            environmental_results = {}
        
        # Calculate security scores
        security_assessment = self.calculate_security_score(puf, attack_results, environmental_results)
        
        # Generate threat assessments
        threat_assessments = {}
        for threat_model in [ThreatModel.ACADEMIC, ThreatModel.CRIMINAL, ThreatModel.MILITARY, ThreatModel.NATION_STATE]:
            threat_assessments[threat_model.value] = self.generate_threat_assessment_report(puf, threat_model)
        
        # Evaluate compliance
        compliance_reports = self.evaluate_compliance(puf, security_assessment)
        
        # Determine overall readiness
        avg_compliance = np.mean([report.compliance_score for report in compliance_reports.values()])
        max_threat_risk = max([assessment.success_probability for assessment in threat_assessments.values()])
        
        if security_assessment['total_score'] >= 0.9 and avg_compliance >= 0.8 and max_threat_risk <= 0.3:
            operational_readiness = "FULLY_OPERATIONAL"
        elif security_assessment['total_score'] >= 0.8 and avg_compliance >= 0.7 and max_threat_risk <= 0.5:
            operational_readiness = "OPERATIONALLY_CAPABLE"
        elif security_assessment['total_score'] >= 0.7 and avg_compliance >= 0.6:
            operational_readiness = "LIMITED_OPERATIONAL"
        else:
            operational_readiness = "NOT_OPERATIONAL"
        
        # Generate executive summary
        executive_summary = {
            'puf_type': type(puf).__name__,
            'assessment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'security_score': security_assessment['total_score'],
            'military_grade': security_assessment['military_grade'],
            'deployment_status': security_assessment['deployment_status'],
            'operational_readiness': operational_readiness,
            'clearance_recommendation': security_assessment['clearance_recommendation'].value,
            'compliance_summary': {
                standard.value: report.status for standard, report in compliance_reports.items()
            },
            'highest_threat_risk': max_threat_risk,
            'critical_vulnerabilities': len([m for m in security_assessment['individual_metrics'].values() 
                                           if m.compliance_status == 'FAIL'])
        }
        
        return {
            'executive_summary': executive_summary,
            'security_assessment': security_assessment,
            'threat_assessments': {name: {
                'threat_model': assessment.threat_model.value,
                'success_probability': assessment.success_probability,
                'risk_level': assessment.risk_level,
                'cost_estimate': assessment.cost_estimate_usd,
                'time_estimate': assessment.time_estimate_hours,
                'detection_probability': assessment.detection_probability
            } for name, assessment in threat_assessments.items()},
            'compliance_reports': {
                standard.value: {
                    'status': report.status,
                    'score': report.compliance_score,
                    'requirements_met': len(report.requirements_met),
                    'requirements_failed': len(report.requirements_failed),
                    'certification_ready': report.certification_ready
                } for standard, report in compliance_reports.items()
            },
            'recommendations': self._generate_consolidated_recommendations(
                security_assessment, threat_assessments, compliance_reports
            )
        }
    
    def _generate_consolidated_recommendations(self, security_assessment, threat_assessments, compliance_reports):
        """Generate consolidated recommendations from all assessments."""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        # Security-based recommendations
        failed_metrics = [m for m in security_assessment['individual_metrics'].values() 
                         if m.compliance_status == 'FAIL']
        
        for metric in failed_metrics:
            if metric.name == 'Reliability':
                recommendations['immediate'].append("Improve PUF reliability mechanisms")
            elif metric.name == 'Attack Resistance':
                recommendations['immediate'].append("Deploy advanced attack countermeasures")
            elif metric.name == 'Environmental Stability':
                recommendations['short_term'].append("Enhance environmental hardening")
        
        # Threat-based recommendations
        for assessment in threat_assessments.values():
            if assessment.risk_level in ['CRITICAL', 'HIGH']:
                recommendations['immediate'].append(f"Address {assessment.threat_model.value} threat vectors")
        
        # Compliance-based recommendations
        for report in compliance_reports.values():
            if not report.certification_ready:
                recommendations['short_term'].extend(report.recommendations[:2])
                recommendations['long_term'].extend(report.recommendations[2:])
        
        # Remove duplicates
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
        
        return recommendations


if __name__ == "__main__":
    print("=== PPET Security Metrics Framework ===")
    print("Testing comprehensive security assessment for military PUF evaluation\n")
    
    # Test with Arbiter PUF
    from .puf_models import ArbiterPUF
    
    test_puf = ArbiterPUF(n_stages=64, seed=42)
    
    # Initialize security analyzer
    analyzer = SecurityMetricsAnalyzer(SecurityClearanceLevel.SECRET)
    
    # Example attack and environmental results
    sample_attack_results = {
        'ml_resistance': 0.8,
        'side_channel_resistance': 0.75,
        'physical_resistance': 0.85
    }
    
    sample_environmental_results = {
        'temperature_resilience': 0.9,
        'humidity_resistance': 0.85,
        'emi_resistance': 0.8
    }
    
    # Generate comprehensive report
    comprehensive_report = analyzer.generate_comprehensive_report(
        test_puf, sample_attack_results, sample_environmental_results
    )
    
    print("Security Assessment Results:")
    print("=" * 60)
    
    # Executive summary
    summary = comprehensive_report['executive_summary']
    print(f"PUF Type: {summary['puf_type']}")
    print(f"Security Score: {summary['security_score']:.3f}")
    print(f"Military Grade: {summary['military_grade']}")
    print(f"Deployment Status: {summary['deployment_status']}")
    print(f"Operational Readiness: {summary['operational_readiness']}")
    print(f"Clearance Recommendation: {summary['clearance_recommendation']}")
    print(f"Critical Vulnerabilities: {summary['critical_vulnerabilities']}")
    
    print("\nThreat Assessment Summary:")
    for threat_name, threat_data in comprehensive_report['threat_assessments'].items():
        print(f"  {threat_name}: {threat_data['risk_level']} risk "
              f"({threat_data['success_probability']:.2f} success probability)")
    
    print("\nCompliance Status:")
    for standard, compliance_data in comprehensive_report['compliance_reports'].items():
        print(f"  {standard}: {compliance_data['status']} "
              f"(Score: {compliance_data['score']:.2f})")
    
    print("\nImmediate Recommendations:")
    for i, rec in enumerate(comprehensive_report['recommendations']['immediate'], 1):
        print(f"  {i}. {rec}")
    
    print("\n=== Security metrics assessment complete ===")
    print("PPET provides comprehensive security evaluation for military applications.")