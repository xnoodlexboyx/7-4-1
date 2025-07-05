"""
Physical Attack Framework for PPET
=================================

This module implements physical attack models for defense-oriented PUF evaluation.
Focuses on modeling sophisticated physical attacks relevant to military and 
national security applications:

- Invasive attacks (decapsulation, microprobing, FIB modification)
- Semi-invasive attacks (laser fault injection, optical probing)
- Non-invasive attacks (fault injection, glitching, environmental manipulation)
- Supply chain attacks (hardware trojans, circuit modification)
- Reverse engineering attacks (imaging, reconstruction)

Designed for evaluating PUF resilience against nation-state and military adversaries.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from .puf_models import BasePUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
# Try to import optional packages
try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AttackComplexity(Enum):
    """Attack complexity levels for physical attacks."""
    TRIVIAL = 1      # Script kiddie level
    LOW = 2          # Standard lab equipment
    MEDIUM = 3       # Advanced lab equipment
    HIGH = 4         # Nation-state capabilities
    EXTREME = 5      # State-of-the-art research facilities


class AttackVector(Enum):
    """Physical attack vector classification."""
    INVASIVE = "invasive"
    SEMI_INVASIVE = "semi_invasive"
    NON_INVASIVE = "non_invasive"
    SUPPLY_CHAIN = "supply_chain"


@dataclass
class PhysicalAttackResult:
    """
    Result container for physical attack analysis.
    """
    attack_type: str
    attack_vector: AttackVector
    complexity: AttackComplexity
    success_probability: float
    extracted_secrets: Optional[Dict[str, Any]]
    damage_level: str  # "none", "minimal", "moderate", "severe", "destructive"
    detection_probability: float
    cost_estimate_usd: float
    time_estimate_hours: float
    equipment_required: List[str]


class FaultInjectionAttacker:
    """
    Fault injection attack implementation for PUF evaluation.
    Models various fault injection techniques including voltage, clock, 
    laser, and electromagnetic fault injection.
    """
    
    def __init__(self, fault_type: str = 'voltage', precision: float = 0.1):
        """
        Initialize fault injection attacker.
        
        Parameters
        ----------
        fault_type : str
            Type of fault injection ('voltage', 'clock', 'laser', 'em')
        precision : float
            Attack precision (0.0 to 1.0)
        """
        self.fault_type = fault_type
        self.precision = precision
        self.fault_history = []
    
    def inject_voltage_fault(self, puf: BasePUF, challenge: np.ndarray, 
                           fault_voltage: float, fault_duration: float) -> PhysicalAttackResult:
        """
        Simulate voltage fault injection attack.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        challenge : np.ndarray
            Input challenge
        fault_voltage : float
            Fault voltage amplitude (V)
        fault_duration : float
            Fault duration (seconds)
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        # Simulate fault effect on PUF parameters
        original_response = puf.eval(challenge.reshape(1, -1))[0]
        
        # Voltage fault effects depend on PUF type
        fault_probability = self._calculate_fault_probability(fault_voltage, fault_duration)
        
        rng = np.random.default_rng(42)
        
        if rng.random() < fault_probability:
            # Fault successful - response may flip
            if isinstance(puf, ArbiterPUF):
                # Voltage fault affects delay parameters
                fault_factor = 1 + fault_voltage * 0.1  # 10% per volt
                faulty_puf = ArbiterPUF(puf.n_stages, seed=None)
                faulty_puf.delay_params = puf.delay_params * fault_factor
                faulty_response = faulty_puf.eval(challenge.reshape(1, -1))[0]
                
            elif isinstance(puf, SRAMPUF):
                # Voltage fault affects threshold voltages
                fault_factor = fault_voltage * 50  # 50 mV per volt
                faulty_puf = SRAMPUF(puf.n_cells, seed=None, 
                                   radiation_hardening=puf.radiation_hardening,
                                   low_power_mode=puf.low_power_mode)
                faulty_puf.vth_variations = puf.vth_variations + fault_factor
                faulty_response = faulty_puf.eval(challenge.reshape(1, -1))[0]
                
            else:
                # Generic fault model
                faulty_response = -original_response if rng.random() < 0.5 else original_response
        else:
            faulty_response = original_response
        
        # Assess attack success
        response_changed = (faulty_response != original_response)
        
        # Calculate damage and detection risk
        damage_level = self._assess_voltage_damage(fault_voltage, fault_duration)
        detection_prob = min(0.9, fault_voltage * 0.2)  # Higher voltage = easier detection
        
        # Cost and complexity assessment
        if fault_voltage < 1.0:
            complexity = AttackComplexity.LOW
            cost_estimate = 1000  # Basic power supply
            time_estimate = 2
            equipment = ['Variable power supply', 'Oscilloscope', 'Probes']
        elif fault_voltage < 5.0:
            complexity = AttackComplexity.MEDIUM
            cost_estimate = 10000  # Advanced power supply with fast rise time
            time_estimate = 8
            equipment = ['High-speed power supply', 'High-bandwidth oscilloscope', 
                        'Precision probes', 'Timing generator']
        else:
            complexity = AttackComplexity.HIGH
            cost_estimate = 50000  # Specialized fault injection equipment
            time_estimate = 24
            equipment = ['Professional fault injection platform', 'High-voltage amplifiers',
                        'Custom PCB', 'Environmental chamber']
        
        extracted_secrets = None
        if response_changed:
            # Attempt to extract information from fault response
            extracted_secrets = {
                'original_response': original_response,
                'faulty_response': faulty_response,
                'fault_voltage': fault_voltage,
                'fault_duration': fault_duration
            }
        
        return PhysicalAttackResult(
            attack_type='voltage_fault_injection',
            attack_vector=AttackVector.NON_INVASIVE,
            complexity=complexity,
            success_probability=fault_probability,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=detection_prob,
            cost_estimate_usd=cost_estimate,
            time_estimate_hours=time_estimate,
            equipment_required=equipment
        )
    
    def inject_laser_fault(self, puf: BasePUF, challenge: np.ndarray,
                          laser_power: float, spot_size_um: float, 
                          target_location: Tuple[float, float]) -> PhysicalAttackResult:
        """
        Simulate laser fault injection attack.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        challenge : np.ndarray
            Input challenge
        laser_power : float
            Laser power (mW)
        spot_size_um : float
            Laser spot size (micrometers)
        target_location : Tuple[float, float]
            Target coordinates (x, y) in micrometers
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        original_response = puf.eval(challenge.reshape(1, -1))[0]
        
        # Laser fault success depends on power density and targeting accuracy
        power_density = laser_power / (np.pi * (spot_size_um / 2) ** 2)  # mW/um^2
        targeting_accuracy = self.precision
        
        # Critical power density threshold
        critical_power_density = 10.0  # mW/um^2
        fault_probability = min(0.95, (power_density / critical_power_density) * targeting_accuracy)
        
        rng = np.random.default_rng(42)
        
        if rng.random() < fault_probability:
            # Laser fault can cause bit flips or stuck bits
            if laser_power > 50:  # High power can cause permanent damage
                # Permanent bit flip
                faulty_response = -original_response
                damage_level = "severe"
            elif laser_power > 10:  # Medium power causes temporary faults
                # Temporary fault
                faulty_response = -original_response if rng.random() < 0.7 else original_response
                damage_level = "moderate"
            else:
                # Low power may not cause reliable faults
                faulty_response = -original_response if rng.random() < 0.3 else original_response
                damage_level = "minimal"
        else:
            faulty_response = original_response
            damage_level = "none"
        
        # Detection probability based on required equipment
        if laser_power > 20:
            detection_prob = 0.8  # High power requires decapsulation
        elif spot_size_um < 1.0:
            detection_prob = 0.6  # Precision targeting requires advanced setup
        else:
            detection_prob = 0.3  # Basic laser setup
        
        # Cost and complexity based on laser specifications
        if laser_power < 10 and spot_size_um > 5:
            complexity = AttackComplexity.MEDIUM
            cost_estimate = 25000  # Basic laser fault injection setup
            time_estimate = 16
            equipment = ['Diode laser', 'Microscope', 'XY stage', 'Timing controller']
        elif laser_power < 50 and spot_size_um > 1:
            complexity = AttackComplexity.HIGH
            cost_estimate = 100000  # Advanced laser system
            time_estimate = 40
            equipment = ['Pulsed laser system', 'High-resolution microscope', 
                        'Precision positioning system', 'Synchronization equipment']
        else:
            complexity = AttackComplexity.EXTREME
            cost_estimate = 500000  # State-of-the-art laser facility
            time_estimate = 80
            equipment = ['Femtosecond laser', 'Confocal microscope', 
                        'Nanometer positioning system', 'Environmental isolation']
        
        extracted_secrets = None
        if faulty_response != original_response:
            extracted_secrets = {
                'original_response': original_response,
                'faulty_response': faulty_response,
                'laser_power': laser_power,
                'spot_size': spot_size_um,
                'target_location': target_location,
                'power_density': power_density
            }
        
        return PhysicalAttackResult(
            attack_type='laser_fault_injection',
            attack_vector=AttackVector.SEMI_INVASIVE,
            complexity=complexity,
            success_probability=fault_probability,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=detection_prob,
            cost_estimate_usd=cost_estimate,
            time_estimate_hours=time_estimate,
            equipment_required=equipment
        )
    
    def _calculate_fault_probability(self, voltage: float, duration: float) -> float:
        """
        Calculate fault injection success probability.
        
        Parameters
        ----------
        voltage : float
            Fault voltage amplitude
        duration : float
            Fault duration
            
        Returns
        -------
        float
            Success probability (0.0 to 1.0)
        """
        # Simple model: higher voltage and longer duration increase success
        voltage_factor = min(1.0, voltage / 5.0)  # Normalize to 5V
        duration_factor = min(1.0, duration / 1e-6)  # Normalize to 1 microsecond
        precision_factor = self.precision
        
        return voltage_factor * duration_factor * precision_factor * 0.8
    
    def _assess_voltage_damage(self, voltage: float, duration: float) -> str:
        """
        Assess damage level from voltage fault injection.
        
        Parameters
        ----------
        voltage : float
            Fault voltage
        duration : float
            Fault duration
            
        Returns
        -------
        str
            Damage level
        """
        if voltage > 10.0 or duration > 1e-3:
            return "severe"
        elif voltage > 5.0 or duration > 1e-4:
            return "moderate"
        elif voltage > 2.0 or duration > 1e-5:
            return "minimal"
        else:
            return "none"

class InvasiveAttacker:
    """
    Invasive attack implementation for PUF analysis.
    Models attacks requiring physical access and modification.
    """
    
    def __init__(self, expertise_level: AttackComplexity = AttackComplexity.HIGH):
        """
        Initialize invasive attacker.
        
        Parameters
        ----------
        expertise_level : AttackComplexity
            Attacker expertise and available resources
        """
        self.expertise_level = expertise_level
        self.attack_history = []
    
    def decapsulation_attack(self, puf: BasePUF) -> PhysicalAttackResult:
        """
        Simulate package decapsulation for direct die access.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        # Decapsulation success depends on expertise and equipment
        success_probability = {
            AttackComplexity.LOW: 0.3,
            AttackComplexity.MEDIUM: 0.7,
            AttackComplexity.HIGH: 0.9,
            AttackComplexity.EXTREME: 0.98
        }.get(self.expertise_level, 0.5)
        
        rng = np.random.default_rng(42)
        
        if rng.random() < success_probability:
            # Successful decapsulation - can access die
            extracted_secrets = {
                'die_access': True,
                'layout_visible': True,
                'probe_access': True
            }
            
            # Additional analysis for different PUF types
            if isinstance(puf, ArbiterPUF):
                extracted_secrets['delay_path_visible'] = True
                extracted_secrets['estimated_parameters'] = "delay_analysis_possible"
            elif isinstance(puf, SRAMPUF):
                extracted_secrets['memory_cells_visible'] = True
                extracted_secrets['cell_biases'] = "direct_measurement_possible"
            elif isinstance(puf, RingOscillatorPUF):
                extracted_secrets['oscillator_rings_visible'] = True
                extracted_secrets['frequency_measurement'] = "direct_access_possible"
            
            damage_level = "moderate"  # Package destroyed but die intact
        else:
            extracted_secrets = None
            damage_level = "destructive"  # Failed decapsulation destroys device
        
        # Cost and time based on expertise level
        cost_estimates = {
            AttackComplexity.LOW: 5000,
            AttackComplexity.MEDIUM: 15000,
            AttackComplexity.HIGH: 50000,
            AttackComplexity.EXTREME: 200000
        }
        
        time_estimates = {
            AttackComplexity.LOW: 8,
            AttackComplexity.MEDIUM: 16,
            AttackComplexity.HIGH: 24,
            AttackComplexity.EXTREME: 40
        }
        
        equipment_lists = {
            AttackComplexity.LOW: ['Acid etching setup', 'Basic microscope', 'Safety equipment'],
            AttackComplexity.MEDIUM: ['Plasma etching system', 'Stereo microscope', 
                                    'Fume hood', 'Chemical safety'],
            AttackComplexity.HIGH: ['RIE etching system', 'High-resolution SEM',
                                  'Clean room facility', 'Advanced safety'],
            AttackComplexity.EXTREME: ['Ion beam milling', 'TEM preparation',
                                     'Class 100 clean room', 'Full safety suite']
        }
        
        return PhysicalAttackResult(
            attack_type='decapsulation',
            attack_vector=AttackVector.INVASIVE,
            complexity=self.expertise_level,
            success_probability=success_probability,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=1.0,  # Always detectable
            cost_estimate_usd=cost_estimates[self.expertise_level],
            time_estimate_hours=time_estimates[self.expertise_level],
            equipment_required=equipment_lists[self.expertise_level]
        )
    
    def microprobing_attack(self, puf: BasePUF, probe_locations: List[Tuple[float, float]]) -> PhysicalAttackResult:
        """
        Simulate microprobing attack for signal extraction.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        probe_locations : List[Tuple[float, float]]
            Probe contact locations on die (x, y coordinates in micrometers)
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        # Microprobing success depends on expertise and probe placement accuracy
        base_success = {
            AttackComplexity.LOW: 0.1,
            AttackComplexity.MEDIUM: 0.4,
            AttackComplexity.HIGH: 0.8,
            AttackComplexity.EXTREME: 0.95
        }.get(self.expertise_level, 0.3)
        
        # More probe locations increase chance of success
        location_bonus = min(0.3, len(probe_locations) * 0.05)
        success_probability = min(0.98, base_success + location_bonus)
        
        rng = np.random.default_rng(42)
        
        if rng.random() < success_probability:
            # Successful probing - can extract internal signals
            extracted_secrets = {
                'internal_signals': True,
                'probe_locations': probe_locations,
                'signal_traces': "acquired"
            }
            
            # PUF-specific signal extraction
            if isinstance(puf, ArbiterPUF):
                extracted_secrets['delay_measurements'] = "path_delays_measured"
                extracted_secrets['parameter_estimation'] = "feasible"
            elif isinstance(puf, SRAMPUF):
                extracted_secrets['cell_states'] = "individual_cells_readable"
                extracted_secrets['threshold_voltages'] = "measurable"
            
            damage_level = "minimal"  # Probe marks but functional
        else:
            extracted_secrets = None
            damage_level = "moderate"  # Probe damage affects function
        
        # Detection is likely due to probe marks
        detection_prob = 0.7 if self.expertise_level >= AttackComplexity.HIGH else 0.9
        
        cost_estimates = {
            AttackComplexity.LOW: 20000,
            AttackComplexity.MEDIUM: 75000,
            AttackComplexity.HIGH: 200000,
            AttackComplexity.EXTREME: 500000
        }
        
        time_estimates = {
            AttackComplexity.LOW: 24,
            AttackComplexity.MEDIUM: 40,
            AttackComplexity.HIGH: 80,
            AttackComplexity.EXTREME: 120
        }
        
        equipment = ['Probe station', 'Microprobes', 'High-resolution microscope',
                    'Signal analysis equipment', 'Vibration isolation']
        
        return PhysicalAttackResult(
            attack_type='microprobing',
            attack_vector=AttackVector.INVASIVE,
            complexity=self.expertise_level,
            success_probability=success_probability,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=detection_prob,
            cost_estimate_usd=cost_estimates[self.expertise_level],
            time_estimate_hours=time_estimates[self.expertise_level],
            equipment_required=equipment
        )

class SupplyChainAttacker:
    """
    Supply chain attack implementation for PUF evaluation.
    Models attacks during manufacturing, distribution, or deployment phases.
    """
    
    def __init__(self, access_level: str = 'distributor'):
        """
        Initialize supply chain attacker.
        
        Parameters
        ----------
        access_level : str
            Level of supply chain access ('manufacturer', 'distributor', 'integrator')
        """
        self.access_level = access_level
        self.insertion_history = []
    
    def hardware_trojan_insertion(self, puf: BasePUF, trojan_type: str = 'passive') -> PhysicalAttackResult:
        """
        Simulate hardware trojan insertion in PUF circuit.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        trojan_type : str
            Type of trojan ('passive', 'active', 'hybrid')
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        # Success probability depends on access level
        access_success = {
            'manufacturer': 0.95,  # Full design control
            'distributor': 0.3,    # Limited modification capability
            'integrator': 0.6      # PCB-level modifications possible
        }.get(self.access_level, 0.1)
        
        # Trojan complexity affects success
        trojan_complexity = {
            'passive': 0.9,    # Simple monitoring circuits
            'active': 0.6,     # Requires modification of PUF logic
            'hybrid': 0.4      # Complex interaction with PUF
        }.get(trojan_type, 0.5)
        
        success_probability = access_success * trojan_complexity
        
        rng = np.random.default_rng(42)
        
        if rng.random() < success_probability:
            # Successful trojan insertion
            extracted_secrets = {
                'trojan_inserted': True,
                'trojan_type': trojan_type,
                'access_level': self.access_level
            }
            
            if trojan_type == 'passive':
                # Passive trojan can monitor PUF responses
                extracted_secrets['monitoring_capability'] = True
                extracted_secrets['response_logging'] = "enabled"
                damage_level = "none"
                
            elif trojan_type == 'active':
                # Active trojan can modify PUF behavior
                extracted_secrets['modification_capability'] = True
                extracted_secrets['response_control'] = "possible"
                damage_level = "minimal"
                
            else:  # hybrid
                # Hybrid trojan combines monitoring and modification
                extracted_secrets['full_control'] = True
                extracted_secrets['stealth_mode'] = "enabled"
                damage_level = "minimal"
        else:
            extracted_secrets = None
            damage_level = "none"
        
        # Detection probability depends on trojan sophistication
        if self.access_level == 'manufacturer':
            detection_prob = 0.1  # Very difficult to detect insider threats
        elif trojan_type == 'passive':
            detection_prob = 0.2  # Passive trojans are harder to detect
        else:
            detection_prob = 0.4  # Active modifications more visible
        
        # Cost varies significantly by access level
        cost_estimates = {
            'manufacturer': 1000000,  # Insider threat cost
            'distributor': 500000,    # Supply chain infiltration
            'integrator': 50000       # PCB modification
        }
        
        time_estimates = {
            'manufacturer': 2000,     # Long-term insider operation
            'distributor': 200,       # Brief access window
            'integrator': 40          # PCB-level modification
        }
        
        if self.access_level == 'manufacturer':
            complexity = AttackComplexity.EXTREME
            equipment = ['Design tools', 'Fabrication access', 'Insider access']
        elif self.access_level == 'distributor':
            complexity = AttackComplexity.HIGH
            equipment = ['Rework station', 'Microscope', 'Custom hardware']
        else:
            complexity = AttackComplexity.MEDIUM
            equipment = ['PCB modification tools', 'Component placement', 'Testing equipment']
        
        return PhysicalAttackResult(
            attack_type='hardware_trojan',
            attack_vector=AttackVector.SUPPLY_CHAIN,
            complexity=complexity,
            success_probability=success_probability,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=detection_prob,
            cost_estimate_usd=cost_estimates[self.access_level],
            time_estimate_hours=time_estimates[self.access_level],
            equipment_required=equipment
        )

class ReverseEngineeringAttacker:
    """
    Reverse engineering attack implementation for PUF analysis.
    Models attempts to extract PUF structure and parameters.
    """
    
    def __init__(self, imaging_capability: AttackComplexity = AttackComplexity.MEDIUM):
        """
        Initialize reverse engineering attacker.
        
        Parameters
        ----------
        imaging_capability : AttackComplexity
            Available imaging and analysis capabilities
        """
        self.imaging_capability = imaging_capability
        self.analysis_results = []
    
    def circuit_imaging_attack(self, puf: BasePUF) -> PhysicalAttackResult:
        """
        Simulate circuit imaging for PUF reverse engineering.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
            
        Returns
        -------
        PhysicalAttackResult
            Attack result analysis
        """
        # Imaging success depends on capabilities
        imaging_success = {
            AttackComplexity.LOW: 0.2,
            AttackComplexity.MEDIUM: 0.6,
            AttackComplexity.HIGH: 0.9,
            AttackComplexity.EXTREME: 0.98
        }.get(self.imaging_capability, 0.3)
        
        rng = np.random.default_rng(42)
        
        if rng.random() < imaging_success:
            # Successful imaging - can extract circuit details
            extracted_secrets = {
                'circuit_layout': True,
                'imaging_successful': True
            }
            
            # PUF-specific reverse engineering
            if isinstance(puf, ArbiterPUF):
                extracted_secrets['delay_path_structure'] = "visible"
                extracted_secrets['parameter_estimation'] = "feasible"
                extracted_secrets['challenge_mapping'] = "analyzable"
                
            elif isinstance(puf, SRAMPUF):
                extracted_secrets['memory_cell_layout'] = "visible"
                extracted_secrets['cell_characteristics'] = "measurable"
                extracted_secrets['bias_distribution'] = "analyzable"
                
            elif isinstance(puf, RingOscillatorPUF):
                extracted_secrets['oscillator_structure'] = "visible"
                extracted_secrets['frequency_characteristics'] = "measurable"
                extracted_secrets['process_variations'] = "analyzable"
                
            elif isinstance(puf, ButterflyPUF):
                extracted_secrets['latch_structure'] = "visible"
                extracted_secrets['metastability_analysis'] = "possible"
                extracted_secrets['crosstalk_modeling'] = "feasible"
            
            damage_level = "none"  # Non-destructive imaging
        else:
            extracted_secrets = None
            damage_level = "none"
        
        # Detection probability for imaging equipment
        detection_prob = 0.05 if self.imaging_capability >= AttackComplexity.HIGH else 0.15
        
        # Cost and equipment based on imaging capability
        cost_estimates = {
            AttackComplexity.LOW: 10000,
            AttackComplexity.MEDIUM: 50000,
            AttackComplexity.HIGH: 200000,
            AttackComplexity.EXTREME: 1000000
        }
        
        time_estimates = {
            AttackComplexity.LOW: 40,
            AttackComplexity.MEDIUM: 80,
            AttackComplexity.HIGH: 160,
            AttackComplexity.EXTREME: 320
        }
        
        equipment_lists = {
            AttackComplexity.LOW: ['Optical microscope', 'Image analysis software'],
            AttackComplexity.MEDIUM: ['SEM', 'Layer removal tools', 'CAD reconstruction'],
            AttackComplexity.HIGH: ['High-resolution SEM', 'FIB', 'Advanced image processing'],
            AttackComplexity.EXTREME: ['Atomic force microscope', 'TEM', 'AI-assisted analysis']
        }
        
        return PhysicalAttackResult(
            attack_type='circuit_imaging',
            attack_vector=AttackVector.INVASIVE,
            complexity=self.imaging_capability,
            success_probability=imaging_success,
            extracted_secrets=extracted_secrets,
            damage_level=damage_level,
            detection_probability=detection_prob,
            cost_estimate_usd=cost_estimates[self.imaging_capability],
            time_estimate_hours=time_estimates[self.imaging_capability],
            equipment_required=equipment_lists[self.imaging_capability]
        )

class ComprehensivePhysicalAttacker:
    """
    Comprehensive physical attacker combining multiple attack vectors.
    Models sophisticated adversaries with diverse capabilities.
    """
    
    def __init__(self, threat_level: AttackComplexity = AttackComplexity.HIGH):
        """
        Initialize comprehensive physical attacker.
        
        Parameters
        ----------
        threat_level : AttackComplexity
            Overall threat level and available resources
        """
        self.threat_level = threat_level
        self.fault_attacker = FaultInjectionAttacker(precision=0.8)
        self.invasive_attacker = InvasiveAttacker(threat_level)
        self.supply_chain_attacker = SupplyChainAttacker('distributor')
        self.reverse_eng_attacker = ReverseEngineeringAttacker(threat_level)
        self.attack_results = []
    
    def comprehensive_physical_attack(self, puf: BasePUF) -> Dict[str, PhysicalAttackResult]:
        """
        Perform comprehensive physical attack evaluation.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
            
        Returns
        -------
        Dict[str, PhysicalAttackResult]
            Comprehensive attack results
        """
        results = {}
        
        # Generate test challenge
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        test_challenge = rng.integers(0, 2, size=n_stages)
        
        # Fault injection attacks
        results['voltage_fault'] = self.fault_attacker.inject_voltage_fault(
            puf, test_challenge, fault_voltage=3.0, fault_duration=1e-6)
        
        results['laser_fault'] = self.fault_attacker.inject_laser_fault(
            puf, test_challenge, laser_power=20.0, spot_size_um=2.0, 
            target_location=(100.0, 100.0))
        
        # Invasive attacks
        results['decapsulation'] = self.invasive_attacker.decapsulation_attack(puf)
        
        probe_locations = [(i * 10, 50) for i in range(5)]  # Sample probe locations
        results['microprobing'] = self.invasive_attacker.microprobing_attack(
            puf, probe_locations)
        
        # Supply chain attacks
        results['hardware_trojan'] = self.supply_chain_attacker.hardware_trojan_insertion(
            puf, trojan_type='hybrid')
        
        # Reverse engineering
        results['circuit_imaging'] = self.reverse_eng_attacker.circuit_imaging_attack(puf)
        
        self.attack_results = results
        return results
    
    def generate_threat_assessment(self, attack_results: Dict[str, PhysicalAttackResult]) -> Dict[str, Any]:
        """
        Generate comprehensive threat assessment from attack results.
        
        Parameters
        ----------
        attack_results : Dict[str, PhysicalAttackResult]
            Results from comprehensive attack
            
        Returns
        -------
        Dict[str, Any]
            Threat assessment report
        """
        # Analyze attack success rates
        successful_attacks = []
        total_cost = 0
        max_time = 0
        
        for attack_name, result in attack_results.items():
            if result.success_probability > 0.5:
                successful_attacks.append(attack_name)
            total_cost += result.cost_estimate_usd
            max_time = max(max_time, result.time_estimate_hours)
        
        # Determine overall threat level
        success_rate = len(successful_attacks) / len(attack_results)
        
        if success_rate >= 0.8:
            overall_threat = 'CRITICAL'
        elif success_rate >= 0.6:
            overall_threat = 'HIGH'
        elif success_rate >= 0.4:
            overall_threat = 'MEDIUM'
        else:
            overall_threat = 'LOW'
        
        # Identify highest risk attacks
        high_risk_attacks = []
        for attack_name, result in attack_results.items():
            if (result.success_probability > 0.7 and 
                result.detection_probability < 0.3):
                high_risk_attacks.append(attack_name)
        
        # Generate countermeasure recommendations
        countermeasures = self._generate_countermeasures(attack_results)
        
        return {
            'overall_threat_level': overall_threat,
            'success_rate': success_rate,
            'successful_attacks': successful_attacks,
            'high_risk_attacks': high_risk_attacks,
            'total_cost_estimate': total_cost,
            'max_time_estimate': max_time,
            'countermeasures': countermeasures,
            'detailed_results': attack_results
        }
    
    def _generate_countermeasures(self, attack_results: Dict[str, PhysicalAttackResult]) -> List[str]:
        """
        Generate countermeasure recommendations based on attack results.
        
        Parameters
        ----------
        attack_results : Dict[str, PhysicalAttackResult]
            Attack analysis results
            
        Returns
        -------
        List[str]
            Recommended countermeasures
        """
        countermeasures = []
        
        # Check for fault injection vulnerabilities
        if any(result.attack_type.endswith('fault_injection') and result.success_probability > 0.5
               for result in attack_results.values()):
            countermeasures.extend([
                'Implement voltage monitoring and brown-out detection',
                'Add temporal and spatial redundancy for fault detection',
                'Use error correction codes for response integrity',
                'Deploy environmental shielding against EM/laser attacks'
            ])
        
        # Check for invasive attack vulnerabilities
        if any(result.attack_vector == AttackVector.INVASIVE and result.success_probability > 0.5
               for result in attack_results.values()):
            countermeasures.extend([
                'Implement tamper-evident packaging',
                'Add active tamper detection circuits',
                'Use security meshes and sensors',
                'Deploy self-destruct mechanisms for critical applications'
            ])
        
        # Check for supply chain vulnerabilities
        if any(result.attack_vector == AttackVector.SUPPLY_CHAIN and result.success_probability > 0.3
               for result in attack_results.values()):
            countermeasures.extend([
                'Implement comprehensive supply chain security',
                'Use trusted foundries and assembly facilities',
                'Deploy hardware trojan detection mechanisms',
                'Perform incoming inspection and verification'
            ])
        
        return countermeasures

if __name__ == "__main__":
    print("=== PPET Physical Attack Framework ===")
    print("Testing physical attack vectors for military PUF evaluation\
")
    
    # Test with different PUF types
    from .puf_models import ArbiterPUF, SRAMPUF
    
    print("--- Arbiter PUF Physical Attack Analysis ---")
    arbiter_puf = ArbiterPUF(n_stages=64, seed=42)
    
    # Comprehensive physical attack
    physical_attacker = ComprehensivePhysicalAttacker(AttackComplexity.HIGH)
    attack_results = physical_attacker.comprehensive_physical_attack(arbiter_puf)
    
    # Generate threat assessment
    threat_assessment = physical_attacker.generate_threat_assessment(attack_results)
    
    print(f"Overall threat level: {threat_assessment['overall_threat_level']}")
    print(f"Attack success rate: {threat_assessment['success_rate']:.2f}")
    print(f"Total estimated cost: ${threat_assessment['total_cost_estimate']:,}")
    print(f"Maximum time estimate: {threat_assessment['max_time_estimate']} hours")
    
    print("\
Successful attacks:")
    for attack in threat_assessment['successful_attacks']:
        result = attack_results[attack]
        print(f"  - {attack}: {result.success_probability:.2f} probability, "
              f"${result.cost_estimate_usd:,}, {result.time_estimate_hours}h")
    
    print("\
High-risk attacks (high success, low detection):")
    for attack in threat_assessment['high_risk_attacks']:
        result = attack_results[attack]
        print(f"  - {attack}: {result.success_probability:.2f} success, "
              f"{result.detection_probability:.2f} detection")
    
    print("\
Recommended countermeasures:")
    for i, countermeasure in enumerate(threat_assessment['countermeasures'], 1):
        print(f"  {i}. {countermeasure}")
    
    print("\
--- SRAM PUF Physical Attack Analysis ---")
    sram_puf = SRAMPUF(n_cells=64, seed=42)
    
    # Focused fault injection analysis
    fault_attacker = FaultInjectionAttacker('laser', precision=0.9)
    test_challenge = np.random.randint(0, 2, 64)
    
    laser_result = fault_attacker.inject_laser_fault(
        sram_puf, test_challenge, laser_power=15.0, spot_size_um=1.5, 
        target_location=(50.0, 75.0))
    
    print(f"Laser fault injection success: {laser_result.success_probability:.2f}")
    print(f"Damage level: {laser_result.damage_level}")
    print(f"Detection probability: {laser_result.detection_probability:.2f}")
    print(f"Attack complexity: {laser_result.complexity.name}")
    
    print("\
=== Physical attack framework testing complete ===")
    print("PPET ready for comprehensive physical security evaluation.")