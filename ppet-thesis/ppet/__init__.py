"""
PPET - Physical Unclonable Function Emulation and Analysis Framework
Defense-oriented PUF emulation for military security applications.
"""

__version__ = "0.1.0"
__author__ = "PPET Development Team"

# Core PUF models
from .puf_models import BasePUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF

# Environmental stressors
from .stressors import apply_temperature, apply_voltage, apply_aging, apply_radiation, apply_emi

# Attack framework
from .attacks import MLAttacker, CNNAttacker, AdversarialAttacker

# Analysis utilities
from .analysis import bit_error_rate, uniqueness, simulate_ecc

# Visualization suite
from .visualization import generate_all_thesis_plots
from .bit_analysis import plot_bit_aliasing_heatmap
from .statistical_plots import generate_statistical_suite
from .defense_dashboard import create_defense_dashboard, generate_military_compliance_report

# Military analysis modules (with graceful import handling)
try:
    from .military_scenarios import MilitaryScenarioSimulator
    HAS_MILITARY_SCENARIOS = True
except ImportError:
    HAS_MILITARY_SCENARIOS = False

try:
    from .security_metrics import SecurityMetricsAnalyzer, SecurityClearanceLevel
    HAS_SECURITY_METRICS = True
except ImportError:
    HAS_SECURITY_METRICS = False

try:
    # Side channel module temporarily disabled due to syntax issues
    # from .side_channel import MultiChannelAttacker  
    HAS_SIDE_CHANNEL = False
except ImportError:
    HAS_SIDE_CHANNEL = False

try:
    from .physical_attacks import ComprehensivePhysicalAttacker, AttackComplexity
    HAS_PHYSICAL_ATTACKS = True
except ImportError:
    HAS_PHYSICAL_ATTACKS = False

__all__ = [
    'BasePUF', 'ArbiterPUF', 'SRAMPUF', 'RingOscillatorPUF', 'ButterflyPUF',
    'apply_temperature', 'apply_voltage', 'apply_aging', 'apply_radiation', 'apply_emi',
    'MLAttacker', 'CNNAttacker', 'AdversarialAttacker',
    'bit_error_rate', 'uniqueness', 'simulate_ecc',
    'generate_all_thesis_plots', 'plot_bit_aliasing_heatmap',
    'generate_statistical_suite', 'create_defense_dashboard', 
    'generate_military_compliance_report'
]

# Add optional exports if available
if HAS_MILITARY_SCENARIOS:
    __all__.append('MilitaryScenarioSimulator')

if HAS_SECURITY_METRICS:
    __all__.extend(['SecurityMetricsAnalyzer', 'SecurityClearanceLevel'])

if HAS_SIDE_CHANNEL:
    __all__.append('MultiChannelAttacker')

if HAS_PHYSICAL_ATTACKS:
    __all__.extend(['ComprehensivePhysicalAttacker', 'AttackComplexity'])