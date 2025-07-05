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

__all__ = [
    'BasePUF', 'ArbiterPUF', 'SRAMPUF', 'RingOscillatorPUF', 'ButterflyPUF',
    'apply_temperature', 'apply_voltage', 'apply_aging', 'apply_radiation', 'apply_emi',
    'MLAttacker', 'CNNAttacker', 'AdversarialAttacker',
    'bit_error_rate', 'uniqueness', 'simulate_ecc',
    'generate_all_thesis_plots', 'plot_bit_aliasing_heatmap',
    'generate_statistical_suite'
]