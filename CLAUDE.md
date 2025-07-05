# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPET (Physical Unclonable Function Emulation and Analysis) is a defense-oriented PUF emulation framework designed for security evaluation under environmental stress conditions. The framework focuses on modeling how environmental stressors affect both PUF reliability and vulnerability to ML attacks.

## Common Commands

### Installation & Setup
```bash
# Install basic requirements
cd ppet-thesis && pip install -r requirements.txt

# Install as editable package with all features
cd ppet-thesis && pip install -e ".[full,test]"

# Alternative: Install specific feature sets
pip install -e ".[full]"    # All visualization dependencies
pip install -e ".[test]"    # All testing dependencies
```

### Core Experiment Execution
```bash
# Run main experiment with comprehensive thesis visualizations
cd ppet-thesis && python scripts/main_experiment.py

# Force regeneration of challenge/response data
cd ppet-thesis && python scripts/main_experiment.py --regenerate

# Run individual components for testing
cd ppet-thesis && python ppet/puf_models.py    # Demo PUF evaluation
cd ppet-thesis && python ppet/stressors.py     # Test temperature effects
cd ppet-thesis && python ppet/attacks.py       # ML attacker demo
cd ppet-thesis && python ppet/analysis.py      # Analysis utilities test

# Generate data at repo root level
python generate_data.py --n_chal 5000 --n_stages 32

# Validate all PUF implementations
python validate_new_pufs.py
```

### Thesis-Quality Visualizations (All Commands Work)
```bash
# Generate comprehensive visualization suite
cd ppet-thesis && python ppet/visualization.py --output-dir figures/

# Generate bit-aliasing security analysis for specific PUF types
cd ppet-thesis && python ppet/bit_analysis.py --puf-type ArbiterPUF --n-instances 30
cd ppet-thesis && python ppet/bit_analysis.py --puf-type SRAMPUF --n-instances 20

# Generate statistical analysis plots
cd ppet-thesis && python ppet/statistical_plots.py --output-dir figures/stats/

# Generate defense operations dashboard
cd ppet-thesis && python ppet/defense_dashboard.py --format png --dpi 300

# Use command-line help for all options
cd ppet-thesis && python ppet/visualization.py --help
cd ppet-thesis && python ppet/bit_analysis.py --help
```

### Testing & Quality Assurance
```bash
# Run all tests
cd ppet-thesis && python -m pytest tests/

# Run specific test file with verbose output
cd ppet-thesis && python -m pytest tests/test_puf_models.py -v

# Run tests with coverage reporting
cd ppet-thesis && python -m pytest tests/ --cov=ppet --cov-report=html

# Run tests in parallel for speed
cd ppet-thesis && python -m pytest tests/ -n auto

# Run specific test method
cd ppet-thesis && python -m pytest tests/test_puf_models.py::test_arbiter_puf_eval -v
```

### Package Entry Points (After Installation)
```bash
# These work after: pip install -e ".[full]"
ppet-experiment              # Main experiment runner
ppet-visualize              # Comprehensive visualization suite
ppet-bit-analysis           # Bit-aliasing security analysis
ppet-statistical-plots      # Statistical analysis plots
ppet-defense-dashboard      # Defense operations dashboard
```

## Architecture Overview

### Core Framework Structure
The repository has a dual-level structure optimized for both development and thesis presentation:

**Repository Structure:**
- `/ppet-thesis/` - Main framework package with installable setup
- `/ppet-thesis/ppet/` - Core PUF implementation modules (4 PUF types, attacks, analysis)
- `/ppet-thesis/scripts/` - Main experiment orchestration with thesis visualization generation
- `/ppet-thesis/tests/` - Comprehensive unit tests for all core modules
- Root level utilities - `generate_data.py`, `validate_new_pufs.py` for standalone operations

**Data Flow Architecture:**
1. **PUF Models** (`ppet/puf_models.py`) - Four PUF architectures: Arbiter, SRAM, Ring Oscillator, Butterfly
2. **Environmental Stressors** (`ppet/stressors.py`) - Temperature, voltage, aging perturbations with realistic modeling
3. **Attack Framework** (`ppet/attacks.py`) - ML attacks (logistic regression, CNN, adversarial), side-channel attacks, physical attacks
4. **Analysis Engine** (`ppet/analysis.py`) - Reliability metrics, ECC simulation, uniqueness calculations
5. **Main Experiment** (`scripts/main_experiment.py`) - Orchestrates multi-PUF temperature sweeps with automatic thesis visualization generation

### Advanced Attack Modeling

**Multi-Vector Attack Framework:**
- **ML Attacks**: Logistic regression with parity features, CNN-based attacks, adversarial ML
- **Side-Channel Attacks**: Power analysis (DPA, CPA), timing attacks, EM emanation analysis
- **Physical Attacks**: Fault injection (voltage, laser), invasive attacks, supply chain tampering

**Defense Scenarios Implementation:**
- **Satellite Communications**: Radiation-hardened SRAM PUFs with cosmic ray modeling
- **Drone Authentication**: Ring oscillator PUFs with EMI resistance for RF-heavy environments
- **IoT Field Deployment**: Butterfly PUFs optimized for low-power, high-density scenarios
- **Supply Chain Integrity**: Multi-architecture PUF verification with tamper detection

### Key Design Patterns

**Immutable PUF Design:**
- `BasePUF` abstract base class defines evaluation interface for all PUF types
- All stressor functions return new PUF instances rather than modifying in-place
- Ensures reproducible experiments and prevents state corruption

**Feature Engineering Consistency:**
- ML attacks use parity transform: `Φ_i(C) = Π_{j=i}^n (1 - 2c_j)`
- Same transform implemented in both `ArbiterPUF._transform_challenge()` and `MLAttacker.feature_map()`
- Ensures attack-defense parity and realistic threat modeling

**Temperature Modeling (Validated):**
- Linear perturbation: `W_stressed = W_nominal * (1 + k_T*(T_current - T_nominal)) + noise`
- Default parameters: `k_T = 0.0005`, `sigma_noise = 0.01` (based on literature)
- Temperature range: -20°C to 100°C (military specification MIL-STD-810)

### Comprehensive Visualization Suite

The framework includes five categories of thesis-quality visualizations:

**1. Multi-PUF Architecture Comparison** (`ppet/visualization.py`)
- 2x2 subplot comparison across all metrics: BER, Attack Accuracy, Uniqueness, ECC Failure
- 95% confidence intervals, professional styling, 300 DPI publication quality

**2. Bit-Aliasing Security Analysis** (`ppet/bit_analysis.py`)
- Correlation heatmaps with hierarchical clustering dendrograms
- Per-bit entropy analysis and statistical significance testing
- Security level assessment with clear recommendations

**3. 3D Defense Threat Surface** (`ppet/visualization.py`)
- Interactive Plotly visualizations (temperature vs voltage vs attack success)
- Critical threshold planes, contour projections, hover tooltips
- Supports both HTML (interactive) and PNG (static) export

**4. Statistical Analysis Suite** (`ppet/statistical_plots.py`)
- Violin plots, box plots with outlier detection, correlation matrices
- ROC curves for ML attack performance, probability density analysis
- ANOVA testing, normality tests, publication-ready formatting

**5. Defense Operations Dashboard** (`ppet/defense_dashboard.py`)
- Military-style real-time monitoring with dark theme
- Mission timeline, threat gauges, environmental stress indicators
- Attack probability tracking with countermeasure effectiveness

### Experiment Configuration & Reproducibility

**Default Experimental Parameters:**
- Temperature range: -20°C to 100°C (6 points for thesis experiments)
- Challenge count: 10,000 CRPs per experiment
- PUF stages: 64-bit challenges (industry standard)
- ECC capability: t=4 error correction (BCH-style)
- Reproducible seeds: PUF=123, challenges=42, analysis=varied per trial

**Output Structure:**
```
figures/
├── multi_puf/           # Architecture comparison plots
├── bit_analysis/        # Security analysis per PUF type
├── threat_surface/      # 3D threat modeling (HTML + PNG)
├── statistical/         # Publication-quality statistical plots
└── dashboard/           # Defense operations monitoring
```

**Data Artifacts:**
- `data/challenges.npy` - Challenge vectors (reproducible with seed=42)
- `data/responses_golden.npy` - Nominal temperature responses
- `data/results.json` - Complete experimental metrics with visualization summary

### Testing Strategy & Quality Assurance

**Comprehensive Test Coverage:**
- **Unit Tests**: Mathematical correctness of PUF evaluation, stressor effects, attack accuracy
- **Integration Tests**: End-to-end experiment pipeline validation
- **Regression Tests**: Ensure visualization generation doesn't break core functionality
- **Performance Tests**: Memory usage and computation time for large-scale experiments

**Error Handling & Graceful Degradation:**
- Visualization modules work with or without optional dependencies (plotly, seaborn, scipy)
- Fallback modes provide simplified analysis when advanced packages unavailable
- Comprehensive error messages guide users to install missing dependencies

### Framework Extension Points & Future Work

**Current PUF Implementations (Fully Functional):**
- **Arbiter PUF**: Linear additive delay model (reference implementation)
- **SRAM PUF**: Threshold voltage variations with radiation hardening parameters
- **Ring Oscillator PUF**: Frequency variations with EMI resistance modeling
- **Butterfly PUF**: Metastability resolution with crosstalk resistance

**Defense-Specific Features:**
- Radiation hardening parameters for satellite/aerospace applications
- EMI resistance modeling for military electronics environments
- Low-power operation modes for energy-constrained IoT devices
- Crosstalk resistance for high-density military systems

**Research Extensions Ready for Implementation:**
- Hybrid/cascaded PUF designs combining multiple architectures
- Enhanced ML attacks using deep learning and optimization techniques
- Advanced side-channel attack modeling with real hardware measurements
- Supply chain integrity verification with blockchain integration

The framework is designed for defense-oriented security research, providing comprehensive tools for modeling how environmental conditions affect both PUF reliability and vulnerability to sophisticated attacks.

## Important Development Notes

**Dependency Management:**
- `requirements.txt` contains all dependencies including optional visualization packages
- `setup.py` uses `extras_require` for clean separation of core vs. full features
- All visualization modules include graceful fallbacks for missing optional dependencies

**Command Execution:**
- All documented commands have been tested and work correctly
- Standalone script execution handles relative import issues automatically
- Entry point commands work after package installation with `pip install -e ".[full]"`

**Performance Considerations:**
- Main experiment generates comprehensive data but can be memory-intensive for large parameter sweeps
- Visualization generation is automatic but can be disabled by modifying `scripts/main_experiment.py`
- Parallel testing available with `pytest -n auto` for faster development cycles