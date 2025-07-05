# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPET (Physical Unclonable Function Emulation and Analysis) is a defense-oriented PUF emulation framework designed for security evaluation under environmental stress conditions. The framework focuses on modeling how environmental stressors affect both PUF reliability and vulnerability to ML attacks.

## Common Commands

### Current Implementation
```bash
# Run main experiment with default parameters
cd ppet-thesis && python scripts/main_experiment.py

# Force regeneration of challenge/response data
cd ppet-thesis && python scripts/main_experiment.py --regenerate

# Run individual components
cd ppet-thesis && python ppet/puf_models.py    # Demo PUF evaluation
cd ppet-thesis && python ppet/stressors.py     # Test temperature effects
cd ppet-thesis && python ppet/attacks.py       # ML attacker demo
cd ppet-thesis && python ppet/analysis.py      # Analysis utilities test

# Generate data at repo root level
python generate_data.py --n_chal 5000 --n_stages 32

# Validate new PUF implementations
python validate_new_pufs.py
```

### Testing
```bash
# Run all tests (from ppet-thesis directory)
cd ppet-thesis && python -m pytest tests/

# Run specific test file
cd ppet-thesis && python -m pytest tests/test_puf_models.py -v

# Run with coverage
cd ppet-thesis && python -m pytest tests/ --cov=ppet --cov-report=html

# Alternative: Run validation without pytest
python validate_new_pufs.py
```

### Dependencies
```bash
# Install requirements
pip install -r ppet-thesis/requirements.txt

# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r ppet-thesis/requirements.txt

# Alternative: Install as package
cd ppet-thesis && pip install -e .
```

## Architecture Overview

### Core Framework Structure
The codebase implements a Physical Unclonable Function (PUF) emulation and analysis framework focused on security evaluation under environmental stress. The repository has a dual-level structure:

**Repository Structure:**
- `/ppet-thesis/` - Main framework package with installable setup
- `/ppet-thesis/ppet/` - Core PUF implementation modules
- `/ppet-thesis/scripts/` - Main experiment orchestration
- `/ppet-thesis/tests/` - Unit tests for core modules  
- Root level utilities - `generate_data.py`, `validate_new_pufs.py` for standalone operations

**Data Flow:**
1. **PUF Models** (`ppet/puf_models.py`) - Multiple PUF types: Arbiter, SRAM, Ring Oscillator, Butterfly
2. **Stressors** (`ppet/stressors.py`) - Apply environmental perturbations (temperature, voltage, aging)
3. **Attacks** (`ppet/attacks.py`) - ML-based attacks using logistic regression on parity features
4. **Analysis** (`ppet/analysis.py`) - Reliability metrics, ECC simulation, and visualization
5. **Main Experiment** (`scripts/main_experiment.py`) - Orchestrates temperature sweep experiments

### Key Design Patterns

**Abstract Base Classes:**
- `BasePUF` in `ppet/puf_models.py` defines the evaluation interface for all PUF types
- All stressor functions return new PUF instances (immutable design)
- Multiple PUF implementations: ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF

**Feature Engineering:**
- ML attacks use parity transform: `Φ_i(C) = Π_{j=i}^n (1 - 2c_j)` 
- Same transform used in both `ArbiterPUF._transform_challenge()` and `MLAttacker.feature_map()`

**Temperature Modeling:**
- Linear perturbation: `W_stressed = W_nominal * (1 + k_T*(T_current - T_nominal)) + noise`
- Default parameters: `k_T = 0.0005`, `sigma_noise = 0.01`

### Experiment Configuration
- Default temperature range: -20°C to 100°C (military spec)
- Challenge count: 10,000 CRPs
- PUF stages: 64-bit
- ECC capability: t=4 (BCH-style)
- Seeds: Reproducible across runs (seed=123 for PUF, seed=42 for challenges)

### Output Artifacts
- `data/challenges.npy` - Challenge vectors
- `data/responses_golden.npy` - Nominal temperature responses  
- `data/results.json` - Experiment metrics
- `figures/` - Temperature vs. reliability/attack accuracy plots

### Testing Strategy
Tests verify mathematical correctness of core algorithms:
- PUF evaluation consistency
- Stressor parameter effects
- Attack accuracy on known data
- ECC simulation edge cases

## Framework Extension Points

The current implementation provides a solid foundation but requires expansion for thesis-level scope:

### Planned Defense Applications
- Satellite communication PUF simulation under radiation stress
- Drone swarm authentication protocols
- Battlefield IoT device verification
- Supply chain hardware integrity verification

### Advanced Attack Modeling
- Side-channel attacks (power analysis, timing attacks, EM emanation)
- Enhanced ML attacks (CNN-based, CMA-ES optimization)
- Physical tampering scenarios (fault injection, laser manipulation)

### Current PUF Implementations

The framework now includes multiple PUF architectures for comprehensive defense evaluation:

**Implemented PUF Types:**
- **Arbiter PUF** - Linear additive delay model (reference implementation)
- **SRAM PUF** - Threshold voltage variations with radiation hardening parameters
- **Ring Oscillator PUF** - Frequency variations with EMI resistance modeling
- **Butterfly PUF** - Metastability resolution with crosstalk resistance

**Defense-Specific Features:**
- Radiation hardening parameters for satellite/aerospace applications
- EMI resistance modeling for military electronics
- Low-power operation modes for energy-constrained devices
- Crosstalk resistance for high-density military systems

### Enhanced Visualization
- 3D threat surface modeling
- Interactive defense dashboards
- Statistical analysis with Seaborn
- Bit-aliasing heatmaps

The framework is designed for defense-oriented security research, modeling how environmental conditions affect both PUF reliability and vulnerability to ML attacks.

