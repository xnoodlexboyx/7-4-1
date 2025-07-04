# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running Experiments
```bash
# Run main experiment with default parameters
cd ppet-thesis && python main_experiment.py

# Force regeneration of challenge/response data
cd ppet-thesis && python main_experiment.py --regenerate

# Run individual components
cd ppet-thesis && python puf_models.py    # Demo PUF evaluation
cd ppet-thesis && python stressors.py     # Test temperature effects
cd ppet-thesis && python attacks.py       # ML attacker demo
cd ppet-thesis && python analysis.py      # Analysis utilities test
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_puf_models.py -v

# Run with coverage
python -m pytest tests/ --cov=ppet-thesis --cov-report=html
```

### Dependencies
```bash
# Install requirements
pip install -r ppet-thesis/requirements.xt

# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r ppet-thesis/requirements.xt
```

## Architecture Overview

### Core Framework Structure
The codebase implements a Physical Unclonable Function (PUF) emulation and analysis framework focused on security evaluation under environmental stress:

**Data Flow:**
1. **PUF Models** (`puf_models.py`) - Generate challenge-response pairs using linear additive delay model
2. **Stressors** (`stressors.py`) - Apply environmental perturbations (temperature, voltage, aging)
3. **Attacks** (`attacks.py`) - ML-based attacks using logistic regression on parity features
4. **Analysis** (`analysis.py`) - Reliability metrics, ECC simulation, and visualization
5. **Main Experiment** (`main_experiment.py`) - Orchestrates temperature sweep experiments

### Key Design Patterns

**Abstract Base Classes:**
- `BasePUF` in `puf_models.py` defines the evaluation interface
- All stressor functions return new PUF instances (immutable design)

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

The framework is designed for defense-oriented security research, modeling how environmental conditions affect both PUF reliability and vulnerability to ML attacks.