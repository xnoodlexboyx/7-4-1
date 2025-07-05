# PPET Thesis

**A Defense-Oriented Software Framework for PUF Emulation and Analysis**

## Overview
This repository provides a modular Python framework to:
- Simulate an Arbiter PUF under nominal and stressed conditions  
- Model temperature‐induced noise and apply logistic‐regression attacks  
- Simulate simple ECC (BCH-style) error-correction outcomes  
- Generate all plots and data needed for a master’s-level thesis

## Features
- **PUF Models**: Linear additive delay Arbiter PUF  
- **Stressors**: Temperature perturbation model (± noise)  
- **Attacks**: MLAttacker wrapper around scikit-learn’s LogisticRegression  
- **Analysis**: Bit error rate, uniqueness, ECC simulation, and plots  
- **Extensible**: Hooks for voltage, aging, additional PUF types

## Requirements
- Python 3.8+  
- NumPy  
- SciPy  
- scikit-learn  
- Matplotlib  
- Seaborn  

## Installation
```bash
git clone <repo_url>
cd ppet-thesis
pip install -r requirements.txt
# ppet-thesis
