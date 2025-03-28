# SOFA Neural Modes

A framework that combines SOFA simulations with neural networks to learn nonlinear deformation modes for fast mechanical simulations.

## Overview

The main scripts are:

1. **Linear Mode Visualization** (`sofa_linear_modes_viz.py`)
   - Computes and visualizes linear modes directly in SOFA
   - Exports mass and stiffness matrices for further processing
   - Provides real-time visualization of mode shapes

2. **Neural Network Training** (`train.py`)
   - Automatically generates linear modes using FEniCSx.
   - Trains a neural network to capture nonlinear deformation modes.
   - Uses the exported matrices from SOFA to compute linear basis
   - Supports different hyperelastic energy formulations
   - Includes visualization tools for latent space exploration

3. **Model Validation** (`validate_twist.py`)
   - Validates the trained model against full FEM simulation
   - Implements twisting beam test case with dynamic loading
   - Provides side-by-side comparison visualization
   - Computes error metrics and energy preservation

## Quick Start for Linear Modes Visualization


1. Generate linear modes:
```bash
python sofa_linear_modes_viz.py --gui
```
## Quick Start for Neural Network Training (here linear modes are automatically generated)

1. Train the neural model:
```bash
python train.py 
```

2. Validate the model:
```bash
python validate_twist.py 
```

## Configuration

The model behavior can be controlled through `configs/default.yaml`:

```yaml
material:
  youngs_modulus: 1000.0
  poissons_ratio: 0.3

model:
  latent_dim: 8
  hid_layers: 2
  hid_dim: 64

training:
  num_epochs: 1000
  learning_rate: 0.001
```

## Results Visualization

The framework includes visualization tools for:
- Linear mode shapes in SOFA
- Neural network training progress
- Latent space exploration
- Dynamic validation comparison
- Energy preservation analysis

Visualization results are saved to:
- Linear modes: `modal_data/`
- Training progress: `tensorboard/`
- Validation results: `validation_results/`
