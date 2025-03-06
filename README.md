# SOFA Neural Modes

A computational framework that connects SOFA simulations with neural network-based reduced order models for fast and accurate mechanical simulations.

## Project Overview

SOFA Neural Modes creates a seamless workflow between SOFA (Simulation Open Framework Architecture) and FEniCS/DOLFINx-based neural networks for mechanical simulations. The framework exports mass and stiffness matrices from SOFA, computes linear modes in FEniCS, and trains neural networks to predict nonlinear deformations.

## Key Features

Matrix Export: Generate and export mass and stiffness matrices from SOFA simulations with metadata
Seamless Transfer: Automatic loading of matrices between SOFA and FEniCS
Linear Mode Computation: Efficient computation of linear modes using SLEPc eigensolvers
Neural Network Training: Train models with different hyperelastic energy formulations
Advanced Visualization: Tools for visualizing deformations, modes, and stress fields
Multiple Energy Models: Support for Neo-Hookean, Saint Venant-Kirchhoff, and other material models

## Installation

### Prerequisites

SOFA (v21.12 or newer)
FEniCS/DOLFINx
PyTorch (1.10+)
PETSc and SLEPc
PyVista for visualization
GMSH (optional, for mesh generation)

### Setup

Clone the repository:

Install Python dependencies:

Ensure SOFA is properly installed and environment variables are set

## Workflow Usage

1. Generate Matrices in SOFA
Run the SOFA simulation to generate matrices:

This creates mass and stiffness matrices in the matrices directory with a timestamp.

2. Train Neural Model
Train a neural network model using the generated matrices:

To use a specific matrix set, provide the timestamp:

3. Validate Model

Validate the trained model with dynamic simulations:

## Configuration

Modify default.yaml to set:

Material properties (Young's modulus, Poisson ratio)
Neural network architecture (hidden layers, dimensions)
Training parameters (epochs, learning rate)
Mesh file paths
Physics parameters (gravity, etc.)

## Visualization Examples

The framework includes tools for visualizing:

Linear modes
Neural mode predictions
Latent space exploration
Stress fields

