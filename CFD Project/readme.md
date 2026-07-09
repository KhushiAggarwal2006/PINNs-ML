# Physics-Informed Neural Networks for Incompressible Navier–Stokes Equations

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **steady two-dimensional incompressible Navier–Stokes equations** for the classical **lid-driven cavity flow** problem.

Unlike conventional Computational Fluid Dynamics (CFD) methods that discretize the governing equations over a computational mesh, the PINN directly learns the velocity and pressure fields by minimizing the residuals of the governing equations while enforcing the prescribed boundary conditions.

The project is implemented using **PyTorch** and benchmarked against reference solutions generated using **OpenFOAM**.

---

## Objectives

- Develop a Physics-Informed Neural Network (PINN) for solving the steady incompressible Navier–Stokes equations.
- Enforce continuity and momentum equations through automatic differentiation.
- Improve PINN convergence using **Fourier Feature Embedding** and **Gradient-Norm Balancing**.
- Benchmark the predicted velocity and pressure fields against conventional OpenFOAM CFD solutions.
- Perform sparse sensor data assimilation to reconstruct complete flow fields from limited velocity observations.
- Recover the pressure field without direct pressure supervision.

---

## Problem Statement

The governing equations for steady incompressible flow are

### Continuity

\[
\nabla \cdot \mathbf{u}=0
\]

### Momentum

\[
(\mathbf{u}\cdot\nabla)\mathbf{u}
=
-\nabla p
+
\nu\nabla^2\mathbf{u}
\]

where

- **u, v** : velocity components
- **p** : pressure
- **ν** : kinematic viscosity

The neural network learns

\[
(x,y)\rightarrow(u,v,p)
\]

using only the governing equations and boundary conditions.

---

## Methodology

The project consists of the following stages:

1. Generate collocation points inside the computational domain.
2. Construct a fully-connected neural network using PyTorch.
3. Compute first- and second-order derivatives using automatic differentiation.
4. Formulate the Navier–Stokes residuals.
5. Compute physics and boundary-condition losses.
6. Train the network using gradient-based optimization.
7. Compare the predicted solution with OpenFOAM simulations.

---

## Physics Loss

The total loss function is

\[
L =
L_{continuity}
+
L_{momentum}
+
L_{boundary}
\]

where

- Continuity loss enforces incompressibility.
- Momentum loss enforces the Navier–Stokes equations.
- Boundary loss satisfies the lid-driven cavity boundary conditions.

---

## Sparse Sensor Data Assimilation

The project further extends the PINN framework by incorporating sparse velocity measurements into the loss function.

Instead of providing the complete velocity field, only a limited number of sensor observations are supplied to the network. The PINN combines these observations with the governing equations to reconstruct the complete velocity field while simultaneously recovering the pressure field.

The total loss becomes

\[
L=
L_{physics}
+
L_{boundary}
+
L_{data}
\]

where

- **Lphysics** represents the Navier–Stokes residuals.
- **Lboundary** enforces the wall boundary conditions.
- **Ldata** minimizes the error between measured and predicted sensor values.

Experiments will be conducted using different numbers of sensor observations to study reconstruction accuracy.

---

## Pressure Recovery

Pressure measurements are not provided during training.

Instead, the PINN predicts pressure as part of the network output and learns a physically consistent pressure field by satisfying the Navier–Stokes momentum equations.

The recovered pressure field will be compared against OpenFOAM reference solutions.

---

## Techniques Used

- Physics-Informed Neural Networks (PINNs)
- PyTorch
- Automatic Differentiation
- Fourier Feature Embedding
- Gradient-Norm Balancing
- Sparse Sensor Data Assimilation
- OpenFOAM Benchmarking

---

## Software Requirements

- Python 3.x
- PyTorch
- DeepXDE
- NumPy
- Matplotlib
- OpenFOAM
- ParaView

---

## Expected Results

- Accurate prediction of the velocity field.
- Pressure recovery without pressure measurements.
- Satisfaction of the incompressibility constraint.
- Benchmark agreement with OpenFOAM.
- Improved convergence using Fourier Feature Embedding and Gradient-Norm Balancing.
- Improved reconstruction accuracy with increasing sensor density.

---

## References

1. S. Cai, Z. Mao, Z. Wang, M. Yin, and G. E. Karniadakis, *Physics-informed neural networks (PINNs) for fluid mechanics: A review*, arXiv:2105.09506, 2021.

2. X. Jin, S. Cai, H. Li, and G. E. Karniadakis, *NSFnets (Navier–Stokes flow nets): Physics-informed neural networks for the incompressible Navier–Stokes equations*, Journal of Computational Physics, 426, 109951, 2021.

3. M. Raissi, A. Yazdani, and G. E. Karniadakis, *Hidden Fluid Mechanics: A Navier–Stokes Informed Deep Learning Framework for Assimilating Flow Visualization Data*, arXiv:1808.04327, 2018.
