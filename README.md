# PINN surrogate model for the Heston stochastic volatility system

This repository contains a research-oriented implementation of a
physics-informed neural network (PINN) used as a surrogate model for
the Heston stochastic volatility model.

The goal of this work is to replace repeated numerical solution of the
Heston pricing equation with a learned approximation that is fast,
differentiable, and suitable for calibration and optimization tasks.

The emphasis is on methodology and modeling rather than production
performance.

The Heston model describes asset price dynamics with stochastic
volatility and leads to a partial differential equation for option
prices. Solving this equation repeatedly during parameter calibration
is computationally expensive, particularly when constraints on the
model parameters must be enforced.

In this work, a neural network is trained to approximate the solution
of the Heston pricing equation while embedding the governing physics
directly into the loss function. Boundary conditions and structural
constraints, including the Feller condition, are enforced during
training.

The resulting PINN acts as a surrogate model that maps state variables
and model parameters to option prices without requiring a numerical
solver.

The implementation is written in PyTorch and is structured to separate
the network architecture, physics-based losses, and data handling.
Automatic differentiation is used to compute the required derivatives
of the pricing function.

Once trained, the surrogate can be used for rapid parameter calibration,
sensitivity analysis, and optimization under uncertainty.

This project reflects ongoing research interests in physics-informed
machine learning, surrogate modeling, stochastic systems, and inverse
problems.

Author: Aman Kumar Chaudhary
