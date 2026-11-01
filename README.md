# Heston Stochastic Volatility Modelling using Physics-Informed Neural Networks

## Exploratory Computational Research

[Python] [PyTorch] [QuantLib] [Numerical Optimisation]

## Description

This project focuses on numerically modelling the Heston Stochastic Volatility Model. This project looks into using Neural Networks to approximate Heston options pricing, calibrating parameters and looking into Physics-Informed Neural Networks.

This project is an exploratory computational project. There is no new stochastic volatility model presented, nor a validated improved method for options pricing. There is also no convergence result presented.

## 1. Background


The Heston model describes an asset price \(S_t\) and a stochastic
variance process \(v_t\).

The asset-price dynamics are:

$$
dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S
$$

The stochastic variance dynamics are:

$$
dv_t = \kappa(\theta-v_t)\,dt
+ \sigma\sqrt{v_t}\,dW_t^v
$$

The two Brownian motions are correlated:

$$
d\langle W^S,W^v\rangle_t = \rho\,dt
$$

Here:

- $S_t$: Asset price
- $v_t$: Instantaneous variance
- $\mu$: Asset-price drift
- $\kappa$: Mean-reversion speed
- $\theta$: Long-run variance
- $\sigma$: Volatility of variance (volatility of volatility)
- $\rho$: Correlation between the Brownian motions
- $W_t^S, W_t^v$: Brownian motions


Under risk-neutral dynamics, the option price $V(S,v,t)$
satisfies the Heston pricing PDE:

$$
\begin{aligned}
0 ={}&
\frac{\partial V}{\partial t}
+ \frac{1}{2}vS^2\frac{\partial^2 V}{\partial S^2}\\
&+ \rho\sigma vS\frac{\partial^2 V}{\partial S\partial v}\\
&+ \frac{1}{2}\sigma^2v\frac{\partial^2 V}{\partial v^2}\\
&+ rS\frac{\partial V}{\partial S}
+ \kappa_Q(\theta_Q-v)\frac{\partial V}{\partial v}
- rV.
\end{aligned}
$$

Here, $r$ is the risk-free interest rate, and
$\kappa_Q,\theta_Q$ are the risk-neutral variance parameters.


This provides a mathematical link between stochastic processes, differential equations and numerical learning.

## 2. Objectives

This project looks to explore the following questions:

Can Neural Networks approximate Heston options pricing using market or synthetic data?

How does incorporating the Heston PDE residual impact the training of the model?

What challenges are there to learning parameters for stochastic volatility models with mathematical constraints?

## 3. Experiments

### A) LS Calibration and Neural Network Comparison

[Heston_LS_vs_Trainable_PINN_Enhanced.ipynb]

This experiment looks at the traditional calibration of the Heston model and Neural Networks for options pricing.

This experiment includes:

- Options Pricing using the Heston model in QuantLib

- Parameter optimisation using bounded least squares

- Gathering financial options chain data from Yahoo Finance

- Training Neural Networks in PyTorch

- Comparing options pricing errors and implied volatility errors

- Diagnostics of training for multiple underlying assets

The Neural Network used in this notebook is using supervised learning to minimise errors in pricing. This should not be considered to be a Physics-Informed Neural Network due to the class name.

### B) Exploratory PDE Constrained Heston PINN

[PINN_Heston_Standalone_Colab_Trial_BGFS.ipynb]

This experiment looks at using a Neural Network with the Heston PDE residual.

This experiment includes:

- Automatic differentiation of options pricing

- First and second order partial derivatives

- Mixed derivative due to correlated stochastic drivers

- Trainable parameters for the Heston model

- Generating synthetic options pricing data using QuantLib

- Minimising the data fitting loss and PDE residual loss

- Diagnostics of training and tracking parameters

- Stress testing and comparison of loss weights


The Feller condition for the CIR variance process is:

$$
2\kappa\theta \geq \sigma^2
$$

where:

- $\kappa$ is the mean-reversion speed.
- $\theta$ is the long-run variance.
- $\sigma$ is the volatility-of-variance parameter.

When $2\kappa\theta \geq \sigma^2$, with suitable positive
parameters and initial variance, the variance process does
not reach zero.

An exploratory penalty is:

$$
\mathcal{L}_{\mathrm{Feller}}
=
\max\left(0,\sigma^2-2\kappa\theta\right).
$$


## 4. Evaluation

The notebooks look at various aspects to evaluate the numerical performance:

| Evaluation | Description |

|---|---|

| Options Pricing MSE | Error in approximating options pricing |

| Implied Volatility Error | Comparing reconstructed market volatility information |

| PDE Residual | How well the model fits the differential operator |

| Parameter Tracking | Optimisation stability |

| Synthetic Parameter Recovery | How well the model can recover parameters |

| Training Loss Curves | Diagnostics of optimisation and convergence |

There are some initial experiments and results saved in the notebooks, however they are not a consistent and repeatable benchmark.

## 5. Challenges

There are a number of challenges with the current methodology:

- Inconsistencies in the PINN notebook relating to variance, strike and maturity parameters

- The penalty function is not providing the desired differentiable parameter constraint

- Parameters are not consistently constrained

- Initial/terminal conditions for pricing need to be more explicit

- The pricing PDE and numerical derivatives need to be checked for consistency

- The models are not being compared using consistent data sets, budgets and evaluation

- The simple Python modules in the repository do not implement the full pricing operator for the Heston model

The results cannot be relied upon to show parameter recovery or that the model outperforms traditional calibration.

## 6. Improvements

Future improvements will include:

Correcting the definition of state variables and PDE.

Implementing differentiable parameter constraints and valid parameter transformations.

Implementing appropriate payoffs and boundary conditions.

Comparing Neural Network pricing with QuantLib using consistent experimental conditions.

Repeat experiments using controlled random seeds and reporting uncertainty.

Separate synthetic data evaluation from market data evaluation.

## 7. Relation to Stochastic Processes and Scientific Machine Learning

The Heston model is useful to demonstrate computationally as it relates to stochastic differential equations and a deterministic pricing PDE.

This project has provided experience with numerically modelling problems with correlated stochastic processes, PDE constraints, parameter estimation and numerical optimisation.

My wider research interests are around reliable scientific machine learning and understanding how constraints and optimisation affect the behaviour of learnt models.

This project is a computational investigation and does not contribute to information theoretical stochastic process theory.

## 8. Status

Status: Exploratory Research / Implementation under Validation.

Author: Aman Kumar Chaudhary

Institution: Southeast Missouri State University

Research Interests: Scientific Machine Learning, Stochastic Modelling, Numerical Optimisation, Physics-Informed Neural Networks
