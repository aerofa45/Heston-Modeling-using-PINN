# Heston Stochastic Volatility Modelling using Physics-Informed Neural Networks

## Exploratory Computational Research

[Python] [PyTorch] [QuantLib] [Numerical Optimisation]

## Description

This project focuses on numerically modelling the Heston Stochastic Volatility Model. This project looks into using Neural Networks to approximate Heston options pricing, calibrating parameters and looking into Physics-Informed Neural Networks.

This project is an exploratory computational project. There is no new stochastic volatility model presented, nor a validated improved method for options pricing. There is also no convergence result presented.

## 1. Background

Unlike constant volatility models the Heston model assumes that the asset price and the instantaneous variance follow stochastic processes.

This model can be defined as follows:

$$

dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S

[
dS_t=\mu S_t,dt+\sqrt{v_t}S_t,dW_t^S
]

[
dv_t=\kappa(\theta-v_t),dt+\sigma\sqrt{v_t},dW_t^v
]

Where:

- \(S_t\): Asset Price

- \(v_t\): Instantaneous Variance

- \(\kappa\): Rate of mean reversion

- \(\theta\): Long run variance

- \(\sigma\): Variance of variance

- \(\rho\): Correlation between brownian motions

The risk neutral pricing can be defined using a partial differential equation. Using the correct risk neutral parameters the Heston pricing operator can be defined as:

[
\begin{aligned}
0={}&V_t+\frac12vS^2V_{SS}
+\rho\sigma vS V_{Sv}\
&+\frac12\sigma^2vV_{vv}
+rSV_S+\kappa(\theta-v)V_v-rV.
\end{aligned}
]

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

This experiment also looks at the Feller condition for the Heston model:

$$

2\kappa\theta \geq \sigma^2

$$

However the penalty function needs to be corrected to be a valid differentiable constraint.

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
