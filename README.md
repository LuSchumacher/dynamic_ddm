# Dynamic Diffusion Decision Model

This repository contains all data and code from the paper "Neural Superstatistics: A Bayesian Method for Estimating Dynamic Models of Cognition" by Lukas Schumacher, Paul-Christian Bürkner, Andreas Voss, Ullrich Köthe, and Stefan T. (https://arxiv.org/abs/2211.13165).

In this work, we propose to augment mechanistic cognitive models with a temporal dimension and estimate the resulting dynamics from a superstatistics perspective. In its simplest form, such a model entails a hierarchy between a low-level observation model and a high-level transition model. The observation model describes the local behavior of a system, and the transition model specifies how the parameters of the observation model evolve over time.

### Low-Level Observation Model
$x_t = \mathcal{G}(x_{1:t-1}, \theta_t, z_t) \quad \text{with}\quad z_t \sim p(z)$


### High-Level Transition Model
$\theta_t = \mathcal{T}(\theta_{0:t-1}, \eta, \xi_t) \quad \text{with}\quad \xi_t \sim p(\xi)$

### Low-Level Parameter Recovery
The following animation shows the parameter recovery performance of the GP-DDM model over all 3200 time steps.


![](param_recovery_animation.gif)
