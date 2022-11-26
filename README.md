# Dynamic Diffusion Decision Model

This repository contains all data and code from the paper "Neural Superstatistics: A Bayesian Method for Estimating Dynamic Models of Cognition" by Lukas Schumacher, Paul-Christian Bürkner, Andreas Voss, Ullrich Köthe, and Stefan T. Radev (https://arxiv.org/abs/2211.13165).

In this work, we propose to augment mechanistic cognitive models with a temporal dimension and estimate the resulting dynamics from a superstatistics perspective. In its simplest form, such a model entails a hierarchy between a low-level observation model and a high-level transition model. The observation model describes the local behavior of a system, and the transition model specifies how the parameters of the observation model evolve over time.

### Low-Level Observation Model
Simple diffusion decision model (DDM)

$\mathrm{d}x_j = v\mathrm{d}t_s + \xi \sqrt{\mathrm{d}t_s} \quad\text{with}\quad \xi\sim\mathcal{N}(0, 1)$


### High-Level Transition Model
The trial-by-trial dynamics of the DDM parameters follow a Gaussian Process (GP)

$\theta_{1:T} \sim \mathcal{GP}(\mu_{\theta}, K_{\theta})$

with mean function $\mu_{\theta}$ and covariance function $K_{\theta}$ defined through the vector of time indices.
The high-level parameters $\eta$ in this case are the kernel parameters, such as the amplitude $\sigma$ and the length-scale $l$ of a Gaussian kernel

$k(\theta_t, \theta_{t'}) = \sigma^2 \exp\left(\frac{||\theta_t - \theta_{t'} ||^2}{2l^2}\right).$

### Low-Level Parameter Recovery
The following animation shows the parameter recovery performance of the GP-DDM model over all 3200 time steps.


![](param_recovery_animation.gif)
