Crime is a complex spatio-temporal phenomenon influenced by interactions between susceptible individuals, criminals, and prison populations. Mathematical compartmental models provide a useful framework for studying these dynamics, but estimating model parameters from real-world data and performing large-scale simulations remain computationally challenging.

This repository presents a hybrid **Physics-Informed Neural Network (PINN)** and **Fourier Neural Operator (FNO)** framework for modeling crime dynamics. First, an Inverse PINN is employed to estimate the unknown parameters of a compartmental crime model using NCRB crime data. The estimated parameters are then used to generate spatio-temporal solutions of the governing reaction-diffusion system through numerical simulations. Finally, a Fourier Neural Operator is trained on these simulations to learn the underlying solution operator of the partial differential equation.

The proposed framework combines data-driven parameter discovery with neural operator learning, enabling accurate and computationally efficient prediction of crime dynamics across space and time.

## Methodology

1. Estimate unknown crime model parameters using Inverse PINNs and NCRB crime data.
2. Generate large-scale spatio-temporal trajectories using the calibrated PDE model.
3. Train a Fourier Neural Operator on the generated dataset.
4. Use the trained neural operator as a fast surrogate for predicting crime evolution.

## Key Contributions

* Physics-informed parameter estimation from real crime data.
* Spatio-temporal crime modeling using reaction-diffusion equations.
* Neural operator learning for rapid surrogate prediction.
* Significant reduction in computational cost compared to repeated numerical simulations.

## Repository Goal

The objective of this work is to bridge mathematical crime modeling and scientific machine learning by combining PINNs and neural operators into a unified framework for understanding and forecasting spatio-temporal crime dynamics.
