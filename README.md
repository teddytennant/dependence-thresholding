dependence-thresholding implements Dependence Thresholding (Kim and Zhou, arXiv:2609.32926): recover latent factor count and support by thresholding Spearman dependence and extracting independent maximal cliques, then fit a structure-constrained map.

Install with `pip install -e .` and run tests with `pytest`.

This does not rerun the 100-dataset simulation, the three latent DAGs, or the high-dimensional robustness study. Model selection uses validation MSE rather than validation log-likelihood. Latent rows are free parameters trained with Adam on mean squared error, and each observed variable has its own one-hidden-layer tanh network on its parent latents only. That is not the paper's stochastic EM procedure, MALA latent sampler, or mask-layer likelihood. An edge is kept when absolute Spearman correlation is at least the threshold. Validation refits latent rows with the decoder held fixed.
