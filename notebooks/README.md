# Executable English curriculum

Ten fresh-kernel CPU walkthroughs connect targets, exact/proposal oracles, measured diagnostics and portable inference. No full benchmark is retrained by opening a notebook. Each has worked assertions, recorded evidence and static visual outputs.

1. [Targets, scores and exact reference draws](01_targets_and_rejection.ipynb)
2. [Metropolis, exact conditionals and slice sampling](02_metropolis_gibbs_slice.ipynb)
3. [Langevin drift, discretization bias and the MALA correction](03_ula_bias_and_mala.ipynb)
4. [Hamiltonian dynamics, mass and adaptation](04_hamiltonian_mass_and_adaptation.ipynb)
5. [Maintained NUTS and rank-normalized diagnostics](05_nuts_and_rank_diagnostics.ipynb)
6. [Multimodality, tempering and misleading warm starts](06_tempering_and_warmstarts.ipynb)
7. [Geometry changes the inference problem](07_funnel_and_reparameterization.ipynb)
8. [Real-data classification and stochastic gradients](08_bayesian_logistic_and_sg_gradients.ipynb)
9. [Conjugate real-data regression as an accuracy oracle](09_conjugate_regression_prediction.ipynb)
10. [Energy learning: CD, persistent chains, replay and likelihood](10_energy_learning_and_normalization.ipynb)

Run `python scripts/execute_notebooks.py` after installing the notebook extra. For an interactive experience without Python, open `docs/explorer.html`.
