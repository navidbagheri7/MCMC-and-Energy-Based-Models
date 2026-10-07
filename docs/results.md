# Measured results

84 sampler protocols with four independent chains each; 36 neural-energy fits with three fitting seeds per task/method. All preset results, including failures, appear in the CSV tables.

19/84 protocols meet the stated rank-Rhat/ESS/divergence flag. This flag is a diagnostic screen, not a proof of convergence or absence of finite-step bias.

| Task | Negative phase | Fits | Mean test NLL | Fitting SD |
|---|---|---:|---:|---:|
| mixture | cd1 | 3 | 3.784 | 0.151 |
| mixture | cd10 | 3 | 3.014 | 0.031 |
| mixture | pcd10 | 3 | 3.188 | 0.144 |
| mixture | pcd50 | 3 | 3.080 | 0.074 |
| mixture | quadrature_mle | 3 | 2.948 | 0.006 |
| mixture | replay20 | 3 | 3.138 | 0.026 |
| moons | cd1 | 3 | 3.204 | 0.045 |
| moons | cd10 | 3 | 2.456 | 0.040 |
| moons | pcd10 | 3 | 2.879 | 0.048 |
| moons | pcd50 | 3 | 2.735 | 0.065 |
| moons | quadrature_mle | 3 | 2.527 | 0.035 |
| moons | replay20 | 3 | 2.908 | 0.008 |

NLL uses a bounded-residual Gaussian-base model and explicit quadrature refinement. No image-generation or exact high-dimensional normalization claim is made. CD-1 is visibly weaker on the mixture; persistent/replay methods do not uniformly outperform CD-10.

The centered funnel exposes divergences and poor mixing. Non-centering is an invertible change of sampling coordinates; diagnostics use the same original-space observables.

[Sampling CSV](sampling_comparison.csv) · [Energy CSV](energy_comparison.csv) · [Interactive explorer](explorer.html)
