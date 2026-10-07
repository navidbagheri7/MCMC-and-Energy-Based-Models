# Reproduction and scientific records

Install the package and optional notebook/development tools as described in the
README. The measured environment versions are saved in each run record. The main
protocols use CPU float64 for sampling/targets, float32 neural energies and float64
quadrature weights. Full runs set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
and `MKL_NUM_THREADS=1`; independent target or model-seed processes may run in
parallel. Concurrency affects wall time and is not normalized away.

```bash
sampling-lab sampling
sampling-lab ebm
python -m sampling_lab.reporting
python scripts/create_explorer.py
python scripts/validate_release.py
python scripts/execute_notebooks.py
python -m build
python scripts/check_distribution.py
```

A completed reference fit is reused only if its full protocol/source identity and
numeric digest match. An incomplete run is rejected and retained for inspection.
The repository does not claim exact interrupted training or chain continuation.
Use a new output directory or checkout to investigate changes. Smoke fitting
always gets small budgets and its own numeric evidence; it never writes the main
reference arrays.

Public evidence contains all 84 sampler and 36 energy protocols, recorded chain
starts/retained states, relevant warmup arrays, diagnostic histories, train-only
real-data scalers and split indices, portable energy weights, plots, executed
notebooks and a self-contained explorer. Original course materials, private
virtual environments and smoke/superseded records are excluded.

`validate_release.py` recomputes diagnostics, reference comparisons, predictive
scores and energy likelihoods, checks all source/numeric/model digests and
independently replays public inference. It also verifies every browser binding
and executed notebook. Distribution checks additionally run tests against an
installed wheel outside the checkout and fresh kernels from the extracted
source archive. These local checks are distinct from the supplied GitHub CI;
remote CI is not claimed to have run before the owner publishes.

For optional browser verification, install Playwright in a development environment,
install its Chromium runtime into your chosen external cache, then run
`python scripts/check_explorer_browser.py`. It checks all 120 presets, 252 exact
chain replay states, 28 independent live calculations and eight mobile concept
areas. Browser tools are not runtime sampling/model dependencies.
