# Start by sampling a distribution

Open [the explorer](explorer.html). It starts on **Chain diagnostics → gaussian →
gibbs · dispersed · preset 1**, showing all 2,000 retained draws from each of four
independent chains. Gray contours are the target density; colored points are the
samples produced by the selected algorithm. The larger points mark the current
positions. The two marginal histograms compare the accumulated samples with
100,000 independent exact reference draws.

## Three concrete experiments

1. **Correlated Gaussian:** stay on `gaussian` and compare Gibbs, HMC, Metropolis,
   ULA and MALA. The target is a tilted Gaussian with mean `(2, −2)` and correlation
   `0.95`. Compare `ula · dispersed · preset 0.08` with MALA at the same nominal
   step. ULA's finite-step variance inflation persists even with long sampling;
   MALA corrects the asymmetric proposal. Different step conventions and costs
   remain explicit in the protocol rather than establishing an equal-cost ranking.
2. **Eight islands:** choose **Multimodality and tempering → ring**. Compare
   `gibbs · dispersed · preset 1` and `parallel_tempering · dispersed · preset 0.6`.
   Gibbs has exact conditionals but can remain in a single island. Tempering spends
   extra work on eight temperatures to improve movement between islands. Check
   which target modes the cloud occupies and the full-chain diagnostics.
3. **Curved geometry:** choose **Geometry and reparameterization → banana** and
   compare Metropolis with HMC. Then compare `funnel` and `funnel_noncentered` under
   NUTS. Both funnel views show the same original coordinates; non-centering changes
   the coordinates used for sampling, not the distribution being estimated.

Click **Play retained chain states** to watch the samples accumulate. Starting at
the final draw restarts replay from the beginning. Change **Displayed chain** to
separate the four independent runs. **Sample cloud** reveals the distribution;
**Connected trajectories** reveals movement between retained endpoints. HMC's
internal leapfrog steps are demonstrated separately under **Sampler mechanics**.

Diagnostic cards always use all four complete retained chains. Changing a plot
window, displayed chain or replay position does not recompute these full-protocol
diagnostics.

**Target focus** uses independently determined reference quantiles and explicitly
reports how many accumulated samples fall outside the view. **Full recorded extent**
shows every retained state. Axes have independent labeled scales. Neither view
changes inference, discards repeated states, or modifies full-chain diagnostics.
The reference marginal curve spans the focused target window; its histogram
uses all 100,000 reference draws in its normalization.

## What is actually implemented and run?

| Method | Evaluated targets | Implementation |
|---|---|---|
| Metropolis, adaptive Metropolis, slice, ULA, MALA, HMC, adaptive HMC | All eight targets | Owned NumPy transitions |
| Gibbs | Gaussian, Gaussian regression posterior, eight-component mixture | Exact Gaussian conditionals or augmented label/position conditionals |
| NUTS | All eight targets | Maintained Pyro implementation connected to the same targets |
| Parallel tempering | Eight-component mixture | Eight-temperature replica/swap kernel |
| SGLD and approximate SGHMC | Wisconsin Diagnostic logistic posterior | Training-only minibatch gradients; approximation declared |

The eight targets are Gaussian, sinusoidally perturbed Gaussian, ring mixture,
banana, centered funnel, non-centered funnel, Diabetes regression posterior and
Wisconsin Diagnostic classification posterior. The original 84 protocols contain
four independent chains each, with 1,000 warmup and 2,000 retained draws. The demo
replays these actual NumPy/Pyro outputs; dropdown changes do not run a new sampler.

For regression, the first two coefficients have an exact Gaussian marginal. For
classification, the two plotted coefficients are a projection of a 31-dimensional
posterior; no exact 2D density or exact-draw reference is invented. The focus window
uses the training-only Laplace approximation strictly as a viewing aid. Predictive
results use all coefficients and independent test data.

The [sampler source](../src/sampling_lab/samplers.py),
[target equations](../src/sampling_lab/targets.py),
[measured protocols](sampling_comparison.csv) and
[executable notebook lessons](../notebooks/README.md) provide reproducible examples.
For a short new Gaussian run in a separate smoke directory:

```bash
sampling-lab sampling --quick --filter '"target": "gaussian"'
```

The [target-geometry validation](target_views_validation.json) independently checks
all contour-grid values against SciPy densities and reproduces exact-reference
marginals. The [interaction audit](demo_review.md) covers every enabled display,
chain and plot-window combination and checks histogram counts against original
saved chain arrays. All reference data and figures are included for offline use.
