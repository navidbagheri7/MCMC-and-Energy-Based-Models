# Read diagnostics as evidence, not a guarantee

The laboratory uses ArviZ 0.21 rank-normalized folded split Rhat, bulk ESS, tail
ESS and mean MCSE. An independent rank-transform/split/folded calculation is
checked against the library. Four independently seeded chains retain every
state after 1,000 warmup transitions; each contributes 2,000 draws. No thinning
is used for inference. A plotted subset is labeled separately.

The displayed screen requires every dimension's Rhat <1.01, bulk and tail ESS
>400 and zero recorded divergences. It is useful for identifying problems, but
not a proof of stationarity, global mode exploration or absence of discretization
bias. A warm-started mixture can pass local diagnostics while every chain misses
most modes. Inspect exact/reference moments and chain plots together. Negative
correlations can produce ESS above the raw draw count; this is not automatically
an error. Constant chains produce unavailable diagnostics and fail the screen.

MCSE describes uncertainty in a Monte Carlo estimate conditional on the target,
kernel and diagnosed mixing. It does not measure model misspecification or
finite-step bias. The 100,000-draw reference itself has Monte Carlo error, especially
for heavy funnel tails. The synthetic rejection/reference sampler is independently
seeded; it never initializes chains or tunes presets.

Owned HMC records total refreshed-momentum Hamiltonian energy and E-BFMI.
For other owned kernels the stored energy is negative log target; its difference
statistic is only a potential trace diagnostic. Pyro's public NUTS interface does
not supply paired momentum energy, so NUTS E-BFMI is explicitly unavailable.
Divergence thresholds and warmup-versus-retained scope are declared in the theory.
A divergent transition can still be rejected, but its geometry deserves inspection.

SGLD/SGHMC use a decreasing schedule and an approximate/noise-uncorrected kernel.
Their Rhat/ESS are descriptive chain summaries, not exact-posterior certificates.
ULA's Gaussian stationary variance oracle directly demonstrates a failure that
can remain after low Rhat. Stochastic-gradient fits are retained even when their
finite-budget diagnostics are poor.

Time includes warmup and retained sampling. Owned gradient calls and Pyro potential
calls are reported, including warmup; MH/Gibbs/slice can have zero gradient calls
while requiring substantial density/conditional work. Tempering has eight replicas.
Those quantities and wall time measure different costs and should not be collapsed
into one purportedly universal efficiency ranking. Hardware, process concurrency
and library overhead also affect elapsed time.

Primary references: [rank normalization, folding and localization](https://arxiv.org/abs/1903.08008),
[ArviZ diagnostics](https://python.arviz.org/en/v0.21.0/api/diagnostics.html),
[HMC](https://arxiv.org/abs/1206.1901), and [NUTS](https://arxiv.org/abs/1111.4246).
