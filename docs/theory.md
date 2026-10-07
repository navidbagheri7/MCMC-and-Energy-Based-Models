# Equations, state conventions and implementation contracts

Every kernel targets an explicitly defined density. Normalization is unnecessary
for Metropolis, Gibbs, Langevin and Hamiltonian transitions, but necessary for
reported energy-model likelihood. Separate statistical correctness, numerical
integration, finite-chain mixing and predictive quality.

## Metropolis and Gibbs

Random-walk MH proposes `y=x+s Lz`, with symmetric covariance `s²P`, and accepts
`min(1, exp(log pi(y)-log pi(x)))`. The complete state trajectory includes
rejected repeats. Adaptive RWM uses a decreasing warmup-only Robbins–Monro
log-scale update toward acceptance .234. This heuristic is not universally
optimal in small dimensions. The sampling kernel freezes after warmup.

Gaussian Gibbs uses precision-matrix conditional means and variance `1/Qkk`.
The ring uses an augmented component label sampled conditional on position,
then an exact component-conditional position draw. Those conditionals are exact
while transitions between distant modes may be extremely rare. Coordinate slice
uses a random bounded stepping-out budget and shrinking interval; it is defined
for the continuous targets with no numerical density floor.

## Langevin, MALA and stochastic gradients

The convention is `y = x + eta P grad(log pi(x)) + sqrt(2 eta) Lz`, `LLᵀ=P`.
ULA accepts this discretization unconditionally when finite. For a Gaussian
covariance eigenvalue lambda and identity P, its stationary covariance eigenvalue
is `lambda/(1-eta/(2 lambda))`, provided `eta < 2 min(lambda)`. This persistent
bias is independent of how long a stable chain runs.

MALA evaluates the asymmetric proposal densities in both directions:
`log a = log pi(y)-log pi(x)+log q(x|y)-log q(y|x)`.
The implementation includes both position-dependent drift terms and the
preconditioner inverse. A test checks log detailed balance against an independent
Gaussian proposal expression.

Logistic minibatches sample 64 training rows without replacement and multiply
the likelihood gradient by `N/batch_size`; the Gaussian-prior gradient is added
once. SGLD uses `eta_t=eta_0/(1+t/1000)^.55` and the same explicit P convention.
The finite-budget empirical distribution is approximate, and its decreasing
step size creates a changing kernel. Diagnostics alone do not establish accuracy.
SGHMC uses friction .05 with identity geometry and the momentum update
`v <- .95 v + eta_t grad(log pi) + sqrt(2*.05*eta_t) z; x <- x+v`.
It assumes zero estimated gradient-noise correction (B=0); actual minibatch noise
is not cancelled. This pedagogical approximation is labeled and compared with
full-gradient NUTS. It is not claimed to reproduce an exact invariant posterior.

## Hamiltonian dynamics and adaptive geometry

Momentum covariance is `M=inv(P)`; kinetic energy is `pᵀPp/2` and position
velocity is `Pp`. Leapfrog uses half momentum kicks and full position updates,
then flips momentum. Metropolis corrects finite integration error. Integration
length is independently randomized between half and the full declared step
count to reduce fixed-length resonance. A nonfinite proposal is rejected; an
absolute energy error above 1,000 is flagged as divergent. Tests check reversibility
and the expected second-order energy-error scaling.

Adaptive HMC uses warmup dual averaging toward .8 acceptance and one warmup-only
diagonal covariance window, with regularized variance bounds [.001,100]. The
final step and geometry freeze for retained draws. NUTS comes from maintained
[Pyro](https://docs.pyro.ai/en/stable/_modules/pyro/infer/mcmc/nuts.html), with target
acceptance .85, full mass adaptation for these dimensions, maximum tree depth 8
and independent sequential chains. Pyro's public diagnostics expose divergences
and an acceptance statistic, but not paired Hamiltonian energies or tree-depth
saturation; the release does not invent those quantities.

## Tempering, initialization and coordinates

Parallel tempering uses eight inverse temperatures geometrically spaced from
1 to .03. Each replica performs a random-walk MH step scaled by `1/sqrt(beta)`;
alternating adjacent swaps use `(beta_i-beta_j)*(log pi(x_j)-log pi(x_i))`.
Only the cold replica estimates the target. Eight replicas entail additional
work; a single cold transition is not a single ordinary MH evaluation.

Synthetic warm starts use the declared model center with independent jitter;
ring centers can place all chains near one mode. Real posterior starts use the
training-only MAP/conjugate mean with independent jitter. Dispersed starts are
recorded separately. A warm start is neither an independent posterior draw nor
proof of global convergence.

The funnel has `y~N(0,9), x|y~N(0,exp(y))`. Non-centering samples independent
`y~N(0,9), z~N(0,1)` and transforms `x=exp(y/2)*z`. Diagnostics and figures use
original-space `(y,x)` in both cases. The finite reference sample has 100,000
exact draws; its covariance estimate is particularly variable for heavy funnel
tails and is not called an exact population moment.

## Neural energies and normalization

The owned energy is `E(x)=||x||²/18 + 8 tanh(f_theta(x)/8)` with a 64×64 tanh MLP.
The residual is bounded, so a Gaussian envelope proves integrability. CD begins
negative chains at a current positive minibatch; PCD retains a 1,024-state buffer;
replay refreshes 5% of sampled states from the Gaussian base. Langevin negatives
use eta .02 and the method's declared step count, detach their states and never
backpropagate through the sampling trajectory. Parameter gradients use the
positive-minus-negative energy objective. Finite negative phases introduce bias.

The quadrature-MLE baseline differentiates the finite-grid log partition directly.
Training uses 81×81 over ±14; validation selection uses 161×161 over ±14 every
100 updates; final test partition is independently refined to 401×401 over ±18.
Refinement differences estimate integration error rather than rigorously bounding
it. The outside-square partition tail is bounded by the Gaussian outside-circle
envelope `2*pi*9*exp(8)*exp(-18²/18)`. This is an absolute partition bound, not
a likelihood-error bound without further scaling. Longer generated ULA negatives
are approximate visual diagnostics and never define the normalized test NLL.
