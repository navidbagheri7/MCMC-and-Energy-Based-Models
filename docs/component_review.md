# Implementation and artifact review

The original notebook is privately preserved with a digest. The authored public
implementation is independent English code, organized by scientific contract.
Foundational methods remain useful baselines; maintained NUTS, rank-normalized
multichain diagnostics, coordinate reparameterization, stochastic-gradient
comparisons and persistent/replay energy training extend that foundation.
No claim is made that the repository implements every conceivable sampler.

| Component | Reviewed contract and evidence |
|---|---|
| targets.py | Log targets/analytic scores agree with float64 autograd and finite differences; exact Gaussian/mixture/banana/funnel/rejection references; train-only real posterior construction |
| samplers.py | Correct asymmetric MH ratios, exact conditionals, retained rejections, reversible mass-consistent leapfrog, warmup-only adaptation, detailed-balance replica swaps, explicit approximate stochastic kernels |
| reference.py | Maintained Pyro NUTS, independent sequential chain seeds, train-only potential, declared adaptation/tree limit, no invented public energy statistics |
| diagnostics.py | ArviZ rank/folded split Rhat, bulk/tail ESS/MCSE, independent diagnostic oracle, unavailable diagnostics retained, finite-step/global-mode limitations |
| experiments.py | Four-chain fixed budgets, independent reference draws, no test tuning, recorded source/config/numeric identity, real-data predictive scores |
| ebm.py | Bounded residual over Gaussian base, detached negative chains, persistent/replay state, validation-only selection, quadrature refinement/tail scope, safe exported weights |
| posterior.py | Raw-feature schema and finite checks, saved train-only scaler, all-chain predictive averages and explicit interval meaning |
| artifacts.py | Atomic finite JSON, source digests and actual dependency environment |
| cli.py | Bundled presets, smoke/reference separation, rejected incomplete records and explicit alternative output |
| reporting.py | Every measured protocol retained, numerical-source figures/CSV summaries, unthinned inference and labeled display subsampling |
| scripts and HTML | Independent public-evidence/inference checks, fresh kernels, exact chain replay, live numerical oracles, mobile controls and isolated package verification |
| tests | Gradient/target, detailed-balance, invariant-law, reversibility, ULA-bias, rank-diagnostic, minibatch, conjugacy, energy-normalization and portable-inference oracles |

The implementation review corrected a library import, the small-gradient behavior
of the Torch logistic potential, finite-grid spacing/weight precision and two
SGHMC preconditioning flags before final publication. Historical private records
from the flag correction remain locally preserved; the main reference was rerun
with truthful identity-geometry metadata. No fitted algorithm is relabeled as a
strong result because its file exists: all failed diagnostic screens remain.

The [component inventory](component_review.json) covers public Python, notebook,
configuration, documentation and workflow files, parses code/configs, checks
English text and local links, and binds the audited source files. The numerical,
browser, notebook and distribution checks have separate scopes. Source inspection
plus passing tests is not proof that arbitrary new settings are correct.
