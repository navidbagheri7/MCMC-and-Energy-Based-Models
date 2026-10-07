# Local release checks

This release is prepared locally for publication by the repository owner. No hosted
CI run, GitHub deployment or external publication is claimed.

- All 84 sampler protocols completed with four independent chains, 1,000 warmup
  iterations and 2,000 retained draws per chain.
- All 36 energy fits completed their 2,000-update protocol. Their selected exported
  weights, partition refinements, validation selection and held-out likelihoods
  are independently checked.
- The 25 tests check gradients, invariant distributions, detailed balance,
  integrator reversibility, finite-step ULA bias and portable inference.
- All ten lessons execute in fresh kernels: 30 code cells with real outputs.
- The browser audit covers all 120 measured presets, 28 independent live numerical
  checks, 252 full-precision chain replay checks and all eight mobile views.
- The evidence validator recomputes diagnostics and real held-out predictions,
  invokes two independent CSV prediction commands and verifies source identities.
- The public inventory parses code, notebooks and configurations, checks English
  text and local links. Manual scientific contracts are documented separately.
- The distribution checker installs the wheel outside the checkout, runs all
  25 tests and both bundled CLI families, then extracts the complete source
  archive and reruns all notebooks, evidence and public-content checks. Its
  local artifact report records actual pass/fail results and package digests.

See [numerical validation](validation.json), [browser validation](explorer_browser_validation.json),
[notebook execution](notebook_execution.json) and [component review](component_review.md).
The distribution report is produced after building by `scripts/check_distribution.py`.
Files and checks are evidence for these fixed protocols, not proof that arbitrary
new settings converge or that approximate samplers become exact.
