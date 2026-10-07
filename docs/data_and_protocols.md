# Data, provenance and evaluation protocol

No external dataset download is required. Procedural targets use explicit seeded
NumPy generators. The perturbed Gaussian preserves the original coursework's
cosine/sine potential and correct score signs; exact rejection sampling uses the
Gaussian envelope with `cos(x0)+sin(x1) <= 2`. The original notebook is preserved
byte for byte privately and identified by [its manifest](../legacy/manifest.json).
The new stress preset uses correlation .95 rather than the original .8, while retaining its mean and periodic terms. Its single-chain unadjusted Langevin output is not evidence of exact convergence.

## Real data

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnosti),
Wolberg, Mangasarian, Street and Street (1993),
[DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), is distributed under CC BY 4.0.
The laboratory uses the public scikit-learn copy, all 569 observations and 30
continuous features, with deterministic 70/15/15 stratified index splits. Public
numeric records include those split indices and derived scaled arrays; the
dataset's attribution/terms remain separate from the code's MIT license.
Scikit-learn class 1 is benign; the predictive probability is explicitly for that
class. This is a statistical benchmark, not a clinical decision tool.

[Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
has 442 observations and 10 features, distributed through scikit-learn's built-in
benchmark copy and associated with Efron, Hastie, Johnstone and Tibshirani's
least-angle-regression study. The loader's feature representation already has
benchmark scaling; an additional StandardScaler fits on the training split only.
Do not interpret this copy as a prospectively collected external validation set.
The source description and library BSD attribution remain relevant to the included
small derived benchmark arrays; this release makes no independent patient-data
ownership claim.

Index splitting uses seeds 741 and 742. Model preprocessing, MAP geometry,
regression target scaling and all posterior likelihood terms use training data
only. Logistic coefficients have independent Normal(0,2.5²) priors, including
intercept. Regression coefficients have Normal(0,4) priors and fixed standardized
observation sigma .75; its Gaussian posterior is computed analytically from
training normal equations. Validation and test results are reported separately;
no sampler preset, noise variance or hyperparameter is chosen from test scores.

Posterior predictions average all retained states from all four chains. Logistic
scores include log loss, Brier score and ROC AUC; probability credible intervals
measure posterior uncertainty in p(class 1), not an outcome interval. Regression
95% intervals use Gaussian predictive moments and include observation noise;
they are labeled approximations. Poorly mixed sampler predictions remain in
comparison, with their failed diagnostics. The maintained NUTS posterior provides
a stronger real-data reference; it is independently diagnosed rather than assumed
correct by library name.

## Energy models

The two procedural tasks have fixed independent 8,000/1,000/1,000 training,
validation and test arrays. One is the eight-component mixture with radius 3 and
std .35; the other uses noisy sklearn moons with .08 noise and declared centering/
scale. Three model seeds 42–44 share the same task data. Each recipe gets 2,000
optimizer updates and batch 128 with Adam 5e-4. Negative-phase steps and cost
differ. The first minimum validation normalized NLL selects the model; the test
set is evaluated only after selection. The initial state is a candidate, and
weak/failed comparisons are not removed.

Each public energy file includes only owned safe numeric weights, architecture
metadata and a digest. Test likelihood uses explicit finite-grid integration;
normalizability does not imply negligible quadrature error. Model samples from
1,000-step ULA with eta .005 are approximate visual diagnostics, not exact IID
samples. Their seed is fixed independently of training and selection.
