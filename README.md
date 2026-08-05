# rfjax

A JAX-based Python port of the `RFtest` R package: permutation-based
hypothesis tests for random forests.

> **Note:** Tree growth still uses Python control flow for the dynamic queue of
> nodes, but the per-node split search and the full prediction pass are now
> JIT-compiled JAX functions.  This avoids per-node recompilation and
> unnecessary host-device synchronisations while keeping the tree structure
> explicit.

## Installation

You need Python 3.9+ and a working JAX installation. For CPU-only:

```bash
pip install -e ".[dev]"
```

or, if you prefer to install JAX separately:

```bash
pip install jax[cpu] jaxlib
pip install -e .
```

## Quick start

```python
import jax.numpy as jnp
from jax import random
import rfjax

# Synthetic data
key = random.PRNGKey(0)
X = random.normal(key, (200, 5))
y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * random.normal(key, (200,))

# Permutation test for variable 0
result = rfjax.mse_test(
    X, y, var=0, n_test=50, B=1000, ntree=200,
    k=50, mtry=3, max_depth=5, random_state=0,
)
print(result["Pvalue"])
print(result["Importance"])

# Variable importance for all variables
imp = rfjax.permutation_importance(
    X, y, single_forest=True, n_test=50, B=500,
    nbtree=50, k=50, mtry=3, max_depth=5, random_state=0,
)
print(imp["Importance_Table"])

# Holdout forest importance
himp = rfjax.holdout_rf(
    X, y, n_test=50, B=500, mintree=20, max_trees=500,
    mtry=2, k=50, max_depth=5, random_state=0,
)
print(himp["Importance_Table"])
```

## API overview

### Trees and forests

- `rfjax.DecisionTreeRegressor` — greedy CART regression tree using JAX for
  split search and prediction.
- `rfjax.RandomForestRegressor` — bagged ensemble of decision trees with
  subsampling (`k`) and random feature selection (`mtry`).

### Tests

- `rfjax.mse_test(X, y, var, ...)` — permutation F-test for a subset of
  variables.
- `rfjax.mse_compare(m1, m2, X_test, ...)` — permutation test comparing two
  fitted forests.
- `rfjax.permutation_importance(X, y, ...)` — marginal permutation importance
  for every variable.
- `rfjax.holdout_rf(X, y, ...)` — efficient variable importance via holdout
  forests.

## Differences from the R package

| R (`RFtest`) | Python (`rfjax`) |
|--------------|------------------|
| Supports `rpart`, `ctree`, `rtree`, `lm` base learners | Decision trees only |
| Uses `ranger` / `randomForest` / `party` | Custom JAX-backed trees |
| Formula interface | NumPy/JAX array interface |
| Character variable names | Integer column indices |

## Running tests

```bash
pytest
```

## Status

This is a first-pass implementation. The code has not yet been executed in this
environment because JAX is not installed here. After installing the package,
run the smoke tests in `tests/test_rfjax.py` and report any failures.
