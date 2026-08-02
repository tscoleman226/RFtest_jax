"""RFJAX: Random forest permutation tests in JAX.

This package provides a JAX-based implementation of the permutation-based
hypothesis tests for random forests originally implemented in the R package
``RFtest``.

The tree induction uses JAX for numerical operations (split search, prediction)
but uses Python control flow for tree growth, because decision trees have
dynamic structure that is not naturally expressed in a single XLA graph.
"""

from rfjax.tree import DecisionTreeRegressor
from rfjax.forest import RandomForestRegressor
from rfjax.tests import mse_test, mse_compare, permutation_importance, holdout_rf

__all__ = [
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "mse_test",
    "mse_compare",
    "permutation_importance",
    "holdout_rf",
]

__version__ = "0.1.0"
