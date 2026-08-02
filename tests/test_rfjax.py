"""Basic smoke tests for rfjax.

These tests assume JAX is installed. They are not run in this environment
because JAX is not currently available, but they should pass after installing
``rfjax`` with its dependencies.
"""

import jax.numpy as jnp
from jax import random

import rfjax


def _make_data(n=100, p=5, key=0):
    key = random.PRNGKey(key)
    k1, k2 = random.split(key)
    X = random.normal(k1, (n, p))
    beta = jnp.array([1.0, -0.5, 0.0, 0.0, 0.0])
    y = X @ beta + 0.1 * random.normal(k2, (n,))
    return X, y


def test_decision_tree_regressor():
    X, y = _make_data()
    tree = rfjax.DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == y.shape


def test_random_forest_regressor():
    X, y = _make_data()
    rf = rfjax.RandomForestRegressor(ntree=10, k=30, mtry=3, max_depth=3, random_state=0)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == y.shape


def test_mse_test():
    X, y = _make_data()
    result = rfjax.mse_test(
        X, y, var=0, n_test=30, B=100, ntree=20,
        k=30, mtry=3, max_depth=3, random_state=0,
    )
    assert "Pvalue" in result
    assert "PermDiffs" in result
    assert result["PermDiffs"].shape == (100,)


def test_mse_compare():
    X, y = _make_data()
    rf1 = rfjax.RandomForestRegressor(ntree=10, k=30, mtry=3, max_depth=3, random_state=0)
    rf2 = rfjax.RandomForestRegressor(ntree=10, k=30, mtry=3, max_depth=3, random_state=1)
    rf1.fit(X[:70], y[:70])
    rf2.fit(X[:70], y[:70])
    result = rfjax.mse_compare(rf1, rf2, X[70:], y[70:], B=100, random_state=0)
    assert "Pvalue" in result


def test_permutation_importance():
    X, y = _make_data()
    result = rfjax.permutation_importance(
        X, y, single_forest=True, n_test=30, B=100,
        nbtree=10, k=30, mtry=3, max_depth=3, random_state=0,
    )
    assert result["Importance_Table"].shape == (3, 5)


def test_holdout_rf():
    X, y = _make_data(n=200, p=5)
    result = rfjax.holdout_rf(
        X, y, n_test=50, B=100, mintree=5, max_trees=200,
        mtry=2, k=50, max_depth=3, random_state=0,
    )
    assert result["Importance_Table"].shape == (3, 5)
