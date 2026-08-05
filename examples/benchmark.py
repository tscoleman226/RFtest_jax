"""Benchmark rfjax tree/forest fitting and permutation tests."""

import time

import jax.numpy as jnp
from jax import random

import rfjax


def _make_data(n=200, p=5, key=0):
    key = random.PRNGKey(key)
    k1, k2 = random.split(key)
    X = random.normal(k1, (n, p))
    beta = jnp.array([1.0, -0.5, 0.0, 0.0, 0.0])
    y = X @ beta + 0.1 * random.normal(k2, (n,))
    return X, y


def timeit(name, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")
    return result


def main():
    X, y = _make_data(n=200, p=5, key=0)

    print("=== rfjax benchmark ===")

    tree = timeit(
        "DecisionTreeRegressor.fit",
        lambda: rfjax.DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=0).fit(X, y),
    )
    _ = timeit("DecisionTreeRegressor.predict", lambda: tree.predict(X))

    rf = timeit(
        "RandomForestRegressor.fit (ntree=100)",
        lambda: rfjax.RandomForestRegressor(
            ntree=100, k=50, mtry=3, max_depth=5, min_samples_leaf=5, random_state=0
        ).fit(X, y),
    )
    _ = timeit("RandomForestRegressor.predict", lambda: rf.predict(X))

    _ = timeit(
        "mse_test",
        lambda: rfjax.mse_test(
            X, y, var=0, n_test=50, B=500, ntree=100,
            k=50, mtry=3, max_depth=5, random_state=0,
        ),
    )

    _ = timeit(
        "permutation_importance (single_forest)",
        lambda: rfjax.permutation_importance(
            X, y, single_forest=True, n_test=50, B=500,
            nbtree=30, k=50, mtry=3, max_depth=5, random_state=0,
        ),
    )


if __name__ == "__main__":
    main()
