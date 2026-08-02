"""JAX-based random forest regressor.

The forest is an ensemble of :class:`rfjax.tree.DecisionTreeRegressor` base
learners, each fit on a subsample of the training data and using a random
subset of features at each split.
"""

import jax.numpy as jnp
from jax import random

from rfjax.tree import DecisionTreeRegressor


def _default_subsample_size(n, p=0.5):
    """Default subsample size matching RFtest's ``ceiling(n^p)`` convention."""
    return int(jnp.ceil(n ** p))


class RandomForestRegressor:
    """Subsampled bagged ensemble of JAX regression trees.

    Parameters
    ----------
    ntree : int, default=500
        Number of base learners in the ensemble.
    k : int or None, default=None
        Number of observations sampled without replacement for each base
        learner. ``None`` uses ``ceil(n ** 0.5)`` to match the RFtest default.
    mtry : int or None, default=None
        Number of features sampled at each split. ``None`` uses all features.
    max_depth : int, default=10
        Maximum depth of each tree.
    min_samples_leaf : int, default=5
        Minimum number of samples in each leaf.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        ntree=500,
        k=None,
        mtry=None,
        max_depth=10,
        min_samples_leaf=5,
        random_state=None,
    ):
        self.ntree = ntree
        self.k = k
        self.mtry = mtry
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []
        self.n_features_ = None

    def fit(self, X, y):
        """Fit the random forest.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        n_samples = X.shape[0]
        self.n_features_ = X.shape[1]
        k = self.k if self.k is not None else _default_subsample_size(n_samples)
        k = min(k, n_samples)
        mtry = self.mtry if self.mtry is not None else self.n_features_

        key = random.PRNGKey(self.random_state if self.random_state is not None else 0)
        keys = random.split(key, self.ntree)

        trees = []
        for tree_key in keys:
            subkey, split_key = random.split(tree_key)
            indices = random.choice(subkey, n_samples, shape=(k,), replace=False)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mtry,
                random_state=int(split_key[0]),
            )
            tree.fit(X[indices], y[indices])
            trees.append(tree)

        self.trees_ = trees
        return self

    def _predict_all(self, X):
        """Return predictions from every tree as a (n_samples, ntree) array."""
        X = jnp.asarray(X, dtype=jnp.float32)
        if not self.trees_:
            raise ValueError("RandomForestRegressor must be fitted before predicting.")
        preds = jnp.stack([tree.predict(X) for tree in self.trees_], axis=1)
        return preds

    def predict(self, X):
        """Return the mean prediction across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        predictions : jnp.ndarray, shape (n_samples,)
        """
        return jnp.mean(self._predict_all(X), axis=1)
