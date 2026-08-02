"""JAX-based regression tree.

Tree growth uses Python control flow because the tree structure is dynamic,
but all numerical work (split search, prediction routing) is done with JAX
arrays and operations.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import random


def _best_split_one_feature(x_sorted, y_sorted, min_samples_leaf):
    """Find the best split threshold for a single already-sorted feature.

    Parameters
    ----------
    x_sorted : jnp.ndarray, shape (n,)
        Feature values sorted in ascending order.
    y_sorted : jnp.ndarray, shape (n,)
        Responses in the same order as ``x_sorted``.
    min_samples_leaf : int
        Minimum number of samples required in each child.

    Returns
    -------
    threshold : float
        Best threshold (midpoint between two consecutive values).
    impurity : float
        Weighted MSE of the best split; ``inf`` if no valid split exists.
    """
    n = x_sorted.shape[0]
    if n < 2 * min_samples_leaf:
        return jnp.inf, jnp.inf

    counts_left = jnp.arange(1, n)
    counts_right = n - counts_left
    valid = (counts_left >= min_samples_leaf) & (counts_right >= min_samples_leaf)

    y_cumsum = jnp.cumsum(y_sorted)
    y_sq_cumsum = jnp.cumsum(y_sorted ** 2)

    sum_left = y_cumsum[:-1]
    sum_right = y_cumsum[-1] - y_cumsum[:-1]
    ss_left = y_sq_cumsum[:-1]
    ss_right = y_sq_cumsum[-1] - y_sq_cumsum[:-1]

    # Variance = E[y^2] - E[y]^2; weighted by counts
    mse_left = ss_left / counts_left - (sum_left / counts_left) ** 2
    mse_right = ss_right / counts_right - (sum_right / counts_right) ** 2
    weighted_mse = (counts_left * mse_left + counts_right * mse_right) / n
    weighted_mse = jnp.where(valid, weighted_mse, jnp.inf)

    thresholds = (x_sorted[:-1] + x_sorted[1:]) / 2.0
    best_idx = jnp.argmin(weighted_mse)
    return thresholds[best_idx], weighted_mse[best_idx]


@partial(jax.vmap, in_axes=(None, None, None, 0, None))
def _best_split_for_features(X, y, indices, feature, min_samples_leaf):
    """Vectorised best-split search over a set of feature indices."""
    x = X[indices, feature]
    y_node = y[indices]
    order = jnp.argsort(x)
    threshold, impurity = _best_split_one_feature(x[order], y_node[order], min_samples_leaf)
    return threshold, impurity


def _choose_best_split(thresholds, impurities, features):
    """Pick the feature and threshold with the lowest impurity."""
    best_idx = jnp.argmin(impurities)
    best_feature = features[best_idx]
    best_threshold = thresholds[best_idx]
    best_impurity = impurities[best_idx]

    # If every candidate was invalid, signal "no split".
    best_feature = jnp.where(jnp.isinf(best_impurity), -1, best_feature)
    best_threshold = jnp.where(jnp.isinf(best_impurity), jnp.inf, best_threshold)
    return best_feature, best_threshold, best_impurity


class DecisionTreeRegressor:
    """Greedy CART regression tree backed by JAX arrays.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_leaf : int, default=5
        Minimum number of training samples in a leaf.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.
        ``None`` means all features.
    random_state : int or None, default=None
        Seed for the random number generator used to sample features.
    """

    def __init__(self, max_depth=10, min_samples_leaf=5, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.n_features_ = None

    def _new_tree(self):
        max_nodes = 2 ** (self.max_depth + 1) - 1
        return {
            "feature": jnp.full(max_nodes, -1, dtype=jnp.int32),
            "threshold": jnp.zeros(max_nodes, dtype=jnp.float32),
            "value": jnp.zeros(max_nodes, dtype=jnp.float32),
            "left_child": jnp.full(max_nodes, -1, dtype=jnp.int32),
            "right_child": jnp.full(max_nodes, -1, dtype=jnp.int32),
            "is_leaf": jnp.zeros(max_nodes, dtype=bool),
        }

    def _set_leaf(self, tree, node_idx, value):
        tree = tree.copy()
        tree["value"] = tree["value"].at[node_idx].set(float(value))
        tree["is_leaf"] = tree["is_leaf"].at[node_idx].set(True)
        return tree

    def fit(self, X, y):
        """Fit the regression tree.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        self.n_features_ = X.shape[1]
        self.max_features_ = self.max_features or self.n_features_

        key = random.PRNGKey(self.random_state if self.random_state is not None else 0)
        tree = self._new_tree()

        # Each queue entry: (node_index, sample_indices, depth, rng_key)
        node_queue = [(0, jnp.arange(X.shape[0]), 0, key)]
        next_free_idx = 1

        while node_queue:
            node_idx, indices, depth, key = node_queue.pop(0)
            n_samples = indices.shape[0]
            node_value = float(jnp.mean(y[indices]))

            if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
                tree = self._set_leaf(tree, node_idx, node_value)
                continue

            key, subkey = random.split(key)
            features = random.choice(
                subkey, self.n_features_, shape=(self.max_features_,), replace=False
            )

            thresholds, impurities = _best_split_for_features(
                X, y, indices, features, self.min_samples_leaf
            )
            best_feature, best_threshold, best_impurity = _choose_best_split(
                thresholds, impurities, features
            )

            if int(best_feature) < 0:
                tree = self._set_leaf(tree, node_idx, node_value)
                continue

            x_feature = X[indices, int(best_feature)]
            left_mask = x_feature <= float(best_threshold)
            n_left = int(jnp.sum(left_mask))
            n_right = n_samples - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                tree = self._set_leaf(tree, node_idx, node_value)
                continue

            left_child = next_free_idx
            right_child = next_free_idx + 1
            next_free_idx += 2

            tree = tree.copy()
            tree["feature"] = tree["feature"].at[node_idx].set(int(best_feature))
            tree["threshold"] = tree["threshold"].at[node_idx].set(float(best_threshold))
            tree["left_child"] = tree["left_child"].at[node_idx].set(left_child)
            tree["right_child"] = tree["right_child"].at[node_idx].set(right_child)
            tree["is_leaf"] = tree["is_leaf"].at[node_idx].set(False)

            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
            node_queue.append((left_child, left_indices, depth + 1, key))
            node_queue.append((right_child, right_indices, depth + 1, key))

        self.tree_ = tree
        self.n_nodes_ = next_free_idx
        return self

    def predict(self, X):
        """Predict regression targets for ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        predictions : jnp.ndarray, shape (n_samples,)
        """
        if self.tree_ is None:
            raise ValueError("DecisionTreeRegressor must be fitted before predicting.")

        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]
        node = jnp.zeros(n_samples, dtype=jnp.int32)

        for _ in range(self.max_depth + 1):
            is_leaf = self.tree_["is_leaf"][node]
            if bool(jnp.all(is_leaf)):
                break

            # Avoid invalid -1 indexing for already-leaf nodes.
            feature = jnp.where(is_leaf, 0, self.tree_["feature"][node])
            threshold = self.tree_["threshold"][node]
            left_child = self.tree_["left_child"][node]
            right_child = self.tree_["right_child"][node]

            x_feature = X[jnp.arange(n_samples), feature]
            go_left = x_feature <= threshold
            next_node = jnp.where(go_left, left_child, right_child)
            node = jnp.where(is_leaf, node, next_node)

        return self.tree_["value"][node]
