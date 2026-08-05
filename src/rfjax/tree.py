"""JAX-based regression tree.

Tree growth uses Python control flow because the tree structure is dynamic,
but all numerical work (split search, prediction routing) is done with JAX
arrays and operations.  The split search is JIT-compiled once per training set
size using a fixed-length sample mask, and tree arrays are updated in place
instead of copied.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import random


def _best_split_one_feature(x_sorted, y_sorted, mask_sorted, min_samples_leaf):
    """Find the best split threshold for a single sorted feature.

    Parameters
    ----------
    x_sorted : jnp.ndarray, shape (n,)
        Feature values sorted in ascending order (all training samples).
    y_sorted : jnp.ndarray, shape (n,)
        Responses in the same order as ``x_sorted``.
    mask_sorted : jnp.ndarray, shape (n,)
        Boolean mask indicating which sorted samples belong to the current
        node.  Split points are only evaluated between consecutive sorted
        positions; positions where ``mask_sorted`` is False simply do not
        contribute to the left/right statistics.
    min_samples_leaf : int
        Minimum number of masked samples required in each child.

    Returns
    -------
    threshold : float
        Best threshold (midpoint between two consecutive sorted values).
    impurity : float
        Weighted MSE of the best split; ``inf`` if no valid split exists.
    """
    n = x_sorted.shape[0]
    n_node = jnp.sum(mask_sorted)

    counts_left = jnp.cumsum(mask_sorted)
    counts_right = n_node - counts_left

    # Valid split points: between i and i+1 for i = 0..n-2
    valid = (counts_left[:-1] >= min_samples_leaf) & (counts_right[:-1] >= min_samples_leaf)

    y_weighted = y_sorted * mask_sorted
    y_sq_weighted = y_sorted ** 2 * mask_sorted

    y_cumsum = jnp.cumsum(y_weighted)
    y_sq_cumsum = jnp.cumsum(y_sq_weighted)

    sum_left = y_cumsum[:-1]
    sum_right = y_cumsum[-1] - y_cumsum[:-1]
    ss_left = y_sq_cumsum[:-1]
    ss_right = y_sq_cumsum[-1] - y_sq_cumsum[:-1]

    # Avoid division by zero when a position has no masked samples.
    safe_counts_left = jnp.where(counts_left[:-1] > 0, counts_left[:-1], 1.0)
    safe_counts_right = jnp.where(counts_right[:-1] > 0, counts_right[:-1], 1.0)

    # Variance = E[y^2] - E[y]^2; weighted by counts
    mse_left = ss_left / safe_counts_left - (sum_left / safe_counts_left) ** 2
    mse_right = ss_right / safe_counts_right - (sum_right / safe_counts_right) ** 2
    weighted_mse = (counts_left[:-1] * mse_left + counts_right[:-1] * mse_right) / jnp.where(n_node > 0, n_node, 1.0)
    weighted_mse = jnp.where(valid, weighted_mse, jnp.inf)

    # If the node is too small to split, every entry is already inf.
    too_small = n_node < 2 * min_samples_leaf
    weighted_mse = jnp.where(too_small, jnp.inf, weighted_mse)

    thresholds = (x_sorted[:-1] + x_sorted[1:]) / 2.0
    best_idx = jnp.argmin(weighted_mse)
    return thresholds[best_idx], weighted_mse[best_idx]


@jax.jit
@partial(jax.vmap, in_axes=(None, None, None, 0, None))
def _best_split_for_features(X, y, mask, feature, min_samples_leaf):
    """Vectorised best-split search over a set of feature indices.

    Operates on a fixed-shape sample mask so the function is compiled once per
    ``(n_samples, max_features)`` shape rather than once per node size.
    """
    x = X[:, feature]
    order = jnp.argsort(x)
    threshold, impurity = _best_split_one_feature(
        x[order], y[order], mask[order], min_samples_leaf
    )
    return threshold, impurity


@jax.jit
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


@partial(jax.jit, static_argnums=(2,))
def _predict_tree(tree, X, max_depth):
    """Route every row of ``X`` through a fixed tree and return leaf values.

    Parameters
    ----------
    tree : dict
        Tree arrays produced by :class:`DecisionTreeRegressor`.
    X : jnp.ndarray, shape (n_samples, n_features)
    max_depth : int
        Maximum depth of the tree (static).

    Returns
    -------
    predictions : jnp.ndarray, shape (n_samples,)
    """
    n_samples = X.shape[0]
    node = jnp.zeros(n_samples, dtype=jnp.int32)

    # Always walk the fixed maximum depth.  Leaf rows are kept unchanged via
    # jnp.where, so the early-exit optimisation is unnecessary and would block
    # JIT compilation.
    for _ in range(max_depth + 1):
        is_leaf = tree["is_leaf"][node]

        # Avoid invalid -1 indexing for already-leaf nodes.
        feature = jnp.where(is_leaf, 0, tree["feature"][node])
        threshold = tree["threshold"][node]
        left_child = tree["left_child"][node]
        right_child = tree["right_child"][node]

        x_feature = X[jnp.arange(n_samples), feature]
        go_left = x_feature <= threshold
        next_node = jnp.where(go_left, left_child, right_child)
        node = jnp.where(is_leaf, node, next_node)

    return tree["value"][node]


def _new_tree(max_depth):
    max_nodes = 2 ** (max_depth + 1) - 1
    return {
        "feature": jnp.full(max_nodes, -1, dtype=jnp.int32),
        "threshold": jnp.zeros(max_nodes, dtype=jnp.float32),
        "value": jnp.zeros(max_nodes, dtype=jnp.float32),
        "left_child": jnp.full(max_nodes, -1, dtype=jnp.int32),
        "right_child": jnp.full(max_nodes, -1, dtype=jnp.int32),
        "is_leaf": jnp.zeros(max_nodes, dtype=bool),
        "n_nodes": jnp.asarray(0, dtype=jnp.int32),
    }


def _set_leaf(tree, node_idx, value):
    tree["value"] = tree["value"].at[node_idx].set(value)
    tree["is_leaf"] = tree["is_leaf"].at[node_idx].set(True)
    return tree


def _fit_tree(X, y, max_depth, min_samples_leaf, max_features, key):
    """Fit a single regression tree and return its fixed-size array dict."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    tree = _new_tree(max_depth)

    # Each queue entry: (node_index, sample_mask, depth, rng_key)
    node_queue = [(0, jnp.ones(n_samples, dtype=bool), 0, key)]
    next_free_idx = 1

    while node_queue:
        node_idx, mask, depth, key = node_queue.pop(0)
        n_node = int(jnp.sum(mask))
        node_value = jnp.mean(y[mask])

        if depth >= max_depth or n_node < 2 * min_samples_leaf:
            tree = _set_leaf(tree, node_idx, node_value)
            continue

        key, subkey = random.split(key)
        features = random.choice(subkey, n_features, shape=(max_features,), replace=False)

        thresholds, impurities = _best_split_for_features(
            X, y, mask, features, min_samples_leaf
        )
        best_feature, best_threshold, best_impurity = _choose_best_split(
            thresholds, impurities, features
        )

        if bool(best_feature < 0):
            tree = _set_leaf(tree, node_idx, node_value)
            continue

        x_feature = X[:, best_feature]
        left_mask = mask & (x_feature <= best_threshold)
        right_mask = mask & ~left_mask
        n_left = int(jnp.sum(left_mask))
        n_right = n_node - n_left

        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            tree = _set_leaf(tree, node_idx, node_value)
            continue

        left_child = next_free_idx
        right_child = next_free_idx + 1
        next_free_idx += 2

        tree["feature"] = tree["feature"].at[node_idx].set(best_feature)
        tree["threshold"] = tree["threshold"].at[node_idx].set(best_threshold)
        tree["left_child"] = tree["left_child"].at[node_idx].set(left_child)
        tree["right_child"] = tree["right_child"].at[node_idx].set(right_child)
        tree["is_leaf"] = tree["is_leaf"].at[node_idx].set(False)

        node_queue.append((left_child, left_mask, depth + 1, key))
        node_queue.append((right_child, right_mask, depth + 1, key))

    tree["n_nodes"] = jnp.asarray(next_free_idx, dtype=jnp.int32)
    return tree


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
        return _new_tree(self.max_depth)

    def _set_leaf(self, tree, node_idx, value):
        return _set_leaf(tree, node_idx, value)

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
        self.tree_ = _fit_tree(
            X, y, self.max_depth, self.min_samples_leaf, self.max_features_, key
        )
        self.n_nodes_ = int(self.tree_["n_nodes"])
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
        return _predict_tree(self.tree_, X, self.max_depth)
