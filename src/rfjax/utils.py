"""Small helpers shared across rfjax modules."""

import jax.numpy as jnp


def ensure_2d(X):
    """Ensure an array is at least 2-D."""
    X = jnp.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def mse(pred, target):
    """Mean squared error."""
    return jnp.mean((pred - target) ** 2)
