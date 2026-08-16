"""Matplotlib plotting helpers for rfjax permutation-test results.

These functions are intentionally kept separate from the core package so that
``matplotlib`` is only required when plots are actually produced.  Install the
plotting extra with::

    pip install "rfjax[plot]"
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rfjax plotting requires matplotlib. "
        'Install it with: pip install "rfjax[plot]"'
    ) from exc


def _to_numpy(arr):
    """Convert JAX/NumPy arrays to a host NumPy array."""
    return np.asarray(arr)


def _infer_observed(result):
    """Infer the observed test statistic from an rfjax result dict."""
    orig = result["originalStat"]

    if isinstance(orig, dict):
        if "Permuted MSE" in orig and "Original MSE" in orig:
            return float(orig["Permuted MSE"] - orig["Original MSE"])
        raise ValueError(
            "Cannot infer observed statistic from originalStat dict with keys: "
            f"{list(orig.keys())}"
        )

    arr = _to_numpy(orig)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.ravel()[0])
    if arr.shape == (2,):
        # mse_compare returns [stat_m1, stat_m2]; the permutation differences
        # are computed as stat_m1 - stat_m2.
        return float(arr[0] - arr[1])

    raise ValueError(
        f"Cannot infer observed statistic from originalStat array with shape {arr.shape}"
    )


def plot_null_distribution(result, ax=None, bins=30, observed=None, **kwargs):
    """Plot the permutation null distribution for an rfjax test result.

    Parameters
    ----------
    result : dict
        Output from :func:`rfjax.mse_test` or :func:`rfjax.mse_compare`.
        Must contain ``PermDiffs`` and ``originalStat``.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  A new figure is created if not provided.
    bins : int, default=30
        Number of histogram bins.
    observed : float, optional
        Observed test statistic.  If omitted, it is inferred from
        ``result["originalStat"]``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.hist``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    perms = _to_numpy(result["PermDiffs"])
    if observed is None:
        observed = _infer_observed(result)
    observed = float(observed)

    ax.hist(
        perms,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
        **kwargs,
    )
    ax.axvline(
        observed,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"observed={observed:.3g}",
    )
    ax.set_xlabel("Permutation test statistic")
    ax.set_ylabel("Frequency")
    ax.set_title("Null distribution")
    ax.legend()
    return ax


def plot_importance(result, kind="SDImp", feature_names=None, ax=None, **kwargs):
    """Plot variable importance from a permutation-importance result.

    Works with the output of :func:`rfjax.permutation_importance` and
    :func:`rfjax.holdout_rf`, whose ``Importance_Table`` has rows
    ``SDImp``, ``MSEImp``, and ``Pval``.

    Parameters
    ----------
    result : dict
        Must contain ``Importance_Table`` with shape ``(3, n_features)``.
    kind : {"SDImp", "MSEImp", "Pval"}, default="SDImp"
        Which importance metric to plot.
    feature_names : sequence of str, optional
        Labels for the variables.  Defaults to ``X0``, ``X1``, ...
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  A new figure is created if not provided.
    **kwargs
        Additional keyword arguments forwarded to ``ax.barh``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for the plot.
    """
    row_map = {"SDImp": 0, "MSEImp": 1, "Pval": 2}
    if kind not in row_map:
        raise ValueError(f"kind must be one of {list(row_map)}, got {kind!r}")

    table = _to_numpy(result["Importance_Table"])
    if table.ndim != 2 or table.shape[0] != 3:
        raise ValueError(
            f"Importance_Table must have shape (3, n_features), got {table.shape}"
        )

    values = table[row_map[kind]]
    n_vars = len(values)
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_vars)]
    elif len(feature_names) != n_vars:
        raise ValueError(
            f"feature_names has length {len(feature_names)} but {n_vars} variables present"
        )

    if ax is None:
        _, ax = plt.subplots()

    y_pos = np.arange(n_vars)
    ax.barh(
        y_pos,
        values,
        align="center",
        color="steelblue",
        edgecolor="white",
        **kwargs,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel(kind)
    ax.set_title(f"Variable importance ({kind})")

    if kind == "Pval":
        ax.axvline(
            0.05,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=r"$\alpha=0.05$",
        )
        ax.legend()

    return ax
