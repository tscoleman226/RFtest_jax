"""Smoke tests for the optional rfjax plotting helpers.

These tests skip automatically if matplotlib is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib
import matplotlib.pyplot as plt

from rfjax.plotting import plot_importance, plot_null_distribution


@pytest.fixture(autouse=True)
def _non_interactive_matplotlib():
    """Use a non-interactive backend so tests run on headless hosts."""
    matplotlib.use("Agg", force=True)
    yield


def _make_null_result():
    rng = np.random.default_rng(0)
    return {
        "PermDiffs": rng.normal(size=100),
        "originalStat": {"Original MSE": 0.5, "Permuted MSE": 0.8},
        "test_stat": "MSE",
    }


def _make_compare_result():
    rng = np.random.default_rng(0)
    return {
        "PermDiffs": rng.normal(size=100),
        "originalStat": np.array([0.5, 0.8]),
        "test_stat": "MSE",
    }


def _make_importance_result():
    return {
        "Importance_Table": np.array([
            [2.0, -1.0, 0.5, 0.0, -0.2],   # SDImp
            [0.3, -0.1, 0.05, 0.0, -0.02], # MSEImp
            [0.01, 0.3, 0.6, 0.8, 0.9],    # Pval
        ]),
    }


def test_plot_null_distribution_from_mse_test_dict():
    result = _make_null_result()
    fig, ax = plt.subplots()
    out = plot_null_distribution(result, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_null_distribution_from_mse_compare_array():
    result = _make_compare_result()
    fig, ax = plt.subplots()
    out = plot_null_distribution(result, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_null_distribution_with_explicit_observed():
    result = _make_null_result()
    fig, ax = plt.subplots()
    out = plot_null_distribution(result, ax=ax, observed=0.25)
    assert out is ax
    plt.close(fig)


def test_plot_importance_sd():
    result = _make_importance_result()
    fig, ax = plt.subplots()
    out = plot_importance(result, kind="SDImp", ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_importance_pval():
    result = _make_importance_result()
    fig, ax = plt.subplots()
    out = plot_importance(result, kind="Pval", ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_importance_with_feature_names():
    result = _make_importance_result()
    fig, ax = plt.subplots()
    names = ["age", "income", "height", "weight", "score"]
    out = plot_importance(result, kind="MSEImp", feature_names=names, ax=ax)
    assert out is ax
    assert ax.get_yticklabels()[0].get_text() == names[0]
    plt.close(fig)


def test_plot_importance_bad_kind_raises():
    result = _make_importance_result()
    with pytest.raises(ValueError, match="kind must be one of"):
        plot_importance(result, kind="Unknown")


def test_plot_importance_bad_table_shape_raises():
    result = {"Importance_Table": np.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="Importance_Table must have shape"):
        plot_importance(result)
