"""Permutation-based hypothesis tests for JAX random forests.

These functions mirror the main procedures in the R package ``RFtest``:
``MSE_Test``, ``MSE_compare``, ``permtestImp``, and ``f_holdoutRF``.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap

from rfjax.forest import RandomForestRegressor, _default_subsample_size


def _train_test_split(X, y, n_test, key):
    """Randomly split data into train and test sets."""
    n = X.shape[0]
    key, subkey = random.split(key)
    perm = random.permutation(subkey, n)
    test_indices = perm[:n_test]
    train_indices = perm[n_test:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices], key


def _permute_columns(X, columns, key):
    """Row-wise permute the specified columns together."""
    n = X.shape[0]
    key, subkey = random.split(key)
    perm = random.permutation(subkey, n)
    X_perm = jnp.array(X)
    X_perm = X_perm.at[:, columns].set(X[perm][:, columns])
    return X_perm, key


def _gather_pool(pool, split_idx, comp_idx, y_test):
    """Compute MSE difference for one permutation of the pooled predictions."""
    pred_full = jnp.mean(pool[:, split_idx], axis=1)
    pred_reduced = jnp.mean(pool[:, comp_idx], axis=1)
    mse_full = jnp.mean((pred_full - y_test) ** 2)
    mse_reduced = jnp.mean((pred_reduced - y_test) ** 2)
    return mse_reduced - mse_full


def _make_forest_kwargs(ntree, k, mtry, max_depth, min_samples_leaf):
    return {
        "ntree": ntree,
        "k": k,
        "mtry": mtry,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }


def mse_test(
    X,
    y,
    var,
    X_test=None,
    y_test=None,
    n_test=None,
    B=1000,
    ntree=500,
    k=None,
    mtry=None,
    max_depth=10,
    min_samples_leaf=5,
    importance=True,
    random_state=None,
):
    """Permutation F-test comparing a full forest to a reduced forest.

    The reduced forest is trained on a copy of the training data in which the
    column(s) ``var`` have been row-wise permuted.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    var : int or sequence of int
        Column index/indices of the variable(s) to test.
    X_test : array-like or None, default=None
        External test covariates. If None, ``n_test`` observations are held
        out from ``X``.
    y_test : array-like or None, default=None
        External test responses.
    n_test : int or None, default=None
        Number of observations to hold out when ``X_test`` is not supplied.
        Defaults to ``n_samples`` (i.e., train and test on the same data,
        matching the RFtest default when ``X.test`` is omitted).
    B : int, default=1000
        Number of permutations.
    ntree : int, default=500
        Number of trees per ensemble.
    k : int or None, default=None
        Subsample size for each tree.
    mtry : int or None, default=None
        Number of features sampled at each split.
    max_depth : int, default=10
        Maximum tree depth.
    min_samples_leaf : int, default=5
        Minimum leaf size.
    importance : bool, default=True
        Compute standardized importance scores.
    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with keys ``variables``, ``originalStat``, ``PermDiffs``,
        ``Importance``, ``Pvalue``, ``test_pts``, ``model_original``,
        ``model_permuted``, ``test_stat``.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32).ravel()
    key = random.PRNGKey(random_state if random_state is not None else 0)

    if isinstance(var, int):
        var = [var]
    var = jnp.asarray(var, dtype=jnp.int32)

    if X_test is None:
        if n_test is None:
            n_test = X.shape[0]
        X_train, y_train, X_test, y_test, key = _train_test_split(X, y, n_test, key)
    else:
        X_train = X
        y_train = y
        X_test = jnp.asarray(X_test, dtype=jnp.float32)
        y_test = jnp.asarray(y_test, dtype=jnp.float32).ravel()

    X_train_perm, key = _permute_columns(X_train, var, key)

    forest_kwargs = _make_forest_kwargs(ntree, k, mtry, max_depth, min_samples_leaf)

    key, k1, k2 = random.split(key, 3)
    rf_og = RandomForestRegressor(**forest_kwargs, random_state=int(k1[0]))
    rf_pm = RandomForestRegressor(**forest_kwargs, random_state=int(k2[0]))
    rf_og.fit(X_train, y_train)
    rf_pm.fit(X_train_perm, y_train)

    P = rf_og._predict_all(X_test)
    PR = rf_pm._predict_all(X_test)

    pred_og = jnp.mean(P, axis=1)
    pred_pm = jnp.mean(PR, axis=1)
    mse_og = jnp.mean((pred_og - y_test) ** 2)
    mse_pm = jnp.mean((pred_pm - y_test) ** 2)
    diff_0 = mse_pm - mse_og

    pool = jnp.concatenate([P, PR], axis=1)
    n_total = pool.shape[1]

    key, subkey = random.split(key)
    perm_keys = random.split(subkey, B)
    perms = vmap(lambda k: random.permutation(k, n_total))(perm_keys)
    splits = perms[:, :ntree]
    complements = perms[:, ntree:]

    mse_diffs = vmap(_gather_pool, in_axes=(None, 0, 0, None))(
        pool, splits, complements, y_test
    )

    p_value = jnp.mean(jnp.concatenate([jnp.ones(1), (diff_0 < mse_diffs).astype(float)]))

    importances = None
    if importance:
        sd_imp = (diff_0 - jnp.mean(mse_diffs)) / jnp.std(mse_diffs)
        z_imp = jax.scipy.stats.norm.cdf(sd_imp)
        importances = {
            "Standard Deviation Importance": float(sd_imp),
            "Standard Normal Importance": float(z_imp),
        }

    return {
        "variables": tuple(int(v) for v in var),
        "originalStat": {
            "Original MSE": float(mse_og),
            "Permuted MSE": float(mse_pm),
        },
        "PermDiffs": mse_diffs,
        "Importance": importances,
        "Pvalue": {"Full Model P": float(p_value)},
        "test_pts": X_test,
        "model_original": rf_og,
        "model_permuted": rf_pm,
        "test_stat": "MSE",
    }


def mse_compare(
    m1,
    m2,
    X_test,
    y_test=None,
    B=1000,
    test_stat="MSE",
    random_state=None,
):
    """Permutation test comparing two pre-trained forests.

    Parameters
    ----------
    m1, m2 : RandomForestRegressor
        The two fitted ensembles.
    X_test : array-like, shape (n_samples, n_features)
    y_test : array-like or None, default=None
        Test responses. If None, the KS-like statistic ``max(|pred1 - pred2|)``
        is used.
    B : int, default=1000
        Number of permutations.
    test_stat : {"MSE", "KS", "diff"} or callable, default="MSE"
        Test statistic. A callable must accept ``(y_test, pred1, pred2)`` and
        return a scalar or a 2-vector (the latter interpreted as
        ``stat1 - stat2`` for ``MSE``).
    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    result : dict
        Dictionary with test results matching the ``mse_test`` output shape.
    """
    X_test = jnp.asarray(X_test, dtype=jnp.float32)
    key = random.PRNGKey(random_state if random_state is not None else 0)

    P1 = m1._predict_all(X_test)
    P2 = m2._predict_all(X_test)
    nt1 = P1.shape[1]
    nt2 = P2.shape[1]
    pool = jnp.concatenate([P1, P2], axis=1)
    n_total = pool.shape[1]

    if y_test is None:
        y_test = jnp.zeros(X_test.shape[0])
    else:
        y_test = jnp.asarray(y_test, dtype=jnp.float32).ravel()

    if callable(test_stat):
        stat_fn = test_stat
    elif test_stat == "MSE":
        stat_fn = lambda yt, p1, p2: jnp.array([
            jnp.mean((p1 - yt) ** 2),
            jnp.mean((p2 - yt) ** 2),
        ])
    elif test_stat == "KS":
        stat_fn = lambda yt, p1, p2: jnp.max(jnp.abs(p1 - p2))
    elif test_stat == "diff":
        stat_fn = lambda yt, p1, p2: jnp.mean(p1 - p2)
    else:
        raise ValueError(f"Unknown test_stat: {test_stat}")

    pred_1 = jnp.mean(P1, axis=1)
    pred_2 = jnp.mean(P2, axis=1)
    ts0 = stat_fn(y_test, pred_1, pred_2)
    ts_temp = ts0

    if test_stat == "MSE":
        ts0 = ts0[0] - ts0[1]

    key, subkey = random.split(key)
    perm_keys = random.split(subkey, B)
    perms = vmap(lambda k: random.permutation(k, n_total))(perm_keys)
    splits = perms[:, :nt1]
    complements = perms[:, nt1:]

    def _one_perm(split_idx, comp_idx):
        pred_t = jnp.mean(pool[:, split_idx], axis=1)
        pred_r = jnp.mean(pool[:, comp_idx], axis=1)
        if test_stat == "MSE":
            temp = stat_fn(y_test, pred_r, pred_t)
            return temp[0] - temp[1]
        return stat_fn(y_test, pred_r, pred_t)

    perm_stats = vmap(_one_perm)(splits, complements)

    if test_stat == "diff":
        p_value = jnp.mean(jnp.concatenate([jnp.ones(1), (jnp.abs(ts0) < jnp.abs(perm_stats)).astype(float)]))
    else:
        p_value = jnp.mean(jnp.concatenate([jnp.ones(1), (ts0 < perm_stats).astype(float)]))

    sd_imp = (ts0 - jnp.mean(perm_stats)) / jnp.std(perm_stats)
    z_imp = jax.scipy.stats.norm.cdf(sd_imp)

    return {
        "variables": None,
        "originalStat": ts_temp,
        "PermDiffs": perm_stats,
        "Importance": {
            "Standard Deviation Importance": float(sd_imp),
            "Standard Normal Importance": float(z_imp),
        },
        "Pvalue": {"Full Model P": float(p_value)},
        "test_pts": X_test,
        "model_original": m1,
        "model_permuted": m2,
        "test_stat": test_stat,
    }


def permutation_importance(
    X,
    y,
    single_forest=True,
    X_test=None,
    y_test=None,
    n_test=None,
    nbtree=30,
    verbose=False,
    keep_forest=False,
    B=1000,
    k=None,
    mtry=None,
    max_depth=10,
    min_samples_leaf=5,
    random_state=None,
):
    """Marginal permutation importance for every variable.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    single_forest : bool, default=True
        If True, a single forest is fit once and each variable is tested by
        training only a reduced forest. Otherwise ``mse_test`` is run
        independently for each variable.
    X_test, y_test, n_test : optional
        Test data; see :func:`mse_test`.
    nbtree : int or sequence of int, default=30
        Number of trees used for each variable. May be a vector with one entry
        per variable.
    verbose : bool, default=False
    keep_forest : bool, default=False
        Return the original forest (only used when ``single_forest=True``).
    B : int, default=1000
    k, mtry, max_depth, min_samples_leaf : optional
        Passed to :class:`RandomForestRegressor`.
    random_state : int or None, default=None

    Returns
    -------
    result : dict
        ``Importance_Table`` with rows ``SDImp``, ``MSEImp``, ``Pval`` and one
        column per variable.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32).ravel()
    n_features = X.shape[1]
    vars = list(range(n_features))

    if isinstance(nbtree, int):
        nbtree = [nbtree] * n_features

    forest_kwargs = {"k": k, "mtry": mtry, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}

    if single_forest:
        if X_test is None:
            if n_test is None:
                n_test = X.shape[0]
            key = random.PRNGKey(random_state if random_state is not None else 0)
            X_train, y_train, X_test, y_test, _ = _train_test_split(X, y, n_test, key)
        else:
            X_train, y_train = X, y
            X_test = jnp.asarray(X_test, dtype=jnp.float32)
            y_test = jnp.asarray(y_test, dtype=jnp.float32).ravel()

        m0 = RandomForestRegressor(
            ntree=max(nbtree), **forest_kwargs,
            random_state=random_state if random_state is not None else 0,
        )
        m0.fit(X_train, y_train)

        def _test_var(v, ntree_v):
            X_perm = jnp.array(X_train)
            perm = random.permutation(random.PRNGKey(v + 123), X_train.shape[0])
            X_perm = X_perm.at[:, v].set(X_train[perm][:, v])
            m1 = RandomForestRegressor(
                ntree=ntree_v, **forest_kwargs,
                random_state=(random_state if random_state is not None else 0) + v + 1,
            )
            m1.fit(X_perm, y_train)
            obj = mse_compare(m1, m0, X_test, y_test, B=B, random_state=random_state)
            return {
                "SDImp": obj["Importance"]["Standard Deviation Importance"],
                "MSEImp": obj["originalStat"][1] - obj["originalStat"][0],
                "Pval": obj["Pvalue"]["Full Model P"],
            }
    else:
        def _test_var(v, ntree_v):
            obj = mse_test(
                X, y, var=v, X_test=X_test, y_test=y_test, n_test=n_test,
                B=B, ntree=ntree_v, k=k, mtry=mtry,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=(random_state if random_state is not None else 0) + v,
            )
            return {
                "SDImp": obj["Importance"]["Standard Deviation Importance"],
                "MSEImp": obj["originalStat"]["Permuted MSE"] - obj["originalStat"]["Original MSE"],
                "Pval": obj["Pvalue"]["Full Model P"],
            }

    out = [_test_var(v, nt) for v, nt in zip(vars, nbtree)]
    out_mat = jnp.array([[o["SDImp"], o["MSEImp"], o["Pval"]] for o in out]).T

    result = {
        "Importance_Table": out_mat,
        "call": {
            "single_forest": single_forest,
            "nbtree": nbtree,
            "B": B,
        },
    }
    if keep_forest and single_forest:
        result["TrainedModel"] = m0
    return result


def holdout_rf(
    X,
    y,
    X_test=None,
    y_test=None,
    n_test=None,
    B=1000,
    mintree=30,
    max_trees=None,
    verbose=False,
    mtry=None,
    k=None,
    max_depth=10,
    min_samples_leaf=5,
    keep_forest=False,
    random_state=None,
):
    """Variable importance via holdout forests.

    Each base learner is trained on a random ``mtry`` subset of features. For
    each variable, trees that did and did not include the variable are compared
    via a permutation test.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    X_test, y_test, n_test : optional
        Test data; see :func:`mse_test`.
    B : int, default=1000
        Number of permutations per variable.
    mintree : int, default=30
        Minimum number of trees that must contain each variable.
    max_trees : int or None, default=None
        Upper bound on the total number of trees built. Defaults to
        ``5 * n_features * mintree``.
    verbose : bool, default=False
    mtry : int or None, default=None
        Number of features per tree. Defaults to ``n_features // 3``.
    k, max_depth, min_samples_leaf : optional
        Passed to :class:`RandomForestRegressor`.
    keep_forest : bool, default=False
        Return the fitted holdout forest.
    random_state : int or None, default=None

    Returns
    -------
    result : dict
        ``Importance_Table`` with rows ``SDImp``, ``MSEImp``, ``Pval``.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32).ravel()
    n_features = X.shape[1]
    vars = list(range(n_features))

    if mtry is None:
        mtry = max(1, n_features // 3)
    if mtry >= n_features:
        raise ValueError("mtry should be less than total number of columns to use holdout procedure")

    if X_test is None:
        if n_test is None:
            n_test = X.shape[0]
        key = random.PRNGKey(random_state if random_state is not None else 0)
        X_train, y_train, X_test, y_test, key = _train_test_split(X, y, n_test, key)
    else:
        X_train, y_train = X, y
        X_test = jnp.asarray(X_test, dtype=jnp.float32)
        y_test = jnp.asarray(y_test, dtype=jnp.float32).ravel()
        key = random.PRNGKey(random_state if random_state is not None else 0)

    if max_trees is None:
        max_trees = 5 * n_features * mintree

    ntree_with_var = {v: 0 for v in vars}
    tree_var_mat = []
    trees = []
    feats_list = []

    base_seed = random_state if random_state is not None else 0
    build_key = random.PRNGKey(base_seed)

    def _build_one(seed):
        key_tree = random.PRNGKey(seed)
        feats = random.choice(key_tree, n_features, shape=(mtry,), replace=False)
        feats_sorted = jnp.sort(feats)
        tree = RandomForestRegressor(
            ntree=1, k=k, mtry=mtry, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=seed,
        )
        tree.fit(X_train[:, feats_sorted], y_train)
        return tree, feats

    tree0, feats0 = _build_one(base_seed)
    trees.append(tree0)
    feats_list.append(feats0)
    tree_var_mat.append([int(v in feats0) for v in vars])
    for v in feats0:
        ntree_with_var[int(v)] += 1

    b_n = 1
    while any(c < mintree for c in ntree_with_var.values()) and b_n < max_trees:
        tree, feats = _build_one(base_seed + b_n)
        trees.append(tree)
        feats_list.append(feats)
        tree_var_mat.append([int(v in feats) for v in vars])
        for v in feats:
            ntree_with_var[int(v)] += 1
        b_n += 1
        if verbose and b_n % 50 == 0:
            to_go = jnp.mean(jnp.array([mintree - ntree_with_var[v] for v in vars]))
            print(f"NumTrees Built: {b_n} || NumTrees per var to go: {to_go}")

    if any(c < mintree for c in ntree_with_var.values()):
        import warnings
        warnings.warn("Some models may not have enough base learners; consider running again with more models")

    preds = jnp.stack([
        tree.predict(X_test[:, jnp.sort(feats)])
        for tree, feats in zip(trees, feats_list)
    ], axis=0)
    tree_var_mat = jnp.array(tree_var_mat, dtype=bool)

    key_state = [key]

    def _compare_var(v):
        flags = tree_var_mat[:, v]
        preds_v = preds[flags, :]
        preds_no_v = preds[~flags, :]

        n_models = int(min(jnp.sum(flags), jnp.sum(~flags)))
        if n_models == 0:
            return jnp.array([jnp.nan, jnp.nan, jnp.nan])

        current_key = key_state[0]
        k1, k2, current_key = random.split(current_key, 3)
        key_state[0] = current_key

        left_idx = random.choice(k1, preds_v.shape[0], shape=(n_models,), replace=False)
        right_idx = random.choice(k2, preds_no_v.shape[0], shape=(n_models,), replace=False)
        preds_v = preds_v[left_idx, :]
        preds_no_v = preds_no_v[right_idx, :]

        avg_v = jnp.mean(preds_v, axis=0)
        avg_no_v = jnp.mean(preds_no_v, axis=0)
        mse_v = jnp.mean((y_test - avg_v) ** 2)
        mse_no_v = jnp.mean((y_test - avg_no_v) ** 2)
        d0 = mse_no_v - mse_v

        if verbose:
            print(f"{v} MSE difference: {d0}")

        pool = jnp.concatenate([preds_v, preds_no_v], axis=0)
        k3, current_key = random.split(current_key)
        key_state[0] = current_key

        perm_keys = random.split(k3, B)
        perms = vmap(lambda k: random.permutation(k, 2 * n_models))(perm_keys)
        splits = perms[:, :n_models]
        complements = perms[:, n_models:]

        def _one_perm(split_idx, comp_idx):
            avg_t = jnp.mean(pool[split_idx, :], axis=0)
            avg_c = jnp.mean(pool[comp_idx, :], axis=0)
            mse_t = jnp.mean((y_test - avg_t) ** 2)
            mse_c = jnp.mean((y_test - avg_c) ** 2)
            return mse_c - mse_t

        perm_stats = vmap(_one_perm)(splits, complements)
        sd_imp = (d0 - jnp.mean(perm_stats)) / jnp.std(perm_stats)
        pval = jnp.mean(jnp.concatenate([jnp.ones(1), (d0 < perm_stats).astype(float)]))
        return jnp.array([sd_imp, d0, pval])

    out_list = [_compare_var(v) for v in vars]
    out = jnp.stack(out_list, axis=1)

    result = {
        "Importance_Table": out,
        "call": {"mintree": mintree, "max_trees": max_trees, "B": B},
    }
    if keep_forest:
        result["TrainedModel"] = trees
    return result
