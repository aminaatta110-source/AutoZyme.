"""Layer 2 -- ranking candidate genes inside a case's ROH blocks.

The task is not "is this variant pathogenic". It is "of the genes inside this
patient's homozygous blocks, which one explains the disease". That is a ranking
problem, so the model is scored with ranking metrics: how often the true causal
gene lands at rank 1, inside the top 5, inside the top 10, and where it sits on
average.

Two details keep the evaluation honest:

* Cross-validation is grouped by case, so no case contributes rows to both the
  training and the test fold.
* Several features are re-expressed as within-case ranks before training. What
  matters clinically is whether a gene looks worse than its neighbours in the
  same block, not its raw score.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .features import FEATURE_COLUMNS

__all__ = [
    "RELATIVE_FEATURES",
    "add_within_case_features",
    "rank_metrics",
    "baseline_rankings",
    "cross_validate_ranker",
    "fit_final_model",
    "score_and_rank",
]

# Features that only mean something relative to the other candidates in the case.
RELATIVE_FEATURES = [
    "max_cadd",
    "neg_log_af",
    "phenotype_match",
    "gene_p_rec",
    "position_in_block",
]


def add_within_case_features(candidates):
    """Add within-case percentile versions of the competitive features."""
    table = candidates.copy()
    for column in RELATIVE_FEATURES:
        table[f"{column}_rel"] = table.groupby("sample_id")[column].rank(pct=True)
    return table


def model_feature_columns():
    return FEATURE_COLUMNS + [f"{c}_rel" for c in RELATIVE_FEATURES]


def rank_metrics(scored, score_column="score", label_column="is_causal",
                 group_column="sample_id", ks=(1, 3, 5, 10)):
    """Ranking metrics computed per case, then averaged.

    Ties are handled with average ranking so a model that gives every candidate
    the same score cannot look good by accident.
    """
    table = scored.copy()
    table["rank"] = (
        table.groupby(group_column)[score_column]
        .rank(ascending=False, method="average")
    )

    causal = table[table[label_column] == 1]
    if causal.empty:
        return {}

    ranks = causal["rank"].to_numpy()
    candidates_per_case = table.groupby(group_column).size().mean()

    metrics = {"n_cases": int(causal[group_column].nunique())}
    for k in ks:
        metrics[f"top_{k}"] = float((ranks <= k).mean())
    metrics["mrr"] = float((1.0 / ranks).mean())
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(ranks.mean())
    metrics["mean_candidates_per_case"] = float(candidates_per_case)
    return metrics


def baseline_rankings(candidates):
    """Reference rankings the model has to beat.

    ``cadd_only`` is what the previous AutoZyme dashboard did: sort by CADD and
    show the doctor the top of the list.
    """
    baselines = {}
    rng = np.random.default_rng(0)

    cadd = candidates.copy()
    cadd["score"] = cadd["max_cadd"]
    baselines["cadd_only"] = rank_metrics(cadd)

    pheno = candidates.copy()
    pheno["score"] = pheno["phenotype_match"]
    baselines["phenotype_only"] = rank_metrics(pheno)

    combined = candidates.copy()
    combined["score"] = (
        combined.groupby("sample_id")["max_cadd"].rank(pct=True)
        + combined.groupby("sample_id")["phenotype_match"].rank(pct=True)
    )
    baselines["cadd_plus_phenotype"] = rank_metrics(combined)

    shuffled = candidates.copy()
    shuffled["score"] = rng.random(len(shuffled))
    baselines["random"] = rank_metrics(shuffled)
    return baselines


def _new_estimator(seed=0):
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=300,
        learning_rate=0.06,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=seed,
    )


def cross_validate_ranker(candidates, n_splits=5, seed=0):
    """Grouped cross-validation. Returns out-of-fold scores and metrics."""
    table = add_within_case_features(candidates)
    columns = model_feature_columns()
    X = table[columns].to_numpy(dtype=float)
    y = table["is_causal"].to_numpy(dtype=int)
    groups = table["sample_id"].to_numpy()

    out_of_fold = np.zeros(len(table))
    splitter = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        model = _new_estimator(seed + fold)
        model.fit(X[train_idx], y[train_idx])
        out_of_fold[test_idx] = model.predict_proba(X[test_idx])[:, 1]

        fold_table = table.iloc[test_idx].copy()
        fold_table["score"] = out_of_fold[test_idx]
        fold_metrics.append({"fold": fold, **rank_metrics(fold_table)})

    table["score"] = out_of_fold
    return table, rank_metrics(table), pd.DataFrame(fold_metrics)


def fit_final_model(candidates, seed=0):
    """Train on the whole cohort for deployment."""
    table = add_within_case_features(candidates)
    columns = model_feature_columns()
    model = _new_estimator(seed)
    model.fit(table[columns].to_numpy(dtype=float), table["is_causal"].to_numpy(dtype=int))
    return model, columns


def score_and_rank(model, columns, candidates):
    """Score a new case and return its candidate genes in ranked order."""
    table = add_within_case_features(candidates)
    table["score"] = model.predict_proba(table[columns].to_numpy(dtype=float))[:, 1]
    table["rank"] = (
        table.groupby("sample_id")["score"].rank(ascending=False, method="first").astype(int)
    )
    return table.sort_values(["sample_id", "rank"]).reset_index(drop=True)


def permutation_importance_by_rank(table, model, columns, n_repeats=3, seed=0):
    """Feature importance measured in the metric that matters: mean reciprocal rank."""
    rng = np.random.default_rng(seed)
    base = table.copy()
    base["score"] = model.predict_proba(base[columns].to_numpy(dtype=float))[:, 1]
    baseline_mrr = rank_metrics(base)["mrr"]

    records = []
    for column in columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = table.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            shuffled["score"] = model.predict_proba(
                shuffled[columns].to_numpy(dtype=float)
            )[:, 1]
            drops.append(baseline_mrr - rank_metrics(shuffled)["mrr"])
        records.append({"feature": column, "mrr_drop": float(np.mean(drops))})

    return pd.DataFrame(records).sort_values("mrr_drop", ascending=False).reset_index(drop=True)
