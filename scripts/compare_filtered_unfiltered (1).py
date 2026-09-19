"""Phase 3 -- compare FILTERED (block-restricted) vs UNFILTERED candidate ranking.

This is the actual redesigned experiment: same cases, same causal-gene labels,
one arm's candidates were pre-restricted to a detected ROH block before
scoring and one arm's were not. If block restriction is pulling its weight,
the unfiltered arm should rank the causal gene worse on average, since it is
choosing among many more genes without being told which ones sit in an
autozygous segment.

    python scripts/compare_filtered_unfiltered.py \
        --filtered cohort/ \
        --unfiltered cohort_unfiltered/ \
        --out cohort/

Expects --unfiltered to contain spikein_variants.csv built the same way as
the filtered cohort (phase 2 / spikein.build_spikein_case), just from the
rare_variants_unfiltered.csv + its own CADD scores, and using the SAME
causal_gene per sample_id as the filtered cohort (build_unfiltered_candidates
.py enforces this by reading --existing-cases).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme import model  # noqa: E402
from autozyme.cohort import VARIANT_FEATURES, gene_level, load_cohort  # noqa: E402


def add_relative(table, columns):
    out = table.copy()
    for column in columns:
        out[f"{column}_rel"] = out.groupby("sample_id")[column].rank(pct=True)
    return out


def cross_validate(table, columns, n_splits=5, seed=0):
    X = table[columns].to_numpy(dtype=float)
    y = table["is_causal"].to_numpy(dtype=int)
    groups = table["sample_id"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))

    oof = np.zeros(len(table))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        est = HistGradientBoostingClassifier(
            max_depth=4, max_iter=300, learning_rate=0.06,
            min_samples_leaf=15, l2_regularization=1.0, random_state=seed + fold)
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]

    scored = table.copy()
    scored["score"] = oof
    return scored, model.rank_metrics(scored)


def paired_top1_ci(a_scored, b_scored, n_boot=10_000, seed=0):
    """Bootstrap CI for (top-1 recall of a - top-1 recall of b), matched by
    sample_id so only cases present in both arms are compared."""
    def hits(scored):
        t = scored.copy()
        t["rank"] = t.groupby("sample_id")["score"].rank(ascending=False, method="average")
        return t[t["is_causal"] == 1].set_index("sample_id")["rank"].le(1).astype(float)

    ha, hb = hits(a_scored), hits(b_scored)
    common = ha.index.intersection(hb.index)
    ha, hb = ha.loc[common].to_numpy(), hb.loc[common].to_numpy()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(common), len(common))
        diffs[i] = ha[idx].mean() - hb[idx].mean()
    return len(common), float(diffs.mean()), (float(np.percentile(diffs, 2.5)),
                                              float(np.percentile(diffs, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtered", required=True)
    p.add_argument("--unfiltered", required=True)
    p.add_argument("--out")
    args = p.parse_args()

    outdir = Path(args.out) if args.out else Path(args.filtered)
    outdir.mkdir(parents=True, exist_ok=True)

    fv, fc, fb = load_cohort(args.filtered)
    truth = fc.set_index("sample_id")["causal_gene"]

    uv = pd.read_csv(Path(args.unfiltered) / "spikein_variants.csv")

    arms = {}
    scored_arms = {}
    for name, variants in [("filtered", fv), ("unfiltered", uv)]:
        t = gene_level(variants)   # NOTE: no blocks argument -- both arms scored
        t["is_causal"] = (t["gene"] == t["sample_id"].map(truth)).astype(int)
        t = t[t["sample_id"].isin(truth.index)]
        t = add_relative(t, ["max_cadd", "neg_log_af"])
        cols = [c for c in VARIANT_FEATURES + ["max_cadd_rel", "neg_log_af_rel"]
               if c in t.columns]
        scored, metrics = cross_validate(t, cols)
        arms[name] = metrics
        scored_arms[name] = scored
        print(f"{name:<12} n={metrics['n_cases']:<4} top-1={metrics['top_1']:.3f} "
              f"top-5={metrics['top_5']:.3f} top-10={metrics['top_10']:.3f} "
              f"MRR={metrics['mrr']:.3f} "
              f"mean_candidates={metrics['mean_candidates_per_case']:.1f}")

    n_common, diff_mean, diff_ci = paired_top1_ci(
        scored_arms["filtered"], scored_arms["unfiltered"])
    print(f"\nfiltered vs. unfiltered, top-1, {n_common} paired cases: "
          f"{diff_mean:+.3f} (95% CI {diff_ci[0]:+.3f} to {diff_ci[1]:+.3f})")
    if diff_ci[0] > 0:
        print("--> block restriction shows a measurable benefit "
             "(interval excludes zero)")
    elif diff_ci[1] < 0:
        print("--> unfiltered ranking is measurably BETTER "
             "(interval excludes zero, on the other side)")
    else:
        print("--> interval includes zero: block restriction neither clearly "
             "helps nor hurts ranking at this cohort size")

    with open(outdir / "filtered_vs_unfiltered.json", "w") as handle:
        json.dump({
            "arms": arms,
            "paired_top1_diff": {"n_cases": n_common, "mean": diff_mean,
                                 "ci95": list(diff_ci)},
        }, handle, indent=2)
    print(f"\nwrote {outdir}/filtered_vs_unfiltered.json")


if __name__ == "__main__":
    main()
