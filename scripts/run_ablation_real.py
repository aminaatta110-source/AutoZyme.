"""Run the variant-only / ROH-only / integrated ablation on a REAL cohort.

Reads the files produced by the cohort builders: real genomes, real ROH blocks,
real CADD scores, with one known pathogenic variant introduced per case.

    python scripts/run_ablation_real.py --cohort cohort/

The three arms answer the question the companion review posed:

* variant-only  -- every gene in the case carrying a rare variant, scored on
                   variant and gene features. No ROH restriction, no block
                   features.
* roh-only      -- candidates restricted to genes inside detected blocks,
                   scored on block geometry alone.
* integrated    -- restricted to blocks, all features.

Because the ROH blocks are called from real genotypes, a difference between the
arms is evidence about the data rather than an artefact of how it was made.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme import model  # noqa: E402
from autozyme.cohort import (INTEGRATED_FEATURES, ROH_FEATURES,  # noqa: E402
                             VARIANT_FEATURES, gene_level, load_cohort)

RELATIVE = {
    "variant_only": ["max_cadd", "neg_log_af"],
    "roh_only": ["position_in_block"],
    "integrated": ["max_cadd", "neg_log_af", "position_in_block"],
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    cohort = Path(args.cohort)
    outdir = Path(args.out) if args.out else cohort
    outdir.mkdir(parents=True, exist_ok=True)

    variants, cases, blocks = load_cohort(cohort)
    print(f"cohort: {len(cases)} cases, {len(variants):,} variant calls, "
          f"{len(blocks):,} ROH blocks")
    print("Normal controls with pathogenic variants introduced in silico.\n")

    truth = cases.set_index("sample_id")["causal_gene"]

    # --- variant-only: no ROH restriction, no block features ---
    v = gene_level(variants)
    v["is_causal"] = (v["gene"] == v["sample_id"].map(truth)).astype(int)
    v = v[v["sample_id"].isin(cases["sample_id"])]

    # --- ROH-restricted arms ---
    r = gene_level(variants, blocks)
    r["is_causal"] = (r["gene"] == r["sample_id"].map(truth)).astype(int)
    r = r[r["sample_id"].isin(cases["sample_id"])]

    captured = r.groupby("sample_id")["is_causal"].max()
    print(f"causal gene inside a detected block: {captured.mean():.1%} of cases")
    keep = captured[captured == 1].index
    v = v[v["sample_id"].isin(keep)]
    r = r[r["sample_id"].isin(keep)]
    print(f"evaluating {len(keep)} cases where ranking is possible\n")

    arms, scored_arms = {}, {}
    for name, table, base in [
        ("variant_only", v, VARIANT_FEATURES),
        ("roh_only", r, ROH_FEATURES),
        ("integrated", r, INTEGRATED_FEATURES),
    ]:
        t = add_relative(table, RELATIVE[name])
        cols = base + [f"{c}_rel" for c in RELATIVE[name]]
        cols = [c for c in cols if c in t.columns]
        scored, metrics = cross_validate(t, cols)
        arms[name] = metrics
        scored_arms[name] = scored

    print(f"{'arm':<15}{'top-1':>8}{'top-5':>8}{'top-10':>8}{'MRR':>8}"
          f"{'median':>8}{'cands':>8}")
    for name, m in arms.items():
        print(f"{name:<15}{m['top_1']:>8.3f}{m['top_5']:>8.3f}{m['top_10']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['median_rank']:>8.0f}"
              f"{m['mean_candidates_per_case']:>8.1f}")

    # Baselines on the integrated candidate set.
    print()
    base_table = scored_arms["integrated"]
    for label, column in [("CADD only", "max_cadd"), ("rarity only", "neg_log_af")]:
        tmp = base_table.copy()
        tmp["score"] = tmp[column]
        m = model.rank_metrics(tmp)
        print(f"baseline {label:<14} top-1 {m['top_1']:.3f}  top-10 {m['top_10']:.3f}  "
              f"MRR {m['mrr']:.3f}")

    y = base_table["is_causal"].to_numpy()
    brier = float(brier_score_loss(y, base_table["score"].to_numpy()))
    print(f"\nBrier score (integrated): {brier:.4f}")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ks = np.arange(1, 21)
    for name, colour in [("variant_only", "#a03030"), ("roh_only", "#c08a30"),
                         ("integrated", "#2f6f4f")]:
        t = scored_arms[name].copy()
        t["rank"] = t.groupby("sample_id")["score"].rank(ascending=False, method="average")
        rk = t.loc[t["is_causal"] == 1, "rank"].to_numpy()
        ax.plot(ks, [(rk <= k).mean() for k in ks], marker="o", ms=3,
                color=colour, label=name.replace("_", " "))
    ax.set_xlabel("Rank cut-off $k$"); ax.set_ylabel("Causal gene recovered")
    ax.set_ylim(0, 1.02); ax.set_title("Ablation on real genomes")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_ablation_real.png", dpi=150, bbox_inches="tight")

    with open(outdir / "ablation_real_metrics.json", "w") as handle:
        json.dump({
            "data_source": "SPIKE-IN on real 1000 Genomes samples",
            "n_cases_built": int(len(cases)),
            "n_cases_evaluated": int(len(keep)),
            "causal_gene_in_block_rate": float(captured.mean()),
            "arms": arms,
            "brier_integrated": brier,
        }, handle, indent=2)
    print(f"\nwrote results to {outdir}")


if __name__ == "__main__":
    main()
