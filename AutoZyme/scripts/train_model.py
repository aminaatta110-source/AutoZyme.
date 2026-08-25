"""Fit the ranking model on a full cohort and save it for reuse.

    python scripts/train_model.py --cohort cohort/ --out cohort/ranker.joblib

The saved model is fitted on every case in the cohort. It is intended for
inspection and for scoring new individuals, not for reproducing the reported
metrics: those come from out-of-fold predictions, where each case is scored by a
model that never saw it. Scoring the training cohort with this file would give
optimistic numbers.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme.cohort import (INTEGRATED_FEATURES, RELATIVE_FEATURES,  # noqa: E402
                             add_within_case_features, gene_level, load_cohort)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cohort = Path(args.cohort)
    out = Path(args.out) if args.out else cohort / "ranker.joblib"

    variants, cases, blocks = load_cohort(cohort)
    table = gene_level(variants, blocks)
    truth = cases.set_index("sample_id")["causal_gene"]
    table["is_causal"] = (table["gene"] == table["sample_id"].map(truth)).astype(int)
    table = table[table["sample_id"].isin(cases["sample_id"])]
    table = add_within_case_features(table)

    columns = [c for c in INTEGRATED_FEATURES
               + [f"{c}_rel" for c in RELATIVE_FEATURES] if c in table.columns]
    X = table[columns].to_numpy(dtype=float)
    y = table["is_causal"].to_numpy(dtype=int)

    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.06,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0)
    model.fit(X, y)

    joblib.dump({"model": model, "columns": columns,
                 "n_cases": int(table["sample_id"].nunique()),
                 "n_candidates": int(len(table)),
                 "note": "Fitted on the full cohort. Reported metrics are "
                         "out-of-fold; see run_ablation_real.py."}, out)
    print(f"fitted on {table['sample_id'].nunique()} cases, "
          f"{len(table):,} candidates, {len(columns)} features")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
