"""Re-plant the causal variants using the cohort's own feature distributions.

The first spike-in pass drew planted allele frequencies from a fixed range
(10^-6 to 10^-4). In a panel of a few thousand samples the rarest observable
frequency is around 10^-4, so every planted variant was rarer than anything
real in the case. Rarity alone then identified the answer, and top-1 recall of
1.000 followed -- a leak, not a result.

This script replaces the planted variant's features with values drawn from the
distribution of real variants in the same cohort:

* allele frequency is sampled from the observed rare-variant frequencies, so
  the causal variant is rare in the way real rare variants are rare;
* CADD is sampled from the upper tail of the observed distribution, so the
  causal variant is damaging but overlaps with the damaging variants the case
  already carries.

A pathogenic variant should look like the most alarming thing in the case, not
like a value no real variant could take.

    python scripts/replant.py --cohort cohort/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cadd-percentile", type=float, default=90.0,
                        help="Planted CADD is drawn above this percentile of "
                             "the observed distribution.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cohort = Path(args.cohort)
    variants = pd.read_csv(cohort / "spikein_variants.csv")
    cases = pd.read_csv(cohort / "spikein_cases.csv")
    rng = np.random.default_rng(args.seed)

    truth = cases.set_index("sample_id")["causal_gene"]
    planted_mask = variants["gene"] == variants["sample_id"].map(truth)
    real = variants[~planted_mask]

    print(f"{planted_mask.sum()} planted variants, {len(real):,} real variants")
    print("\nBEFORE re-planting:")
    print(f"  planted AF   : {variants.loc[planted_mask,'gnomad_af'].min():.2e} "
          f"to {variants.loc[planted_mask,'gnomad_af'].max():.2e}")
    print(f"  real AF      : {real['gnomad_af'].min():.2e} "
          f"to {real['gnomad_af'].max():.2e}")
    print(f"  planted CADD : {variants.loc[planted_mask,'cadd_phred'].min():.1f} "
          f"to {variants.loc[planted_mask,'cadd_phred'].max():.1f}")
    print(f"  real CADD    : median {real['cadd_phred'].median():.1f}, "
          f"p90 {real['cadd_phred'].quantile(0.90):.1f}, "
          f"max {real['cadd_phred'].max():.1f}")

    # Allele frequency: resample from the real rare-variant frequencies. Draw
    # from the lower half, since a disease allele should be at the rare end of
    # what is observable, but still inside the observable range.
    af_pool = real["gnomad_af"].to_numpy()
    af_pool = af_pool[af_pool <= np.quantile(af_pool, 0.50)]

    # CADD: draw from the observed upper tail. This keeps the planted variant
    # damaging while overlapping the damaging variants already in the case.
    cadd_pool = real["cadd_phred"].to_numpy()
    cadd_pool = cadd_pool[cadd_pool >= np.percentile(cadd_pool, args.cadd_percentile)]

    n = int(planted_mask.sum())
    variants.loc[planted_mask, "gnomad_af"] = rng.choice(af_pool, size=n, replace=True)
    variants.loc[planted_mask, "cadd_phred"] = rng.choice(cadd_pool, size=n, replace=True)

    print("\nAFTER re-planting:")
    print(f"  planted AF   : {variants.loc[planted_mask,'gnomad_af'].min():.2e} "
          f"to {variants.loc[planted_mask,'gnomad_af'].max():.2e}")
    print(f"  planted CADD : {variants.loc[planted_mask,'cadd_phred'].min():.1f} "
          f"to {variants.loc[planted_mask,'cadd_phred'].max():.1f}")

    # How often is the planted variant simply the extreme value in its case?
    def share_extreme(column, ascending):
        wins = 0
        for sample_id, chunk in variants.groupby("sample_id"):
            causal = truth.get(sample_id)
            hit = chunk[chunk["gene"] == causal]
            if hit.empty:
                continue
            best = chunk[column].min() if ascending else chunk[column].max()
            if (hit[column].iloc[0] <= best if ascending
                    else hit[column].iloc[0] >= best):
                wins += 1
        return wins / max(1, variants["sample_id"].nunique())

    print(f"\n  planted variant is the rarest in its case : "
          f"{share_extreme('gnomad_af', True):.1%}")
    print(f"  planted variant has the top CADD in its case: "
          f"{share_extreme('cadd_phred', False):.1%}")
    print("  (both should be well below 100%; at 100% a single feature "
          "identifies the answer)")

    variants.to_csv(cohort / "spikein_variants.csv", index=False)
    print(f"\nrewrote {cohort/'spikein_variants.csv'}")


if __name__ == "__main__":
    main()
