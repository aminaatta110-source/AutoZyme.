"""Merge per-chromosome evaluation cohorts into a single cohort.

Each per-chromosome run selects one causal gene per individual, so an
individual with qualifying ROH blocks on more than one chromosome appears once
per chromosome. Keeping both would give two cases drawn from the same genome
and the same rare-variant background, which are not independent observations.

One case per individual is retained, chosen at random under a fixed seed so the
selection is reproducible and does not depend on which case the model happens to
rank better. Block identifiers are made unique across chromosomes, and each
individual's variants are restricted to the chromosome of the case retained.

    python merge_cohorts.py --inputs cohort_chr1 cohort_chr2 --out cohort_merged
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    path = Path(path)
    return (pd.read_csv(path / "spikein_variants.csv"),
            pd.read_csv(path / "spikein_cases.csv"),
            pd.read_csv(path / "blocks.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    variants, cases, blocks = [], [], []

    for source in args.inputs:
        v, c, b = load(source)
        tag = Path(source).name
        # block_id is only unique within a run, so qualify it
        b = b.copy()
        b["block_id"] = tag + ":" + b["block_id"].astype(str)
        for frame in (v, c, b):
            frame["source"] = tag
        variants.append(v)
        cases.append(c)
        blocks.append(b)
        print(f"  {tag}: {len(c)} cases, {len(b)} blocks, {len(v):,} variant calls")

    cases = pd.concat(cases, ignore_index=True)
    variants = pd.concat(variants, ignore_index=True)
    blocks = pd.concat(blocks, ignore_index=True)

    duplicated = cases["sample_id"].duplicated(keep=False)
    n_dup = cases.loc[duplicated, "sample_id"].nunique()
    print(f"\n{n_dup} individuals have a case on more than one chromosome")

    keep_rows = []
    for sample_id, group in cases.groupby("sample_id", sort=False):
        keep_rows.append(group.index[rng.integers(len(group))] if len(group) > 1
                         else group.index[0])
    cases = cases.loc[sorted(keep_rows)].reset_index(drop=True)
    print(f"retained one case per individual: {len(cases)} cases")

    # keep only the rows belonging to each retained case
    keep = cases.set_index("sample_id")["source"].to_dict()
    variants = variants[
        variants.apply(lambda r: keep.get(r["sample_id"]) == r["source"], axis=1)]
    blocks = blocks[
        blocks.apply(lambda r: keep.get(r["sample_id"]) == r["source"], axis=1)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants.drop(columns="source").to_csv(out / "spikein_variants.csv", index=False)
    cases.drop(columns="source").to_csv(out / "spikein_cases.csv", index=False)
    blocks.drop(columns="source").to_csv(out / "blocks.csv", index=False)

    print(f"\nmerged cohort: {len(cases)} cases, {len(blocks)} blocks, "
          f"{len(variants):,} variant calls")
    print("chromosomes represented:",
          ", ".join(sorted(cases["causal_chrom"].unique())))
    print(f"distinct causal genes: {cases['causal_gene'].nunique()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
