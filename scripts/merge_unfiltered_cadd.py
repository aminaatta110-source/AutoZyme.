"""Phase 2 (unfiltered) -- merge CADD scores into the unfiltered candidate set.

Takes rare_variants_unfiltered.csv (from build_unfiltered_candidates.py) plus
one or more CADD result files, and produces a spikein_variants.csv in the
unfiltered output directory with the same schema as the filtered cohort --
so compare_filtered_unfiltered.py can score both arms identically.

Critically, this also copies over each case's PLANTED causal variant from the
existing filtered spikein_variants.csv. The disease variant was synthetically
introduced, not a real call in that individual, so without carrying it over
the causal gene would have nothing to rank on in the unfiltered candidate set
either, and the comparison would be meaningless. The planted row is uniquely
identifiable: it is always written with consequence_severity == 5, a value no
real variant in this cohort takes (see autozyme/spikein.py, and the paper's
"A note on molecular consequence").

    python scripts/merge_unfiltered_cadd.py \
        --rare-variants cohort_unfiltered_chr2/rare_variants_unfiltered.csv \
        --cadd cadd_scores.tsv.gz \
        --cadd "GRCh38-v1.7_anno_2e367ab6....tsv.gz" \
        --filtered-variants cohort/spikein_variants.csv \
        --out cohort_unfiltered_chr2/
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

SEVERITY = {
    "frameshift": 5, "stop_gained": 5, "nonsense": 5, "splice_donor": 5,
    "splice_acceptor": 5, "start_lost": 4, "splice": 4, "missense": 3,
    "inframe": 3, "synonymous": 1, "utr": 1, "intron": 1,
}


def severity_from_consequence(text):
    if not isinstance(text, str):
        return 2
    lowered = text.lower()
    for key, score in SEVERITY.items():
        if key in lowered:
            return score
    return 2


def normalise_chrom(value):
    value = str(value).strip()
    return value if value.startswith("chr") else f"chr{value}"


def read_cadd(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        skip = 0
        for line in handle:
            if line.startswith("##"):
                skip += 1
            else:
                break
    frame = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, low_memory=False)
    frame.columns = [str(c).strip().lstrip("#").lower() for c in frame.columns]
    rename = {"chrom": "chrom", "pos": "pos", "ref": "ref", "alt": "alt",
              "phred": "cadd_phred", "consequence": "consequence"}
    frame = frame.rename(columns={c: rename.get(c, c) for c in frame.columns})
    for needed in ("chrom", "pos", "ref", "alt", "cadd_phred"):
        if needed not in frame.columns:
            raise ValueError(f"{path}: missing {needed!r} column")
    frame["chrom"] = frame["chrom"].map(normalise_chrom)
    frame["pos"] = pd.to_numeric(frame["pos"], errors="coerce")
    frame["cadd_phred"] = pd.to_numeric(frame["cadd_phred"], errors="coerce")
    frame = frame.sort_values("cadd_phred", ascending=False)
    frame = frame.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    keep = ["chrom", "pos", "ref", "alt", "cadd_phred"]
    if "consequence" in frame.columns:
        keep.append("consequence")
    return frame[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rare-variants", required=True)
    p.add_argument("--cadd", action="append", required=True,
                   help="Repeatable -- pass every CADD result file you have.")
    p.add_argument("--filtered-variants", required=True,
                   help="Existing (filtered) spikein_variants.csv, to copy "
                        "each case's planted causal variant from.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rare = pd.read_csv(args.rare_variants)
    print(f"unfiltered candidate variants: {len(rare):,} rows, "
          f"{rare['sample_id'].nunique()} samples")

    cadd = pd.concat([read_cadd(path) for path in args.cadd], ignore_index=True)
    cadd = cadd.sort_values("cadd_phred", ascending=False).drop_duplicates(
        subset=["chrom", "pos", "ref", "alt"])
    print(f"combined CADD scores: {len(cadd):,} unique sites")

    merged = rare.merge(cadd, on=["chrom", "pos", "ref", "alt"], how="left")
    coverage = merged["cadd_phred"].notna().mean()
    print(f"CADD coverage on unfiltered candidates: {coverage:.1%}")
    if coverage < 0.999:
        missing = merged[merged["cadd_phred"].isna()]
        print(f"WARNING: {len(missing):,} candidate rows have no CADD score "
              "and will be dropped -- this shouldn't happen if the earlier "
              "build_unfiltered_candidates.py run reported 0 sites still needed.")
    merged = merged.dropna(subset=["cadd_phred"])

    merged["consequence_severity"] = (
        merged["consequence"].map(severity_from_consequence)
        if "consequence" in merged.columns else 2)

    unfiltered = merged[["sample_id", "gene", "chrom", "pos", "cadd_phred",
                         "gnomad_af", "zygosity", "consequence_severity"]]

    # --- carry over each case's planted causal variant ---
    filtered = pd.read_csv(args.filtered_variants)
    planted = filtered[filtered["consequence_severity"] == 5].copy()
    planted = planted[planted["sample_id"].isin(unfiltered["sample_id"].unique())]
    print(f"\ncarrying over {len(planted)} planted causal variants "
          f"(consequence_severity == 5) from {args.filtered_variants}")

    # Drop any real candidate row that happens to sit at the exact planted
    # position, so the planted variant is unambiguous, mirroring
    # build_spikein_case's own "remove real calls in the causal gene" step.
    key_cols = ["sample_id", "gene"]
    already_has_real_call_in_causal_gene = unfiltered.merge(
        planted[key_cols], on=key_cols, how="inner")
    if len(already_has_real_call_in_causal_gene):
        print(f"removing {len(already_has_real_call_in_causal_gene)} real "
              "variant(s) that coincide with a causal gene, same as the "
              "filtered cohort's own construction")
    unfiltered = unfiltered.merge(planted[key_cols].assign(_drop=1),
                                  on=key_cols, how="left")
    unfiltered = unfiltered[unfiltered["_drop"].isna()].drop(columns="_drop")

    final = pd.concat([unfiltered, planted[unfiltered.columns]], ignore_index=True)
    final.to_csv(outdir / "spikein_variants.csv", index=False)

    per_case = final.groupby("sample_id")["gene"].nunique()
    print(f"\nwrote {outdir}/spikein_variants.csv: {len(final):,} rows, "
         f"mean {per_case.mean():.1f} candidate genes per case "
         f"(including the planted causal gene in every case)")
    print("Ready for: python scripts/compare_filtered_unfiltered.py "
         f"--filtered cohort/ --unfiltered {outdir}/")


if __name__ == "__main__":
    main()
