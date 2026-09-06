"""Loading an evaluation cohort and building the gene-level candidate table.

This is the shared path used by both the ablation script and the dashboard, so
that what the dashboard displays is produced by exactly the same code that
generated the reported results.
"""

import numpy as np
import pandas as pd

__all__ = [
    "CONSEQUENCE_SEVERITY", "VARIANT_FEATURES", "ROH_FEATURES",
    "INTEGRATED_FEATURES", "RELATIVE_FEATURES",
    "load_cohort", "gene_level", "add_within_case_features",
]

# Severity ranks for CADD's consequence annotation.
CONSEQUENCE_SEVERITY = {
    "STOP_GAINED": 5, "FRAME_SHIFT": 5, "SPLICE_SITE": 5, "STOP_LOST": 5,
    "CANONICAL_SPLICE": 5, "NON_SYNONYMOUS": 3, "INFRAME": 3,
    "SYNONYMOUS": 1, "5PRIME_UTR": 1, "3PRIME_UTR": 1, "INTRONIC": 1,
    "REGULATORY": 2, "UPSTREAM": 1, "DOWNSTREAM": 1, "INTERGENIC": 1,
}

# Molecular consequence severity is deliberately excluded. Variants introduced
# during cohort construction carry a fixed severity of 5, a value no real
# variant in the cohort takes, so the feature identifies the introduced variant
# outright. It is computed and stored for inspection but never used for scoring.
# Reinstating it requires resampling introduced severity from the observed
# distribution, as replant.py does for allele frequency and CADD.
VARIANT_FEATURES = [
    "max_cadd", "neg_log_af", "n_hom_variants", "n_rare_hom_damaging",
    "has_hom_damaging", "n_candidates_in_case",
]
ROH_FEATURES = [
    "block_length_kb", "position_in_block", "n_candidates_in_block",
    "n_candidates_in_case", "case_total_roh_kb", "case_n_blocks",
]
INTEGRATED_FEATURES = sorted(set(VARIANT_FEATURES + ROH_FEATURES))
RELATIVE_FEATURES = ["max_cadd", "neg_log_af", "position_in_block"]


def _severity(text):
    if not isinstance(text, str):
        return 2
    return CONSEQUENCE_SEVERITY.get(text.strip().upper(), 2)


def load_cohort(path):
    """Read the three files an evaluation cohort consists of."""
    from pathlib import Path
    path = Path(path)
    variants = pd.read_csv(path / "spikein_variants.csv")
    cases = pd.read_csv(path / "spikein_cases.csv")
    blocks = pd.read_csv(path / "blocks.csv")
    return variants, cases, blocks


def gene_level(variants, blocks=None, damaging_cadd=20.0):
    """Collapse variant calls to one row per (case, gene) with features.

    Passing ``blocks`` restricts candidates to genes inside a detected block and
    adds block-geometry features. Omitting it gives the unrestricted
    variant-level view used by the variant-only ablation arm.
    """
    frame = variants.copy()
    frame["is_hom"] = (frame["zygosity"] == "hom").astype(int)
    frame["is_damaging"] = (frame["cadd_phred"] >= damaging_cadd).astype(int)
    frame["hom_damaging"] = frame["is_hom"] * frame["is_damaging"]
    if "consequence_severity" in frame.columns:
        frame["sev"] = pd.to_numeric(frame["consequence_severity"],
                                     errors="coerce").fillna(2)
    elif "consequence" in frame.columns:
        frame["sev"] = frame["consequence"].map(_severity)
    else:
        frame["sev"] = 2

    genes = frame.groupby(["sample_id", "gene"], as_index=False).agg(
        max_cadd=("cadd_phred", "max"),
        min_af=("gnomad_af", "min"),
        n_hom_variants=("is_hom", "sum"),
        n_rare_hom_damaging=("hom_damaging", "sum"),
        max_consequence_severity=("sev", "max"),
        variant_pos=("pos", "first"),
        chrom=("chrom", "first"),
    )
    genes["neg_log_af"] = -np.log10(np.clip(genes["min_af"], 1e-7, None))
    genes["has_hom_damaging"] = (genes["n_rare_hom_damaging"] > 0).astype(int)
    genes["n_candidates_in_case"] = genes.groupby("sample_id")["gene"].transform("count")

    if blocks is None:
        return genes

    merged = genes.merge(blocks, on=["sample_id", "chrom"], how="inner")
    inside = ((merged["variant_pos"] >= merged["start"])
              & (merged["variant_pos"] <= merged["end"]))
    merged = merged.loc[inside].drop_duplicates(subset=["sample_id", "gene"]).copy()

    half = (merged["end"] - merged["start"]) / 2.0
    centre = (merged["start"] + merged["end"]) / 2.0
    merged["position_in_block"] = (
        1.0 - (merged["variant_pos"] - centre).abs() / half.clip(lower=1.0)
    ).clip(0.0, 1.0)
    merged["block_length_kb"] = merged["length_kb"]
    merged["n_candidates_in_block"] = merged.groupby("block_id")["gene"].transform("count")
    merged["n_candidates_in_case"] = merged.groupby("sample_id")["gene"].transform("count")

    burden = blocks.groupby("sample_id").agg(
        case_total_roh_kb=("length_kb", "sum"),
        case_n_blocks=("block_id", "count"),
    ).reset_index()
    return merged.merge(burden, on="sample_id", how="left")


def add_within_case_features(table, columns=None):
    """Add within-case percentile versions of the competitive features."""
    columns = columns or RELATIVE_FEATURES
    out = table.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}_rel"] = out.groupby("sample_id")[column].rank(pct=True)
    return out
