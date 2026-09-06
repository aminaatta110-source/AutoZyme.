"""Candidate assembly and feature engineering.

Turns detected ROH blocks plus a case's rare variant calls into the table the
ranking model consumes: one row per candidate gene, grouped by case.

A gene becomes a candidate when it falls inside one of the case's ROH blocks
and carries at least one rare variant. Everything else in the genome is out of
scope, which is the whole point of autozygosity mapping -- the search space
drops from twenty thousand genes to a few dozen.
"""

import numpy as np
import pandas as pd

__all__ = ["FEATURE_COLUMNS", "build_candidates"]

FEATURE_COLUMNS = [
    "max_cadd",
    "neg_log_af",
    "n_hom_variants",
    "n_rare_hom_damaging",
    "max_consequence_severity",
    "has_hom_damaging",
    "gene_p_rec",
    "known_disease_gene",
    "phenotype_match",
    "block_length_kb",
    "block_n_snps",
    "position_in_block",
    "n_candidates_in_block",
    "n_candidates_in_case",
    "case_f_roh",
    "case_n_blocks",
]


def _assign_variants_to_blocks(variants, blocks):
    """Keep only variants sitting inside one of that same case's ROH blocks."""
    merged = variants.merge(blocks, on=["sample_id", "chrom"], how="inner")
    inside = (merged["pos"] >= merged["start"]) & (merged["pos"] <= merged["end"])
    return merged.loc[inside].copy()


def build_candidates(variants, blocks, genes, cases, patient_profiles, gene_profiles,
                     sample_ids, rare_af_threshold=0.01, damaging_cadd=20.0):
    """Assemble the candidate gene table with features and labels.

    Parameters
    ----------
    variants : DataFrame
        Rare variant calls with sample_id, gene, chrom, pos, cadd_phred,
        gnomad_af, zygosity, consequence_severity.
    blocks : DataFrame
        Output of ``roh.detect_roh``.
    genes : DataFrame
        Gene map with gene, chrom, start, end, p_rec, known_disease_gene.
    cases : DataFrame
        Per-case metadata. Needs sample_id, and causal_gene when labels exist.
    patient_profiles, gene_profiles : arrays
        Phenotype vectors, used to compute similarity between the patient's
        presentation and each gene's known phenotype.
    sample_ids : sequence
        Row order of ``patient_profiles``.

    Returns
    -------
    DataFrame
        One row per candidate gene, with FEATURE_COLUMNS plus sample_id, gene,
        block_id and is_causal.
    """
    if blocks.empty:
        return pd.DataFrame(columns=["sample_id", "gene", "block_id", *FEATURE_COLUMNS])

    variants = variants[variants["gnomad_af"] < rare_af_threshold].copy()
    hits = _assign_variants_to_blocks(variants, blocks)
    if hits.empty:
        return pd.DataFrame(columns=["sample_id", "gene", "block_id", *FEATURE_COLUMNS])

    hits["is_hom"] = (hits["zygosity"] == "hom").astype(int)
    hits["is_damaging"] = (hits["cadd_phred"] >= damaging_cadd).astype(int)
    hits["hom_damaging"] = hits["is_hom"] * hits["is_damaging"]

    grouped = hits.groupby(["sample_id", "block_id", "gene"], as_index=False).agg(
        max_cadd=("cadd_phred", "max"),
        min_af=("gnomad_af", "min"),
        n_hom_variants=("is_hom", "sum"),
        n_rare_hom_damaging=("hom_damaging", "sum"),
        max_consequence_severity=("consequence_severity", "max"),
        chrom=("chrom", "first"),
        variant_pos=("pos", "first"),
        n_variants=("pos", "count"),
    )

    grouped["neg_log_af"] = -np.log10(np.clip(grouped["min_af"], 1e-7, None))
    grouped["has_hom_damaging"] = (grouped["n_rare_hom_damaging"] > 0).astype(int)

    # Gene-level annotation.
    gene_cols = genes[["gene", "p_rec", "known_disease_gene"]].rename(
        columns={"p_rec": "gene_p_rec"}
    )
    grouped = grouped.merge(gene_cols, on="gene", how="left")

    # Block geometry. A variant near the middle of a long block is more likely
    # to be the ancestral one than a variant clinging to a block edge.
    block_cols = blocks[["block_id", "length_kb", "n_snps", "start", "end"]].rename(
        columns={"length_kb": "block_length_kb", "n_snps": "block_n_snps"}
    )
    grouped = grouped.merge(block_cols, on="block_id", how="left")
    half_span = (grouped["end"] - grouped["start"]) / 2.0
    centre = (grouped["start"] + grouped["end"]) / 2.0
    grouped["position_in_block"] = (
        1.0 - (grouped["variant_pos"] - centre).abs() / half_span.clip(lower=1.0)
    ).clip(0.0, 1.0)

    # Competition: how crowded is this block, and this case.
    grouped["n_candidates_in_block"] = grouped.groupby("block_id")["gene"].transform("count")
    grouped["n_candidates_in_case"] = grouped.groupby("sample_id")["gene"].transform("count")

    # Case-level burden.
    case_summary = (
        blocks.groupby("sample_id")
        .agg(total_roh_kb=("length_kb", "sum"), case_n_blocks=("block_id", "count"))
        .reset_index()
    )
    from .genome import GENOME_LENGTH

    case_summary["case_f_roh"] = case_summary["total_roh_kb"] * 1000.0 / GENOME_LENGTH
    grouped = grouped.merge(
        case_summary[["sample_id", "case_f_roh", "case_n_blocks"]],
        on="sample_id",
        how="left",
    )

    # Phenotype similarity between the patient and each candidate gene.
    sample_row = {s: i for i, s in enumerate(sample_ids)}
    gene_row = {g: i for i, g in enumerate(genes["gene"])}
    patient_index = grouped["sample_id"].map(sample_row).to_numpy()
    gene_index = grouped["gene"].map(gene_row).to_numpy()
    valid = ~pd.isna(patient_index) & ~pd.isna(gene_index)

    similarity = np.zeros(len(grouped))
    if valid.any():
        rows = patient_index[valid].astype(int)
        cols = gene_index[valid].astype(int)
        similarity[valid] = np.einsum(
            "ij,ij->i", patient_profiles[rows], gene_profiles[cols]
        )
    grouped["phenotype_match"] = similarity

    # Labels, when the truth is known.
    if "causal_gene" in cases.columns:
        truth = cases.set_index("sample_id")["causal_gene"]
        grouped["is_causal"] = (
            grouped["gene"] == grouped["sample_id"].map(truth)
        ).astype(int)
    else:
        grouped["is_causal"] = np.nan

    keep = ["sample_id", "block_id", "gene", "chrom", "variant_pos", "n_variants",
            "min_af", "is_causal", *FEATURE_COLUMNS]
    keep = [c for c in keep if c in grouped.columns]
    return grouped[keep].reset_index(drop=True)
