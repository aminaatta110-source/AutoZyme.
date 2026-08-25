"""Spike-in benchmarking: real genomes with a planted known answer.

Evaluation requires cases whose answer is known. Rather than generate genomes,
this takes a real individual's genotypes, uses their real runs of homozygosity,
keeps their real rare-variant background, and introduces exactly one known
pathogenic variant into a gene that genuinely falls inside one of their ROH
blocks.

Everything the ranker sees is then real except the single planted variant:
real linkage disequilibrium, real block lengths and positions, real allele
frequencies, real gene density. The label is known because we planted it.

The result is semi-synthetic and should be described that way. The background
individuals are normal controls, no phenotype is observed, and clinical
ascertainment is not reproduced. This is evaluation, not validation.
"""

import numpy as np
import pandas as pd

__all__ = ["genes_in_blocks", "choose_causal_gene", "build_spikein_case"]


def genes_in_blocks(genes, blocks):
    """Every gene falling entirely inside one of the sample's ROH blocks."""
    if blocks.empty or genes.empty:
        return pd.DataFrame(columns=[*genes.columns, "block_id"])

    merged = genes.merge(blocks, on="chrom", how="inner", suffixes=("", "_block"))
    inside = (
        (merged["start"] >= merged["start_block"])
        & (merged["end"] <= merged["end_block"])
    )
    out = merged.loc[inside, [*genes.columns, "block_id", "sample_id"]]
    return out.reset_index(drop=True)


def choose_causal_gene(candidate_genes, disease_genes=None, rng=None,
                       require_known=True):
    """Pick which gene will carry the planted variant.

    Restricting to genes with an established recessive disease association
    keeps the benchmark clinically plausible: a spike-in placed in a random
    gene would ask the model to find something no clinician would call causal.
    Set ``require_known`` to False to also sample novel-gene cases.
    """
    rng = rng or np.random.default_rng()
    pool = candidate_genes

    if disease_genes is not None:
        eligible = pool[pool["gene"].isin(set(disease_genes))]
        if require_known and eligible.empty:
            return None
        if not eligible.empty:
            pool = eligible

    if pool.empty:
        return None
    return pool.iloc[int(rng.integers(len(pool)))]


def build_spikein_case(sample_id, annotated_variants, blocks, genes,
                       causal_gene, planted_variant, rng=None):
    """Assemble one labelled case from a real individual.

    Parameters
    ----------
    sample_id : str
    annotated_variants : DataFrame
        The individual's real variants after VEP or ANNOVAR annotation. Needs
        chrom, pos, gene, cadd_phred, gnomad_af, zygosity, consequence_severity.
    blocks : DataFrame
        Real ROH blocks for this individual, from AutoMap or Layer 1.
    genes : DataFrame
        Gene annotation table.
    causal_gene : str
        Gene that will carry the planted variant. Must sit inside a block.
    planted_variant : dict
        The ClinVar pathogenic variant to insert. Needs cadd_phred,
        gnomad_af and consequence_severity; pos is drawn within the gene if
        absent. It is always inserted homozygous, because a recessive causal
        allele inside an autozygous segment is homozygous by construction.

    Returns
    -------
    (variants, case_row) ready for ``features.build_candidates``.
    """
    rng = rng or np.random.default_rng()

    variants = annotated_variants.copy()
    variants["sample_id"] = sample_id

    # Remove anything already called in the causal gene, so the planted variant
    # is unambiguously the answer rather than competing with a real call there.
    variants = variants[variants["gene"] != causal_gene]

    gene_row = genes.loc[genes["gene"] == causal_gene]
    if gene_row.empty:
        raise ValueError(f"{causal_gene} is not in the gene table.")
    gene_row = gene_row.iloc[0]

    position = planted_variant.get("pos")
    if position is None:
        position = int(rng.integers(gene_row["start"], gene_row["end"] + 1))

    planted = {
        "sample_id": sample_id,
        "gene": causal_gene,
        "chrom": gene_row["chrom"],
        "pos": int(position),
        "cadd_phred": float(planted_variant["cadd_phred"]),
        "gnomad_af": float(planted_variant["gnomad_af"]),
        "zygosity": "hom",
        "consequence_severity": int(planted_variant["consequence_severity"]),
    }

    columns = ["sample_id", "gene", "chrom", "pos", "cadd_phred",
               "gnomad_af", "zygosity", "consequence_severity"]
    for column in columns:
        if column not in variants.columns:
            raise ValueError(f"annotated_variants is missing {column!r}.")

    combined = pd.concat(
        [variants[columns], pd.DataFrame([planted])], ignore_index=True
    )

    case_row = {
        "sample_id": sample_id,
        "causal_gene": causal_gene,
        "causal_chrom": gene_row["chrom"],
        "n_blocks": int(len(blocks)),
        "spikein": 1,
    }
    return combined, case_row
