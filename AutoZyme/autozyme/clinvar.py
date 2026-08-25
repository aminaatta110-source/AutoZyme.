"""Load real ClinVar exports into the pipeline's variant schema.

Takes the ClinVar
tab-delimited exports (the PH1 / LCA / PCG files) and maps them onto the columns
``features.build_candidates`` expects.

One thing ClinVar cannot give you, and the reason the loader warns about it:
ClinVar is a variant knowledge base, not a patient. It has no zygosity and no
sample. Zygosity has to come from the patient's own VCF, and without it the
autozygosity logic has nothing to stand on -- a heterozygous variant inside an
ROH block is almost certainly not the recessive cause.
"""

import warnings

import numpy as np
import pandas as pd

__all__ = ["CONSEQUENCE_SEVERITY", "load_clinvar_export", "attach_patient_genotypes"]

# Rough severity ordering for the ClinVar molecular consequence field.
CONSEQUENCE_SEVERITY = {
    "nonsense": 5,
    "frameshift variant": 5,
    "splice donor variant": 5,
    "splice acceptor variant": 5,
    "initiator codon variant": 4,
    "missense variant": 3,
    "inframe deletion": 3,
    "inframe insertion": 3,
    "splice site variant": 4,
    "synonymous variant": 1,
    "3 prime UTR variant": 1,
    "5 prime UTR variant": 1,
    "intron variant": 1,
    "non-coding transcript variant": 1,
}

PATHOGENIC_LABELS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_LABELS = {"benign", "likely benign", "benign/likely benign"}


def _parse_location(value):
    """Split a ClinVar ``chrom:pos`` location string."""
    if not isinstance(value, str) or ":" not in value:
        return None, np.nan
    chrom, _, pos = value.partition(":")
    pos = pos.split(" ")[0].split("-")[0].replace(",", "")
    try:
        return f"chr{chrom.strip()}", int(pos)
    except ValueError:
        return f"chr{chrom.strip()}", np.nan


def _severity(consequence):
    if not isinstance(consequence, str):
        return 2
    lowered = consequence.lower()
    for key, score in CONSEQUENCE_SEVERITY.items():
        if key in lowered:
            return score
    return 2


def load_clinvar_export(path, assembly="GRCh38"):
    """Read a ClinVar TSV export into the variant schema.

    Returns a frame with gene, chrom, pos, consequence_severity, clinvar_label
    and, when the file carries one, cadd_phred. It deliberately does not invent
    sample_id, zygosity or gnomad_af -- those come from the patient.
    """
    table = pd.read_csv(path, sep="\t", dtype=str)
    location_column = f"{assembly} Location"
    if location_column not in table.columns:
        raise ValueError(f"{path} has no '{location_column}' column.")

    parsed = table[location_column].map(_parse_location)
    out = pd.DataFrame(
        {
            "gene": table["Genes"].fillna("").str.split("|").str[0],
            "chrom": [p[0] for p in parsed],
            "pos": [p[1] for p in parsed],
            "variant": table.get("Variation"),
            "condition": table.get("Condition"),
            "consequence_severity": table.get("Molecular consequence").map(_severity)
            if "Molecular consequence" in table.columns
            else 2,
            "clinvar_raw": table.get("Classification"),
        }
    )

    label = (
        out["clinvar_raw"].fillna("").str.replace(r"^[A-Z]:\s*", "", regex=True).str.lower()
    )
    out["clinvar_label"] = np.where(
        label.isin(PATHOGENIC_LABELS), "pathogenic",
        np.where(label.isin(BENIGN_LABELS), "benign", "uncertain"),
    )

    for candidate in ("CADD_PHRED", "cadd_phred"):
        if candidate in table.columns:
            out["cadd_phred"] = pd.to_numeric(table[candidate], errors="coerce")
            break

    out = out.dropna(subset=["chrom", "pos"])
    out["pos"] = out["pos"].astype(int)
    return out.reset_index(drop=True)


def attach_patient_genotypes(variants, patient_calls, sample_id):
    """Join ClinVar annotation onto one patient's actual calls.

    ``patient_calls`` must carry chrom, pos, zygosity ('hom' or 'het') and
    ideally gnomad_af. This is the step that turns a knowledge base into a case.
    """
    required = {"chrom", "pos", "zygosity"}
    missing = required - set(patient_calls.columns)
    if missing:
        raise ValueError(f"patient_calls is missing {sorted(missing)}.")

    merged = patient_calls.merge(variants, on=["chrom", "pos"], how="left")
    merged["sample_id"] = sample_id

    if "gnomad_af" not in merged.columns:
        warnings.warn(
            "No gnomad_af column. Rarity is one of the strongest ranking features, "
            "so annotate the calls with population frequency before ranking.",
            stacklevel=2,
        )
        merged["gnomad_af"] = 0.001

    if "cadd_phred" not in merged.columns:
        warnings.warn(
            "No cadd_phred column. Score the variants with CADD or VEP before ranking.",
            stacklevel=2,
        )
        merged["cadd_phred"] = np.nan

    merged["consequence_severity"] = merged["consequence_severity"].fillna(2).astype(int)
    return merged[
        ["sample_id", "gene", "chrom", "pos", "cadd_phred", "gnomad_af",
         "zygosity", "consequence_severity"]
    ].dropna(subset=["gene"])
