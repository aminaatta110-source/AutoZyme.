"""Reading real genome data: VCF files and AutoMap ROH output.

This is the input layer for cohort construction. Everything here expects
real files -- a 1000 Genomes sample VCF, an AutoMap ROH report, a gene
annotation table.
"""

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "read_vcf",
    "genotype_matrix",
    "read_automap_roh",
    "read_gene_table",
    "normalise_chrom",
]

_GT_SPLIT = re.compile(r"[/|]")


def normalise_chrom(value):
    """Return chromosome names in a single style: chr1, chr2, ... chrX."""
    text = str(value).strip()
    if not text.lower().startswith("chr"):
        text = "chr" + text
    return text


def _open_maybe_gzip(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _parse_genotype(field):
    """Map a VCF GT field onto 0 (hom ref), 1 (het), 2 (hom alt), -1 (missing)."""
    if not field or field == ".":
        return -1
    call = field.split(":", 1)[0]
    alleles = _GT_SPLIT.split(call)
    if any(a in (".", "") for a in alleles):
        return -1
    try:
        values = [int(a) for a in alleles]
    except ValueError:
        return -1
    if len(values) == 1:
        # Hemizygous calls (male X, for example) count as homozygous.
        return 2 if values[0] > 0 else 0
    if all(v == 0 for v in values):
        return 0
    if all(v > 0 for v in values):
        return 2
    return 1


def read_vcf(path, sample=None, biallelic_only=True, min_qual=None,
             pass_only=True, chroms=None):
    """Read one sample out of a VCF into a DataFrame.

    Parameters
    ----------
    path : str
        VCF or VCF.gz. Single-sample or multi-sample.
    sample : str, optional
        Which sample column to take. Defaults to the first.
    biallelic_only : bool
        Skip multi-allelic sites, which complicate zygosity coding.
    min_qual : float, optional
        Drop sites below this QUAL.
    pass_only : bool
        Keep only sites with FILTER of PASS or '.'.
    chroms : set, optional
        Restrict to these chromosomes. Defaults to the 22 autosomes.

    Returns
    -------
    DataFrame with chrom, pos, ref, alt, genotype, and the sample name.
    """
    if chroms is None:
        chroms = {f"chr{i}" for i in range(1, 23)}
    chroms = {normalise_chrom(c) for c in chroms}

    records = []
    sample_index = None
    sample_name = sample

    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")

            if line.startswith("#CHROM"):
                if len(fields) < 10:
                    raise ValueError(f"{path} has no genotype columns.")
                samples = fields[9:]
                if sample is None:
                    sample_index, sample_name = 9, samples[0]
                elif sample in samples:
                    sample_index = 9 + samples.index(sample)
                else:
                    raise ValueError(
                        f"Sample {sample!r} not in {path}. Found: {samples[:5]}..."
                    )
                continue

            if sample_index is None:
                raise ValueError(f"{path} is missing its #CHROM header line.")

            chrom = normalise_chrom(fields[0])
            if chrom not in chroms:
                continue
            if pass_only and fields[6] not in ("PASS", ".", ""):
                continue
            if min_qual is not None:
                try:
                    if float(fields[5]) < min_qual:
                        continue
                except (ValueError, IndexError):
                    pass

            alt = fields[4]
            if biallelic_only and ("," in alt or alt in (".", "*")):
                continue

            genotype = _parse_genotype(fields[sample_index])
            info = fields[7] if len(fields) > 7 else ""
            records.append((chrom, int(fields[1]), fields[3], alt, genotype, info))

    frame = pd.DataFrame(
        records, columns=["chrom", "pos", "ref", "alt", "genotype", "info"]
    )
    frame["sample_id"] = sample_name

    # Sort by chromosome in numeric order, then position.
    frame["_order"] = frame["chrom"].str.replace("chr", "", regex=False)
    frame["_order"] = pd.to_numeric(frame["_order"], errors="coerce").fillna(99)
    frame = frame.sort_values(["_order", "pos"]).drop(columns="_order")
    return frame.reset_index(drop=True)


def genotype_matrix(vcf_frame):
    """Split a VCF frame into the marker map and genotype array Layer 1 expects."""
    marker_map = vcf_frame[["chrom", "pos"]].reset_index(drop=True)
    genotypes = vcf_frame["genotype"].to_numpy(dtype=np.int8)[None, :]
    sample_ids = [vcf_frame["sample_id"].iloc[0]] if len(vcf_frame) else []
    return genotypes, marker_map, sample_ids


def read_automap_roh(path, sample_id):
    """Parse an AutoMap ROH report into the block schema used downstream.

    AutoMap writes a tab-delimited file whose first three columns are
    chromosome, start and end. Header and comment lines are skipped, so minor
    version differences in the trailing columns do not matter.
    """
    records = []
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            try:
                start, end = int(float(fields[1])), int(float(fields[2]))
            except ValueError:
                continue  # header row
            records.append(
                {
                    "sample_id": sample_id,
                    "chrom": normalise_chrom(fields[0]),
                    "start": start,
                    "end": end,
                    "length_kb": round((end - start + 1) / 1000.0, 1),
                    "n_snps": int(float(fields[3])) if len(fields) > 3
                    and fields[3].replace(".", "").isdigit() else -1,
                    "n_het": -1,
                }
            )

    blocks = pd.DataFrame(
        records,
        columns=["sample_id", "chrom", "start", "end", "length_kb", "n_snps", "n_het"],
    )
    if blocks.empty:
        return blocks
    blocks = blocks.sort_values(["chrom", "start"]).reset_index(drop=True)
    blocks.insert(1, "block_id", blocks["sample_id"] + ":" + blocks.index.astype(str))
    return blocks


def _read_annotation_table(path):
    """Read a gene table whether or not it has a header, and whether or not
    that header is prefixed with '#' in the UCSC style.

    A headerless BED file is assumed to be chrom, start, end, name.
    """
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.strip():
                first = line.rstrip("\n")
                break
        else:
            raise ValueError(f"{path} is empty.")

    fields = first.lstrip("#").split("\t")
    # A header row has a non-numeric second column; a data row has a position.
    has_header = not fields[1].strip().replace(".", "").isdigit() if len(fields) > 1 else True

    if has_header:
        frame = pd.read_csv(path, sep="\t", dtype=str)
        frame.columns = [str(c).lstrip("#").strip() for c in frame.columns]
        return frame

    frame = pd.read_csv(path, sep="\t", dtype=str, header=None)
    names = ["chrom", "start", "end", "gene"]
    frame.columns = names[: frame.shape[1]] + [
        f"col{i}" for i in range(len(names), frame.shape[1])
    ]
    return frame


def read_gene_table(path):
    """Read a gene annotation table.

    Expects at least chrom, start, end and gene columns, in a BED-like or
    tab-delimited file. Optional columns are carried through if present:
    p_rec (constraint against biallelic loss) and known_disease_gene.
    """
    frame = _read_annotation_table(path)
    lowered = {c.lower().lstrip("#").strip(): c for c in frame.columns}

    def pick(*names):
        for name in names:
            if name in lowered:
                return lowered[name]
        return None

    chrom_col = pick("chrom", "chromosome", "chr")
    start_col = pick("start", "txstart", "chromstart")
    end_col = pick("end", "txend", "chromend")
    gene_col = pick("gene", "name2", "gene_name", "genename", "symbol",
                    "gene_id", "name")

    missing = [n for n, c in
               [("chrom", chrom_col), ("start", start_col),
                ("end", end_col), ("gene", gene_col)] if c is None]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Found {list(frame.columns)}")

    out = pd.DataFrame(
        {
            "gene": frame[gene_col],
            "chrom": frame[chrom_col].map(normalise_chrom),
            "start": pd.to_numeric(frame[start_col], errors="coerce"),
            "end": pd.to_numeric(frame[end_col], errors="coerce"),
        }
    ).dropna(subset=["start", "end"])

    for optional, default in [("p_rec", 0.5), ("known_disease_gene", 0)]:
        source = pick(optional)
        if source:
            out[optional] = pd.to_numeric(frame[source], errors="coerce").fillna(default)
        else:
            out[optional] = default

    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)

    # Collapse transcript rows to one span per gene. This must be done per
    # (gene, chromosome) and per locus: the same symbol can appear on several
    # chromosomes and at unrelated positions, and taking a global min and max
    # would weld those into a single span hundreds of megabases wide, which
    # then swallows every variant on the chromosome.
    out = out.sort_values(["gene", "chrom", "start"])
    grouped = out.groupby(["gene", "chrom"], sort=False)

    # A transcript starting more than max_locus_gap beyond the running end is
    # treated as a separate locus rather than an extension of the same gene.
    max_locus_gap = 1_000_000
    starts = out["start"].to_numpy()
    ends = out["end"].to_numpy()
    running_end = grouped["end"].cummax().to_numpy()
    previous_end = np.roll(running_end, 1)
    new_group = out.groupby(["gene", "chrom"], sort=False).cumcount().to_numpy() == 0
    is_new_locus = new_group | (starts > np.where(new_group, starts, previous_end) + max_locus_gap)
    out["locus"] = np.cumsum(is_new_locus)

    out = out.groupby(["gene", "chrom", "locus"], as_index=False).agg(
        start=("start", "min"), end=("end", "max"),
        p_rec=("p_rec", "max"), known_disease_gene=("known_disease_gene", "max"),
    )
    # Keep the longest locus per gene, which is the canonical one.
    out["span"] = out["end"] - out["start"]
    out = (out.sort_values("span", ascending=False)
              .drop_duplicates(subset=["gene"])
              .drop(columns=["locus", "span"]))
    return out[["gene", "chrom", "start", "end", "p_rec",
                "known_disease_gene"]].reset_index(drop=True)
