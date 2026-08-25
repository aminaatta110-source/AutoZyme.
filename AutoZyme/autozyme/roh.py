"""Layer 1 -- runs of homozygosity (ROH) detection.

Sliding-window homozygosity scan in the style of PLINK ``--homozyg``.

A run of homozygosity is a stretch of the genome where an individual carries
two identical copies of a haplotype. In a consanguineous family those two
copies are identical by descent, so a recessive disease allele carried by the
shared ancestor sits homozygous inside one of these blocks. Finding the blocks
is what turns a whole-genome variant list into a short list of regions.

Input is a genotype matrix coded 0 = homozygous reference, 1 = heterozygous,
2 = homozygous alternate, -1 = missing.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ROHParams", "detect_roh", "genome_wide_f_roh",
           "EXOME_PARAMS", "ARRAY_PARAMS", "params_for_data"]


@dataclass(frozen=True)
class ROHParams:
    """Scan parameters.

    Defaults follow the PLINK homozygosity defaults, loosened slightly for
    exome-density marker sets. ``window_max_het`` above zero is what keeps a
    single genotyping error from splitting a real block in two.
    """

    window_snps: int = 50
    window_max_het: int = 1
    window_max_missing: int = 5
    window_threshold: float = 0.05
    min_snps: int = 100
    min_length_kb: float = 1000.0
    max_gap_kb: float = 1000.0
    max_density_kb_per_snp: float = 50.0
    max_block_het_rate: float = 0.02


def _window_hit_rate(is_het, is_missing, params):
    """Fraction of overlapping windows around each marker that look homozygous.

    Every window of ``window_snps`` consecutive markers is scored as homozygous
    or not, then each marker inherits the mean score of the windows covering
    it. Implemented with cumulative sums so a genome-wide scan stays linear.
    """
    n = is_het.size
    width = min(params.window_snps, n)
    if width == 0:
        return np.zeros(n)

    het_cumsum = np.concatenate([[0], np.cumsum(is_het)])
    missing_cumsum = np.concatenate([[0], np.cumsum(is_missing)])

    starts = np.arange(0, n - width + 1)
    ends = starts + width
    window_ok = (
        (het_cumsum[ends] - het_cumsum[starts] <= params.window_max_het)
        & (missing_cumsum[ends] - missing_cumsum[starts] <= params.window_max_missing)
    ).astype(np.float64)

    # Spread each window's verdict back over the markers it covers.
    hits = np.zeros(n + 1)
    np.add.at(hits, starts, window_ok)
    np.add.at(hits, ends, -window_ok)
    hits = np.cumsum(hits)[:n]

    coverage = np.zeros(n + 1)
    np.add.at(coverage, starts, 1.0)
    np.add.at(coverage, ends, -1.0)
    coverage = np.cumsum(coverage)[:n]

    return np.divide(hits, coverage, out=np.zeros(n), where=coverage > 0)


def _runs_from_mask(mask):
    """Start and end indices of every contiguous True run in a boolean mask."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts, ends))


def _split_on_gaps(positions, start, end, max_gap_bp):
    """Break a run wherever consecutive markers are further apart than the gap limit."""
    if end <= start:
        return [(start, end)]
    gaps = np.diff(positions[start : end + 1])
    breaks = np.flatnonzero(gaps > max_gap_bp)
    if breaks.size == 0:
        return [(start, end)]

    segments = []
    cursor = start
    for offset in breaks:
        segments.append((cursor, start + offset))
        cursor = start + offset + 1
    segments.append((cursor, end))
    return segments


def detect_roh(genotypes, marker_map, sample_ids=None, params=None):
    """Call ROH blocks for every sample.

    Parameters
    ----------
    genotypes : array of shape (n_samples, n_markers)
        Coded 0/1/2, with -1 for missing calls.
    marker_map : DataFrame
        Needs ``chrom`` and ``pos`` columns, sorted by chromosome then position.
    sample_ids : sequence, optional
        Sample labels. Defaults to positional integers.
    params : ROHParams, optional

    Returns
    -------
    DataFrame
        One row per block: sample_id, chrom, start, end, length_kb, n_snps,
        n_het, and the marker index range of the block.
    """
    params = params or ROHParams()
    genotypes = np.asarray(genotypes)
    if genotypes.ndim == 1:
        genotypes = genotypes[None, :]

    n_samples, n_markers = genotypes.shape
    if len(marker_map) != n_markers:
        raise ValueError(
            f"marker_map has {len(marker_map)} rows but genotypes has {n_markers} columns."
        )
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]

    chroms = marker_map["chrom"].to_numpy()
    positions = marker_map["pos"].to_numpy().astype(np.int64)
    max_gap_bp = params.max_gap_kb * 1000.0
    min_length_bp = params.min_length_kb * 1000.0

    # Marker index ranges for each chromosome, so runs never cross a boundary.
    chrom_blocks = []
    boundaries = np.flatnonzero(chroms[1:] != chroms[:-1]) + 1
    cut_points = np.concatenate([[0], boundaries, [n_markers]])
    for i in range(len(cut_points) - 1):
        chrom_blocks.append((chroms[cut_points[i]], cut_points[i], cut_points[i + 1]))

    records = []
    for sample_index in range(n_samples):
        row = genotypes[sample_index]
        is_het = (row == 1).astype(np.float64)
        is_missing = (row < 0).astype(np.float64)

        for chrom, lo, hi in chrom_blocks:
            chrom_positions = positions[lo:hi]
            hit_rate = _window_hit_rate(is_het[lo:hi], is_missing[lo:hi], params)
            mask = hit_rate > params.window_threshold

            for run_start, run_end in _runs_from_mask(mask):
                for start, end in _split_on_gaps(
                    chrom_positions, run_start, run_end, max_gap_bp
                ):
                    n_snps = end - start + 1
                    span_bp = chrom_positions[end] - chrom_positions[start] + 1
                    n_het = int(is_het[lo + start : lo + end + 1].sum())

                    if n_snps < params.min_snps or span_bp < min_length_bp:
                        continue
                    if n_het / n_snps > params.max_block_het_rate:
                        continue
                    if span_bp / n_snps > params.max_density_kb_per_snp * 1000.0:
                        continue

                    records.append(
                        {
                            "sample_id": sample_ids[sample_index],
                            "chrom": chrom,
                            "start": int(chrom_positions[start]),
                            "end": int(chrom_positions[end]),
                            "length_kb": round(span_bp / 1000.0, 1),
                            "n_snps": int(n_snps),
                            "n_het": n_het,
                            "marker_start": int(lo + start),
                            "marker_end": int(lo + end),
                        }
                    )

    columns = [
        "sample_id",
        "chrom",
        "start",
        "end",
        "length_kb",
        "n_snps",
        "n_het",
        "marker_start",
        "marker_end",
    ]
    blocks = pd.DataFrame.from_records(records, columns=columns)
    if blocks.empty:
        return blocks

    blocks = blocks.sort_values(["sample_id", "chrom", "start"]).reset_index(drop=True)
    blocks.insert(1, "block_id", blocks["sample_id"] + ":" + blocks.index.astype(str))
    return blocks


def genome_wide_f_roh(blocks, genome_length_bp):
    """Inbreeding coefficient estimated from ROH, per sample.

    F_ROH is the share of the autosomal genome sitting inside called blocks.
    First-cousin offspring land near 0.0625, double first cousins near 0.125.
    """
    if blocks.empty:
        return pd.DataFrame(columns=["sample_id", "total_roh_kb", "n_blocks", "f_roh"])

    summary = (
        blocks.groupby("sample_id")
        .agg(total_roh_kb=("length_kb", "sum"), n_blocks=("block_id", "count"))
        .reset_index()
    )
    summary["f_roh"] = (summary["total_roh_kb"] * 1000.0 / genome_length_bp).round(4)
    return summary


# Marker density differs by an order of magnitude between SNP arrays and exome
# sequencing, and one parameter set cannot serve both. Array-tuned settings
# applied to exome data fragment or miss real blocks entirely -- the failure
# mode that motivated exome-specific tools such as H3M2 and AutoMap.
ARRAY_PARAMS = ROHParams()

EXOME_PARAMS = ROHParams(
    window_snps=20,
    window_max_het=2,
    window_max_missing=3,
    window_threshold=0.05,
    min_snps=25,
    min_length_kb=1000.0,
    max_gap_kb=10_000.0,
    max_density_kb_per_snp=2_000.0,
    max_block_het_rate=0.12,
)


def params_for_data(marker_map, verbose=True):
    """Pick array or exome parameters from the observed marker spacing.

    SNP arrays place markers every few tens of kilobases. Exome capture leaves
    long uncovered stretches between targets, so median spacing is far larger
    and much more variable.
    """
    positions = marker_map["pos"].to_numpy()
    chroms = marker_map["chrom"].to_numpy()
    gaps = np.diff(positions)
    same = chroms[1:] == chroms[:-1]
    gaps = gaps[same & (gaps > 0)]

    if gaps.size == 0:
        return ARRAY_PARAMS

    median_gap_kb = float(np.median(gaps)) / 1000.0
    choice = ARRAY_PARAMS if median_gap_kb <= 60.0 else EXOME_PARAMS
    if verbose:
        kind = "array-like" if choice is ARRAY_PARAMS else "exome-like"
        print(f"      median marker spacing {median_gap_kb:.1f} kb -> {kind} parameters")
    return choice
