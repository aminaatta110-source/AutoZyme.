"""Build a spike-in cohort from ONE multi-sample VCF (e.g. a 1000 Genomes panel).

The single-sample script re-reads a file per individual, which is wrong when
every individual lives in the same VCF. This version streams the file once and
keeps two separate marker sets, because ROH detection and candidate assembly
need opposite things:

* ROH detection wants COMMON variants at even spacing. Rare variants carry
  almost no information about homozygosity and thinning them out keeps memory
  flat regardless of how dense the panel is.
* Candidate assembly wants RARE variants inside genes, since those are what
  could plausibly be causal.

    python scripts/build_spikein_multisample.py \
        --vcf pjl_chr2.vcf.gz --genes genes.bed \
        --disease-genes recessive_genes.txt --out cohort/
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme import roh, spikein, vcf_io  # noqa: E402

_GT = re.compile(r"[/|]")
_AF = re.compile(r"(?:^|;)AF=([0-9.eE+-]+)")
_BASES = set("ACGT")


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _codes(fields, start):
    """Genotype codes for every sample on one VCF line."""
    out = np.empty(len(fields) - start, dtype=np.int8)
    for i in range(start, len(fields)):
        call = fields[i].split(":", 1)[0]
        if call in ("0|0", "0/0"):
            out[i - start] = 0
        elif call in ("1|1", "1/1"):
            out[i - start] = 2
        elif call in ("0|1", "1|0", "0/1", "1/0"):
            out[i - start] = 1
        else:
            alleles = _GT.split(call)
            if any(a in (".", "") for a in alleles):
                out[i - start] = -1
            else:
                values = [int(a) for a in alleles if a.isdigit()]
                if not values:
                    out[i - start] = -1
                elif all(v == 0 for v in values):
                    out[i - start] = 0
                elif all(v > 0 for v in values):
                    out[i - start] = 2
                else:
                    out[i - start] = 1
    return out


def _gene_lookup(genes):
    """Position -> gene, per chromosome, using sorted starts for bisection."""
    table = {}
    for chrom, chunk in genes.groupby("chrom"):
        chunk = chunk.sort_values("start")
        table[chrom] = (
            chunk["start"].to_numpy(),
            chunk["end"].to_numpy(),
            chunk["gene"].to_numpy(),
        )
    return table


def stream_vcf(path, genes, common_af=0.05, rare_af=0.01, thin_bp=10_000,
               report_every=500_000):
    """One pass over the VCF, collecting ROH markers and rare genic variants."""
    lookup = _gene_lookup(genes)
    samples = None

    roh_rows, roh_chrom, roh_pos = [], [], []
    rare_records = []
    last_kept = {}
    seen_sites = set()
    total = 0

    with _open(path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")

            if line.startswith("#CHROM"):
                samples = fields[9:]
                print(f"      {len(samples)} samples in VCF")
                continue

            total += 1
            if total % report_every == 0:
                print(f"      ...{total:,} variants read, "
                      f"{len(roh_rows):,} ROH markers, {len(rare_records):,} rare calls")

            ref, alt = fields[3], fields[4]
            # Biallelic SNVs only. Structural alleles like <DEL> and multi-allelic
            # rows cannot be scored by CADD and confuse zygosity coding.
            if len(ref) != 1 or len(alt) != 1 or ref not in _BASES or alt not in _BASES:
                continue

            chrom = vcf_io.normalise_chrom(fields[0])
            pos = int(fields[1])
            if (chrom, pos) in seen_sites:
                continue          # duplicate position -> spacing of zero
            seen_sites.add((chrom, pos))

            match = _AF.search(fields[7])
            af = float(match.group(1).split(",")[0]) if match else np.nan

            # ROH marker selection. The AF field in a multi-ancestry panel is
            # the GLOBAL frequency, which is the wrong filter here: a variant
            # common worldwide but absent in this cohort is homozygous
            # reference in every sample, passes the window test, and
            # manufactures blocks that are not autozygous at all. Frequency is
            # therefore recomputed from the cohort's own genotypes, and only
            # markers polymorphic within the cohort are kept.
            previous = last_kept.get(chrom, -10**9)
            if pos - previous >= thin_bp:
                codes_here = _codes(fields, 9)
                called = codes_here >= 0
                if called.sum() >= 0.9 * codes_here.size:
                    cohort_af = codes_here[called].sum() / (2.0 * called.sum())
                    if common_af <= cohort_af <= 1.0 - common_af:
                        roh_rows.append(codes_here)
                        roh_chrom.append(chrom)
                        roh_pos.append(pos)
                        last_kept[chrom] = pos

            # Rare genic variant: a possible candidate.
            if np.isnan(af) or af >= rare_af:
                continue
            if chrom not in lookup:
                continue
            starts, ends, names = lookup[chrom]
            slot = np.searchsorted(starts, pos, side="right") - 1
            if slot < 0 or pos > ends[slot]:
                continue

            codes = _codes(fields, 9)
            carriers = np.flatnonzero(codes > 0)
            gene = names[slot]
            for index in carriers:
                rare_records.append(
                    (samples[index], gene, chrom, pos, ref, alt,
                     float(af), "hom" if codes[index] == 2 else "het")
                )

    marker_map = pd.DataFrame({"chrom": roh_chrom, "pos": roh_pos})
    genotypes = (np.vstack(roh_rows).T if roh_rows
                 else np.zeros((len(samples), 0), dtype=np.int8))
    rare = pd.DataFrame(
        rare_records,
        columns=["sample_id", "gene", "chrom", "pos", "ref", "alt",
                 "gnomad_af", "zygosity"],
    )
    print(f"      done: {total:,} variants, {genotypes.shape[1]:,} ROH markers, "
          f"{len(rare):,} rare genic calls")
    return genotypes, marker_map, samples, rare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--disease-genes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--common-af", type=float, default=0.05)
    parser.add_argument("--rare-af", type=float, default=0.01)
    parser.add_argument("--thin-bp", type=int, default=10_000)
    parser.add_argument("--min-length-kb", type=float, default=1000.0)
    parser.add_argument("--min-snps", type=int, default=40)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = vcf_io.read_gene_table(args.genes)
    print(f"gene table: {len(genes):,} genes")

    disease_genes = None
    if args.disease_genes:
        disease_genes = {
            l.strip().split("\t")[0] for l in open(args.disease_genes)
            if l.strip() and not l.startswith("#")
        }
        print(f"disease genes: {len(disease_genes):,}")

    print("\n[1/4] streaming VCF")
    genotypes, marker_map, samples, rare = stream_vcf(
        args.vcf, genes, args.common_af, args.rare_af, args.thin_bp
    )
    if genotypes.shape[1] == 0:
        sys.exit("No ROH markers found. Try lowering --common-af.")

    gaps = np.diff(marker_map["pos"].to_numpy())
    gaps = gaps[gaps > 0]
    print(f"      median marker spacing {np.median(gaps)/1000:.1f} kb")

    print("\n[2/4] detecting ROH blocks")
    params = roh.ROHParams(
        window_snps=50, window_max_het=1, min_snps=args.min_snps,
        min_length_kb=args.min_length_kb, max_gap_kb=1000.0,
        max_density_kb_per_snp=100.0, max_block_het_rate=0.02,
    )
    blocks = roh.detect_roh(genotypes, marker_map, samples, params=params)
    per_sample = blocks.groupby("sample_id").size() if not blocks.empty else pd.Series(dtype=int)
    print(f"      {len(blocks):,} blocks across "
          f"{per_sample.size} of {len(samples)} samples")
    if not blocks.empty:
        # Inside a genuine autozygous segment nearly every call is homozygous.
        # A low rate here means the blocks are not really runs of homozygosity.
        checks = []
        for sample_id, chunk in rare.groupby("sample_id"):
            sample_blocks = blocks[blocks["sample_id"] == sample_id]
            if sample_blocks.empty:
                continue
            inside = np.zeros(len(chunk), dtype=bool)
            positions = chunk["pos"].to_numpy()
            chroms_c = chunk["chrom"].to_numpy()
            for row in sample_blocks.itertuples():
                inside |= ((chroms_c == row.chrom) & (positions >= row.start)
                           & (positions <= row.end))
            if inside.any():
                checks.append((chunk["zygosity"].to_numpy()[inside] == "hom").mean())
        if checks:
            print(f"      homozygous rate inside blocks: {np.mean(checks):.1%} "
                  "(expect >85% for genuine autozygosity)")
    if not blocks.empty:
        print(f"      median blocks per sample: {per_sample.median():.0f}, "
              f"median length {blocks['length_kb'].median()/1000:.1f} Mb")

    print("\n[3/4] choosing causal genes")
    cases, kept_blocks = [], []
    no_gene = 0
    for sample_id in samples:
        sample_blocks = blocks[blocks["sample_id"] == sample_id]
        if sample_blocks.empty:
            continue
        inside = spikein.genes_in_blocks(genes, sample_blocks)
        rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
        causal = spikein.choose_causal_gene(inside, disease_genes, rng=rng)
        if causal is None:
            no_gene += 1
            continue
        cases.append({
            "sample_id": sample_id, "causal_gene": causal["gene"],
            "causal_chrom": causal["chrom"], "causal_block": causal["block_id"],
            "n_blocks": len(sample_blocks),
            "genes_in_blocks": len(inside),
        })
        kept_blocks.append(sample_blocks)

    print(f"      {len(cases)} usable cases; {no_gene} samples had blocks "
          f"but no disease gene inside them")
    if not cases:
        sys.exit("No usable cases. Widen the disease gene list, or lower "
                 "--min-length-kb so more of the genome is captured.")

    cases_frame = pd.DataFrame(cases)
    print("      causal genes chosen:",
          ", ".join(cases_frame["causal_gene"].value_counts().head(8).index))

    print("\n[4/4] writing outputs")
    pd.concat(kept_blocks, ignore_index=True).to_csv(outdir / "blocks.csv", index=False)
    cases_frame.to_csv(outdir / "cases.csv", index=False)
    genes.to_csv(outdir / "genes.csv", index=False)
    rare = rare[rare["sample_id"].isin(cases_frame["sample_id"])]
    rare.to_csv(outdir / "rare_variants.csv", index=False)

    sites = rare[["chrom", "pos", "ref", "alt"]].drop_duplicates().sort_values(
        ["chrom", "pos"])
    with open(outdir / "needs_cadd.vcf", "w") as handle:
        handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in sites.itertuples(index=False):
            handle.write(f"{row.chrom.replace('chr','')}\t{row.pos}\t.\t"
                         f"{row.ref}\t{row.alt}\t.\t.\t.\n")

    print(f"      {len(cases)} cases, {len(rare):,} rare calls, "
          f"{len(sites):,} unique sites to score")
    print(f"\nUpload {outdir}/needs_cadd.vcf to https://cadd.gs.washington.edu/score "
          "(GRCh38), save as cadd_scores.tsv, then run phase2.")


if __name__ == "__main__":
    main()
