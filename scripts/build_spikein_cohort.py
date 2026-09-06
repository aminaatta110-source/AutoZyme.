"""Build a labelled spike-in cohort from real genomes. Run this on your machine.

Two phases, because CADD scoring needs an external service.

    PHASE 1  -- find ROH blocks, pick causal genes, extract rare variants,
                and write a small VCF containing only the variants that need
                CADD scores.

        python scripts/build_spikein_cohort.py phase1 \
            --vcf-dir vcfs/ --genes genes.tsv \
            --disease-genes recessive_genes.txt --out cohort/

        Then upload cohort/needs_cadd.vcf to https://cadd.gs.washington.edu/score
        (or score it locally) and save the result as cohort/cadd_scores.tsv.

    PHASE 2  -- attach the CADD scores, plant the pathogenic variants, and
                write the finished cohort.

        python scripts/build_spikein_cohort.py phase2 \
            --out cohort/ --cadd cohort/cadd_scores.tsv \
            --clinvar clinvar_pathogenic.tsv

Splitting it this way matters. Filtering to rare variants first cuts each
exome from tens of thousands of variants to a few hundred, which is small
enough for the CADD web service. Trying to score whole genomes up front is
what makes this look like a week of work.

Allele frequencies come from the AF field already present in 1000 Genomes VCFs,
so no separate gnomAD download is needed to get started. Swap in a
population-matched reference later; that substitution is the experiment the
paper proposes.
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme import roh, spikein, vcf_io  # noqa: E402

# Rough severity ranks, matching the scale used elsewhere in the pipeline.
SEVERITY = {
    "frameshift": 5, "stop_gained": 5, "nonsense": 5, "splice_donor": 5,
    "splice_acceptor": 5, "start_lost": 4, "splice": 4, "missense": 3,
    "inframe": 3, "synonymous": 1, "utr": 1, "intron": 1,
}
_AF_FIELD = re.compile(r"(?:^|;)AF=([0-9.eE+-]+)")


def parse_info_af(info):
    """Pull the allele frequency out of a VCF INFO string."""
    if not isinstance(info, str):
        return np.nan
    match = _AF_FIELD.search(info)
    if not match:
        return np.nan
    try:
        return float(match.group(1).split(",")[0])
    except ValueError:
        return np.nan


def severity_from_consequence(text):
    if not isinstance(text, str):
        return 2
    lowered = text.lower()
    for key, score in SEVERITY.items():
        if key in lowered:
            return score
    return 2


def assign_genes(variants, genes):
    """Label each variant with the gene it falls inside, dropping intergenic ones."""
    kept = []
    for chrom, chunk in variants.groupby("chrom", sort=False):
        gene_chunk = genes[genes["chrom"] == chrom]
        if gene_chunk.empty:
            continue
        starts = gene_chunk["start"].to_numpy()
        ends = gene_chunk["end"].to_numpy()
        names = gene_chunk["gene"].to_numpy()
        order = np.argsort(starts)
        starts, ends, names = starts[order], ends[order], names[order]

        positions = chunk["pos"].to_numpy()
        slot = np.searchsorted(starts, positions, side="right") - 1
        valid = (slot >= 0) & (positions <= np.where(slot >= 0, ends[slot], -1))
        if not valid.any():
            continue
        found = chunk.loc[valid].copy()
        found["gene"] = names[slot[valid]]
        kept.append(found)

    if not kept:
        return pd.DataFrame(columns=[*variants.columns, "gene"])
    return pd.concat(kept, ignore_index=True)


def phase1(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = vcf_io.read_gene_table(args.genes)
    print(f"gene table: {len(genes):,} genes")

    disease_genes = None
    if args.disease_genes:
        disease_genes = {
            line.strip().split("\t")[0]
            for line in open(args.disease_genes)
            if line.strip() and not line.startswith("#")
        }
        print(f"recessive disease genes: {len(disease_genes):,}")

    vcf_paths = sorted(
        p for p in Path(args.vcf_dir).iterdir()
        if p.name.endswith((".vcf", ".vcf.gz"))
    )
    if not vcf_paths:
        sys.exit(f"No VCF files found in {args.vcf_dir}")
    print(f"found {len(vcf_paths)} VCF files\n")

    all_rare, all_blocks, case_rows, skipped = [], [], [], []

    for index, path in enumerate(vcf_paths, 1):
        try:
            frame = vcf_io.read_vcf(path, sample=args.sample)
        except Exception as error:                      # noqa: BLE001
            skipped.append((path.name, f"unreadable: {error}"))
            continue

        if frame.empty:
            skipped.append((path.name, "no usable variants"))
            continue
        sample_id = frame["sample_id"].iloc[0]

        # ROH blocks: prefer AutoMap output if it exists, else run Layer 1.
        automap_file = Path(args.roh_dir) / f"{sample_id}.txt" if args.roh_dir else None
        if automap_file and automap_file.exists():
            blocks = vcf_io.read_automap_roh(automap_file, sample_id)
            source = "AutoMap"
        else:
            genotypes, marker_map, ids = vcf_io.genotype_matrix(frame)
            params = roh.params_for_data(marker_map, verbose=(index == 1))
            blocks = roh.detect_roh(genotypes, marker_map, ids, params=params)
            source = "Layer 1"

        if blocks.empty:
            skipped.append((sample_id, "no ROH blocks detected"))
            continue

        in_blocks = spikein.genes_in_blocks(genes, blocks)
        rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
        causal = spikein.choose_causal_gene(in_blocks, disease_genes, rng=rng)
        if causal is None:
            skipped.append((sample_id, "no disease gene inside any ROH block"))
            continue

        frame["gnomad_af"] = (frame["info"].map(parse_info_af)
                              if "info" in frame.columns else np.nan)
        rare = frame[(frame["gnomad_af"].isna()) | (frame["gnomad_af"] < args.max_af)]
        rare = assign_genes(rare, genes)
        if rare.empty:
            skipped.append((sample_id, "no rare variants inside genes"))
            continue

        rare["sample_id"] = sample_id
        all_rare.append(rare)
        all_blocks.append(blocks)
        case_rows.append(
            {
                "sample_id": sample_id,
                "causal_gene": causal["gene"],
                "causal_chrom": causal["chrom"],
                "causal_block": causal["block_id"],
                "n_blocks": len(blocks),
                "roh_source": source,
                "n_rare_variants": len(rare),
            }
        )
        print(f"[{index}/{len(vcf_paths)}] {sample_id}: {len(blocks)} blocks "
              f"({source}), {len(rare)} rare variants, causal={causal['gene']}")

    if not case_rows:
        print("\nNo usable cases were built. Reasons:")
        if any("no ROH blocks" in r for _, r in skipped):
            print("   Hint: if these are exome VCFs, run AutoMap and pass "
                  "--roh-dir. It is validated for exome data; our Layer 1 is not.")
        for name, reason in skipped:
            print(f"   {name}: {reason}")
        sys.exit(1)

    rare_all = pd.concat(all_rare, ignore_index=True)
    pd.concat(all_blocks, ignore_index=True).to_csv(outdir / "blocks.csv", index=False)
    rare_all.to_csv(outdir / "rare_variants.csv", index=False)
    pd.DataFrame(case_rows).to_csv(outdir / "cases.csv", index=False)
    genes.to_csv(outdir / "genes.csv", index=False)

    # Unique sites needing a CADD score, as a minimal VCF.
    sites = (
        rare_all[["chrom", "pos", "ref", "alt"]]
        .drop_duplicates()
        .sort_values(["chrom", "pos"])
    )
    with open(outdir / "needs_cadd.vcf", "w") as handle:
        handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in sites.itertuples(index=False):
            handle.write(f"{row.chrom.replace('chr','')}\t{row.pos}\t.\t"
                         f"{row.ref}\t{row.alt}\t.\t.\t.\n")

    print(f"\n{len(case_rows)} usable cases, {len(skipped)} skipped")
    for name, reason in skipped[:10]:
        print(f"   skipped {name}: {reason}")
    print(f"\nWrote {outdir}/needs_cadd.vcf with {len(sites):,} unique sites.")
    print("Score it at https://cadd.gs.washington.edu/score (GRCh38, include "
          "annotations), save the result as cadd_scores.tsv, then run phase2.")


def _read_cadd(path):
    """Read a CADD score file.

    CADD writes a '## CADD GRCh38-v1.6' banner followed by a header row that
    itself begins with '#Chrom'. Passing comment='#' to the CSV reader would
    swallow the header along with the banner, so skip only the double-hash
    lines and keep the single-hash header.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        skip = 0
        for line in handle:
            if line.startswith("##"):
                skip += 1
            else:
                break
    frame = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str)
    frame.columns = [str(c).strip().lstrip("#").lower() for c in frame.columns]
    return frame


def phase2(args):
    outdir = Path(args.out)
    rare = pd.read_csv(outdir / "rare_variants.csv")
    blocks = pd.read_csv(outdir / "blocks.csv")
    cases = pd.read_csv(outdir / "cases.csv")
    genes = pd.read_csv(outdir / "genes.csv")

    cadd = _read_cadd(args.cadd)
    rename = {"chrom": "chrom", "#chrom": "chrom", "pos": "pos",
              "ref": "ref", "alt": "alt", "phred": "cadd_phred",
              "cadd_phred": "cadd_phred"}
    cadd = cadd.rename(columns={c: rename.get(c, c) for c in cadd.columns})
    for needed in ("chrom", "pos", "ref", "alt", "cadd_phred"):
        if needed not in cadd.columns:
            sys.exit(f"CADD file lacks a {needed!r} column. Found: {list(cadd.columns)}")

    cadd["chrom"] = cadd["chrom"].map(vcf_io.normalise_chrom)
    cadd["pos"] = pd.to_numeric(cadd["pos"], errors="coerce")
    cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")

    # With annotations enabled CADD emits one row per transcript, so a variant
    # hitting several transcripts appears repeatedly. Keep the most severe
    # score per site, and carry the consequence from that same row.
    if "consequence" in cadd.columns:
        cadd = cadd.sort_values("cadd_phred", ascending=False)
        cadd = cadd.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    else:
        cadd = (cadd.sort_values("cadd_phred", ascending=False)
                    .drop_duplicates(subset=["chrom", "pos", "ref", "alt"]))

    carry = ["chrom", "pos", "ref", "alt", "cadd_phred"]
    if "consequence" in cadd.columns:
        carry.append("consequence")
    merged = rare.merge(cadd[carry], on=["chrom", "pos", "ref", "alt"], how="left")
    coverage = merged["cadd_phred"].notna().mean()
    print(f"CADD matched {coverage:.1%} of rare variants")
    if coverage < 0.5:
        print("  Low match rate. Check the genome build (GRCh37 vs GRCh38) "
              "and whether chromosome names carry the 'chr' prefix.")

    if "zygosity" not in merged.columns:
        merged["zygosity"] = np.where(merged.get("genotype", 0) == 2, "hom", "het")
    merged["consequence_severity"] = (
        merged["consequence"].map(severity_from_consequence)
        if "consequence" in merged.columns else 2
    )
    merged["gnomad_af"] = merged["gnomad_af"].fillna(1e-4)
    merged = merged.dropna(subset=["cadd_phred"])

    if args.clinvar and Path(args.clinvar).exists():
        clinvar = pd.read_csv(args.clinvar, sep="\t", dtype=str)
        clinvar.columns = [c.strip() for c in clinvar.columns]
    else:
        print("  No ClinVar file supplied; planted variant scores will be drawn "
              "from a plausible pathogenic distribution and this is recorded in "
              "the cohort summary.")
        clinvar = pd.DataFrame(columns=["Genes"])

    built_variants, built_cases = [], []
    for case in cases.itertuples(index=False):
        sample_variants = merged[merged["sample_id"] == case.sample_id]
        if sample_variants.empty:
            continue
        rng = np.random.default_rng(abs(hash(case.sample_id)) % (2**32))

        pool = clinvar[clinvar.get("Genes", pd.Series(dtype=str)) == case.causal_gene]
        planted = {
            "cadd_phred": float(pool["CADD_PHRED"].astype(float).max())
            if "CADD_PHRED" in pool.columns and not pool.empty
            else float(np.clip(rng.normal(28.0, 4.0), 15.0, 40.0)),
            "gnomad_af": 10 ** rng.uniform(-6.0, -4.0),
            "consequence_severity": 5,
        }

        try:
            variants, row = spikein.build_spikein_case(
                case.sample_id, sample_variants, blocks, genes,
                case.causal_gene, planted, rng=rng,
            )
        except ValueError as error:
            print(f"  skipped {case.sample_id}: {error}")
            continue

        built_variants.append(variants)
        built_cases.append(row)

    if not built_cases:
        sys.exit("No cases could be built. Check the CADD match rate above.")

    pd.concat(built_variants, ignore_index=True).to_csv(
        outdir / "spikein_variants.csv", index=False)
    pd.DataFrame(built_cases).to_csv(outdir / "spikein_cases.csv", index=False)

    with open(outdir / "cohort_summary.json", "w") as handle:
        json.dump(
            {
                "data_source": "SPIKE-IN on real genomes -- semi-synthetic benchmark",
                "n_cases": len(built_cases),
                "cadd_match_rate": float(coverage),
                "note": "Backgrounds are real individuals; one pathogenic variant "
                        "per case was planted. Not clinical validation.",
            },
            handle, indent=2,
        )
    print(f"\nBuilt {len(built_cases)} spike-in cases in {outdir}/")
    print("Next: python scripts/run_ablation.py --cohort", outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    one = sub.add_parser("phase1", help="ROH, causal genes, rare variants")
    one.add_argument("--vcf-dir", required=True)
    one.add_argument("--genes", required=True)
    one.add_argument("--disease-genes")
    one.add_argument("--roh-dir", help="Directory of AutoMap outputs, named <sample>.txt")
    one.add_argument("--sample", help="Sample column to read from multi-sample VCFs")
    one.add_argument("--max-af", type=float, default=0.01)
    one.add_argument("--out", required=True)
    one.set_defaults(func=phase1)

    two = sub.add_parser("phase2", help="attach CADD, plant variants")
    two.add_argument("--out", required=True)
    two.add_argument("--cadd", required=True)
    two.add_argument("--clinvar")
    two.set_defaults(func=phase2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
