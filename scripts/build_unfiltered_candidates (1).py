"""Phase 1b -- build the UNFILTERED candidate set for the block-restriction ablation.

This is the redesigned experiment: an arm that ranks every rare variant inside
the gene panel genome-wide, with NO ROH-block gate at all, on the SAME cases
already published (same sample_ids, same causal_gene per case, read straight
from your existing spikein_cases.csv). That makes the comparison paired and
direct: same individuals, same "correct answer" per case, the only thing that
changes is whether candidates were pre-restricted to a detected block.

    python scripts/build_unfiltered_candidates.py \
        --vcf pjl_chr2.vcf.gz --genes genes.bed \
        --existing-cases cohort/spikein_cases.csv \
        --out cohort_unfiltered_chr2/

Repeat once per chromosome (chr1, chr2), then concatenate the outputs before
phase 2. This only performs extraction -- it does not call CADD. It ends by
writing needs_cadd_unfiltered.vcf, which is what actually has to be submitted
to https://cadd.gs.washington.edu/score. That submission is the one part of
this redesign that cannot be automated: CADD is an external batch service,
and for the full genome-wide panel this file will likely contain tens of
thousands of sites rather than the ~3,700 scored before, so budget real time
for it to come back.
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autozyme import vcf_io  # noqa: E402

_AF = re.compile(r"(?:^|;)AF=([0-9.eE+-]+)")
_BASES = set("ACGT")


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _gene_lookup(genes):
    table = {}
    for chrom, chunk in genes.groupby("chrom"):
        chunk = chunk.sort_values("start")
        table[chrom] = (chunk["start"].to_numpy(), chunk["end"].to_numpy(),
                        chunk["gene"].to_numpy())
    return table


def extract_unfiltered_rare_variants(vcf_path, genes, wanted_samples, rare_af=0.01):
    """One pass over the VCF: every rare (AF<rare_af) biallelic SNV inside any
    panel gene, for any of the wanted samples, with NO block restriction.

    This mirrors stream_vcf's candidate-collection logic in
    build_spikein_multisample.py exactly, so the definition of "rare genic
    variant" is identical to what was used before -- the only thing removed
    is any later step that narrowed this down to variants inside a block.
    """
    lookup = _gene_lookup(genes)
    samples = None
    wanted_idx = None
    records = []
    seen_sites = set()
    total = 0

    with _open(vcf_path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")

            if line.startswith("#CHROM"):
                samples = fields[9:]
                wanted_idx = [i for i, s in enumerate(samples) if s in wanted_samples]
                print(f"      {len(samples)} samples in VCF, "
                      f"{len(wanted_idx)} match the existing cohort")
                continue

            total += 1
            if total % 1_000_000 == 0:
                print(f"      ...{total:,} variants read, {len(records):,} rare calls")

            ref, alt = fields[3], fields[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in _BASES or alt not in _BASES:
                continue

            chrom = vcf_io.normalise_chrom(fields[0])
            pos = int(fields[1])
            if (chrom, pos) in seen_sites:
                continue
            seen_sites.add((chrom, pos))

            match = _AF.search(fields[7])
            af = float(match.group(1).split(",")[0]) if match else np.nan
            if np.isnan(af) or af >= rare_af:
                continue
            if chrom not in lookup:
                continue
            starts, ends, names = lookup[chrom]
            slot = np.searchsorted(starts, pos, side="right") - 1
            if slot < 0 or pos > ends[slot]:
                continue
            gene = names[slot]

            for i in wanted_idx:
                call = fields[9 + i].split(":", 1)[0]
                if call in ("0|0", "0/0", ".", "./.", ".|."):
                    continue
                code = 2 if call in ("1|1", "1/1") else 1
                records.append((samples[i], gene, chrom, pos, ref, alt, af,
                               "hom" if code == 2 else "het"))

    print(f"      done: {total:,} variants scanned, {len(records):,} rare genic calls "
          f"across {len(wanted_idx)} samples, no block restriction applied")
    return pd.DataFrame(records, columns=["sample_id", "gene", "chrom", "pos",
                                          "ref", "alt", "gnomad_af", "zygosity"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--existing-cases", required=True,
                   help="Your current spikein_cases.csv -- reuses the same "
                        "sample_ids and causal_gene per case for a paired comparison.")
    p.add_argument("--existing-cadd", action="append", default=[],
                   help="Path to a cadd_scores file already obtained (repeatable). "
                        "Sites already scored are skipped in the output VCF.")
    p.add_argument("--rare-af", type=float, default=0.01)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = vcf_io.read_gene_table(args.genes)
    existing = pd.read_csv(args.existing_cases)
    wanted = set(existing["sample_id"])
    print(f"gene panel: {len(genes):,} genes; {len(wanted)} existing cases to match")

    print("\n[1/3] extracting unfiltered rare genic variants (no block gate)")
    rare = extract_unfiltered_rare_variants(args.vcf, genes, wanted, args.rare_af)
    rare.to_csv(outdir / "rare_variants_unfiltered.csv", index=False)

    per_case = rare.groupby("sample_id")["gene"].nunique()
    print(f"\n      mean unfiltered candidate genes per case: {per_case.mean():.1f} "
          f"(compare to the block-restricted ~52.8 already published)")

    print("\n[2/3] checking against already-scored sites")
    already = set()
    for path in args.existing_cadd:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            skip = 0
            for line in handle:
                if line.startswith("##"):
                    skip += 1
                else:
                    break
        prior = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str)
        prior.columns = [c.strip().lstrip("#").lower() for c in prior.columns]
        for row in prior.itertuples(index=False):
            already.add((vcf_io.normalise_chrom(row.chrom), int(row.pos)))
    print(f"      {len(already):,} sites already have a CADD score")

    print("\n[3/3] writing needs_cadd_unfiltered.vcf")
    sites = rare[["chrom", "pos", "ref", "alt"]].drop_duplicates().sort_values(
        ["chrom", "pos"])
    new_sites = sites[~sites.apply(lambda r: (r.chrom, r.pos) in already, axis=1)]
    with open(outdir / "needs_cadd_unfiltered.vcf", "w") as handle:
        handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in new_sites.itertuples(index=False):
            handle.write(f"{row.chrom.replace('chr','')}\t{row.pos}\t.\t"
                         f"{row.ref}\t{row.alt}\t.\t.\t.\n")

    print(f"      {len(sites):,} unique unfiltered sites total, "
          f"{len(new_sites):,} still need scoring "
          f"({len(sites) - len(new_sites):,} already covered)")
    print(f"\nSubmit {outdir}/needs_cadd_unfiltered.vcf to "
          "https://cadd.gs.washington.edu/score (GRCh38, include annotations).")
    print("This will likely be a much larger job than before -- expect a longer "
         "turnaround. Once it returns, run phase 2 / merge as before, using "
         "gene_level(variants) WITHOUT the blocks argument to score this arm.")


if __name__ == "__main__":
    main()
