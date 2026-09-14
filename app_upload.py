"""AutoZyme upload prototype -- real pipeline, honest about the one external step.

    streamlit run app_upload.py -- --cohort cohort/

What is genuinely computed here, live, on whatever VCF you upload:

    1. Parsing        (autozyme.vcf_io.read_vcf)          -- real
    2. ROH detection   (autozyme.roh.detect_roh)            -- real
    3. Candidate genes  (rare variants inside blocks/genes)  -- real
       Allele frequency comes straight from the VCF's own AF INFO field,
       exactly as scripts/build_spikein_cohort.py does it. No external
       gnomAD lookup is needed for this step.

What cannot run live, and why:

    4. Ranking needs a CADD score per candidate variant. CADD is a web
       service (https://cadd.gs.washington.edu/score) that runs as a batch
       job, not something that returns in the seconds after a click. So
       this page stops after step 3, lets you download the small VCF of
       exactly the sites that need scoring, and picks back up once you
       upload the CADD result. The ranking step that follows uses the
       actual trained model in cohort/ranker.joblib -- nothing here is
       simulated or hand-typed.

This is research code accompanying a manuscript. It is a prototype, has not
been validated on patient data, and is not for clinical use.
"""

import gzip
import re
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from autozyme import roh, vcf_io
from autozyme.cohort import (RELATIVE_FEATURES, add_within_case_features,
                             gene_level)
from autozyme.genome import CHROM_LENGTHS

# ---------------------------------------------------------------------------
# Small helpers ported from scripts/build_spikein_cohort.py so that a variant
# uploaded here is annotated in exactly the same way as the cohort that
# produced the reported results. Kept local because the scripts/ helpers
# aren't part of the importable autozyme package.
# ---------------------------------------------------------------------------

_AF_FIELD = re.compile(r"(?:^|;)AF=([0-9.eE+-]+)")

SEVERITY = {
    "frameshift": 5, "stop_gained": 5, "nonsense": 5, "splice_donor": 5,
    "splice_acceptor": 5, "start_lost": 4, "splice": 4, "missense": 3,
    "inframe": 3, "synonymous": 1, "utr": 1, "intron": 1,
}


def parse_info_af(info):
    """Allele frequency straight out of the VCF's own INFO field."""
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
    """Label each variant with the gene it falls inside; drop intergenic ones."""
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


def read_cadd(path):
    """Read a CADD score file, matching scripts/build_spikein_cohort.py."""
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
    rename = {"chrom": "chrom", "#chrom": "chrom", "pos": "pos",
              "ref": "ref", "alt": "alt", "phred": "cadd_phred",
              "cadd_phred": "cadd_phred"}
    frame = frame.rename(columns={c: rename.get(c, c) for c in frame.columns})
    for needed in ("chrom", "pos", "ref", "alt", "cadd_phred"):
        if needed not in frame.columns:
            raise ValueError(f"CADD file lacks a {needed!r} column. Found: {list(frame.columns)}")
    frame["chrom"] = frame["chrom"].map(vcf_io.normalise_chrom)
    frame["pos"] = pd.to_numeric(frame["pos"], errors="coerce")
    frame["cadd_phred"] = pd.to_numeric(frame["cadd_phred"], errors="coerce")
    frame = frame.sort_values("cadd_phred", ascending=False)
    frame = frame.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    return frame


def genome_map(case_blocks, highlight=None):
    names = list(CHROM_LENGTHS)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for row, chrom in enumerate(names):
        ax.broken_barh([(0, CHROM_LENGTHS[chrom] / 1e6)], (row - 0.3, 0.6),
                       facecolors="#ececec", edgecolors="#cccccc", linewidth=0.5)
        spans = case_blocks[case_blocks["chrom"] == chrom]
        if not spans.empty:
            colour = "#c25b3a" if chrom == highlight else "#2f6f4f"
            ax.broken_barh(
                [(s / 1e6, (e - s) / 1e6) for s, e in zip(spans["start"], spans["end"])],
                (row - 0.3, 0.6), facecolors=colour)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([c.replace("chr", "") for c in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig


def cohort_path():
    if "--cohort" in sys.argv:
        return Path(sys.argv[sys.argv.index("--cohort") + 1])
    return Path("cohort")


def save_upload(uploaded_file, suffix):
    """Streamlit gives us an in-memory file; the real pipeline wants a path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AutoZyme -- upload", layout="wide")
st.title("AutoZyme")
st.caption("Autozygosity-aware gene prioritisation, prototype -- upload a new VCF")

st.warning(
    "Research prototype. Not validated on patient data, not for clinical use. "
    "Nothing on this page produces a diagnosis.")

for key in ("blocks", "hits", "sample_id", "genes", "scored"):
    st.session_state.setdefault(key, None)

# ---- Step 1-3: real, live -------------------------------------------------

st.header("1. Upload")
col1, col2 = st.columns(2)
with col1:
    vcf_file = st.file_uploader("Patient VCF (single sample)", type=["vcf", "gz"])
with col2:
    gene_file = st.file_uploader(
        "Gene annotation table (BED-like: chrom, start, end, gene)",
        type=["bed", "tsv", "txt", "gz"])

rare_af_threshold = st.slider("Rare allele frequency threshold", 0.0001, 0.05, 0.01,
                              format="%.4f")

run = st.button("Parse VCF and detect ROH", type="primary",
                disabled=not (vcf_file and gene_file))

if run:
    with st.spinner("Parsing VCF..."):
        vcf_path = save_upload(vcf_file, ".vcf.gz" if vcf_file.name.endswith(".gz") else ".vcf")
        frame = vcf_io.read_vcf(vcf_path)
        sample_id = frame["sample_id"].iloc[0]

    with st.spinner("Detecting runs of homozygosity..."):
        genotypes, marker_map, sample_ids = vcf_io.genotype_matrix(frame)
        params = roh.params_for_data(marker_map, verbose=False)
        blocks = roh.detect_roh(genotypes, marker_map, sample_ids, params=params)

    if blocks.empty:
        st.error("No ROH blocks detected in this sample -- nothing to rank. "
                 "This is a real result, not an error in the app.")
        st.session_state.blocks = None
    else:
        with st.spinner("Assembling candidate genes..."):
            gene_path = save_upload(gene_file, ".bed")
            genes = vcf_io.read_gene_table(gene_path)

            frame["gnomad_af"] = frame["info"].map(parse_info_af)
            rare = frame[(frame["gnomad_af"].isna()) | (frame["gnomad_af"] < rare_af_threshold)]
            rare = assign_genes(rare, genes)

            merged = rare.merge(blocks, on="chrom", how="inner")
            hits = merged[(merged["pos"] >= merged["start"]) & (merged["pos"] <= merged["end"])].copy()

        st.session_state.blocks = blocks
        st.session_state.hits = hits
        st.session_state.sample_id = sample_id
        st.session_state.genes = genes
        st.session_state.scored = None

if st.session_state.blocks is not None:
    blocks = st.session_state.blocks
    hits = st.session_state.hits
    sample_id = st.session_state.sample_id

    st.header("2. Runs of homozygosity and candidates")
    st.caption(f"Sample {sample_id} -- computed live from the uploaded VCF")

    left, right = st.columns([1.05, 1])
    with left:
        cols = st.columns(2)
        cols[0].metric("ROH blocks", len(blocks))
        cols[1].metric("Candidate genes", hits["gene"].nunique() if not hits.empty else 0)
        st.pyplot(genome_map(blocks))
    with right:
        if hits.empty:
            st.info("No rare variants fall inside a candidate gene within a "
                    "detected block at this frequency threshold.")
        else:
            preview = (hits[["gene", "chrom", "pos", "gnomad_af", "block_id"]]
                      .drop_duplicates(subset=["gene"])
                      .sort_values("gene"))
            st.dataframe(preview, hide_index=True, height=380)

    # ---- Step 4: the one step that cannot run live -------------------------

    if not hits.empty:
        st.header("3. Score with CADD")
        st.info(
            "This is the one step that genuinely can't run in the seconds after "
            "a click. CADD is an external batch service. Download the VCF of "
            "just the sites that need scoring, submit it at "
            "https://cadd.gs.washington.edu/score (GRCh38, include annotations), "
            "then upload the result below to finish ranking.")

        sites = (hits[["chrom", "pos", "ref", "alt"]].drop_duplicates()
                .sort_values(["chrom", "pos"]))
        vcf_lines = ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
        for row in sites.itertuples(index=False):
            vcf_lines.append(f"{row.chrom.replace('chr', '')}\t{row.pos}\t.\t"
                             f"{row.ref}\t{row.alt}\t.\t.\t.")
        st.download_button("Download needs_cadd.vcf", "\n".join(vcf_lines) + "\n",
                           f"{sample_id}_needs_cadd.vcf", "text/plain")

        cadd_file = st.file_uploader("Upload CADD result (cadd_scores.tsv or .tsv.gz)",
                                     type=["tsv", "gz", "txt"])

        if cadd_file and st.button("Score and rank candidates", type="primary"):
            with st.spinner("Merging CADD scores and scoring with the trained model..."):
                cadd_path = save_upload(cadd_file, ".tsv.gz" if cadd_file.name.endswith(".gz") else ".tsv")
                cadd = read_cadd(cadd_path)
                carry = ["chrom", "pos", "ref", "alt", "cadd_phred"]
                if "consequence" in cadd.columns:
                    carry.append("consequence")

                merged = hits.merge(cadd[carry], on=["chrom", "pos", "ref", "alt"], how="left")
                coverage = merged["cadd_phred"].notna().mean() if len(merged) else 0.0

                merged["zygosity"] = np.where(merged["genotype"] == 2, "hom", "het")
                merged["consequence_severity"] = (
                    merged["consequence"].map(severity_from_consequence)
                    if "consequence" in merged.columns else 2)
                merged["gnomad_af"] = merged["gnomad_af"].fillna(1e-4)
                merged = merged.dropna(subset=["cadd_phred"])

                if merged.empty:
                    st.error("None of the uploaded CADD scores matched a candidate site. "
                             "Check the genome build (GRCh38) and chromosome naming.")
                else:
                    variants = merged[["sample_id", "gene", "chrom", "pos", "cadd_phred",
                                       "gnomad_af", "zygosity", "consequence_severity"]]
                    table = gene_level(variants, blocks)
                    table = add_within_case_features(table)

                    bundle = joblib.load(cohort_path() / "ranker.joblib")
                    model, columns = bundle["model"], bundle["columns"]
                    missing = [c for c in columns if c not in table.columns]
                    for c in missing:
                        table[c] = 0.0
                    X = table[columns].to_numpy(dtype=float)
                    table["score"] = model.predict_proba(X)[:, 1]
                    table = table.sort_values("score", ascending=False).reset_index(drop=True)
                    table.insert(0, "rank", np.arange(1, len(table) + 1))

                    st.session_state.scored = table
                    st.session_state.coverage = coverage

if st.session_state.scored is not None:
    table = st.session_state.scored
    st.header("4. Ranked candidates")
    st.caption(f"CADD matched {st.session_state.coverage:.0%} of candidate sites -- "
              "scored with cohort/ranker.joblib, the actual trained model.")

    view = table[["rank", "gene", "score", "chrom", "max_cadd", "min_af",
                  "n_rare_hom_damaging", "block_length_kb"]].rename(columns={
        "rank": "Rank", "gene": "Gene", "score": "Score", "chrom": "Chr",
        "max_cadd": "Top CADD", "min_af": "Rarest AF",
        "n_rare_hom_damaging": "Hom. damaging", "block_length_kb": "Block (kb)",
    })
    styled = view.style.format({
        "Score": "{:.3f}", "Top CADD": "{:.1f}", "Rarest AF": "{:.2e}",
        "Block (kb)": "{:,.0f}"})
    if len(table):
        styled = styled.apply(
            lambda r: ["background-color: #e6f2ea" if r.name == 0 else "" for _ in r], axis=1)
    st.dataframe(styled, hide_index=True, height=420)

    st.download_button("Download ranked shortlist", table.to_csv(index=False),
                       f"autozyme_{st.session_state.sample_id}_ranked.csv", "text/csv")

    st.caption(
        "This ranking is produced by the same model and feature pipeline used "
        "for the reported results, run live on this upload. It has not been "
        "validated on patient data and is not diagnostic.")
