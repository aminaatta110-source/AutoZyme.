"""Layer 3: the AutoZyme review dashboard.

    streamlit run app.py -- --cohort cohort/

Shows one case at a time: where that individual's runs of homozygosity lie, the
ranked shortlist of candidate genes inside them, and the evidence behind each
ranking.

Scores are produced out of fold, with folds grouped by individual, so the score
shown for a case comes from a model that never saw that case during training.
This is the same procedure used for the reported results, so the dashboard and
the manuscript cannot disagree.

The cohort consists of normal population controls carrying pathogenic variants
introduced in silico. It is not patient data, and the rankings are not
diagnostic.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from autozyme.cohort import (INTEGRATED_FEATURES, RELATIVE_FEATURES,
                             add_within_case_features, gene_level, load_cohort)
from autozyme.genome import CHROM_LENGTHS

EVIDENCE = {
    "rank": "Rank", "gene": "Gene", "score": "Score", "chrom": "Chr",
    "max_cadd": "Top CADD", "min_af": "Rarest allele freq",
    "n_rare_hom_damaging": "Homozygous damaging",
    "max_consequence_severity": "Consequence severity",
    "block_length_kb": "Block length (kb)",
    "position_in_block": "Position in block",
}

st.set_page_config(page_title="AutoZyme", layout="wide")


def cohort_path():
    if "--cohort" in sys.argv:
        return Path(sys.argv[sys.argv.index("--cohort") + 1])
    return Path("cohort")


@st.cache_data
def prepare(path):
    """Load the cohort and produce out-of-fold scores for every candidate."""
    variants, cases, blocks = load_cohort(path)
    table = gene_level(variants, blocks)
    truth = cases.set_index("sample_id")["causal_gene"]
    table["is_causal"] = (table["gene"] == table["sample_id"].map(truth)).astype(int)
    table = table[table["sample_id"].isin(cases["sample_id"])]

    table = add_within_case_features(table)
    columns = [c for c in INTEGRATED_FEATURES
               + [f"{c}_rel" for c in RELATIVE_FEATURES] if c in table.columns]

    X = table[columns].to_numpy(dtype=float)
    y = table["is_causal"].to_numpy(dtype=int)
    groups = table["sample_id"].to_numpy()

    scores = np.zeros(len(table))
    n_splits = min(5, len(np.unique(groups)))
    for fold, (train_idx, test_idx) in enumerate(
            GroupKFold(n_splits=n_splits).split(X, y, groups)):
        model = HistGradientBoostingClassifier(
            max_depth=4, max_iter=300, learning_rate=0.06,
            min_samples_leaf=15, l2_regularization=1.0, random_state=fold)
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    table["score"] = scores
    return table, cases, blocks


def genome_map(case_blocks, highlight=None):
    """Autosome ideogram with this individual's ROH blocks marked."""
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


st.title("AutoZyme")
st.caption("Autozygosity-aware gene prioritisation, prototype")

path = cohort_path()
if not (path / "spikein_variants.csv").exists():
    st.warning(
        f"No cohort found at {path}. Build one with "
        "scripts/build_spikein_multisample.py, then phase2 and replant.py. "
        "Pass a different location with: streamlit run app.py -- --cohort PATH")
    st.stop()

table, cases, blocks = prepare(path)

st.info(
    "This cohort consists of normal population controls carrying pathogenic "
    "variants introduced in silico. It is not patient data and the rankings "
    "shown are not diagnostic. Scores are out of fold, from models that did not "
    "see the case being scored.")

with st.sidebar:
    st.header("Case")
    case_id = st.selectbox("Individual", sorted(table["sample_id"].unique()))
    top_n = st.slider("Genes to show", 5, 50, 15)
    hom_only = st.checkbox("Homozygous damaging only", value=False)
    reveal = st.checkbox("Mark the introduced gene", value=False,
                         help="Available because the answer is known by construction.")

case = table[table["sample_id"] == case_id].sort_values("score", ascending=False).copy()
case.insert(0, "rank", np.arange(1, len(case) + 1))
case_blocks = blocks[blocks["sample_id"] == case_id]
if hom_only:
    case = case[case["n_rare_hom_damaging"] > 0]

causal_series = cases.loc[cases["sample_id"] == case_id, "causal_gene"]
causal = causal_series.iloc[0] if len(causal_series) else None
causal_rank = case.loc[case["gene"] == causal, "rank"] if causal else pd.Series(dtype=int)

cols = st.columns(4)
cols[0].metric("ROH blocks", len(case_blocks))
cols[1].metric("Autozygous span",
               f"{case_blocks['length_kb'].sum() / 1000:.1f} Mb"
               if not case_blocks.empty else "n/a")
cols[2].metric("Candidate genes", len(case))
if reveal and len(causal_rank):
    cols[3].metric("Introduced gene rank", int(causal_rank.iloc[0]))

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Runs of homozygosity")
    st.pyplot(genome_map(case_blocks,
                         case["chrom"].iloc[0] if not case.empty else None))
with right:
    st.subheader("Ranked candidates")
    view = case.head(top_n)[[c for c in EVIDENCE if c in case.columns]]
    view = view.rename(columns=EVIDENCE)
    styled = view.style.format({
        "Score": "{:.3f}", "Top CADD": "{:.1f}", "Rarest allele freq": "{:.2e}",
        "Position in block": "{:.2f}", "Block length (kb)": "{:,.0f}"})
    if reveal and causal:
        styled = styled.apply(
            lambda r: ["background-color: #e6f2ea" if r.get("Gene") == causal else ""
                       for _ in r], axis=1)
    st.dataframe(styled, hide_index=True, height=430)

st.subheader("Evidence for a selected gene")
if case.empty:
    st.write("No candidates match the current filter.")
else:
    gene = st.selectbox("Gene", case["gene"].head(top_n))
    row = case[case["gene"] == gene].iloc[0]
    facts = {
        "Rank": int(row["rank"]),
        "Model score": f"{row['score']:.4f}",
        "Highest CADD in the gene": f"{row['max_cadd']:.1f}",
        "Rarest allele frequency": f"{row['min_af']:.2e}",
        "Homozygous damaging variants": int(row["n_rare_hom_damaging"]),
        "Most severe consequence (1 to 5)": int(row["max_consequence_severity"]),
        "Position within block (1.0 = centre)":
            f"{row.get('position_in_block', float('nan')):.2f}",
        "Competing genes in the same block": int(row.get("n_candidates_in_block", 0)),
    }
    st.table(pd.DataFrame({"Evidence": list(facts),
                           "Value": [str(v) for v in facts.values()]}))

st.download_button("Download ranked shortlist", case.to_csv(index=False),
                   f"autozyme_{case_id}_ranked.csv", "text/csv")
