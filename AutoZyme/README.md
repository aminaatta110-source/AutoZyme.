# AutoZyme

A prototype for autozygosity-aware ranking of candidate disease genes.

In a consanguineous family an affected child can inherit the same ancestral
segment twice, so a recessive disease allele sits homozygous inside a run of
homozygosity (ROH). Detecting those blocks is a solved problem. Deciding which
of the dozens of genes inside them explains the disease is not. AutoZyme treats
that second step as a ranking problem and returns a shortlist with the evidence
behind each gene.

This is research code accompanying a manuscript. It is a prototype, has not
been validated on patient data, and is not for clinical use.

## Results

Evaluated on 27 cases constructed from 146 Punjabi normal controls
(1000 Genomes, chromosome 2), with one pathogenic variant introduced in silico
per case:

| Method | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|
| Rarity only | 0.037 | 0.185 | 0.481 | 0.190 |
| CADD only | 0.148 | 0.481 | 0.815 | 0.319 |
| ROH features only | 0.037 | 0.222 | 0.407 | 0.154 |
| Integrated | 0.593 | 0.926 | 0.963 | 0.743 |
| Variant features | 0.667 | 0.926 | 1.000 | 0.777 |

Mean 52.8 candidate genes per case. Cross-validation grouped by individual.

These are normal controls carrying planted variants, not patients. The numbers
describe prototype behaviour, not clinical performance.

## Layers

**Layer 1** (`autozyme/roh.py`) — sliding-window homozygosity scan with
filters on block length, marker count, density and heterozygous call rate.
Marker frequency is computed from the cohort's own genotypes, not from a global
reference; see the note below.

**Layer 2** (`autozyme/features.py`, `autozyme/model.py`) — genes inside a
block carrying a rare variant become candidates, described by variant, gene and
block-geometry features plus within-case percentile versions of the competitive
ones. Scored with gradient-boosted trees and sorted within each case.

**Layer 3** (`app.py`) — Streamlit dashboard: autosome map of the individual's
blocks, ranked shortlist, per-gene evidence panel.

## Try it

The evaluation cohort is included, so the dashboard runs immediately:

```bash
pip install -r requirements.txt
streamlit run app.py -- --cohort cohort/
```

Pick an individual to see their runs of homozygosity, the ranked candidate
genes, and the evidence behind each ranking. The sidebar has a toggle to mark
the introduced gene and show its rank.

To reproduce the reported metrics:

```bash
python scripts/run_ablation_real.py --cohort cohort/
```

`cohort/ranker.joblib` is a model fitted on the whole cohort, provided for
inspection. It is not what produced the reported numbers; those come from
out-of-fold predictions where each case is scored by a model that never saw it.
Regenerate it with `python scripts/train_model.py --cohort cohort/`.

## A note on molecular consequence

Consequence severity is computed and stored but excluded from scoring. Variants
introduced during cohort construction carry a fixed severity of 5, a value no
real variant in this cohort takes, so including it would identify the introduced
variant outright and return a top-1 recall of 1.000. Reinstating the feature
requires resampling introduced severity from the observed distribution, in the
way `replant.py` handles allele frequency and CADD.

## Reproduce the reported results

The genomic data is not redistributed here. See `DATA.md` for how to obtain
it. In outline:

```bash
# 1. build the evaluation cohort from a multi-sample VCF
python scripts/build_spikein_multisample.py \
    --vcf pjl_chr2.vcf.gz --genes genes.bed \
    --disease-genes recessive_genes.txt --out cohort/ \
    --min-length-kb 3000

# 2. score cohort/needs_cadd.vcf at https://cadd.gs.washington.edu/score
#    (GRCh38-v1.7, include annotations), save as cadd_scores.tsv.gz

# 3. attach scores and introduce the causal variants
python scripts/build_spikein_cohort.py phase2 \
    --out cohort/ --cadd cadd_scores.tsv.gz

# 4. resample planted feature values from the cohort's own distributions
python scripts/replant.py --cohort cohort/

# 5. run the ablation
python scripts/run_ablation_real.py --cohort cohort/
```

Step 4 is not optional. See below.

## Two failure modes worth knowing about

Both produced apparently excellent results and both are general hazards for
anyone building a similar evaluation.

**A feature that only the introduced variant can take.** Every introduced
variant was assigned a molecular consequence severity of 5, which no real
variant in the cohort carries. Any model given that feature reaches perfect
top-1 recall without learning anything. The feature is excluded from scoring
and the exclusion is enforced in `autozyme/cohort.py` rather than left to
chance.

**Planting outside the observable range.** An early version drew planted allele
frequencies from 1e-6 to 1e-4, but the rarest frequency observable in a panel of
a few thousand samples is around 3e-4. Every planted variant was therefore rarer
than anything real, and ranking by rarity alone identified the causal gene in
100% of cases. `scripts/replant.py` resamples planted values from the cohort's
own distributions, which removes the leak.

**Collapsing gene records by symbol.** Taking the minimum start and maximum end
across transcript rows sharing a gene symbol produces spans over 100 Mb, because
the same symbol occurs at unrelated loci. Those spans then absorb every variant
on the chromosome. `autozyme/vcf_io.py` collapses per gene, chromosome and
locus. The error surfaced because CADD annotated the affected variants as
intergenic while the pipeline reported them as genic.

## Known limitations

- Not validated on patients. The evaluation cohort is healthy individuals.
- 27 cases; the ablation comparing feature sets is underpowered.
- Chromosome 2 only in the reported run.
- Layer 1 was not compared against an established caller. AutoMap requires AD
  and DP annotations, which the 1000 Genomes phased panel does not carry.
- Ranks genes, not variants within a gene.
- Ancestry is Punjabi, not the Saudi population the tool targets.

## Citation

Atta AM, Barradah AA, Marghalani SAM, Bawazir OA, AlEissa MM. AutoZyme: A
Prototype for Autozygosity-Aware Ranking of Candidate Disease Genes, Evaluated
on Normal Population Controls. Manuscript in preparation.

## License

MIT. See `LICENSE`.
