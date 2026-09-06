# Building an evaluation cohort

Evaluation needs cases whose answer is known. Since patient data was not
available, cases are constructed from normal population controls: take a real
individual's genome, use their real runs of homozygosity, keep their real
rare-variant background, and introduce one known pathogenic variant into a gene
that genuinely falls inside one of their ROH blocks.

Everything the model sees is then real except that one variant. Block structure
comes from the individual's own genotypes rather than from any generative
assumption, so a difference between the ablation arms is evidence about the
data.

The result is semi-synthetic and should be described that way. The background
individuals are normal controls with no phenotype, and clinical ascertainment is
not reproduced. This is evaluation, not validation.

---

## What to download

**1. Genomes.** 1000 Genomes Project phase 3 or the 30x NYGC release, from the
IGSR site (`internationalgenome.org`) or the EBI FTP mirror. Take 100 to 200
individuals. One VCF per sample, or a multi-sample VCF you subset.

Note the limitation up front: 1000 Genomes has very few Middle Eastern samples.
That weakens the population angle — and is itself further evidence for the gap
you are claiming. If you can get Greater Middle East Variome or Saudi Human
Genome Program access, use it instead.

**2. Gene annotation.** A RefSeq or GENCODE gene table with chromosome, start,
end and gene symbol. UCSC Table Browser exports this directly. Any BED-like
file works; `read_gene_table` accepts headered, `#`-headered and headerless
formats.

**3. Recessive disease genes.** So the planted variant lands somewhere
clinically plausible. Export from ClinVar, OMIM, or Genomics England PanelApp.
A plain list of gene symbols is enough.

**4. Pathogenic variants to plant.** ClinVar pathogenic and likely-pathogenic
records for those genes. You already have exports for AGXT, CYP1B1 and RPE65 —
the same procedure extends to any gene list.

**5. Annotation tools.** Ensembl VEP or ANNOVAR, to attach real CADD scores and
real gnomAD allele frequencies to each individual's variants. **This step is not
optional.** Rarity is one of the strongest features in the model, and a made-up
frequency makes every downstream number meaningless.

---

## Pipeline

```bash
# 1. ROH blocks from the real genome. Use AutoMap rather than our Layer 1 --
#    it is validated on 52 real consanguineous families.
python AutoMap_v1.0.sh --vcf HG00096.vcf.gz --out roh/HG00096 --genome hg38

# 2. Annotate that individual's variants with real CADD and gnomAD frequencies
vep -i HG00096.vcf.gz -o annot/HG00096.txt --cache --plugin CADD,...  --af_gnomad

# 3. Build the labelled spike-in cohort
python scripts/build_spikein_cohort.py \
    --vcf-dir vcfs/ --roh-dir roh/ --annot-dir annot/ \
    --genes refseq_genes.tsv --disease-genes recessive_genes.txt \
    --clinvar clinvar_pathogenic.tsv --out spikein_cohort/

# 4. Resample introduced feature values, then run the ablation
python scripts/run_ablation.py --cohort spikein_cohort/
```

Our Layer 1 remains available as a fallback if AutoMap will not run, but
AutoMap is the better choice and citing it as a dependency is stronger than
competing with it.

---

## Watch for

**Do not plant a variant in a gene the individual already carries rare variants
in.** `build_spikein_case` drops existing calls in the causal gene so the
planted variant is unambiguously the answer. Without that, you have two
competing candidates and a label that is arguably wrong.

**Do not reuse the CADD values from the project's earlier answer-key files.**
Those separate pathogenic from benign perfectly — 22.01 to 36.99 against 2.00
to 11.99, with nothing in between — and any model trained on them scores 1.000
and means nothing. Pull fresh scores from the CADD server or VEP.

**Split at the individual level.** One person's candidates share block and
case-level features, so a random split leaks. `GroupKFold` on `sample_id`
already does this.

**Match ancestry where you can.** Rarity is measured against a reference
population; a variant rare in gnomAD may be common in the population your
patient came from.

---

## What to expect

Optimised Exomiser reaches roughly 88% top-10 on real undiagnosed patient
cases. Our reported run gives 66.7% top-1 and 100% top-10 on 27 cases against a
mean of 52.8 candidates. Cohorts differ, so these are not directly comparable,
but they indicate the range to expect.
