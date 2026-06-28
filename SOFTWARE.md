# Software environments — EVOSCZ

The analysis ran in **two locations** (local macOS + the TRUBA HPC cluster) across **three
conda environments**. All versions below were captured with `conda env export --no-builds`
and `pip freeze` on **2026-06-28** — i.e. the *current* state of each environment. Because
the environments were used over several months and may have received package updates after
the May 2026 analysis runs, treat minor versions as the best available record. Two anchors
are independently confirmed: **LDSC v1.0.1** (printed in every S-LDSC run-log banner) and
**scikit-allel 1.3.13** (matches the value recorded at analysis time).

## Environment → pipeline-phase mapping

| Environment | Location | Python | Pipeline phases |
|---|---|---|---|
| **local base** (`environment.yml`) | macOS (`/opt/anaconda3`) | 3.11.15 | phase11 (variant master); phase13 (within-locus); phase14b (GMM, scikit-learn); iHS (scikit-allel); phase14e/phase18 (EUR/EAS partitioned S-LDSC, LDSC v1.0.1 under Python 3); phase14f/h/j; figures |
| **v11** (`environments/environment_truba_v11.yml`) | TRUBA HPC | 3.11.15 | phase14p3–p7 (pure-AFR allele frequency, Akbari overlap, Wohns TMRCA); phase19 (polarity-aware RAF); test1 (age-conditioning robustness) |
| **ldsc_py2** (`environments/environment_truba_ldsc_py2.yml`) | TRUBA HPC | 2.7.15 | phase14p7 AFR partitioned S-LDSC (classic LDSC v1.0.1 under Python 2.7) |

## Local base — key packages (produces the core manuscript numbers)
- Python 3.11.15
- numpy 1.26.4 · scipy 1.16.3 · pandas 2.3.3
- scikit-learn 1.8.0  (Gaussian-mixture clustering; `random_state=42`, deterministic)
- scikit-allel 1.3.13  (iHS / Voight integrated haplotype score)
- statsmodels 0.14.6 · pysam 0.24.0 · pyliftover 0.4.1 · bedtools 2.31.1
- plink2 v2.0.0-a.7.1
- LDSC v1.0.1  (vendored in `scripts/phase14e/ldsc/`; runs under Python 3)
- Ensembl ancestral-allele FASTA, GRCh37 release-71  (iHS polarisation)

## TRUBA v11 — key packages (HPC phases)
- Python 3.11.15 · numpy 2.4.4 · scipy 1.17.1 · pandas 2.3.3 · statsmodels 0.14.6 · pyliftover 0.4.1
- bcftools 1.20 · plink2 2.0.0a.6.9

## TRUBA ldsc_py2 — classic LDSC (Python 2.7)
- Python 2.7.15 · numpy 1.16.5 · pandas 0.20.3 · scipy 1.2.1 · bitarray 0.8.3 · LDSC v1.0.1

## Recreating the environments
```bash
conda env create -f environment.yml                              # local analysis env
conda env create -f environments/environment_truba_v11.yml       # HPC phases
conda env create -f environments/environment_truba_ldsc_py2.yml  # classic LDSC (Py2.7)
```

> **Note on S-LDSC.** The main EUR/EAS partitioned-heritability results (the C0/C1/C2
> enrichments in the manuscript and Supplementary Tables) were produced **locally under
> Python 3.11** with the vendored LDSC v1.0.1. The African-lineage S-LDSC (phase14p7) was
> run on TRUBA under the classic Python-2.7 LDSC (`ldsc_py2`). Both are LDSC v1.0.1; results
> are independent of the interpreter.
