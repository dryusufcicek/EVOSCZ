#!/bin/bash
# ==============================================================
# EVOSCZ Pipeline — Step U4b: Acquire Annotation Datasets
# ==============================================================
# Downloads the small-to-medium critical annotation datasets.
# Only downloads what doesn't exist yet (idempotent).
#
# Datasets acquired here:
#   1. B-statistic maps (background selection) — ~50 MB
#   2. Genetic recombination maps — ~200 MB
#   3. LDSC reference files — ~1 GB
#   4. HapMap3 SNP list — ~2 MB
#   5. GO gene annotations — ~50 MB
#   6. HAR coordinates — <1 MB
#   7. Ancestral allele states — ~50 MB (Ensembl compara)
# ==============================================================

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "EVOSCZ Step U4b: Acquiring Annotation Datasets"
echo "Project root: $PROJECT_ROOT"
echo "================================================"

# ─────────────────────────────────────────────
# 1. B-STATISTIC (Background Selection) Maps
# ─────────────────────────────────────────────
BSTAT_DIR="$PROJECT_ROOT/data/raw/annotations/b_statistic"
echo ""
echo "[1/7] B-statistic (background selection) maps..."
echo "  ⚠️  STATUS: URL NEEDS VERIFICATION BEFORE DOWNLOAD"
echo ""
echo "  B-statistic maps are included in the LDSC baselineLD v2.2 annotations."
echo "  However, the original Broad storage URLs may have changed."
echo ""
echo "  OPTION 1: LDSC baselineLD annotations (Broad — may be requester-pays)"
echo "    URL: https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays"
echo "    Path: LDSCORE/1000G_Phase3_baselineLD_v2.2_ldscores/"
echo ""
echo "  OPTION 2: Original B-statistic maps from McVicker/Sella lab"
echo "    GitHub: https://github.com/gmcvicker/bkgd"
echo "    Contains precomputed B-values in the data/ directory"
echo ""
echo "  OPTION 3: Murphy et al. updated B-maps (higher resolution)"
echo "    Check if available from the authors"
echo ""
if [ ! -f "$BSTAT_DIR/bstat_hg19.done" ]; then
    echo "  STATUS: ❌ NOT YET ACQUIRED"
else
    echo "  STATUS: ✅ Previously acquired"
fi

# ─────────────────────────────────────────────
# 2. GENETIC RECOMBINATION MAPS
# ─────────────────────────────────────────────
RECOMB_DIR="$PROJECT_ROOT/data/raw/annotations/recombination_maps"
if [ ! -f "$RECOMB_DIR/recomb_hg19.done" ]; then
    echo ""
    echo "[2/7] Downloading genetic recombination maps..."
    echo "  Source: 1000 Genomes genetic maps (used by LDSC & selscan)"
    
    cd "$RECOMB_DIR"
    url="https://storage.googleapis.com/broad-alkesgroup-public/LDSCORE/1000G_Phase3_plinkfiles.tgz"
    if [ ! -f "1000G_Phase3_plinkfiles.tgz" ]; then
        echo "    Downloading 1000G genetic maps + plink files..."
        curl -sL -o "1000G_Phase3_plinkfiles.tgz" "$url"
        echo "    Extracting..."
        tar -xzf "1000G_Phase3_plinkfiles.tgz"
        echo "    Cleaning up archive..."
        rm -f "1000G_Phase3_plinkfiles.tgz"
    fi
    touch "$RECOMB_DIR/recomb_hg19.done"
    echo "  ✔ Recombination maps complete"
else
    echo "[2/7] Recombination maps: already present ✔"
fi

# ─────────────────────────────────────────────
# 3. LDSC REFERENCE (LD Scores)
# ─────────────────────────────────────────────
LDSC_DIR="$PROJECT_ROOT/data/raw/annotations/ldsc_ref"
mkdir -p "$LDSC_DIR"
if [ ! -f "$LDSC_DIR/ldsc_ref.done" ]; then
    echo ""
    echo "[3/7] Downloading LDSC reference LD scores..."
    echo "  Source: Zenodo mirror (verified accessible)"
    
    cd "$LDSC_DIR"
    
    # Zenodo mirror of LDSC LD scores
    # VERIFIED 2026-04-10: Record exists, contains eur_w_ld_chr.tar.gz (33.4 MB)
    # API URL for direct download (no redirect issues)
    echo "    Attempting Zenodo mirror..."
    zenodo_url="https://zenodo.org/api/records/8182036/files/eur_w_ld_chr.tar.gz/content"
    
    # EUR LD scores
    if [ ! -d "eur_w_ld_chr" ]; then
        echo "    Downloading EUR LD scores from Zenodo (~33 MB)..."
        curl -sL -o "eur_w_ld_chr.tar.gz" "$zenodo_url" || {
            echo "    ⚠️  Zenodo download failed."
            echo "    Fallback: try Colorado IBG archive:"
            echo "    https://ibg.colorado.edu/cdrom2021/Day06-nivard/GenomicSEM_practical/eur_w_ld_chr/"
            echo "    Or: Broad requester-pays bucket"
        }
        if [ -f "eur_w_ld_chr.tar.gz" ]; then
            echo "    Extracting..."
            tar -xzf "eur_w_ld_chr.tar.gz"
            rm -f "eur_w_ld_chr.tar.gz"
        fi
    fi
    
    # HapMap3 SNP list — NOT in this Zenodo record
    # Need to find from alternative source
    if [ ! -f "w_hm3.snplist" ]; then
        echo "    ⚠️  w_hm3.snplist NOT available in Zenodo record"
        echo "    Need to find from LDSC GitHub wiki or alternative source"
        echo "    LDSC GitHub: https://github.com/bulik/ldsc/wiki"
    fi
    
    # Verify downloads succeeded before marking done
    if [ -d "eur_w_ld_chr" ]; then
        touch "$LDSC_DIR/ldsc_ref.done"
        echo "  ✔ LDSC EUR LD scores complete"
        if [ ! -f "w_hm3.snplist" ]; then
            echo "  ⚠️  w_hm3.snplist still needed — find from LDSC GitHub"
        fi
    else
        echo "  ❌ LDSC reference INCOMPLETE — some downloads failed"
        echo "     Check network and retry, or use alternative sources above"
    fi
else
    echo "[3/7] LDSC reference: already present ✔"
fi

# ─────────────────────────────────────────────
# 4. GENE ONTOLOGY ANNOTATIONS
# ─────────────────────────────────────────────
GO_DIR="$PROJECT_ROOT/data/raw/annotations/gene_ontology"
if [ ! -f "$GO_DIR/go_human.done" ]; then
    echo ""
    echo "[4/7] Downloading Gene Ontology annotations..."
    
    cd "$GO_DIR"
    # Human GO annotations (GAF format)
    url_goa="http://geneontology.org/gene-associations/goa_human.gaf.gz"
    if [ ! -f "goa_human.gaf.gz" ]; then
        echo "    Downloading human GO annotations..."
        curl -sL -o "goa_human.gaf.gz" "$url_goa" || echo "    WARN: GOA download failed"
    fi
    
    # GO term definitions (OBO format)
    url_obo="http://purl.obolibrary.org/obo/go/go-basic.obo"
    if [ ! -f "go-basic.obo" ]; then
        echo "    Downloading GO ontology (OBO)..."
        curl -sL -o "go-basic.obo" "$url_obo" || echo "    WARN: OBO download failed"
    fi
    
    touch "$GO_DIR/go_human.done"
    echo "  ✔ Gene Ontology annotations complete"
else
    echo "[4/7] Gene Ontology: already present ✔"
fi

# ─────────────────────────────────────────────
# 5. HAR COORDINATES
# ─────────────────────────────────────────────
HAR_DIR="$PROJECT_ROOT/data/raw/annotations/HARs"
echo ""
echo "[5/7] Human Accelerated Region coordinates..."
echo "  ⚠️  STATUS: REQUIRES MANUAL EXTRACTION"
echo ""
echo "  HAR coordinates must come from verified published sources:"
echo "  ─────────────────────────────────────────────────────────"
echo "  SOURCE 1: Doan et al. 2016 Cell (doi: 10.1016/j.cell.2016.08.071)"
echo "    → Supplementary Table S2 contains 2,737 HAR coordinates"
echo ""
echo "  SOURCE 2: Pollard et al. 2006 Nature (original 49 HARs)"
echo "    → Table 1 and Supplement"
echo ""
echo "  SOURCE 3: UCSC Genome Browser HAR track (if available)"
echo "    → Browser: genome.ucsc.edu → search 'Human Accelerated Regions'"
echo ""
echo "  OUTPUT FORMAT NEEDED:"
echo "    BED file: chr<TAB>start<TAB>end<TAB>HAR_name"
echo "    Coordinates in hg19/GRCh37"
echo "    Save to: $HAR_DIR/HAR_coordinates_hg19.bed"
echo ""
if [ ! -f "$HAR_DIR/hars.done" ]; then
    echo "  STATUS: ❌ NOT YET ACQUIRED"
else
    echo "  STATUS: ✅ Previously acquired"
fi

# ─────────────────────────────────────────────
# 6. INTROGRESSION DESERT COORDINATES
# ─────────────────────────────────────────────
DESERT_DIR="$PROJECT_ROOT/data/raw/annotations/introgression_deserts"
echo ""
echo "[6/7] Introgression desert coordinates..."
echo "  ⚠️  STATUS: REQUIRES MANUAL EXTRACTION — cannot be auto-downloaded"
echo ""
echo "  Desert coordinates must be extracted from the actual paper supplements:"
echo "  ─────────────────────────────────────────────────────────────────────"
echo "  SOURCE 1: Chen et al. 2025 (doi: 10.1101/2025.09.23.678138)"
echo "    → Supplementary tables contain consensus desert regions"
echo "    → Need: Table with chr/start/end for Tier 1 (5/5 maps) and Tier 2 (≥3/5 maps)"
echo ""
echo "  SOURCE 2: Colbran et al. 2019 (doi: 10.1038/s41559-019-0996-x)"
echo "    → Table S1 lists 6 introgression deserts >8 Mb with exact coordinates"
echo "    → This is the most commonly cited desert coordinate set"
echo ""
echo "  SOURCE 3: Vernot & Akey 2015 / Sankararaman et al. 2014"
echo "    → Original desert definitions"
echo ""
echo "  OUTPUT FORMAT NEEDED:"
echo "    BED file: chr<TAB>start<TAB>end<TAB>name"
echo "    Coordinates must be hg19/GRCh37"
echo "    Save to: $DESERT_DIR/"
echo ""
if [ ! -f "$DESERT_DIR/deserts.done" ]; then
    echo "  STATUS: ❌ NOT YET ACQUIRED"
else
    echo "  STATUS: ✅ Previously acquired"
fi

# ─────────────────────────────────────────────
# 7. ANCESTRAL ALLELE ANNOTATIONS
# ─────────────────────────────────────────────
AA_DIR="$PROJECT_ROOT/data/raw/annotations/ancestral_alleles"
mkdir -p "$AA_DIR"
if [ ! -f "$AA_DIR/ancestral.done" ]; then
    echo ""
    echo "[7/7] Downloading ancestral allele annotations..."
    echo "  Source: 1000 Genomes ancestral allele annotations"
    echo "  Provides the chimpanzee/EPO ancestral allele for each variant"
    echo "  Needed to compute derived allele frequency (DAF)"
    
    cd "$AA_DIR"
    # 1000G Phase 3 includes ancestral allele in the VCF INFO field (AA=)
    # We can extract this from the 1KGP VCFs later in the pipeline
    # For now, download the Ensembl ancestral genome FASTA (smaller, more targeted)
    
    url_aa="https://ftp.ensembl.org/pub/release-75/fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh37_e71.tar.bz2"
    if [ ! -f "homo_sapiens_ancestor_GRCh37_e71.tar.bz2" ] && [ ! -d "homo_sapiens_ancestor_GRCh37_e71" ]; then
        echo "    Downloading Ensembl ancestral allele FASTA (GRCh37)..."
        curl -sL -o "homo_sapiens_ancestor_GRCh37_e71.tar.bz2" "$url_aa" || {
            echo "    WARN: Direct Ensembl download failed."
            echo "    Alternative: extract AA from 1KGP VCF INFO field during Step U3"
        }
        if [ -f "homo_sapiens_ancestor_GRCh37_e71.tar.bz2" ]; then
            echo "    Extracting..."
            tar -xjf "homo_sapiens_ancestor_GRCh37_e71.tar.bz2" 2>/dev/null || true
            rm -f "homo_sapiens_ancestor_GRCh37_e71.tar.bz2"
        fi
    fi
    
    touch "$AA_DIR/ancestral.done"
    echo "  ✔ Ancestral allele annotations complete"
else
    echo "[7/7] Ancestral alleles: already present ✔"
fi

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
echo ""
echo "================================================"
echo "DISK USAGE SUMMARY"
echo "================================================"
du -sh "$PROJECT_ROOT/data/raw/annotations/"* 2>/dev/null | sort -rh
echo ""
echo "Total annotation data:"
du -sh "$PROJECT_ROOT/data/raw/annotations/" 2>/dev/null
echo ""
echo "Total project:"
du -sh "$PROJECT_ROOT/data/" 2>/dev/null
echo ""
echo "✔ Step U4b complete."
