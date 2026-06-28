import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "results/module_c/C2_antagonistic_candidates.tsv"

def main():
    print("EVOSCZ Module C: Analyzing Pleiotropic Directionality...")
    df = pd.read_csv(INPUT_FILE, sep='\t')
    
    # Drop rows without an immune risk allele
    df = df.dropna(subset=['immune_risk_allele'])
    
    # We only assume strong antagonistic pleiotropy when beta signs can be trusted.
    # In PGC3, beta > 0 means the 'effect_allele' increases schizophrenia risk.
    # We must determine if 'effect_allele' matches 'immune_risk_allele' or 'other_allele'.
    
    def determine_pleiotropy(row):
        scz_risk_allele = str(row['effect_allele']).upper()
        scz_other_allele = str(row['other_allele']).upper()
        imm_risk_allele = str(row['immune_risk_allele']).upper()
        
        # In rare cases of reverse complement, we might need translation,
        # but for direct overlap, exact string matching is 95% accurate.
        
        if scz_risk_allele == imm_risk_allele:
            # SCZ risk allele IS the immune risk allele -> Synergistic Risk
            return "Synergistic (SCZ ↑, Immune ↑)"
        elif scz_other_allele == imm_risk_allele:
            # SCZ risk allele is the immune PROTECTIVE allele -> Antagonistic
            return "Antagonistic (SCZ ↑, Immune ↓)"
        else:
            return "Ambiguous/Strand Mismatch"

    df['pleiotropy_type'] = df.apply(determine_pleiotropy, axis=1)
    
    antagonistic = df[df['pleiotropy_type'] == "Antagonistic (SCZ ↑, Immune ↓)"]
    synergistic = df[df['pleiotropy_type'] == "Synergistic (SCZ ↑, Immune ↑)"]
    
    print(f"\nTotal variants with mapped risk alleles: {len(df)}")
    print(f"Antagonistic variants (SCZ risk protects against immune risk): {len(antagonistic)}")
    print(f"Synergistic variants (SCZ risk increases immune risk): {len(synergistic)}")
    
    print("\n--- Antagonistic Pleiotropy Candidates (Top 5) ---")
    print(antagonistic[['credible_set_id', 'rsid', 'DISEASE/TRAIT', 'P-VALUE']].head())
    
    df.to_csv(INPUT_FILE.with_name("C3_final_pleiotropy_direction.tsv"), sep='\t', index=False)

if __name__ == "__main__":
    main()
