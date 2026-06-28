#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Step U6: LiftOver HAR Coordinates hg38 → hg19
================================================================
HAR coordinates from Cui et al. 2025 Nature are in hg38.
PGC3 credible sets and all other data are in hg19.
This script converts HAR BED coordinates from hg38 to hg19
using pyliftover (UCSC chain file-based).

Source: Cui et al. 2025 Nature 640, 991-999
DOI: 10.1038/s41586-025-08622-x
Input: 3,257 HARs in hg38
Output: HAR_coordinates_hg19.bed
"""

from pyliftover import LiftOver
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAR_HG38 = PROJECT_ROOT / "data/raw/annotations/HARs/HAR_coordinates_hg38.bed"
HAR_HG19 = PROJECT_ROOT / "data/raw/annotations/HARs/HAR_coordinates_hg19.bed"
UNMAPPED = PROJECT_ROOT / "data/raw/annotations/HARs/HAR_unmapped_hg38.bed"


def main():
    print("=" * 60)
    print("EVOSCZ Step U6: LiftOver HAR Coordinates hg38 → hg19")
    print("=" * 60)
    
    # Initialize liftover (downloads chain file on first use)
    print("\nInitializing pyliftover (hg38 → hg19)...")
    lo = LiftOver('hg38', 'hg19')
    
    # Read HARs
    hars = []
    with open(HAR_HG38) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            hars.append(fields)
    
    print(f"Input: {len(hars)} HARs in hg38\n")
    
    # LiftOver each HAR
    converted = []
    unmapped = []
    
    for fields in hars:
        chrom, start, end, name = fields[0], int(fields[1]), int(fields[2]), fields[3]
        
        # Convert start
        new_start = lo.convert_coordinate(chrom, start)
        # Convert end
        new_end = lo.convert_coordinate(chrom, end)
        
        if new_start and new_end:
            new_chrom = new_start[0][0]
            new_s = new_start[0][1]
            new_e = new_end[0][1]
            
            # Sanity check: same chromosome, reasonable size
            if new_chrom == new_end[0][0] and abs(new_e - new_s) < 10 * (end - start + 1):
                # Ensure start < end
                if new_s > new_e:
                    new_s, new_e = new_e, new_s
                converted.append(f"{new_chrom}\t{new_s}\t{new_e}\t{name}")
            else:
                unmapped.append(f"{chrom}\t{start}\t{end}\t{name}\t# cross-chrom or size change")
        else:
            unmapped.append(f"{chrom}\t{start}\t{end}\t{name}\t# no mapping")
    
    # Write outputs
    with open(HAR_HG19, 'w') as f:
        f.write('\n'.join(converted) + '\n')
    
    with open(UNMAPPED, 'w') as f:
        f.write('\n'.join(unmapped) + '\n' if unmapped else '')
    
    print(f"Results:")
    print(f"  Converted: {len(converted)} HARs → hg19")
    print(f"  Unmapped:  {len(unmapped)} HARs")
    print(f"  Success rate: {len(converted)/len(hars)*100:.1f}%")
    print(f"\n  Output: {HAR_HG19}")
    print(f"  Unmapped: {UNMAPPED}")


if __name__ == "__main__":
    main()
