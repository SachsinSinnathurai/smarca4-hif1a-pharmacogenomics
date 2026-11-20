#!/usr/bin/env python3
"""Public ChIP-Atlas (hg38) Solid-Tumor ChIP-seq Track Overlap at SLC7A1 and Positive Control VEGFA."""

import os
import urllib.request
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_CHIP_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "chipseq")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "chipseq_slc7a1_occupancy.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Genomic windows (hg38 / GRCh38)
LOCI = {
    "SLC7A1": ("chr13", 29500000, 29620000),  # NCBI GeneID 6541: chr13:29,509,413-29,595,687 (- strand, TSS=29,595,687)
    "VEGFA": ("chr6", 43760000, 43790000),    # NCBI GeneID 7422: chr6:43,770,210-43,786,486 (+ strand, TSS=43,770,210)
}

# Verified ChIP-Atlas hg38 MACS2 (q < 1e-5) experiments across 4 matched human solid-tumor lineages
CHIP_ATLAS_TRACKS = [
    ("ChIP-Atlas", "SRX15691692", "GSM6239122", "HIF1A", "RCC4+VHL (0.5% O2 6h)", "Renal Cell Carcinoma"),
    ("ChIP-Atlas", "SRX666556", "GSM1462475", "HIF1A", "T-47D", "Breast Adenocarcinoma"),
    ("ChIP-Atlas", "SRX2584127", "GSM2501226", "HIF1A", "501mel (1% O2 16h)", "Melanoma"),
    ("ChIP-Atlas", "SRX3342251", "GSM2835770", "HIF1A", "PC-3", "Prostate Carcinoma"),
    ("ChIP-Atlas", "SRX8566545", "GSE148506", "SMARCA4", "786-O", "Renal Clear Cell Carcinoma"),
    ("ChIP-Atlas", "SRX1156541", "GSE71510", "SMARCA4", "MDA-MB-231", "Breast Adenocarcinoma"),
    ("ChIP-Atlas", "SRX1156535", "GSE71510", "SMARCA4", "MDA-MB-231", "Breast Adenocarcinoma"),
    ("ChIP-Atlas", "SRX10640844", "GSE171525", "SMARCA4", "SK-MEL-147", "Melanoma"),
    ("ChIP-Atlas", "SRX6743053", "GSE136803", "SMARCA4", "VCaP", "Prostate Carcinoma"),
]


def fetch_or_cache(url: str, local_name: str, timeout: int = 45) -> str:
    os.makedirs(RAW_CHIP_DIR, exist_ok=True)
    local_path = os.path.join(RAW_CHIP_DIR, local_name)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path


def parse_chip_atlas_peaks(srx: str, meta: tuple) -> list[dict]:
    _, _, sec_acc, factor, cell_line, cancer_type = meta
    fp = fetch_or_cache(
        f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{srx}.05.bed",
        f"{srx}.05.bed",
    )
    rows = []
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, s, e = parts[0], int(parts[1]), int(parts[2])
            for locus, (lc, ls, le) in LOCI.items():
                if chrom == lc and e >= ls and s <= le:
                    rows.append({
                        "Source": "ChIP-Atlas",
                        "Accession": srx,
                        "Secondary_ID": sec_acc,
                        "Factor": factor,
                        "Cell_Line": cell_line,
                        "Context": cancer_type,
                        "Locus": locus,
                        "Chrom": chrom,
                        "Start_hg38": s,
                        "End_hg38": e,
                        "Peak_Width_bp": e - s,
                        "MACS2_Score": int(float(parts[4])),
                        "Fold_Enrichment": round(float(parts[6]), 3),
                        "NegLog10_P": round(float(parts[7]), 3),
                        "NegLog10_Q": round(float(parts[8]), 3),
                    })
    return rows


def bp_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def main() -> None:
    print("PUBLIC SOLID-TUMOR CHIP-ATLAS OVERLAP AT SLC7A1 (chr13:29.50-29.62Mb) & VEGFA (chr6:43.76-43.79Mb, hg38)")
    rows = []
    for t in CHIP_ATLAS_TRACKS:
        rows.extend(parse_chip_atlas_peaks(t[1], t))

    df = pd.DataFrame(rows).sort_values(["Locus", "Factor", "Start_hg38", "Accession"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    for locus in ["SLC7A1", "VEGFA"]:
        sub = df[df["Locus"] == locus]
        print(f"\n[{locus} Locus: {LOCI[locus][0]}:{LOCI[locus][1]:,}-{LOCI[locus][2]:,}] ({len(sub)} peaks)")
        for _, r in sub.iterrows():
            print(
                f"  {r['Source']:<10} | {r['Accession']:<12} ({r['Secondary_ID']:<11}) | {r['Factor']:<7} | "
                f"{r['Cell_Line']:<24} | {r['Chrom']}:{r['Start_hg38']:,}-{r['End_hg38']:,} "
                f"| signal={r['Fold_Enrichment']:6.2f}x | -log10(q)={r['NegLog10_Q']:5.2f}"
            )

    # Report exact lineage-matched HIF1A + SMARCA4 peak co-occupancy overlaps at SLC7A1 & VEGFA
    print("\n[Exact Base-Pair Co-Occupancy Between HIF1A and SMARCA4 Peaks]")
    for locus in ["SLC7A1", "VEGFA"]:
        hif_sub = df[(df["Locus"] == locus) & (df["Factor"] == "HIF1A")]
        sm_sub = df[(df["Locus"] == locus) & (df["Factor"] == "SMARCA4")]
        overlaps = []
        for _, h in hif_sub.iterrows():
            for _, s in sm_sub.iterrows():
                ov = bp_overlap(h["Start_hg38"], h["End_hg38"], s["Start_hg38"], s["End_hg38"])
                if ov > 0:
                    overlaps.append((ov, h["Accession"], h["Cell_Line"], h["Start_hg38"], h["End_hg38"], s["Accession"], s["Cell_Line"], s["Start_hg38"], s["End_hg38"]))
        overlaps.sort(reverse=True)
        print(f"  {locus}: {len(overlaps)} overlapping HIF1A-SMARCA4 peak pairs; top examples:")
        for item in overlaps[:6]:
            print(f"    {item[0]:3d} bp | HIF1A {item[1]} ({item[2]}: {item[3]:,}-{item[4]:,}) <-> SMARCA4 {item[5]} ({item[6]}: {item[7]:,}-{item[8]:,})")

    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
