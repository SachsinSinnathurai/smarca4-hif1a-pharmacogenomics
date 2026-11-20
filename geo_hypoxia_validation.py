#!/usr/bin/env python3
"""Paired 1% vs. 21% O2 Hypoxia Perturbation Validation across Public NCBI GEO Human Solid-Tumor Cell Line Series."""

import gzip
import io
import os
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_GEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "geo_hypoxia")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "geo_hypoxia_perturbation.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}

TARGET_GENES = ["SLC7A1", "SLC1A5", "HIF1A", "SMARCA4", "CA9", "VEGFA"]
ENSEMBL_TO_SYMBOL = {
    "ENSG00000139514": "SLC7A1",
    "ENSG00000105281": "SLC1A5",
    "ENSG00000100644": "HIF1A",
    "ENSG00000127616": "SMARCA4",
    "ENSG00000107159": "CA9",
    "ENSG00000112715": "VEGFA",
}


def fetch_or_cache(url: str, local_name: str, timeout: int = 45) -> str:
    os.makedirs(RAW_GEO_DIR, exist_ok=True)
    local_path = os.path.join(RAW_GEO_DIR, local_name)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path


def load_gpl10558_gene_probes() -> dict[str, list[str]]:
    fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz",
        "GPL10558.annot.gz",
        timeout=60,
    )
    gmap: dict[str, list[str]] = {g: [] for g in TARGET_GENES}
    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
        hdr = None
        sym_idx = -1
        for line in f:
            ls = line.rstrip("\n")
            if ls.startswith("ID\t"):
                hdr = ls.split("\t")
                sym_idx = hdr.index("Gene symbol")
                continue
            if hdr and not ls.startswith("!"):
                parts = ls.split("\t")
                if len(parts) > sym_idx:
                    for sym in parts[sym_idx].split("///"):
                        s_clean = sym.strip()
                        if s_clean in gmap and parts[0] not in gmap[s_clean]:
                            gmap[s_clean].append(parts[0])
    return gmap


def eval_gse283995() -> list[dict]:
    """GSE283995: HepG2 hepatocellular carcinoma, 21% O2 Normoxia (n=6) vs. 1% O2 24h (n=6) & 6d Hypoxia (n=5)."""
    fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE283nnn/GSE283995/suppl/GSE283995_normalized_counts.csv.gz",
        "GSE283995_normalized_counts.csv.gz",
    )
    df = pd.read_csv(fp, compression="gzip").set_index("Unnamed: 0")
    nx_idx = sorted([i for i in df.index if "Normoxia" in i])
    hx24_idx = sorted([i for i in df.index if "24h-Hypoxia" in i])
    hx6d_idx = sorted([i for i in df.index if "6d-Hypoxia" in i])
    assert len(nx_idx) == 6 and len(hx24_idx) == 6 and len(hx6d_idx) == 5, (
        f"Unexpected GSE283995 sample counts: nx={len(nx_idx)}, 24h={len(hx24_idx)}, 6d={len(hx6d_idx)}"
    )

    rows = []
    for comp_label, hx_idx in [("1% O2 (24h) vs 21% O2", hx24_idx), ("1% O2 (6d) vs 21% O2", hx6d_idx)]:
        for g in TARGET_GENES:
            cols = [c for c in df.columns if c.endswith("|" + g)]
            assert len(cols) == 1, f"Expected 1 column for {g} in GSE283995, found {cols}"
            vx = df.loc[nx_idx, cols[0]].values.astype(float)
            vh = df.loc[hx_idx, cols[0]].values.astype(float)
            t_stat, p_val = stats.ttest_ind(vh, vx, equal_var=False)
            rows.append({
                "GEO_Accession": "GSE283995",
                "Cell_Line": "HepG2",
                "Cancer_Type": "Hepatocellular Carcinoma",
                "Platform": "RNA-seq (log2 norm. counts)",
                "Comparison": comp_label,
                "N_Normoxia": len(vx),
                "N_Hypoxia": len(vh),
                "Paired": False,
                "Test": "Welch two-sample t-test",
                "Gene": g,
                "Mean_Normoxia_Log2": round(float(vx.mean()), 4),
                "Mean_Hypoxia_Log2": round(float(vh.mean()), 4),
                "Log2FC": round(float(vh.mean() - vx.mean()), 4),
                "T_Statistic": round(float(t_stat), 4),
                "P_Value": f"{float(p_val):.3e}",
            })
    return rows


def eval_gse139673() -> list[dict]:
    """GSE139673: AsPC-1 pancreatic ductal adenocarcinoma, paired 1% O2 vs 21% O2 (n=3 pairs, siControl) + siARNT check."""
    fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139673/suppl/GSE139673_all_expression_counts.tsv.gz",
        "GSE139673_all_expression_counts.tsv.gz",
    )
    mat_fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139673/matrix/GSE139673_series_matrix.txt.gz",
        "GSE139673_series_matrix.txt.gz",
    )
    # Inspect exact protocol duration from series matrix
    with gzip.open(mat_fp, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            if line.startswith("!Sample_treatment_protocol_ch1"):
                print("  [GSE139673 protocol]", line.strip().split("\t")[1][:180])
                break

    df = pd.read_csv(fp, compression="gzip", sep="\t").set_index("gene_id")
    df["ens"] = df.index.str.split(".").str[0]
    df_e = df.set_index("ens")
    cpm = df_e.div(df_e.sum(axis=0), axis=1) * 1e6

    nx_cols = ["exp_norm_ctrl_A", "exp_norm_ctrl_B", "exp_norm_ctrl_C"]
    hx_cols = ["exp_hyp_ctrl_A", "exp_hyp_ctrl_B", "exp_hyp_ctrl_C"]
    kd_cols = ["exp_hyp_noHIF_A", "exp_hyp_noHIF_B", "exp_hyp_noHIF_C"]

    rows = []
    for ens, g in ENSEMBL_TO_SYMBOL.items():
        vx = np.log2(cpm.loc[ens, nx_cols].values.astype(float) + 1.0)
        vh = np.log2(cpm.loc[ens, hx_cols].values.astype(float) + 1.0)
        t_stat, p_val = stats.ttest_rel(vh, vx)
        rows.append({
            "GEO_Accession": "GSE139673",
            "Cell_Line": "AsPC-1",
            "Cancer_Type": "Pancreatic Ductal Adenocarcinoma",
            "Platform": "RNA-seq (log2 CPM+1)",
            "Comparison": "1% O2 vs 21% O2 (siControl)",
            "N_Normoxia": len(vx),
            "N_Hypoxia": len(vh),
            "Paired": True,
            "Test": "Paired two-sided t-test",
            "Gene": g,
            "Mean_Normoxia_Log2": round(float(vx.mean()), 4),
            "Mean_Hypoxia_Log2": round(float(vh.mean()), 4),
            "Log2FC": round(float(vh.mean() - vx.mean()), 4),
            "T_Statistic": round(float(t_stat), 4),
            "P_Value": f"{float(p_val):.3e}",
        })
        if g in ("CA9", "SLC7A1"):
            vk = np.log2(cpm.loc[ens, kd_cols].values.astype(float) + 1.0)
            tk, pk = stats.ttest_rel(vk, vh)
            print(f"  [GSE139673 siARNT check in 1% O2] {g}: Hx_siCtrl={vh.mean():.3f} -> Hx_siARNT={vk.mean():.3f} (log2FC={vk.mean()-vh.mean():+.3f}, paired p={pk:.2e})")
    return rows


def eval_gse145108() -> list[dict]:
    """GSE145108: RKO (colon), A549 (lung), H460 (lung), 24h 1% O2 vs 21% O2 (n=6 pooled + per-line pairs)."""
    fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145108/suppl/GSE145108_Normoxia_Hypoxia_RKO_A549_H460_RPKM.txt.gz",
        "GSE145108_Normoxia_Hypoxia_RKO_A549_H460_RPKM.txt.gz",
    )
    df = pd.read_csv(fp, compression="gzip", sep="\t").set_index("GeneID")
    line_specs = [
        (
            "RKO + A549 + H460 (pooled pairs)",
            "Colorectal & Non-Small Cell Lung",
            "1% O2 (24h) vs 21% O2",
            "Paired two-sided t-test",
            ["RKO_normoxia_1", "RKO_normoxia_2", "A549_normoxia_1", "A549_normoxia_2", "H460_normoxia_1", "H460_normoxia_2"],
            ["RKO_hypoxia_1", "RKO_hypoxia_2", "A549_hypoxia_1", "A549_hypoxia_2", "H460_hypoxia_1", "H460_hypoxia_2"],
        ),
        ("RKO (GSE145108 stratified)", "Colorectal Carcinoma", "1% O2 (24h) vs 21% O2 (RKO)", "Paired two-sided t-test (n=2 pairs)", ["RKO_normoxia_1", "RKO_normoxia_2"], ["RKO_hypoxia_1", "RKO_hypoxia_2"]),
        ("A549 (GSE145108 stratified)", "Non-Small Cell Lung", "1% O2 (24h) vs 21% O2 (A549)", "Paired two-sided t-test (n=2 pairs)", ["A549_normoxia_1", "A549_normoxia_2"], ["A549_hypoxia_1", "A549_hypoxia_2"]),
        ("H460 (GSE145108 stratified)", "Non-Small Cell Lung", "1% O2 (24h) vs 21% O2 (H460)", "Paired two-sided t-test (n=2 pairs)", ["H460_normoxia_1", "H460_normoxia_2"], ["H460_hypoxia_1", "H460_hypoxia_2"]),
    ]

    rows = []
    for cl_lbl, ca_lbl, comp_lbl, test_lbl, nx_cols, hx_cols in line_specs:
        for g in TARGET_GENES:
            vx = np.log2(df.loc[g, nx_cols].values.astype(float) + 1.0)
            vh = np.log2(df.loc[g, hx_cols].values.astype(float) + 1.0)
            t_stat, p_val = stats.ttest_rel(vh, vx)
            rows.append({
                "GEO_Accession": "GSE145108",
                "Cell_Line": cl_lbl,
                "Cancer_Type": ca_lbl,
                "Platform": "RNA-seq (log2 RPKM+1)",
                "Comparison": comp_lbl,
                "N_Normoxia": len(vx),
                "N_Hypoxia": len(vh),
                "Paired": True,
                "Test": test_lbl,
                "Gene": g,
                "Mean_Normoxia_Log2": round(float(vx.mean()), 4),
                "Mean_Hypoxia_Log2": round(float(vh.mean()), 4),
                "Log2FC": round(float(vh.mean() - vx.mean()), 4),
                "T_Statistic": round(float(t_stat), 4),
                "P_Value": f"{float(p_val):.3e}",
            })
    return rows


def eval_gse300765(gpl_probes: dict[str, list[str]]) -> list[dict]:
    """GSE300765: U87MG glioblastoma, 48h 1% O2 vs 21% O2 (n=3 paired replicates, GPL10558)."""
    fp = fetch_or_cache(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE300nnn/GSE300765/matrix/GSE300765_series_matrix.txt.gz",
        "GSE300765_series_matrix.txt.gz",
    )
    want_probes = {p for ps in gpl_probes.values() for p in ps}
    titles, cols, prob_rows = [], [], []
    in_t = False
    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            ls = line.rstrip("\n")
            if ls.startswith("!Sample_title\t"):
                titles = [x.strip('"') for x in ls.split("\t")[1:]]
            elif ls.startswith("!series_matrix_table_begin"):
                in_t = True
            elif ls.startswith("!series_matrix_table_end"):
                break
            elif in_t:
                parts = ls.split("\t")
                key = parts[0].strip('"')
                if key == "ID_REF":
                    cols = [x.strip('"') for x in parts[1:]]
                elif key in want_probes:
                    prob_rows.append((key, [float(v) for v in parts[1:]]))

    df_m = pd.DataFrame({k: v for k, v in prob_rows}, index=cols)
    df_m["title"] = titles
    nx_idx = [i for i in df_m.index if "_nox_" in df_m.loc[i, "title"]]
    hx_idx = [i for i in df_m.index if "_hox_" in df_m.loc[i, "title"]]
    assert len(nx_idx) == 3 and len(hx_idx) == 3

    rows = []
    for g in TARGET_GENES:
        ps = gpl_probes[g]
        vx = df_m.loc[nx_idx, ps].mean(axis=1).values.astype(float)
        vh = df_m.loc[hx_idx, ps].mean(axis=1).values.astype(float)
        t_stat, p_val = stats.ttest_rel(vh, vx)
        rows.append({
            "GEO_Accession": "GSE300765",
            "Cell_Line": "U87MG",
            "Cancer_Type": "Glioblastoma",
            "Platform": "Illumina HumanHT-12 v4 (GPL10558, log2)",
            "Comparison": "1% O2 (48h) vs 21% O2",
            "N_Normoxia": len(vx),
            "N_Hypoxia": len(vh),
            "Paired": True,
            "Test": "Paired two-sided t-test",
            "Gene": g,
            "Mean_Normoxia_Log2": round(float(vx.mean()), 4),
            "Mean_Hypoxia_Log2": round(float(vh.mean()), 4),
            "Log2FC": round(float(vh.mean() - vx.mean()), 4),
            "T_Statistic": round(float(t_stat), 4),
            "P_Value": f"{float(p_val):.3e}",
        })
    return rows


def main() -> None:
    print("PUBLIC NCBI GEO 1% VS 21% O2 HYPOXIA PERTURBATION VALIDATION")
    gpl_probes = load_gpl10558_gene_probes()
    all_rows = []
    all_rows.extend(eval_gse283995())
    all_rows.extend(eval_gse139673())
    all_rows.extend(eval_gse145108())
    all_rows.extend(eval_gse300765(gpl_probes))

    df_out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print("\nSummary Table (Log2FC = Hypoxia 1% O2 - Normoxia 21% O2):")
    for (gse, cell_line, comp), sub in df_out.groupby(["GEO_Accession", "Cell_Line", "Comparison"], sort=False):
        print(f"\n  [{gse} | {cell_line} | {comp} | n={sub['N_Normoxia'].iloc[0]} vs {sub['N_Hypoxia'].iloc[0]} | {sub['Test'].iloc[0]}]")
        for _, r in sub.iterrows():
            print(f"    {r['Gene']:<8}: Nx={r['Mean_Normoxia_Log2']:6.3f} -> Hx={r['Mean_Hypoxia_Log2']:6.3f} | log2FC={r['Log2FC']:+6.3f} | t={r['T_Statistic']:+6.2f} | p={r['P_Value']}")
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
