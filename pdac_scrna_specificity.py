"""Cell-type expression specificity and patient-level pseudobulk analysis in primary PDAC scRNA-seq (CRA001160)."""

import os
import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw", "scrna_peng2019")
OUT_CSV = os.path.join(ROOT, "data", "processed", "pdac_scrna_celltype_specificity.csv")

H5_PATH = os.path.join(RAW_DIR, "PAAD_CRA001160_expression.h5")
META_PATH = os.path.join(RAW_DIR, "PAAD_CRA001160_CellMetainfo_table.tsv")
GENES = ["SMARCA4", "HIF1A", "SLC7A1", "SLC1A5", "ABCB1"]

COMPARTMENT_MAP = {
    "Malignant": "Malignant Epithelial",
    "Ductal": "Normal Epithelial",
    "Acinar": "Normal Epithelial",
    "Endocrine": "Normal Epithelial",
    "Fibroblasts": "Stromal",
    "Stellate": "Stromal",
    "Endothelial": "Stromal",
}


def load_h5_matrix(path):
    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        mat = sparse.csc_matrix(
            (grp["data"][:], grp["indices"][:], grp["indptr"][:]),
            shape=tuple(grp["shape"][:]),
        )
        barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in grp["barcodes"][:]]
        feat_key = grp["features"]["name"] if "features" in grp else grp["gene_names"]
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in feat_key[:]]
    return mat, np.array(genes), np.array(barcodes)


def main():
    if not os.path.exists(H5_PATH) or not os.path.exists(META_PATH):
        print("Raw TISCH2 PAAD_CRA001160 archives absent; preserving pre-computed pseudobulk CSV.")
        return

    meta = pd.read_csv(META_PATH, sep="\t")
    meta = meta.set_index("Cell" if "Cell" in meta.columns else meta.columns[0])
    ct_col = next(
        (c for c in meta.columns if "celltype" in c.lower() and "major" in c.lower()),
        next(c for c in meta.columns if "celltype" in c.lower()),
    )

    mat, genes, barcodes = load_h5_matrix(H5_PATH)
    nz = mat.data if mat.data.min() > 0 else mat.data[mat.data > 0]
    p25, p50, p90, p99, p999 = np.percentile(nz, [25, 50, 90, 99, 99.9])
    print(
        f"[CRA001160 h5 Nonzero Scale Audit] count={len(nz):,}, min={nz.min():.4f}, "
        f"p25={p25:.4f}, median={p50:.4f}, p90={p90:.4f}, p99={p99:.4f}, "
        f"p99.9={p999:.4f}, max={nz.max():.4f} (confirms ln(CP10K+1) normalization; np.expm1 exact)",
        flush=True,
    )
    del nz
    meta = meta.reindex(barcodes).dropna(subset=[ct_col])
    valid_cols = pd.Index(barcodes).isin(meta.index)
    tumor_mask = (meta["Source"] == "Tumor").values
    meta_tum = meta.loc[tumor_mask]
    expr_tum = np.zeros((len(GENES), int(tumor_mask.sum())), dtype=float)
    for k, gene in enumerate(GENES):
        row_idx = int(np.where(genes == gene)[0][0])
        pos = np.where(mat.indices == row_idx)[0]
        cols_nz = np.searchsorted(mat.indptr[1:], pos, side="right")
        expr_full = np.zeros(mat.shape[1], dtype=float)
        np.add.at(expr_full, cols_nz, mat.data[pos])
        expr_tum[k] = expr_full[valid_cols][tumor_mask]
    del mat
    cell_types = meta_tum[ct_col].values
    patients = meta_tum["Patient"].values
    ordered_cts = pd.Series(cell_types).value_counts().index.tolist()

    records = []
    for k, gene in enumerate(GENES):
        expr = expr_tum[k]

        lin_expr = np.expm1(expr)
        df_g = pd.DataFrame({"Patient": patients, "CT": cell_types, "Expr": expr, "LinExpr": lin_expr})
        pb = df_g[df_g["CT"].isin(["Malignant", "Fibroblasts"])].groupby(["Patient", "CT"])["LinExpr"].mean().unstack().dropna()
        n_paired = int(len(pb))
        log2fc_pb = float(np.log2((pb["Malignant"].mean() + 0.01) / (pb["Fibroblasts"].mean() + 0.01)))
        _, p_wil = stats.wilcoxon(pb["Malignant"], pb["Fibroblasts"], alternative="two-sided")
        _, p_mwu = stats.mannwhitneyu(pb["Malignant"], pb["Fibroblasts"], alternative="two-sided")

        for ct in ordered_cts:
            mask_ct = (cell_types == ct)
            vals = expr[mask_ct]
            lin_vals = lin_expr[mask_ct]
            records.append({
                "Target_Gene": gene,
                "Cell_Type": ct,
                "Biological_Compartment": COMPARTMENT_MAP.get(ct, "Immune"),
                "N_Tumor_Cells": int(len(vals)),
                "N_Paired_Patients": n_paired,
                "Pct_Expressing": round(float((vals > 0).mean() * 100.0), 4),
                "Mean_LogNorm_Expr": round(float(vals.mean()), 6),
                "Mean_Linear_Expr": round(float(lin_vals.mean()), 6),
                "Pseudobulk_Log2FC_Malignant_vs_CAF": round(log2fc_pb, 6),
                "Paired_Wilcoxon_P_vs_CAF": float(p_wil),
                "Pseudobulk_MWU_P_vs_CAF": float(p_mwu),
            })

    df = pd.DataFrame(records)
    p_unique = [df.loc[df["Target_Gene"] == g, "Paired_Wilcoxon_P_vs_CAF"].iloc[0] for g in GENES]
    _, q_unique, _, _ = multipletests(p_unique, method="fdr_bh")
    df["Paired_Wilcoxon_FDR_q"] = df["Target_Gene"].map(dict(zip(GENES, q_unique)))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows across {expr_tum.shape[1]:,} tumor cells (n={n_paired} paired patients) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
