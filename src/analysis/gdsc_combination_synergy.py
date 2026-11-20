#!/usr/bin/env python3
"""Per-cell-line pooled Sanger GDSC combination Bliss excess and Risk/SMARCA4 correlation analysis."""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")

GDSC_MATRIX_FILES = [
    "gdsc-007_matrix_results.csv",
    "gdsc-008_matrix_results.csv",
    "gdsc-009_matrix_results.csv",
    "gdsc-010_matrix_results.csv",
    "sandpiper-01_matrix_results.csv",
]

PAIRS = [
    ("AZD5153", "AZD5991", "BRD4 + MCL1 Co-Inhibition"),
    ("Olaparib", "SN-38", "PARP1/2 + TOP1 Co-Inhibition"),
    ("Capivasertib", "Selumetinib", "AKT + MEK1/2 Co-Inhibition"),
]


def resolve_gdsc_path(filename: str) -> str:
    p = os.path.join(DATA_RAW, "gdsc", filename)
    return p if os.path.exists(p) else os.path.join(PROJECT_ROOT, filename)


def main():
    model_fp = os.path.join(DATA_RAW, "Model.csv")
    expr_fp = os.path.join(DATA_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    crispr_fp = os.path.join(DATA_RAW, "CRISPRGeneEffect.csv")
    gdsc_paths = [resolve_gdsc_path(fn) for fn in GDSC_MATRIX_FILES]
    if not all(os.path.exists(p) for p in (model_fp, expr_fp, crispr_fp)) or not any(os.path.exists(p) for p in gdsc_paths):
        print("Raw DepMap or GDSC combination archives absent; preserving pre-computed gdsc_combination_synergy_summary.csv.")
        return

    df_m = pd.read_csv(model_fp)
    non_solid = ["Lymphoid", "Myeloid", "Normal", "Fibroblast"]
    norm_ids = set(
        df_m[
            (df_m["OncotreePrimaryDisease"] == "Non-Cancerous")
            | (df_m["OncotreeLineage"].isin(["Normal", "Fibroblast"]))
        ]["ModelID"].dropna().astype(str)
    )
    solid_models = set(
        df_m[
            (~df_m["OncotreeLineage"].isin(non_solid))
            & (~df_m["ModelID"].astype(str).isin(norm_ids))
        ]["ModelID"].dropna().astype(str)
    )
    c2m = {
        int(float(r["COSMICID"])): str(r["ModelID"])
        for _, r in df_m.iterrows()
        if pd.notna(r.get("COSMICID")) and pd.notna(r.get("ModelID"))
    }

    gdsc_dfs = []
    for fp in gdsc_paths:
        if os.path.exists(fp):
            gdsc_dfs.append(
                pd.read_csv(
                    fp,
                    usecols=["COSMIC_ID", "lib1_name", "lib2_name", "Bliss_window", "HSA_window"],
                )
            )
    df_gdsc = pd.concat(gdsc_dfs, ignore_index=True)
    total_rows = int(len(df_gdsc))
    df_gdsc["ModelID"] = [c2m.get(int(x)) if pd.notna(x) else None for x in df_gdsc["COSMIC_ID"]]
    df_solid = df_gdsc[df_gdsc["ModelID"].isin(solid_models)].copy()

    lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()

    # Load Risk Score, SLFN11, and BAK1 across deduplicated unique solid DepMap lines
    eh = pd.read_csv(expr_fp, nrows=1).columns
    sm_col = next(c for c in eh if c.split(" ")[0] == "SMARCA4")
    hf_col = next(c for c in eh if c.split(" ")[0] == "HIF1A")
    sl_col = next(c for c in eh if c.split(" ")[0] == "SLFN11")
    bk_col = next(c for c in eh if c.split(" ")[0] == "BAK1")
    df_expr = pd.read_csv(expr_fp, usecols=["ModelID", "IsDefaultEntryForModel", sm_col, hf_col, sl_col, bk_col]).rename(
        columns={sm_col: "SMARCA4", hf_col: "HIF1A", sl_col: "SLFN11", bk_col: "BAK1"}
    )
    df_expr = df_expr[df_expr["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").dropna(subset=["SMARCA4", "HIF1A", "SLFN11", "BAK1"]).copy()
    df_expr["ModelID"] = df_expr["ModelID"].astype(str)
    df_expr = df_expr[df_expr["ModelID"].isin(solid_models)].copy()
    df_expr["Lineage"] = df_expr["ModelID"].map(lin_map)
    sm_z = (df_expr["SMARCA4"] - df_expr["SMARCA4"].mean()) / df_expr["SMARCA4"].std()
    hf_z = (df_expr["HIF1A"] - df_expr["HIF1A"].mean()) / df_expr["HIF1A"].std()
    raw_sum = sm_z + hf_z
    df_expr["Risk_Score"] = (raw_sum - raw_sum.mean()) / raw_sum.std()

    # Load SMARCA4 Chronos score across DepMap lines
    ch = pd.read_csv(crispr_fp, nrows=1).columns
    id_col = "ModelID" if "ModelID" in ch else ch[0]
    sm_c = next(c for c in ch if c.split(" ")[0].strip("()") == "SMARCA4")
    df_crispr = pd.read_csv(crispr_fp, usecols=[id_col, sm_c]).rename(
        columns={id_col: "ModelID", sm_c: "SMARCA4_Chronos"}
    ).dropna()
    df_crispr["ModelID"] = df_crispr["ModelID"].astype(str)

    # Empirical all-pair background benchmark across canonical drug pairs with >= 30 unique solid lines
    df_solid["Pair_Key"] = np.where(
        df_solid["lib1_name"].astype(str) <= df_solid["lib2_name"].astype(str),
        df_solid["lib1_name"].astype(str) + " + " + df_solid["lib2_name"].astype(str),
        df_solid["lib2_name"].astype(str) + " + " + df_solid["lib1_name"].astype(str),
    )
    all_pair_lines = df_solid.groupby(["Pair_Key", "ModelID"], as_index=False)["Bliss_window"].mean().dropna()
    pair_stats = all_pair_lines.groupby("Pair_Key").agg(n_lines=("ModelID", "nunique"), med_bliss=("Bliss_window", "median"))
    bg_pairs = pair_stats[pair_stats["n_lines"] >= 30]
    bg_median = float(bg_pairs["med_bliss"].median())

    rows = []
    print(f"GDSC COMBINATION SYNERGY (Total Pooled Rows = {total_rows:,}; Solid-Tumor Rows = {len(df_solid):,}; Background k = {len(bg_pairs)} pairs, median Bliss = {bg_median:+.4f})")
    for d1, d2, mech in PAIRS:
        sub = df_solid[
            ((df_solid["lib1_name"] == d1) & (df_solid["lib2_name"] == d2))
            | ((df_solid["lib1_name"] == d2) & (df_solid["lib2_name"] == d1))
        ].copy()
        per_line = sub.groupby("ModelID", as_index=False)[["Bliss_window", "HSA_window"]].mean().dropna(subset=["Bliss_window"])
        # Evaluate extreme-value peak 3x3 Bliss window against the 109-pair empirical background median (+0.0662), not against 0
        _, w_bg_p = stats.wilcoxon(per_line["Bliss_window"].values - bg_median, alternative="two-sided")
        _, w_bg_gt_p = stats.wilcoxon(per_line["Bliss_window"].values - bg_median, alternative="greater")

        n_pos = int((per_line["Bliss_window"] > 0).sum())
        n_gt_bg = int((per_line["Bliss_window"] > bg_median).sum())
        n_ge10 = int((per_line["Bliss_window"] >= 0.10).sum())
        med_val = float(per_line["Bliss_window"].median())
        pct_rank = float((bg_pairs["med_bliss"] < med_val).mean() * 100.0)

        m_risk = per_line.merge(df_expr[["ModelID", "Risk_Score", "Lineage", "SLFN11", "BAK1"]], on="ModelID").dropna()
        rho_risk, p_risk = stats.spearmanr(m_risk["Bliss_window"], m_risk["Risk_Score"])
        r_slfn11, p_slfn11 = stats.pearsonr(m_risk["Bliss_window"], m_risk["SLFN11"])
        r_bak1, p_bak1 = stats.pearsonr(m_risk["Bliss_window"], m_risk["BAK1"])
        aov = sm.stats.anova_lm(smf.ols("Bliss_window ~ Risk_Score + C(Lineage)", data=m_risk).fit(), typ=2)
        f_lin = float(aov.loc["Risk_Score", "F"])
        p_lin = float(aov.loc["Risk_Score", "PR(>F)"])
        df2_lin = int(aov.loc["Residual", "df"])

        m_ch = per_line.merge(df_crispr[["ModelID", "SMARCA4_Chronos"]], on="ModelID").dropna()
        rho_ch, p_ch = stats.spearmanr(m_ch["Bliss_window"], m_ch["SMARCA4_Chronos"])

        row = {
            "Drug_Pair": f"{d1} + {d2}",
            "Mechanism_Class": mech,
            "Total_Combo_Rows_Pooled": total_rows,
            "Solid_Tumor_Rows_Pooled": int(len(df_solid)),
            "Solid_Tumor_Matrix_Screens": int(len(sub)),
            "Unique_Solid_Cell_Lines": int(len(per_line)),
            "Positive_Bliss_Cell_Lines": n_pos,
            "Above_Background_Median_Cell_Lines": n_gt_bg,
            "Bliss_GE_010_Cell_Lines": n_ge10,
            "Median_Bliss_Excess": round(med_val, 4),
            "Mean_Bliss_Excess": round(float(per_line["Bliss_window"].mean()), 4),
            "Median_HSA_Excess": round(float(per_line["HSA_window"].median()), 4),
            "Empirical_Background_Median": round(bg_median, 4),
            "Median_Minus_Background": round(med_val - bg_median, 4),
            "Background_Median_PctRank": round(pct_rank, 1),
            "Wilcoxon_Vs_Background_P": float(w_bg_p),
            "Wilcoxon_Gt_Background_P": float(w_bg_gt_p),
            "N_Risk_Score_Overlap": int(len(m_risk)),
            "Spearman_Rho_Risk_Score": round(float(rho_risk), 4),
            "Spearman_P_Risk_Score": float(p_risk),
            "Lineage_Adjusted_TypeII_F": round(f_lin, 2),
            "Lineage_Adjusted_DF_Den": df2_lin,
            "Lineage_Adjusted_TypeII_P": p_lin,
            "Pearson_R_SLFN11": round(float(r_slfn11), 4),
            "Pearson_P_SLFN11": float(p_slfn11),
            "Pearson_R_BAK1": round(float(r_bak1), 4),
            "Pearson_P_BAK1": float(p_bak1),
            "N_SMARCA4_Chronos_Overlap": int(len(m_ch)),
            "Spearman_Rho_SMARCA4_Chronos": round(float(rho_ch), 4),
            "Spearman_P_SMARCA4_Chronos": float(p_ch),
        }
        rows.append(row)
        print(
            f"  * {row['Drug_Pair']:<25} | n_mat={row['Solid_Tumor_Matrix_Screens']:3d} | "
            f">bg({bg_median:+.4f})={n_gt_bg}/{row['Unique_Solid_Cell_Lines']} (>=0.10: {n_ge10}/{row['Unique_Solid_Cell_Lines']} [{100*n_ge10/len(per_line):.1f}%]) | "
            f"med_Bliss={row['Median_Bliss_Excess']:+.4f} (Delta={row['Median_Minus_Background']:+.4f}, {pct_rank:.1f}th pct, vs bg 2s p={w_bg_p:.2e}, >bg p={w_bg_gt_p:.2e}) | "
            f"Risk rho={row['Spearman_Rho_Risk_Score']:+.4f} "
            f"(p={row['Spearman_P_Risk_Score']:.2e}, n={row['N_Risk_Score_Overlap']}; Lin-adj F(1, {df2_lin})={f_lin:.2f}, p={p_lin:.4f}) | "
            f"SLFN11 r={r_slfn11:+.4f} (p={p_slfn11:.2e}), BAK1 r={r_bak1:+.4f} (p={p_bak1:.4f})"
        )

    out_fp = os.path.join(DATA_PROC, "gdsc_combination_synergy_summary.csv")
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    print(f"Saved -> {out_fp}")


if __name__ == "__main__":
    main()
