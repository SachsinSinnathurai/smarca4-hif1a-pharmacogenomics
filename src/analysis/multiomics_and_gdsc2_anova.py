#!/usr/bin/env python3
"""Full-Screen GDSC2 Type II ANOVA, MSigDB Hallmark GSEA & CAMERA, TCPA RPPA Proteomics, CIN Orthogonality, and IMvigor210 Survival."""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")


def _zscore(vals: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    arr = np.asarray(vals, dtype=float)
    sd = np.nanstd(arr, ddof=1)
    out = (arr - np.nanmean(arr)) / sd if (sd > 0 and not np.isnan(sd)) else np.zeros_like(arr)
    return pd.Series(out, index=vals.index) if isinstance(vals, pd.Series) else out


def _unweighted_apriori_score(smarca4_vals, hif1a_vals):
    """Compute within-cohort standardized unweighted score Z(Z(SMARCA4) + Z(HIF1A))."""
    return _zscore(_zscore(smarca4_vals) + _zscore(hif1a_vals))


def get_solid_models(model_fp: str) -> tuple[set[str], dict, dict, dict]:
    """Build unified malignant solid-tumor ModelID set excluding hematopoietic and non-cancerous lines."""
    df_m = pd.read_csv(model_fp)
    non_solid = ["Lymphoid", "Myeloid", "Normal", "Fibroblast"]
    norm_ids = set(
        df_m[
            (df_m["OncotreePrimaryDisease"] == "Non-Cancerous")
            | (df_m["OncotreeLineage"].isin(["Normal", "Fibroblast"]))
        ]["ModelID"]
        .dropna()
        .astype(str)
    )
    solid_ids = set(
        df_m[
            (~df_m["OncotreeLineage"].isin(non_solid))
            & (~df_m["ModelID"].astype(str).isin(norm_ids))
        ]["ModelID"]
        .dropna()
        .astype(str)
    )
    c2m = {
        int(float(r["COSMICID"])): str(r["ModelID"])
        for _, r in df_m.iterrows()
        if pd.notna(r.get("COSMICID")) and pd.notna(r.get("ModelID"))
    }
    s2m = {
        str(r["SangerModelID"]): str(r["ModelID"])
        for _, r in df_m.iterrows()
        if pd.notna(r.get("SangerModelID")) and pd.notna(r.get("ModelID"))
    }
    lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()
    return solid_ids, c2m, s2m, lin_map


def run_gdsc2_anova():
    """Run lineage-controlled Type II OLS ANOVA across Sanger GDSC2 compounds using primary 1 - AUC and ln(IC50) concordance."""
    out_fp = os.path.join(DATA_PROC, "gdsc2_lineage_controlled_anova.csv")
    gdsc_fp = os.path.join(DATA_RAW, "gdsc", "GDSC2_fitted_dose_response_27Oct23.xlsx")
    model_fp = os.path.join(DATA_RAW, "Model.csv")
    expr_fp = os.path.join(DATA_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")

    if not all(os.path.exists(p) for p in (gdsc_fp, model_fp, expr_fp)):
        if os.path.exists(out_fp):
            df_aov = pd.read_csv(out_fp)
            n_fdr10 = int((df_aov["FDR_Q_Value"] < 0.10).sum())
            n_nom = int((df_aov["P_Value"] < 0.05).sum())
            print(f"[1] Raw GDSC2 or DepMap archives absent; GDSC2 1-AUC ANOVA ({len(df_aov)} drugs; {n_fdr10} FDR q < 0.10; {n_nom} nominal p < 0.05)")
        else:
            print("[1] Raw GDSC2 or DepMap archives absent; skipping GDSC2 ANOVA recomputation.")
        return

    solid_ids, c2m, s2m, lin_map = get_solid_models(model_fp)
    eh = pd.read_csv(expr_fp, nrows=1).columns
    gate_genes = ["SMARCA4", "HIF1A", "MKI67", "SLFN11", "BAK1", "BAX"]
    col_map = {next(c for c in eh if c.split(" ")[0] == g): g for g in gate_genes}
    df_expr = pd.read_csv(expr_fp, usecols=["ModelID", "IsDefaultEntryForModel"] + list(col_map.keys())).rename(columns=col_map)
    df_expr = (
        df_expr[df_expr["IsDefaultEntryForModel"] == "Yes"]
        .drop_duplicates("ModelID")
        .dropna(subset=["SMARCA4", "HIF1A", "MKI67"])
        .copy()
    )
    df_expr["ModelID"] = df_expr["ModelID"].astype(str)
    # Filter to malignant solid-tumor models BEFORE Z-scoring SMARCA4, HIF1A, and MKI67
    df_expr = df_expr[df_expr["ModelID"].isin(solid_ids)].copy()
    df_expr["Lineage"] = df_expr["ModelID"].map(lin_map)
    df_expr["Risk_Score"] = _unweighted_apriori_score(df_expr["SMARCA4"], df_expr["HIF1A"])
    df_expr["MKI67_z"] = _zscore(df_expr["MKI67"])

    # Unified SANGER_MODEL_ID + COSMIC_ID mapping used identically across all models
    g2 = pd.read_excel(gdsc_fp).copy()
    g2["ModelID"] = g2["SANGER_MODEL_ID"].map(s2m)
    miss = g2["ModelID"].isna() & g2["COSMIC_ID"].notna()
    g2.loc[miss, "ModelID"] = [c2m.get(int(c)) for c in g2.loc[miss, "COSMIC_ID"]]
    g2 = g2[g2["ModelID"].isin(solid_ids)].copy()
    g2["OneMinusAUC"] = 1.0 - g2["AUC"].astype(float)

    rows = []
    for drug, grp in g2.groupby("DRUG_NAME"):
        per_line = grp.groupby("ModelID", as_index=False)[["OneMinusAUC", "LN_IC50"]].mean()
        merged = per_line.merge(df_expr[["ModelID", "Risk_Score", "MKI67_z", "Lineage"]], on="ModelID").dropna()
        if len(merged) < 30 or merged["Lineage"].nunique() < 2:
            continue
        f_auc = smf.ols("OneMinusAUC ~ Risk_Score + C(Lineage)", data=merged).fit()
        a_auc = sm.stats.anova_lm(f_auc, typ=2)
        f_auc_mk = smf.ols("OneMinusAUC ~ Risk_Score + MKI67_z + C(Lineage)", data=merged).fit()
        a_auc_mk = sm.stats.anova_lm(f_auc_mk, typ=2)
        f_ic = smf.ols("LN_IC50 ~ Risk_Score + C(Lineage)", data=merged).fit()
        a_ic = sm.stats.anova_lm(f_ic, typ=2)
        f_ic_mk = smf.ols("LN_IC50 ~ Risk_Score + MKI67_z + C(Lineage)", data=merged).fit()
        a_ic_mk = sm.stats.anova_lm(f_ic_mk, typ=2)
        ss_risk = float(a_auc.loc["Risk_Score", "sum_sq"])
        ss_resid = float(a_auc.loc["Residual", "sum_sq"])
        partial_r2 = ss_risk / (ss_risk + ss_resid) if (ss_risk + ss_resid) > 0 else 0.0
        ss_risk_mk = float(a_auc_mk.loc["Risk_Score", "sum_sq"])
        ss_resid_mk = float(a_auc_mk.loc["Residual", "sum_sq"])
        partial_r2_mk = ss_risk_mk / (ss_risk_mk + ss_resid_mk) if (ss_risk_mk + ss_resid_mk) > 0 else 0.0
        rows.append({
            "Compound_Name": drug,
            "Putative_Target": str(grp["PUTATIVE_TARGET"].dropna().iloc[0]) if grp["PUTATIVE_TARGET"].notna().any() else "Unknown",
            "Pathway_Name": str(grp["PATHWAY_NAME"].dropna().iloc[0]) if grp["PATHWAY_NAME"].notna().any() else "Unknown",
            "Data_Source": "Sanger GDSC2 (Full Screen, 1-AUC Primary)",
            "N_Solid_Lines": int(len(merged)),
            "N_Lineages": int(merged["Lineage"].nunique()),
            "DF_Num": 1,
            "DF_Den": int(a_auc.loc["Residual", "df"]),
            "OLS_Beta_Risk": float(f_auc.params["Risk_Score"]),
            "Type_II_ANOVA_F": float(a_auc.loc["Risk_Score", "F"]),
            "Partial_R2": partial_r2,
            "P_Value": float(a_auc.loc["Risk_Score", "PR(>F)"]),
            "FDR_Q_Value": np.nan,
            "Significant_FDR_5pct": False,
            "Significant_FDR_10pct": False,
            "MKI67_Adj_Beta_Risk": float(f_auc_mk.params["Risk_Score"]),
            "MKI67_Adj_ANOVA_F": float(a_auc_mk.loc["Risk_Score", "F"]),
            "MKI67_Adj_Partial_R2": partial_r2_mk,
            "MKI67_Adj_P_Value": float(a_auc_mk.loc["Risk_Score", "PR(>F)"]),
            "LN_IC50_Beta_Risk": float(f_ic.params["Risk_Score"]),
            "LN_IC50_ANOVA_F": float(a_ic.loc["Risk_Score", "F"]),
            "LN_IC50_P_Value": float(a_ic.loc["Risk_Score", "PR(>F)"]),
            "LN_IC50_FDR_Q_Value": np.nan,
            "LN_IC50_MKI67_Adj_F": float(a_ic_mk.loc["Risk_Score", "F"]),
            "LN_IC50_MKI67_Adj_P_Value": float(a_ic_mk.loc["Risk_Score", "PR(>F)"]),
        })
    df_out = pd.DataFrame(rows)
    df_out["FDR_Q_Value"] = multipletests(df_out["P_Value"].values, method="fdr_bh")[1]
    df_out["Significant_FDR_5pct"] = df_out["FDR_Q_Value"] < 0.05
    df_out["Significant_FDR_10pct"] = df_out["FDR_Q_Value"] < 0.10
    df_out["LN_IC50_FDR_Q_Value"] = multipletests(df_out["LN_IC50_P_Value"].values, method="fdr_bh")[1]
    df_out = df_out.sort_values("P_Value").reset_index(drop=True)
    df_out.to_csv(out_fp, index=False)

    n_fdr05 = int(df_out["Significant_FDR_5pct"].sum())
    n_fdr10 = int(df_out["Significant_FDR_10pct"].sum())
    n_nom = int((df_out["P_Value"] < 0.05).sum())
    n_nom_sens = int(((df_out["P_Value"] < 0.05) & (df_out["OLS_Beta_Risk"] > 0)).sum())
    n_ic_fdr10 = int((df_out["LN_IC50_FDR_Q_Value"] < 0.10).sum())
    pr2_med = float(df_out["Partial_R2"].median())
    pr2_max = float(df_out["Partial_R2"].max())
    pr2_min_fdr10 = float(df_out.loc[df_out["Significant_FDR_10pct"], "Partial_R2"].min())
    pr2_max_fdr10 = float(df_out.loc[df_out["Significant_FDR_10pct"], "Partial_R2"].max())
    print(
        f"[1] Recomputed GDSC2 Type II ANOVA (Primary 1-AUC, {len(df_out)} drugs): "
        f"{n_fdr05} FDR q < 0.05; {n_fdr10} FDR q < 0.10; {n_nom} nominal p < 0.05 ({n_nom_sens} sensitization beta_1-AUC > 0) | "
        f"Partial R^2 (eta_p^2): all drugs median={pr2_med*100:.3f}%, max={pr2_max*100:.2f}%, FDR q<0.10 range={pr2_min_fdr10*100:.2f}%-{pr2_max_fdr10*100:.2f}% | "
        f"ln(IC50) concordance: {n_ic_fdr10} FDR q < 0.10 (min q = {df_out['LN_IC50_FDR_Q_Value'].min():.3f})"
    )

    # Print top 1-AUC hits, pathway controls, and MKI67 adjustment comparison
    for _, rr in df_out.head(14).iterrows():
        print(
            f"    * {rr['Compound_Name']:<13}: 1-AUC F(1,{int(rr['DF_Den'])})={rr['Type_II_ANOVA_F']:.2f}, eta_p^2={rr['Partial_R2']*100:.2f}%, beta={rr['OLS_Beta_Risk']:+.5f}, p={rr['P_Value']:.2e}, q={rr['FDR_Q_Value']:.3f} | "
            f"+MKI67 F={rr['MKI67_Adj_ANOVA_F']:.2f} (eta_p^2={rr['MKI67_Adj_Partial_R2']*100:.2f}%, p={rr['MKI67_Adj_P_Value']:.2e}) | lnIC50 F={rr['LN_IC50_ANOVA_F']:.2f} (q={rr['LN_IC50_FDR_Q_Value']:.3f})"
        )
    for ctrl in ["Niraparib", "Ipatasertib", "Rucaparib", "Camptothecin", "Olaparib", "Talazoparib", "Paclitaxel", "AZD5991", "5-Fluorouracil", "AZD5153", "Docetaxel", "SN-38"]:
        r_c = df_out[df_out["Compound_Name"] == ctrl]
        if len(r_c):
            rr = r_c.iloc[0]
            print(
                f"    - {ctrl:<13}: 1-AUC F={rr['Type_II_ANOVA_F']:.2f} (p={rr['P_Value']:.2e}, q={rr['FDR_Q_Value']:.3f}) | "
                f"+MKI67 F={rr['MKI67_Adj_ANOVA_F']:.2f} (p={rr['MKI67_Adj_P_Value']:.2e}) | lnIC50 F={rr['LN_IC50_ANOVA_F']:.2f} (p={rr['LN_IC50_P_Value']:.2e})"
            )

    print("    Axis-Independent GDSC2 Biomarker Gates (Primary 1-AUC & ln(IC50) Concordance):")
    for drg, gene, cut in [("SN-38", "SLFN11", 1.0), ("Olaparib", "SLFN11", 1.0), ("AZD5991", "BAK1", None), ("AZD5991", "BAX", None)]:
        sub_g = (
            g2[g2["DRUG_NAME"] == drg]
            .groupby("ModelID", as_index=False)[["OneMinusAUC", "LN_IC50"]]
            .mean()
            .merge(df_expr[["ModelID", gene, "Lineage"]], on="ModelID")
            .dropna()
        )
        r_raw, p_raw = stats.pearsonr(sub_g[gene], sub_g["OneMinusAUC"])
        rx = smf.ols(f"{gene} ~ C(Lineage)", data=sub_g).fit().resid
        ry_auc = smf.ols("OneMinusAUC ~ C(Lineage)", data=sub_g).fit().resid
        radj_auc, _ = stats.pearsonr(rx, ry_auc)
        dfd = len(sub_g) - sub_g["Lineage"].nunique() - 1
        padj_auc = float(2.0 * stats.t.sf(abs(radj_auc * np.sqrt(dfd / (1.0 - radj_auc ** 2))), df=dfd))
        cval = cut if cut is not None else float(np.median(sub_g[gene]))
        lo_m = sub_g[gene] < cval
        d_oma = float(sub_g.loc[~lo_m, "OneMinusAUC"].mean() - sub_g.loc[lo_m, "OneMinusAUC"].mean())
        _, pmwu_auc = stats.mannwhitneyu(sub_g.loc[~lo_m, "OneMinusAUC"], sub_g.loc[lo_m, "OneMinusAUC"], alternative="two-sided")
        ry_ic = smf.ols("LN_IC50 ~ C(Lineage)", data=sub_g).fit().resid
        radj_ic, _ = stats.pearsonr(rx, ry_ic)
        padj_ic = float(2.0 * stats.t.sf(abs(radj_ic * np.sqrt(dfd / (1.0 - radj_ic ** 2))), df=dfd))
        print(
            f"    - {drg:<9} vs {gene:<6} (n={len(sub_g)}, df={dfd}): 1-AUC raw r={r_raw:+.4f} (p={p_raw:.2e}), "
            f"adj r={radj_auc:+.4f} (p={padj_auc:.2e}), delta(1-AUC)={d_oma:+.4f} (MWU p={pmwu_auc:.2e}) | "
            f"ln(IC50) adj r={radj_ic:+.4f} (p={padj_ic:.2e})"
        )


def _compute_cohort_gsea_camera(expr_df: pd.DataFrame, cohort_label: str, gmt_sets: dict[str, list[str]], n_perm: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Compute Welch-t pre-ranked GSEA and Wu & Smyth (2012) CAMERA across 50 Hallmark gene sets."""
    sm_v = expr_df.loc["SMARCA4"].values.astype(float)
    hf_v = expr_df.loc["HIF1A"].values.astype(float)
    risk = _unweighted_apriori_score(sm_v, hf_v)
    hi_mask = risk >= np.median(risk)
    n_tot = int(expr_df.shape[1])
    n_hi = int(hi_mask.sum())
    n_lo = int(n_tot - n_hi)

    X = expr_df.values.astype(float)
    x1, x2 = X[:, hi_mask], X[:, ~hi_mask]
    m1, m2 = x1.mean(axis=1), x2.mean(axis=1)
    v1, v2 = x1.var(axis=1, ddof=1), x2.var(axis=1, ddof=1)
    se = np.sqrt(v1 / n_hi + v2 / n_lo)
    valid = se > 0
    X = X[valid]
    m1, m2, se = m1[valid], m2[valid], se[valid]
    genes_all = expr_df.index.values[valid]
    t_vals = (m1 - m2) / se

    order = np.argsort(-t_vals, kind="mergesort")
    genes_sorted = genes_all[order]
    t_sorted = t_vals[order]
    abs_t_sorted = np.abs(t_sorted)
    G = len(genes_sorted)
    gene_to_rank = {g: i for i, g in enumerate(genes_sorted)}
    resid_sorted = np.hstack([X[order][:, hi_mask] - m1[order][:, None], X[order][:, ~hi_mask] - m2[order][:, None]])
    df_cam = n_tot - 2

    rng = np.random.default_rng(seed)
    # Pre-generate random ranks per set size using fast weighted random-walk GSEA
    unique_sizes = sorted({sum(1 for g in gs if g in gene_to_rank) for gs in gmt_sets.values()})
    null_es_by_k: dict[int, np.ndarray] = {}
    for k_sz in unique_sizes:
        if k_sz < 5:
            continue
        null_es = np.empty(n_perm, dtype=float)
        miss_step = -1.0 / (G - k_sz)
        for p_i in range(n_perm):
            idx = rng.choice(G, size=k_sz, replace=False)
            idx.sort()
            w_hits = abs_t_sorted[idx]
            nr = w_hits.sum()
            # Evaluate walk only at hit positions and just before hit positions
            prev_n_miss = idx - np.arange(k_sz)
            cum_hits = np.cumsum(w_hits) / nr
            peak_pos = prev_n_miss * miss_step + cum_hits
            pre_pos = peak_pos - (w_hits / nr)
            cand = np.concatenate([peak_pos, pre_pos])
            null_es[p_i] = float(cand[np.argmax(np.abs(cand))])
        null_es_by_k[k_sz] = null_es

    rows = []
    for hm_full, gs_genes in gmt_sets.items():
        hm_short = hm_full.replace("HALLMARK_", "")
        hit_idx = np.array(sorted([gene_to_rank[g] for g in gs_genes if g in gene_to_rank]), dtype=int)
        K = len(hit_idx)
        w_hits = abs_t_sorted[hit_idx]
        nr = float(w_hits.sum())
        miss_step = -1.0 / (G - K)
        prev_n_miss = hit_idx - np.arange(K)
        cum_hits = np.cumsum(w_hits) / nr
        peak_pos = prev_n_miss * miss_step + cum_hits
        pre_pos = peak_pos - (w_hits / nr)
        cand = np.concatenate([peak_pos, pre_pos])
        max_idx_pos = int(np.argmax(np.abs(peak_pos)))
        if abs(peak_pos[max_idx_pos]) >= abs(pre_pos).max():
            es = float(peak_pos[max_idx_pos])
            peak_rank_idx = int(hit_idx[max_idx_pos])
        else:
            max_idx_pre = int(np.argmax(np.abs(pre_pos)))
            es = float(pre_pos[max_idx_pre])
            peak_rank_idx = int(hit_idx[max_idx_pre])

        if es >= 0:
            le_indices = hit_idx[hit_idx <= peak_rank_idx]
            top_le = [genes_sorted[i] for i in le_indices[:8]]
            direction = "Up in High Risk"
        else:
            le_indices = hit_idx[hit_idx >= peak_rank_idx][::-1]
            top_le = [genes_sorted[i] for i in le_indices[:8]]
            direction = "Down in High Risk"

        null_arr = null_es_by_k[K]
        if es >= 0:
            same = null_arr[null_arr >= 0]
            nes = float(es / same.mean()) if len(same) > 0 and same.mean() > 0 else 0.0
            nom_p = float((np.sum(same >= es) + 1) / (len(same) + 1))
        else:
            same = null_arr[null_arr < 0]
            nes = float(es / abs(same.mean())) if len(same) > 0 and abs(same.mean()) > 0 else 0.0
            nom_p = float((np.sum(same <= es) + 1) / (len(same) + 1))

        # Wu & Smyth (2012) CAMERA on group residuals
        Z_sub = resid_sorted[hit_idx, :]
        corr_m = np.corrcoef(Z_sub)
        iu = np.triu_indices(K, k=1)
        mean_rho = float(np.nanmean(corr_m[iu]))
        vif = max(1e-6, 1.0 + (K - 1) * mean_rho)

        in_mask = np.zeros(G, dtype=bool)
        in_mask[hit_idx] = True
        t_in = t_sorted[in_mask]
        t_out = t_sorted[~in_mask]
        s_in = float(np.var(t_in, ddof=1))
        s_out = float(np.var(t_out, ddof=1))
        sp = float(np.sqrt(((K - 1) * s_in + (G - K - 1) * s_out) / (G - 2)))
        cam_t = float((t_in.mean() - t_out.mean()) / (sp * np.sqrt(vif / K + 1.0 / (G - K))))
        cam_p = float(2.0 * stats.t.sf(abs(cam_t), df=df_cam))

        rows.append({
            "Cohort": cohort_label,
            "N_Patients": n_tot,
            "N_High_Risk": n_hi,
            "N_Low_Risk": n_lo,
            "Transcriptome_Genes": G,
            "Hallmark_Set": hm_full,
            "Hallmark_Short": hm_short,
            "Set_Size_GMT": len(gs_genes),
            "Set_Size_Matched": K,
            "Mean_Welch_t": round(float(t_in.mean()), 4),
            "Enrichment_Score_ES": round(es, 4),
            "Normalized_ES_NES": round(nes, 4),
            "GSEA_Nominal_P": nom_p,
            "CAMERA_Intergene_Corr": round(mean_rho, 4),
            "CAMERA_VIF": round(vif, 4),
            "CAMERA_t_Stat": round(cam_t, 4),
            "CAMERA_P_Value": cam_p,
            "Leading_Edge_Count": int(len(le_indices)),
            "Leading_Edge_Fraction": round(float(len(le_indices) / K), 4),
            "Peak_Rank_Pct": round(float((peak_rank_idx + 1) / G * 100.0), 2),
            "Direction": direction,
            "Top_Leading_Edge_Genes": "; ".join(top_le),
        })

    df_res = pd.DataFrame(rows)
    df_res["GSEA_FDR_q"] = multipletests(df_res["GSEA_Nominal_P"].values, method="fdr_bh")[1]
    df_res["CAMERA_FDR_q"] = multipletests(df_res["CAMERA_P_Value"].values, method="fdr_bh")[1]
    return df_res


def run_hallmark_gsea_and_camera():
    """Recompute MSigDB v2024.1 Hallmark GSEA and CAMERA on TCGA-PAAD (N = 176) and METABRIC (N = 1,980)."""
    gmt_fp = os.path.join(DATA_RAW, "h.all.v2024.1.Hs.symbols.gmt")
    paad_rsem_fp = os.path.join(DATA_RAW, "paad_tcga_pan_can_atlas_2018", "data_mrna_seq_v2_rsem.txt")
    paad_clin_fp = os.path.join(DATA_RAW, "tcga", "TCGA_PAAD_expression_survival.csv")
    mb_fp = os.path.join(DATA_RAW, "cbioportal_high_mortality", "brca_metabric", "data_mrna_illumina_microarray.txt")
    out_fp = os.path.join(DATA_PROC, "hallmark_gsea_transcriptome_wide_enrichment.csv")

    if not all(os.path.exists(p) for p in (gmt_fp, paad_rsem_fp, paad_clin_fp, mb_fp)):
        if os.path.exists(out_fp):
            df_g = pd.read_csv(out_fp)
            paad = df_g[df_g["Cohort"].str.contains("PAAD")].set_index("Hallmark_Short")
            meta = df_g[df_g["Cohort"].str.contains("METABRIC")].set_index("Hallmark_Short")
            common = paad.index.intersection(meta.index)
            rho_g, p_g = stats.spearmanr(paad.loc[common, "Normalized_ES_NES"], meta.loc[common, "Normalized_ES_NES"])
            print(f"[2] Raw RNA archives absent; cached Hallmark GSEA ({len(common)} sets): PAAD vs METABRIC rho = {rho_g:+.4f}, p = {p_g:.3e}")
        return

    gmt_sets: dict[str, list[str]] = {}
    with open(gmt_fp, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gmt_sets[parts[0]] = parts[2:]

    # 1. Real N = 176 primary -01 TCGA-PAAD PanCanAtlas RNA-seq log2(RSEM + 1)
    df_paad_clin = pd.read_csv(paad_clin_fp)
    id_col = "Sample_ID" if "Sample_ID" in df_paad_clin.columns else "Sample_Barcode"
    paad_ids = [s for s in df_paad_clin[id_col].astype(str) if s.endswith("-01")]
    paad_raw = pd.read_csv(paad_rsem_fp, sep="\t").dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
    keep_paad = [c for c in paad_ids if c in paad_raw.columns]
    paad_expr = np.log2(paad_raw[keep_paad].apply(pd.to_numeric, errors="coerce").dropna() + 1.0)
    res_paad = _compute_cohort_gsea_camera(paad_expr, f"TCGA-PAAD (Primary -01 PanCanAtlas RNA-seq N={len(keep_paad)})", gmt_sets)

    # 2. METABRIC (N = 1,980 microarray)
    mb_raw = pd.read_csv(mb_fp, sep="\t").dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol").iloc[:, 1:]
    mb_expr = mb_raw.apply(pd.to_numeric, errors="coerce").dropna()
    res_mb = _compute_cohort_gsea_camera(mb_expr, f"METABRIC (Microarray N={mb_expr.shape[1]})", gmt_sets)

    # Cross-cohort concordance and CAMERA Stouffer Z/P
    m_p = res_paad.set_index("Hallmark_Set")
    m_m = res_mb.set_index("Hallmark_Set")
    common_hm = m_p.index
    st_z = []
    st_p = []
    same_sign = []
    for hm in common_hm:
        nes_p = float(m_p.loc[hm, "Normalized_ES_NES"])
        nes_m = float(m_m.loc[hm, "Normalized_ES_NES"])
        same_sign.append((nes_p >= 0) == (nes_m >= 0))
        zp = float(stats.norm.isf(max(float(m_p.loc[hm, "CAMERA_P_Value"]) / 2.0, 1e-300))) * (1.0 if float(m_p.loc[hm, "CAMERA_t_Stat"]) >= 0 else -1.0)
        zm = float(stats.norm.isf(max(float(m_m.loc[hm, "CAMERA_P_Value"]) / 2.0, 1e-300))) * (1.0 if float(m_m.loc[hm, "CAMERA_t_Stat"]) >= 0 else -1.0)
        z_comb = (zp + zm) / np.sqrt(2.0)
        st_z.append(round(float(z_comb), 4))
        st_p.append(float(2.0 * stats.norm.sf(abs(z_comb))))

    cross_map = pd.DataFrame({"Cross_Cohort_Same_Sign": same_sign, "Cross_Cohort_Stouffer_Z": st_z, "Cross_Cohort_Stouffer_P": st_p}, index=common_hm)
    res_paad = res_paad.join(cross_map, on="Hallmark_Set").sort_values("Normalized_ES_NES", ascending=False)
    res_mb = res_mb.join(cross_map, on="Hallmark_Set").sort_values("Normalized_ES_NES", ascending=False)
    df_all = pd.concat([res_paad, res_mb], ignore_index=True)
    df_all.to_csv(out_fp, index=False)

    rho_g, p_g = stats.spearmanr(m_p["Normalized_ES_NES"], m_m.loc[common_hm, "Normalized_ES_NES"])
    print(
        f"[2] Recomputed MSigDB v2024.1 Hallmark GSEA & CAMERA (Z(SMARCA4_z+HIF1A_z)): "
        f"PAAD N={len(keep_paad)} ({(res_paad['GSEA_FDR_q']<0.05).sum()}/50 GSEA q<0.05; {(res_paad['CAMERA_FDR_q']<0.05).sum()}/50 CAMERA q<0.05), "
        f"METABRIC N={mb_expr.shape[1]} ({(res_mb['GSEA_FDR_q']<0.05).sum()}/50 GSEA q<0.05; {(res_mb['CAMERA_FDR_q']<0.05).sum()}/50 CAMERA q<0.05); "
        f"NES rho={rho_g:+.4f}, p={p_g:.3e}"
    )


def run_tcpa_rppa_proteomics():
    """Recompute within-cohort 12-cohort TCPA Level-4 RPPA validation using Z(SMARCA4_z + HIF1A_z)."""
    out_fp = os.path.join(DATA_PROC, "tcga_rppa_proteomic_validation.csv")
    pancan_12 = [
        ("BRCA",     "brca",                                    "brca"),
        ("CESC",     "cesc",                                    "cesc/cesc_tcga_pan_can_atlas_2018"),
        ("COADREAD", "coadread",                                "coadread"),
        ("ESCA",     "esca",                                    "esca/esca_tcga_pan_can_atlas_2018"),
        ("HNSC",     "hnsc",                                    "hnsc/hnsc_tcga_pan_can_atlas_2018"),
        ("LUAD",     "luad",                                    "luad"),
        ("LUSC",     "lusc",                                    "lusc"),
        ("PRAD",     "prad",                                    "prad"),
        ("SARC",     "sarc",                                    "sarc/sarc_tcga_pan_can_atlas_2018"),
        ("STAD",     "stad",                                    "stad/stad_tcga_pan_can_atlas_2018"),
        ("THCA",     "thca",                                    "thca"),
        ("UCEC",     "ucec",                                    "ucec"),
    ]
    pancan_dir = os.path.join(DATA_RAW, "cbioportal_pancan")
    if not os.path.exists(os.path.join(pancan_dir, "brca", "data_rppa.txt")):
        if os.path.exists(out_fp):
            df_r = pd.read_csv(out_fp)
            print(f"[3] Raw RPPA files absent; cached TCPA RPPA: {(df_r['Spearman_FDR_q']<0.05).sum()}/{len(df_r)} FDR q < 0.05")
        return

    ab_cohort_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    ab_n_cohorts: dict[str, int] = {}
    for cohort, m_sub, r_sub in pancan_12:
        m_fp = os.path.join(pancan_dir, m_sub, "data_mrna_seq_v2_rsem.txt")
        r_fp = os.path.join(pancan_dir, r_sub, "data_rppa.txt")
        m_df = pd.read_csv(m_fp, sep="\t")
        m_two = m_df[m_df["Hugo_Symbol"].isin(["SMARCA4", "HIF1A"])].set_index("Hugo_Symbol").iloc[:, 1:].T.apply(pd.to_numeric, errors="coerce").dropna()
        r_df = pd.read_csv(r_fp, sep="\t", index_col=0)
        ov = [c for c in m_two.index if c in r_df.columns]
        m_ov = np.log2(m_two.loc[ov] + 1.0)
        risk_z = _unweighted_apriori_score(m_ov["SMARCA4"].values, m_ov["HIF1A"].values)
        r_ov = r_df[ov].apply(pd.to_numeric, errors="coerce")
        for ab, row_vals in r_ov.iterrows():
            vals = row_vals.values.astype(float)
            ok = ~np.isnan(vals)
            if ok.sum() < 10 or np.nanstd(vals[ok], ddof=1) == 0:
                continue
            vals_z = (vals[ok] - np.mean(vals[ok])) / np.std(vals[ok], ddof=1)
            ab_cohort_pairs.setdefault(str(ab), []).append((risk_z[ok], vals_z))
            ab_n_cohorts[str(ab)] = ab_n_cohorts.get(str(ab), 0) + 1

    rows = []
    for ab, pairs in ab_cohort_pairs.items():
        if ab_n_cohorts[ab] < 7:
            continue
        all_r = np.concatenate([p[0] for p in pairs])
        all_y = np.concatenate([p[1] for p in pairs])
        rho, pval = stats.spearmanr(all_r, all_y)
        target = ab.split("|", 1)[1] if "|" in ab else ab
        rows.append({
            "Antibody_REF": ab,
            "Protein_Target": target,
            "N_Patients": int(len(all_r)),
            "N_Cohorts": int(ab_n_cohorts[ab]),
            "Within_Cohort_Spearman_rho": round(float(rho), 4),
            "Spearman_P_Value": float(pval),
        })
    df_out = pd.DataFrame(rows).sort_values("Spearman_P_Value").reset_index(drop=True)
    df_out["Spearman_FDR_q"] = multipletests(df_out["Spearman_P_Value"].values, method="fdr_bh")[1]
    df_out.to_csv(out_fp, index=False)

    n_sig = int((df_out["Spearman_FDR_q"] < 0.05).sum())
    max_n = int(df_out["N_Patients"].max())
    n_full = int((df_out["N_Patients"] == max_n).sum())
    print(f"[3] Recomputed 12-Cohort TCPA Level-4 RPPA (Z(SMARCA4_z+HIF1A_z), up to N={max_n:,}): {n_sig}/{len(df_out)} antibodies FDR q < 0.05 ({n_full}/{len(df_out)} complete)")
    pos_top = df_out[df_out["Within_Cohort_Spearman_rho"] > 0].head(5)
    neg_top = df_out[df_out["Within_Cohort_Spearman_rho"] < 0].head(6)
    for _, r in pd.concat([pos_top, neg_top]).iterrows():
        print(f"    * {r['Protein_Target']:<14}: rho = {r['Within_Cohort_Spearman_rho']:+.4f}, p = {r['Spearman_P_Value']:.2e}, q = {r['Spearman_FDR_q']:.2e}")


def run_cin_orthogonality():
    """Recompute Knijnenburg HRD and Aneuploidy chromosomal instability orthogonality using Z(SMARCA4_z + HIF1A_z)."""
    out_fp = os.path.join(DATA_PROC, "genomic_scars_hrd_aneuploidy_orthogonality.csv")
    hrd_fp = os.path.join(DATA_RAW, "TCGA.HRD_withSampleID.txt")
    f1_fp = os.path.join(DATA_PROC, "Full_TCGA_PanCan_Patient_Treatment_Registry.csv")
    f2_fp = os.path.join(DATA_PROC, "Batch_Indication_Expansion_Patient_Registry.csv")
    f3_fp = os.path.join(DATA_PROC, "High_Event_10Cohort_Treatment_Registry.csv")

    if not os.path.exists(hrd_fp):
        if os.path.exists(out_fp):
            df_c = pd.read_csv(out_fp)
            dr2_7 = float(df_c.loc[(df_c["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_c["Genomic_Scar_Metric"] == "HRD_Sum"), "OLS_Incremental_Delta_R2"].iloc[0])
            dr2_17 = float(df_c.loc[(df_c["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_c["Genomic_Scar_Metric"] == "HRD_Sum"), "OLS_Incremental_Delta_R2"].iloc[0])
            print(f"[4] Raw HRD archive absent; cached CIN: 7-cohort dR^2={dr2_7*100:.2f}%, 17-cohort dR^2={dr2_17*100:.2f}%")
        return

    hrd_raw = pd.read_csv(hrd_fp, sep="\t", index_col=0).T
    hrd_raw = hrd_raw.apply(pd.to_numeric, errors="coerce").dropna().rename(
        columns={"HRD": "HRD_Sum", "hrd-loh": "HRD_LOH", "ai1": "HRD_TAI", "lst1": "HRD_LST"}
    )

    # Map Aneuploidy scores from 12 pancan clinical sample files
    pancan_dirs = {
        "BRCA": "brca", "COADREAD": "coadread", "LUAD": "luad", "LUSC": "lusc",
        "PRAD": "prad", "THCA": "thca", "UCEC": "ucec",
        "CESC": "cesc/cesc_tcga_pan_can_atlas_2018",
        "ESCA": "esca/esca_tcga_pan_can_atlas_2018",
        "HNSC": "hnsc/hnsc_tcga_pan_can_atlas_2018",
        "SARC": "sarc/sarc_tcga_pan_can_atlas_2018",
        "STAD": "stad/stad_tcga_pan_can_atlas_2018",
    }
    aneu_map = {}
    for _, sub_d in pancan_dirs.items():
        clin = pd.read_csv(os.path.join(DATA_RAW, "cbioportal_pancan", sub_d, "data_clinical_sample.txt"), sep="\t", comment="#")
        sub_c = clin[["SAMPLE_ID", "ANEUPLOIDY_SCORE"]].dropna()
        aneu_map.update(dict(zip(sub_c["SAMPLE_ID"].astype(str), sub_c["ANEUPLOIDY_SCORE"].astype(float))))

    # Build 7-cohort primary registry and 17/18-cohort extended TCGA registry using within-cohort Z(SMARCA4_z + HIF1A_z)
    f1 = pd.read_csv(f1_fp, usecols=["Sample_ID", "Cohort", "SMARCA4_zScore", "HIF1A_zScore"])
    f2 = pd.read_csv(f2_fp, usecols=["Sample_ID", "Cohort", "SMARCA4_zScore", "HIF1A_zScore"])
    f3 = pd.read_csv(f3_fp, usecols=["Sample_ID", "Cohort", "SMARCA4_zScore", "HIF1A_zScore"])
    f3 = f3[f3["Cohort"].str.endswith("_TCGA")].copy()
    f3["Cohort"] = f3["Cohort"].str.replace("_TCGA", "", regex=False)

    df_7 = f1.copy()
    df_7["Risk_Score"] = df_7.groupby("Cohort", group_keys=False).apply(
        lambda g: pd.Series(_unweighted_apriori_score(g["SMARCA4_zScore"].values, g["HIF1A_zScore"].values), index=g.index)
    )
    df_17 = pd.concat([f1, f2, f3], ignore_index=True).drop_duplicates("Sample_ID").copy()
    df_17["Risk_Score"] = df_17.groupby("Cohort", group_keys=False).apply(
        lambda g: pd.Series(_unweighted_apriori_score(g["SMARCA4_zScore"].values, g["HIF1A_zScore"].values), index=g.index)
    )

    for frame in (df_7, df_17):
        for col in ["HRD_Sum", "HRD_LOH", "HRD_TAI", "HRD_LST"]:
            frame[col] = frame["Sample_ID"].map(hrd_raw[col])
        frame["Aneuploidy_Score"] = frame["Sample_ID"].map(aneu_map)

    scars = ["HRD_Sum", "HRD_LOH", "HRD_TAI", "HRD_LST", "Aneuploidy_Score"]
    rows = []

    def _eval_slice(sub_df: pd.DataFrame, scope: str, cohort_nm: str, pooled: bool = False):
        # For joint OLS incremental dR^2 across multi-cohort pooled scopes: Risk_Score vs cohort FE
        if pooled:
            joint_cols = ["Cohort", "Risk_Score", "HRD_Sum", "Aneuploidy_Score"]
            j_df = sub_df.dropna(subset=joint_cols).copy()
            r2_fe = float(smf.ols("Risk_Score ~ C(Cohort)", data=j_df).fit().rsquared)
            r2_full = float(smf.ols("Risk_Score ~ C(Cohort) + HRD_Sum + Aneuploidy_Score", data=j_df).fit().rsquared)
            inc_dr2 = round(r2_full - r2_fe, 4)
        else:
            r2_full, inc_dr2 = np.nan, np.nan

        for sc in scars:
            d = sub_df[["Cohort", "Risk_Score", sc]].dropna().copy()
            hi_m = d["Risk_Score"] >= d["Risk_Score"].median()
            rho, sp_p = stats.spearmanr(d["Risk_Score"], d[sc])
            r_p, pe_p = stats.pearsonr(d["Risk_Score"], d[sc])
            _, mwu_p = stats.mannwhitneyu(d.loc[hi_m, sc], d.loc[~hi_m, sc], alternative="two-sided")
            rows.append({
                "Analysis_Scope": scope,
                "Cohort": cohort_nm,
                "Genomic_Scar_Metric": sc,
                "N_Patients": int(len(d)),
                "Mean_Scar": round(float(d[sc].mean()), 3),
                "Median_Scar": round(float(d[sc].median()), 1),
                "High_Risk_Median_Scar": round(float(d.loc[hi_m, sc].median()), 1),
                "Low_Risk_Median_Scar": round(float(d.loc[~hi_m, sc].median()), 1),
                "Spearman_Rho": round(float(rho), 4),
                "Spearman_P_Value": float(sp_p),
                "Pearson_R": round(float(r_p), 4),
                "Pearson_P_Value": float(pe_p),
                "Pairwise_R2": round(float(r_p ** 2), 6),
                "MWU_P_Value": float(mwu_p),
                "OLS_Model_R2": round(r2_full, 4) if pooled else np.nan,
                "OLS_Incremental_Delta_R2": inc_dr2 if pooled else np.nan,
                "Orthogonal_Criteria_Pass": bool((abs(rho) < 0.15) if not pooled else (inc_dr2 < 0.05)),
            })

    _eval_slice(df_7, "PanCancer_Pooled_Registry", "Pan-Cancer Pooled (7 Cohorts)", pooled=True)
    for c_nm in ["BRCA", "COADREAD", "LUAD", "LUSC", "PRAD", "THCA", "UCEC"]:
        _eval_slice(df_7[df_7["Cohort"] == c_nm], "Cohort_Stratified", c_nm, pooled=False)
    _eval_slice(df_17, "Expanded_TCGA_Sensitivity", "Pan-Cancer Extended TCGA (17 Cohorts)", pooled=True)

    df_out = pd.DataFrame(rows)
    df_out["Spearman_FDR_q"] = multipletests(df_out["Spearman_P_Value"].values, method="fdr_bh")[1]
    df_out.to_csv(out_fp, index=False)

    r7 = df_out[(df_out["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_out["Genomic_Scar_Metric"] == "HRD_Sum")].iloc[0]
    r7a = df_out[(df_out["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_out["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]
    r17 = df_out[(df_out["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_out["Genomic_Scar_Metric"] == "HRD_Sum")].iloc[0]
    r17a = df_out[(df_out["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_out["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]
    rb = df_out[(df_out["Cohort"] == "BRCA") & (df_out["Genomic_Scar_Metric"] == "HRD_Sum")].iloc[0]
    rba = df_out[(df_out["Cohort"] == "BRCA") & (df_out["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]
    print(
        f"[4] Recomputed CIN Orthogonality (Z(SMARCA4_z+HIF1A_z)): "
        f"7-cohort (HRD N={int(r7['N_Patients'])}, Aneu N={int(r7a['N_Patients'])}) partial dR^2={r7['OLS_Incremental_Delta_R2']*100:.2f}%, "
        f"17-cohort (HRD N={int(r17['N_Patients'])}, Aneu N={int(r17a['N_Patients'])}) partial dR^2={r17['OLS_Incremental_Delta_R2']*100:.2f}%; "
        f"BRCA HRD rho={rb['Spearman_Rho']:+.4f} (p={rb['Spearman_P_Value']:.2e}), Aneu rho={rba['Spearman_Rho']:+.4f}"
    )


def run_imvigor210_survival():
    """Evaluate IMvigor210 anti-PD-L1 OS orthogonality using Z(SMARCA4_z + HIF1A_z)."""
    imv_raw = os.path.join(DATA_RAW, "external", "IMvigor210_blca_cached.csv")
    if os.path.exists(imv_raw):
        imv = pd.read_csv(imv_raw)
        imv["Event"] = imv["OS_STATUS"].astype(str).str.startswith("1:").astype(int)
        sub = imv[["OS_MONTHS", "Event", "SMARCA4_zScore", "HIF1A_zScore"]].dropna().copy()
        sub = sub[sub["OS_MONTHS"] > 0].copy()
        sub["Risk_Z"] = _unweighted_apriori_score(sub["SMARCA4_zScore"], sub["HIF1A_zScore"])
        cph = CoxPHFitter(penalizer=0.0).fit(sub[["OS_MONTHS", "Event", "Risk_Z"]], duration_col="OS_MONTHS", event_col="Event")
        s = cph.summary.loc["Risk_Z"]
        print(
            f"[5] IMvigor210 Anti-PD-L1 OS (Z(SMARCA4_z+HIF1A_z), N = {len(sub)}, {int(sub['Event'].sum())} events): "
            f"HR = {s['exp(coef)']:.3f} [{s['exp(coef) lower 95%']:.3f}-{s['exp(coef) upper 95%']:.3f}], p = {s['p']:.3f}"
        )


def main():
    print("GDSC2 MONOTHERAPY ANOVA, HALLMARK GSEA & CAMERA, TCPA RPPA, CIN ORTHOGONALITY & IMVIGOR210 (UNWEIGHTED SCORE)")
    run_gdsc2_anova()
    run_hallmark_gsea_and_camera()
    run_tcpa_rppa_proteomics()
    run_cin_orthogonality()
    run_imvigor210_survival()


if __name__ == "__main__":
    main()
