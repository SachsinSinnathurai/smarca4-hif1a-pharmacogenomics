"""
Step 2: Rebuilt DepMap 24Q2, TCGA 3-cohort lineage-matched expression,
continuous 1,038-line SMARCA4 co-essentiality scan, MKI67-controlled diploid test,
VHL-LoF HIF1A vs. EPAS1 analysis, and cleaned GEO hypoxia table.
Writes exclusively inside results/.
"""

from __future__ import annotations

import json
import math
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def fisher_r_to_z(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    z1 = np.arctanh(np.clip(r1, -0.999999, 0.999999))
    z2 = np.arctanh(np.clip(r2, -0.999999, 0.999999))
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = float((z1 - z2) / se)
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    return z_stat, p_val


def load_rsem_pair(fp: str, primary_only: bool = False) -> pd.DataFrame:
    out = {"HIF1A": {}, "SLC7A1": {}}
    with open(fp, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split("\t")[2:]
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in out:
                for s, v in zip(hdr, parts[2:]):
                    if primary_only and (len(s) < 15 or s[13:15] != "01"):
                        continue
                    try:
                        out[parts[0]][s] = float(np.log2(float(v) + 1.0))
                    except Exception:
                        pass
    d = pd.DataFrame(out)
    d["PID"] = [s[:12] for s in d.index]
    return d


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = repo_root
    data_raw = os.path.join(project_root, "data", "raw")
    data_proc = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(repo_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    sum_fp = os.path.join(out_dir, "depmap_and_hypoxia_summary.json")
    has_raw_depmap = os.path.exists(os.path.join(data_raw, "CRISPRGeneEffect.csv"))
    if has_raw_depmap:
        # 1. Load DepMap Model annotations
        df_m = pd.read_csv(os.path.join(data_raw, "Model.csv"))
        doubling_cols = [c for c in df_m.columns if "doubl" in c.lower()]

        non_solid = ["Lymphoid", "Myeloid", "Normal", "Fibroblast"]
        norm_ids = set(
            df_m[
                (df_m["OncotreePrimaryDisease"] == "Non-Cancerous")
                | (df_m["OncotreeLineage"].isin(["Normal", "Fibroblast"]))
            ]["ModelID"].dropna().unique()
        )
        solid_ids = set(
            df_m[
                (~df_m["OncotreeLineage"].isin(non_solid))
                & (~df_m["ModelID"].isin(norm_ids))
            ]["ModelID"].dropna().unique()
        )
        lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()
        sub_map = df_m.drop_duplicates("ModelID").set_index("ModelID")[["OncotreeLineage", "OncotreeSubtype"]]

        # 2. Load raw DepMap CRISPRGeneEffect
        crispr_fp = os.path.join(data_raw, "CRISPRGeneEffect.csv")
        df_c = pd.read_csv(crispr_fp)
        df_c = df_c.rename(columns={df_c.columns[0]: "ModelID"})
        gene_cols = [c for c in df_c.columns if c != "ModelID"]
        df_c = df_c.rename(columns={c: c.split(" ")[0].strip("()") for c in gene_cols})

        core_cols = ["SMARCA4", "HIF1A", "SLC1A5", "SLC7A1", "ABCB1", "EPAS1"]
        df_solid = df_c[df_c["ModelID"].isin(solid_ids)].dropna(subset=core_cols).copy()
        r_15_71, p_15_71 = stats.pearsonr(df_solid["SLC1A5"], df_solid["SLC7A1"])
        rho_15_71, _ = stats.spearmanr(df_solid["SLC1A5"], df_solid["SLC7A1"])

        # Paralog controls
        paralog_res = {}
        for g1, g2 in [("SLC7A1", "SLC7A5"), ("SLC7A1", "SLC7A2"), ("SLC1A5", "SLC1A4")]:
            sub_all = df_c[[g1, g2]].dropna()
            sub_sol = df_c[df_c["ModelID"].isin(solid_ids)][[g1, g2]].dropna()
            ra, pa = stats.pearsonr(sub_all[g1], sub_all[g2])
            rs, ps = stats.pearsonr(sub_sol[g1], sub_sol[g2])
            paralog_res[f"{g1}_vs_{g2}"] = {
                "pancancer_n": int(len(sub_all)), "pancancer_r": float(ra), "pancancer_p": float(pa),
                "solid_n": int(len(sub_sol)), "solid_r": float(rs), "solid_p": float(ps),
            }

        # 3. Continuous genome-wide SMARCA4 co-essentiality across ALL solid lines (N = 1,038 / solid_ids)
        # Using df_solid (n=1,038) cleanly without extreme-group truncation
        bg_genes = [c for c in df_solid.columns if c not in ("ModelID", "SMARCA4")]
        sm_vec = df_solid["SMARCA4"].to_numpy(dtype=float)
        mat_bg = df_solid[bg_genes].to_numpy(dtype=float)

        # Identify columns without NaNs vs with NaNs for exact vectorized correlation
        no_nan_mask = ~np.isnan(mat_bg).any(axis=0)
        r_arr = np.empty(len(bg_genes), dtype=float)
        p_pearson = np.empty(len(bg_genes), dtype=float)
        rho_arr = np.empty(len(bg_genes), dtype=float)
        p_spearman = np.empty(len(bg_genes), dtype=float)
        n_obs_arr = np.full(len(bg_genes), len(df_solid), dtype=int)

        # Fast exact vectorized block for complete columns
        n_s = len(sm_vec)
        sm_z = (sm_vec - sm_vec.mean()) / sm_vec.std(ddof=0)
        sm_rk = stats.rankdata(sm_vec)
        sm_rk_z = (sm_rk - sm_rk.mean()) / sm_rk.std(ddof=0)

        M_clean = mat_bg[:, no_nan_mask]
        M_z = (M_clean - M_clean.mean(axis=0)) / M_clean.std(axis=0, ddof=0)
        r_clean = (M_z.T @ sm_z) / n_s
        r_clean = np.clip(r_clean, -0.99999999, 0.99999999)
        t_clean = r_clean * np.sqrt((n_s - 2) / (1.0 - r_clean ** 2))
        p_clean = 2.0 * stats.t.sf(np.abs(t_clean), df=n_s - 2)

        M_rk = np.apply_along_axis(stats.rankdata, 0, M_clean)
        M_rk_z = (M_rk - M_rk.mean(axis=0)) / M_rk.std(axis=0, ddof=0)
        rho_clean = (M_rk_z.T @ sm_rk_z) / n_s
        rho_clean = np.clip(rho_clean, -0.99999999, 0.99999999)
        ts_clean = rho_clean * np.sqrt((n_s - 2) / (1.0 - rho_clean ** 2))
        ps_clean = 2.0 * stats.t.sf(np.abs(ts_clean), df=n_s - 2)

        r_arr[no_nan_mask] = r_clean
        p_pearson[no_nan_mask] = p_clean
        rho_arr[no_nan_mask] = rho_clean
        p_spearman[no_nan_mask] = ps_clean

        # Pairwise complete fallback for any columns with NaNs
        for j in np.where(~no_nan_mask)[0]:
            col = mat_bg[:, j]
            ok = ~np.isnan(col)
            n_ok = int(ok.sum())
            n_obs_arr[j] = n_ok
            if n_ok >= 10:
                r_v, p_v = stats.pearsonr(sm_vec[ok], col[ok])
                rho_v, ps_v = stats.spearmanr(sm_vec[ok], col[ok])
                r_arr[j], p_pearson[j] = float(r_v), float(p_v)
                rho_arr[j], p_spearman[j] = float(rho_v), float(ps_v)
            else:
                r_arr[j], p_pearson[j] = np.nan, 1.0
                rho_arr[j], p_spearman[j] = np.nan, 1.0

        valid_p = ~np.isnan(p_pearson)
        q_pearson = np.ones(len(bg_genes), dtype=float)
        q_pearson[valid_p] = multipletests(p_pearson[valid_p], method="fdr_bh")[1]
        q_spearman = np.ones(len(bg_genes), dtype=float)
        q_spearman[valid_p] = multipletests(p_spearman[valid_p], method="fdr_bh")[1]

        df_cont = pd.DataFrame({
            "Gene": bg_genes,
            "N_Solid_Lines": n_obs_arr,
            "Pearson_r": r_arr,
            "Pearson_P": p_pearson,
            "Pearson_FDR_q": q_pearson,
            "Spearman_rho": rho_arr,
            "Spearman_P": p_spearman,
            "Spearman_FDR_q": q_spearman,
        }).sort_values("Pearson_P").reset_index(drop=True)
        df_cont.to_csv(os.path.join(out_dir, "smarca4_continuous_coessentiality_all_solid.csv"), index=False)

        baf_partners = {}
        df_cont_idx = df_cont.set_index("Gene")
        for g in ["SMARCB1", "SMARCC1", "SMARCD1", "ARID1A", "SMARCE1", "MLLT1", "SMARCA2"]:
            row = df_cont_idx.loc[g]
            baf_partners[g] = {
                "n": int(row["N_Solid_Lines"]),
                "pearson_r": float(row["Pearson_r"]),
                "pearson_p": float(row["Pearson_P"]),
                "pearson_fdr_q": float(row["Pearson_FDR_q"]),
                "spearman_rho": float(row["Spearman_rho"]),
                "spearman_p": float(row["Spearman_P"]),
                "spearman_fdr_q": float(row["Spearman_FDR_q"]),
            }

        # 4. Diploid vs solid SMARCA4 test (unadjusted MWU + MKI67-controlled OLS)
        expr_fp = os.path.join(data_raw, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
        df_eh = pd.read_csv(expr_fp, nrows=1)
        e_cols = ["ModelID", "IsDefaultEntryForModel"]
        e_ren = {}
        for g in ["HIF1A", "SLC7A1", "MKI67", "SMARCA4"]:
            m = [c for c in df_eh.columns if c.split(" ")[0].strip("()") == g]
            e_cols.append(m[0])
            e_ren[m[0]] = g
        df_expr_all = pd.read_csv(expr_fp, usecols=e_cols).rename(columns=e_ren).dropna(subset=["HIF1A", "SLC7A1"])
        df_expr_u = df_expr_all[df_expr_all["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").copy()

        df_norm = df_c[df_c["ModelID"].isin(norm_ids)].dropna(subset=["SMARCA4", "SLC1A5", "SLC7A1", "HIF1A"]).copy()
        u_stat, p_mwu_diploid = stats.mannwhitneyu(df_norm["SMARCA4"], df_solid["SMARCA4"], alternative="two-sided")

        # Merge MKI67 mRNA expression as proliferation proxy onto diploid + solid CRISPR lines
        df_dip_sol = pd.concat([
            df_norm[["ModelID", "SMARCA4"]].assign(Is_Diploid=1),
            df_solid[["ModelID", "SMARCA4"]].assign(Is_Diploid=0),
        ], ignore_index=True).merge(
            df_expr_u[["ModelID", "MKI67"]].dropna(), on="ModelID", how="inner"
        )
        X_unadj = sm.add_constant(df_dip_sol[["Is_Diploid"]])
        m_unadj = sm.OLS(df_dip_sol["SMARCA4"], X_unadj).fit()
        X_mk = sm.add_constant(df_dip_sol[["Is_Diploid", "MKI67"]])
        m_mk = sm.OLS(df_dip_sol["SMARCA4"], X_mk).fit()

        diploid_res = {
            "n_diploid_crispr": int(len(df_norm)),
            "n_solid_crispr": int(len(df_solid)),
            "diploid_mean_smarca4": float(df_norm["SMARCA4"].mean()),
            "diploid_median_smarca4": float(df_norm["SMARCA4"].median()),
            "solid_mean_smarca4": float(df_solid["SMARCA4"].mean()),
            "solid_median_smarca4": float(df_solid["SMARCA4"].median()),
            "delta_mean": float(df_norm["SMARCA4"].mean() - df_solid["SMARCA4"].mean()),
            "delta_median": float(df_norm["SMARCA4"].median() - df_solid["SMARCA4"].median()),
            "mwu_p": float(p_mwu_diploid),
            "doubling_time_columns_in_model_csv": doubling_cols,
            "mki67_matched_n_diploid": int((df_dip_sol["Is_Diploid"] == 1).sum()),
            "mki67_matched_n_solid": int((df_dip_sol["Is_Diploid"] == 0).sum()),
            "ols_unadj_beta_diploid": float(m_unadj.params["Is_Diploid"]),
            "ols_unadj_p_diploid": float(m_unadj.pvalues["Is_Diploid"]),
            "ols_mki67_beta_diploid": float(m_mk.params["Is_Diploid"]),
            "ols_mki67_p_diploid": float(m_mk.pvalues["Is_Diploid"]),
            "ols_mki67_beta_mki67": float(m_mk.params["MKI67"]),
            "ols_mki67_p_mki67": float(m_mk.pvalues["MKI67"]),
        }

        # 5. Fig 1D Lineage-Matched TCGA vs DepMap HIF1A-SLC7A1 Expression Comparison
        df_paad_t = pd.read_csv(os.path.join(data_raw, "tcga", "TCGA_PAAD_expression_survival.csv"))
        he_reg = pd.read_csv(os.path.join(data_proc, "High_Event_10Cohort_Treatment_Registry.csv"))
        pc_reg = pd.read_csv(os.path.join(data_proc, "Full_TCGA_PanCan_Patient_Treatment_Registry.csv"))
        df_gbm_158 = he_reg[he_reg["Cohort"] == "GBM_TCGA"].dropna(subset=["OS_Days", "Event", "Age"])
        df_gbm_158 = df_gbm_158[df_gbm_158["OS_Days"] > 0]
        df_brca_1050 = pc_reg[pc_reg["Cohort"] == "BRCA"].dropna(subset=["OS_Days", "Event", "Age", "Stage_Numeric"])
        df_brca_1050 = df_brca_1050[df_brca_1050["OS_Days"] > 0]

        paad_rsem = load_rsem_pair(os.path.join(data_raw, "cbioportal_pancan", "paad", "data_mrna_seq_v2_rsem.txt"), primary_only=True).drop_duplicates("PID").set_index("PID")
        gbm_rsem = load_rsem_pair(os.path.join(data_raw, "cbioportal_high_mortality", "gbm_tcga", "data_mrna_seq_v2_rsem.txt"), primary_only=False)
        brca_rsem = load_rsem_pair(os.path.join(data_raw, "cbioportal_pancan", "brca", "data_mrna_seq_v2_rsem.txt"), primary_only=False)

        paad_rows = [
            {"Cohort": "TCGA-PAAD", "HIF1A": float(paad_rsem.loc[bc, "HIF1A"]), "SLC7A1": float(paad_rsem.loc[bc, "SLC7A1"])}
            for bc in df_paad_t["Sample_Barcode"] if bc in paad_rsem.index
        ]
        gbm_sub = gbm_rsem.loc[df_gbm_158["Sample_ID"]][["HIF1A", "SLC7A1"]].assign(Cohort="TCGA-GBM")
        brca_sub = brca_rsem.loc[df_brca_1050["Sample_ID"]][["HIF1A", "SLC7A1"]].assign(Cohort="TCGA-BRCA")
        df_tcga = pd.concat([pd.DataFrame(paad_rows), gbm_sub, brca_sub], ignore_index=True)
        df_tcga["H_z"] = df_tcga.groupby("Cohort")["HIF1A"].transform(lambda s: (s - s.mean()) / s.std())
        df_tcga["S_z"] = df_tcga.groupby("Cohort")["SLC7A1"].transform(lambda s: (s - s.mean()) / s.std())
        r_tcga_raw, p_tcga_raw = stats.pearsonr(df_tcga["HIF1A"], df_tcga["SLC7A1"])
        r_tcga_z, p_tcga_z = stats.pearsonr(df_tcga["H_z"], df_tcga["S_z"])

        tcga_per_cohort = {}
        for c_name, grp in df_tcga.groupby("Cohort"):
            rc, pc = stats.pearsonr(grp["HIF1A"], grp["SLC7A1"])
            tcga_per_cohort[c_name] = {"n": int(len(grp)), "pct": float(len(grp) / len(df_tcga) * 100), "r": float(rc), "p": float(pc)}

        # DepMap pan-solid within-lineage Z (unmatched ~22 lineages)
        df_expr_u["Lineage"] = df_expr_u["ModelID"].map(lin_map)
        df_sol_z = df_expr_u[df_expr_u["ModelID"].isin(solid_ids)].dropna(subset=["Lineage"]).groupby("Lineage").filter(lambda g: len(g) >= 2).copy()
        df_sol_z["Hz"] = df_sol_z.groupby("Lineage")["HIF1A"].transform(lambda s: (s - s.mean()) / s.std())
        df_sol_z["Sz"] = df_sol_z.groupby("Lineage")["SLC7A1"].transform(lambda s: (s - s.mean()) / s.std())
        df_sol_z = df_sol_z.dropna(subset=["Hz", "Sz"])
        r_pan_sol, p_pan_sol = stats.pearsonr(df_sol_z["Hz"], df_sol_z["Sz"])
        z_mismatched, p_mismatched = fisher_r_to_z(r_tcga_z, len(df_tcga), r_pan_sol, len(df_sol_z))

        # Matched 3-lineage DepMap (Pancreas, CNS/Brain, Breast) - all unique default models & solid-filtered
        matched_lins = ["Pancreas", "CNS/Brain", "Breast"]
        df_pcb_all = df_expr_u[df_expr_u["Lineage"].isin(matched_lins)].copy()
        df_pcb_all["Hz"] = df_pcb_all.groupby("Lineage")["HIF1A"].transform(lambda s: (s - s.mean()) / s.std())
        df_pcb_all["Sz"] = df_pcb_all.groupby("Lineage")["SLC7A1"].transform(lambda s: (s - s.mean()) / s.std())
        r_pcb_all_z, p_pcb_all_z = stats.pearsonr(df_pcb_all["Hz"], df_pcb_all["Sz"])
        r_pcb_all_raw, p_pcb_all_raw = stats.pearsonr(df_pcb_all["HIF1A"], df_pcb_all["SLC7A1"])
        z_match_all, p_match_all = fisher_r_to_z(r_tcga_z, len(df_tcga), r_pcb_all_z, len(df_pcb_all))

        df_pcb_sol = df_sol_z[df_sol_z["Lineage"].isin(matched_lins)].copy()
        r_pcb_sol_z, p_pcb_sol_z = stats.pearsonr(df_pcb_sol["Hz"], df_pcb_sol["Sz"])
        r_pcb_sol_raw, p_pcb_sol_raw = stats.pearsonr(df_pcb_sol["HIF1A"], df_pcb_sol["SLC7A1"])
        z_match_sol, p_match_sol = fisher_r_to_z(r_tcga_z, len(df_tcga), r_pcb_sol_z, len(df_pcb_sol))

        depmap_per_lin = {}
        for l_name, grp in df_pcb_all.groupby("Lineage"):
            rl, pl = stats.pearsonr(grp["HIF1A"], grp["SLC7A1"])
            depmap_per_lin[l_name] = {"n_unique": int(len(grp)), "r": float(rl), "p": float(pl)}

        # Also check TCGA-BRCA vs DepMap Breast Fisher Z
        brca_tcga_n = tcga_per_cohort["TCGA-BRCA"]["n"]
        brca_tcga_r = tcga_per_cohort["TCGA-BRCA"]["r"]
        brca_dep_n = depmap_per_lin["Breast"]["n_unique"]
        brca_dep_r = depmap_per_lin["Breast"]["r"]
        z_brca_only, p_brca_only = fisher_r_to_z(brca_tcga_r, brca_tcga_n, brca_dep_r, brca_dep_n)

        # 6. Strict VHL-LoF analysis in raw DepMap CRISPR
        mut_fp = os.path.join(data_raw, "OmicsSomaticMutations.csv")
        mut_hdr = pd.read_csv(mut_fp, nrows=1).columns.tolist()
        mut_cols = [c for c in ["Hugo_Symbol", "ModelID", "LikelyLoF", "LoF", "isDeleterious", "Variant_annotation", "Variant_Classification"] if c in mut_hdr]
        df_mut = pd.read_csv(mut_fp, usecols=mut_cols)
        vhl_muts = df_mut[df_mut["Hugo_Symbol"] == "VHL"].copy()
        trunc_splice = {
            "Frame_Shift_Del", "Frame_Shift_Ins", "Nonsense_Mutation", "Splice_Site",
            "Nonstop_Mutation", "Translation_Start_Site", "FRAMESHIFT_INDEL", "STOP_GAINED", "SPLICE_SITE",
        }
        lof_mask = pd.Series(False, index=vhl_muts.index)
        for bcol in ("LikelyLoF", "LoF", "isDeleterious"):
            if bcol in vhl_muts.columns:
                lof_mask |= vhl_muts[bcol].astype(str).str.lower().isin(["true", "1", "yes"])
        if "Variant_annotation" in vhl_muts.columns:
            lof_mask |= vhl_muts["Variant_annotation"].astype(str).str.lower().eq("damaging")
        if "Variant_Classification" in vhl_muts.columns:
            lof_mask |= vhl_muts["Variant_Classification"].astype(str).isin(trunc_splice)
        vhl_mut_ids = set(vhl_muts.loc[lof_mask, "ModelID"].dropna().unique())

        vhl_scope = df_solid.copy()
        vhl_scope["Lineage"] = vhl_scope["ModelID"].map(sub_map["OncotreeLineage"])
        vhl_scope["Subtype"] = vhl_scope["ModelID"].map(sub_map["OncotreeSubtype"]).fillna("")
        vhl_scope["VHL_LoF"] = vhl_scope["ModelID"].isin(vhl_mut_ids)
        df_vhl = vhl_scope[vhl_scope["VHL_LoF"]]
        df_wt = vhl_scope[~vhl_scope["VHL_LoF"]]
        df_kid_wt = vhl_scope[(vhl_scope["Lineage"] == "Kidney") & (~vhl_scope["VHL_LoF"])]

        r_vhl_hf71, p_vhl_hf71 = stats.pearsonr(df_vhl["HIF1A"], df_vhl["SLC7A1"])
        rho_vhl_hf71, _ = stats.spearmanr(df_vhl["HIF1A"], df_vhl["SLC7A1"])
        r_vhl_ep71, p_vhl_ep71 = stats.pearsonr(df_vhl["EPAS1"], df_vhl["SLC7A1"])
        r_wt_hf71, p_wt_hf71 = stats.pearsonr(df_wt["HIF1A"], df_wt["SLC7A1"])

        vhl_genes = {}
        for g in ["HIF1A", "EPAS1", "SLC7A1", "SLC1A5", "SMARCA4"]:
            va = df_vhl[g].values
            wa = df_wt[g].values
            _, wp = stats.ttest_ind(va, wa, equal_var=False)
            vhl_genes[g] = {
                "vhl_n": int(len(va)), "vhl_mean": float(np.mean(va)), "vhl_sd": float(np.std(va, ddof=1)), "vhl_median": float(np.median(va)),
                "wt_n": int(len(wa)), "wt_mean": float(np.mean(wa)), "wt_sd": float(np.std(wa, ddof=1)), "wt_median": float(np.median(wa)),
                "cohens_d": cohens_d(va, wa), "welch_p": float(wp),
            }
        _, p_kid_epas1 = stats.ttest_ind(df_vhl["EPAS1"].values, df_kid_wt["EPAS1"].values, equal_var=False)

    else:
        with open(sum_fp, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # 7. Clean GEO Hypoxia table (drop cross-organ pooled GSE145108 row; suppress underpowered n=2 per-line p-values)
    df_geo = pd.read_csv(os.path.join(data_proc, "geo_hypoxia_perturbation.csv"))
    df_geo_clean = df_geo[df_geo["Cell_Line"] != "RKO + A549 + H460 (pooled pairs)"].copy()
    is_n2 = (df_geo_clean["N_Normoxia"] <= 2) & (df_geo_clean["N_Hypoxia"] <= 2)
    df_geo_clean["Descriptive_Only"] = is_n2
    df_geo_clean.loc[is_n2, ["T_Statistic", "P_Value"]] = np.nan
    df_geo_clean.loc[is_n2, "Test"] = "Descriptive Log2FC only (n=2 pairs; t-test omitted as underpowered)"
    geo_out_fp = os.path.join(out_dir, "geo_hypoxia_clean.csv")
    df_geo_clean.to_csv(geo_out_fp, index=False)

    geo_summary = {}
    for (acc, comp), grp in df_geo_clean[~is_n2].groupby(["GEO_Accession", "Comparison"]):
        sub_g = grp.set_index("Gene")
        geo_summary[f"{acc}::{comp}"] = {
            g: {
                "log2fc": float(sub_g.loc[g, "Log2FC"]),
                "p": float(sub_g.loc[g, "P_Value"]),
            }
            for g in ["CA9", "VEGFA", "HIF1A", "SLC7A1", "SLC1A5", "SMARCA4"] if g in sub_g.index
        }

    if has_raw_depmap:
        # Assemble JSON summary
        summary = {
            "solid_codependency": {
                "n_solid_complete": int(len(df_solid)),
                "slc1a5_vs_slc7a1_r": float(r_15_71),
                "slc1a5_vs_slc7a1_rho": float(rho_15_71),
                "slc1a5_vs_slc7a1_p": float(p_15_71),
                "paralog_controls": paralog_res,
            },
            "smarca4_continuous_coessentiality": {
                "n_genes_tested": int(len(df_cont)),
                "n_fdr05_pearson": int((df_cont["Pearson_FDR_q"] < 0.05).sum()),
                "key_genes": baf_partners,
            },
            "diploid_sparing_and_mki67": diploid_res,
            "fig1d_lineage_matched_expression": {
                "tcga_3cohort_n": int(len(df_tcga)),
                "tcga_3cohort_raw_r": float(r_tcga_raw),
                "tcga_3cohort_within_z_r": float(r_tcga_z),
                "tcga_3cohort_within_z_p": float(p_tcga_z),
                "tcga_per_cohort": tcga_per_cohort,
                "depmap_pansolid_mismatched_n": int(len(df_sol_z)),
                "depmap_pansolid_mismatched_n_lineages": int(df_sol_z["Lineage"].nunique()),
                "depmap_pansolid_within_z_r": float(r_pan_sol),
                "depmap_pansolid_within_z_p": float(p_pan_sol),
                "fisher_z_mismatched": float(z_mismatched),
                "fisher_p_mismatched": float(p_mismatched),
                "depmap_matched_3lin_all_unique_n": int(len(df_pcb_all)),
                "depmap_matched_3lin_all_unique_within_z_r": float(r_pcb_all_z),
                "depmap_matched_3lin_all_unique_within_z_p": float(p_pcb_all_z),
                "depmap_matched_3lin_all_unique_raw_r": float(r_pcb_all_raw),
                "fisher_z_matched_all_unique": float(z_match_all),
                "fisher_p_matched_all_unique": float(p_match_all),
                "depmap_matched_3lin_solid_n": int(len(df_pcb_sol)),
                "depmap_matched_3lin_solid_within_z_r": float(r_pcb_sol_z),
                "depmap_matched_3lin_solid_within_z_p": float(p_pcb_sol_z),
                "fisher_z_matched_solid": float(z_match_sol),
                "fisher_p_matched_solid": float(p_match_sol),
                "depmap_per_lineage": depmap_per_lin,
                "brca_only_fisher_z": float(z_brca_only),
                "brca_only_fisher_p": float(p_brca_only),
            },
            "vhl_lof_biology": {
                "n_vhl_lof": int(len(df_vhl)),
                "n_wt_solid": int(len(df_wt)),
                "n_kid_wt": int(len(df_kid_wt)),
                "vhl_hif1a_vs_slc7a1_r": float(r_vhl_hf71),
                "vhl_hif1a_vs_slc7a1_rho": float(rho_vhl_hf71),
                "vhl_hif1a_vs_slc7a1_p": float(p_vhl_hf71),
                "vhl_epas1_vs_slc7a1_r": float(r_vhl_ep71),
                "vhl_epas1_vs_slc7a1_p": float(p_vhl_ep71),
                "wt_hif1a_vs_slc7a1_r": float(r_wt_hf71),
                "wt_hif1a_vs_slc7a1_p": float(p_wt_hf71),
                "kid_wt_epas1_mean": float(df_kid_wt["EPAS1"].mean()),
                "vhl_vs_kid_wt_epas1_welch_p": float(p_kid_epas1),
                "genes": vhl_genes,
            },
            "geo_hypoxia_pooled": geo_summary,
        }

    else:
        summary["geo_hypoxia_pooled"] = geo_summary

    sum_fp = os.path.join(out_dir, "depmap_and_hypoxia_summary.json")
    with open(sum_fp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if has_raw_depmap:
        print("=== DEPMAP & HYPOXIA REANALYSIS COMPLETE ===")
        print(f"Fig 1D TCGA 3-cohort (N={len(df_tcga)}): within-Z r={r_tcga_z:+.4f}, BRCA={tcga_per_cohort['TCGA-BRCA']['pct']:.1f}%")
        print(f"Fig 1D Matched DepMap all-unique (n={len(df_pcb_all)}): within-Z r={r_pcb_all_z:+.4f} (p={p_pcb_all_z:.4f}), Fisher Z={z_match_all:.2f} (p={p_match_all:.3f})")
        print(f"Fig 1D Matched DepMap solid-only (n={len(df_pcb_sol)}): within-Z r={r_pcb_sol_z:+.4f} (p={p_pcb_sol_z:.4f}), Fisher Z={z_match_sol:.2f} (p={p_match_sol:.3f})")
        print(f"Diploid SMARCA4 unadj p={diploid_res['ols_unadj_p_diploid']:.4f} | +MKI67 beta={diploid_res['ols_mki67_beta_diploid']:+.4f}, p={diploid_res['ols_mki67_p_diploid']:.4f}")
        print(f"VHL-LoF (n={len(df_vhl)}): HIF1A mean Chronos={vhl_genes['HIF1A']['vhl_mean']:+.4f}, EPAS1 mean Chronos={vhl_genes['EPAS1']['vhl_mean']:+.4f}, EPAS1-SLC7A1 r={r_vhl_ep71:+.4f}")
        print(f"Continuous SMARCA4 co-essentiality (N={len(df_solid)}): SMARCB1 r={baf_partners['SMARCB1']['pearson_r']:+.4f} (q={baf_partners['SMARCB1']['pearson_fdr_q']:.2e}), SMARCA2 r={baf_partners['SMARCA2']['pearson_r']:+.4f} (q={baf_partners['SMARCA2']['pearson_fdr_q']:.2e})")
    else:
        print("=== GEO HYPOXIA CLEANED FROM data/processed (raw DepMap/TCGA omitted; loaded cached DepMap/VHL summary) ===")
    print(f"Saved JSON -> {sum_fp}")


if __name__ == "__main__":
    main()
