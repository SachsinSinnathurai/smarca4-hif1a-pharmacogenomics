#!/usr/bin/env python3
"""DepMap 24Q2 Co-Dependency, BAF Co-Essentiality, Diploid Sparing, and VHL Restoration."""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ord = np.minimum.accumulate(((ranked * n) / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ord, 0.0, 1.0)
    return q


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def main():
    model_fp = os.path.join(DATA_RAW, "Model.csv")
    crispr_fp = os.path.join(DATA_RAW, "CRISPRGeneEffect.csv")
    expr_fp = os.path.join(DATA_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    mut_fp = os.path.join(DATA_RAW, "OmicsSomaticMutations.csv")

    if not all(os.path.exists(fp) for fp in (model_fp, crispr_fp, expr_fp, mut_fp)):
        print("DepMap 24Q2 raw archives not found in data/raw/; skipping raw recomputation.")
        return

    print("DEPMAP 24Q2 CO-DEPENDENCY, BAF CO-ESSENTIALITY, DIPLOID SPARING & VHL ANALYSIS")

    df_m = pd.read_csv(model_fp)
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

    df_c = pd.read_csv(crispr_fp)
    id_col = df_c.columns[0]
    df_c = df_c.rename(columns={id_col: "ModelID"})
    gene_cols = [c for c in df_c.columns if c != "ModelID"]
    col_rename = {c: c.split(" ")[0].strip("()") for c in gene_cols}
    df_c = df_c.rename(columns=col_rename)

    # 1. Solid tumor SLC1A5-SLC7A1 co-dependency & paralog specificity (n = 1,038)
    core_cols = ["SMARCA4", "HIF1A", "SLC1A5", "SLC7A1", "ABCB1", "EPAS1"]
    df_solid = df_c[df_c["ModelID"].isin(solid_ids)].dropna(subset=core_cols)
    cand5 = ["SMARCA4", "HIF1A", "SLC1A5", "SLC7A1", "ABCB1"]
    pair_list = [(cand5[i], cand5[j]) for i in range(len(cand5)) for j in range(i + 1, len(cand5))]
    p_cand5 = [stats.pearsonr(df_solid[a], df_solid[b])[1] for a, b in pair_list]
    q_cand5 = dict(zip(pair_list, bh_fdr(np.array(p_cand5))))
    r_15_71, p_15_71 = stats.pearsonr(df_solid["SLC1A5"], df_solid["SLC7A1"])
    print(f"\n[1] SOLID TUMOR CO-DEPENDENCY & PARALOG SPECIFICITY (n = {len(df_solid):,})")
    print(f"  * SLC1A5 vs SLC7A1: Pearson r = {r_15_71:+.4f} (rounded {r_15_71:+.3f}), p = {p_15_71:.2e}, BH FDR q (m={len(pair_list)}) = {q_cand5[('SLC1A5', 'SLC7A1')]:.2e}")
    for g_main, paralog in [("SLC7A1", "SLC7A5"), ("SLC7A1", "SLC7A2"), ("SLC1A5", "SLC1A4")]:
        sub_p = df_c[[g_main, paralog]].dropna()
        r_p, p_p = stats.pearsonr(sub_p[g_main], sub_p[paralog])
        print(f"  * Paralog specificity {g_main} vs {paralog} (pan-cancer n = {len(sub_p):,}): r = {r_p:+.4f}, p = {p_p:.2e}")
    sub_71_75_sol = df_c[df_c["ModelID"].isin(solid_ids)][["SLC7A1", "SLC7A5"]].dropna()
    r_71_75_s, p_71_75_s = stats.pearsonr(sub_71_75_sol["SLC7A1"], sub_71_75_sol["SLC7A5"])
    print(f"  * Solid-tumor SLC7A1 vs SLC7A5 (n = {len(sub_71_75_sol):,}): r = {r_71_75_s:+.4f}, p = {p_71_75_s:.2e}")

    # 2. SMARCA4-dependent co-essentiality scan across 18,434 genes (unified malignant solid_ids)
    df_lin = df_c[df_c["ModelID"].isin(solid_ids)].copy()
    dep_mask = df_lin["SMARCA4"] <= -0.50
    ind_mask = df_lin["SMARCA4"] >= -0.10
    bg_genes = [c for c in df_lin.columns if c not in ("ModelID", "SMARCA4")]
    dep_mat = df_lin.loc[dep_mask, bg_genes].to_numpy(dtype=float)
    ind_mat = df_lin.loc[ind_mask, bg_genes].to_numpy(dtype=float)

    t_vals, p_vals = stats.ttest_ind(dep_mat, ind_mat, axis=0, equal_var=False, nan_policy="omit")
    deltas = np.nanmean(dep_mat, axis=0) - np.nanmean(ind_mat, axis=0)
    valid = ~np.isnan(p_vals)
    q_vals = np.ones_like(p_vals)
    q_vals[valid] = bh_fdr(p_vals[valid])

    df_scan = pd.DataFrame({
        "Gene": bg_genes,
        "Delta_Chronos": deltas,
        "T_Stat": t_vals,
        "P_Value": p_vals,
        "FDR_Q": q_vals,
    }).set_index("Gene")

    os.makedirs(DATA_PROC, exist_ok=True)
    df_scan.reset_index().rename(columns={"Delta_Chronos": "Delta"}).to_csv(
        os.path.join(DATA_PROC, "smarca4_coessentiality_genomewide.csv"), index=False
    )
    df_scan.reset_index().query("FDR_Q < 0.05").to_csv(
        os.path.join(DATA_PROC, "smarca4_significant_synthetic_lethal_genes.csv"), index=False
    )

    print(
        f"\n[2] SMARCA4-DEPENDENT CO-ESSENTIALITY SCAN ({len(bg_genes):,} genes; "
        f"N_dep = {int(dep_mask.sum())}, N_ind = {int(ind_mask.sum())})"
    )
    for g in ["SMARCB1", "SMARCC1", "SMARCD1", "ARID1A", "SMARCE1", "MLLT1", "SMARCA2"]:
        row = df_scan.loc[g]
        print(
            f"  * {g:<8} | Delta = {row['Delta_Chronos']:+.4f} | "
            f"t = {row['T_Stat']:+.2f} | p = {row['P_Value']:.2e} | FDR q = {row['FDR_Q']:.2e}"
        )

    # 3. Non-malignant diploid fibroblast/epithelial sparing (n = 16)
    comp_genes = ["SMARCA4", "SLC1A5", "SLC7A1", "HIF1A"]
    df_norm = df_c[df_c["ModelID"].isin(norm_ids)].dropna(subset=comp_genes).copy()
    comp_norm = df_norm[comp_genes].mean(axis=1)
    comp_solid = df_solid[comp_genes].mean(axis=1)
    comp_smarca4_dep = df_solid.loc[df_solid["SMARCA4"] <= -0.50, comp_genes].mean(axis=1)

    _, p_mwu_solid = stats.mannwhitneyu(comp_norm, comp_solid, alternative="two-sided")
    _, p_mwu_dep = stats.mannwhitneyu(comp_norm, comp_smarca4_dep, alternative="two-sided")
    _, p_mwu_smarca4 = stats.mannwhitneyu(df_norm["SMARCA4"], df_solid["SMARCA4"], alternative="two-sided")
    n_lt_half = int((comp_norm < -0.50).sum())

    print(f"\n[3] NON-MALIGNANT DIPLOID SPARING (n = {len(df_norm)})")
    print(f"  * Composite 4-gene < -0.50 in diploid lines: {n_lt_half}/{len(df_norm)} (mean = {comp_norm.mean():+.4f})")
    print(f"  * Composite 4-gene diploid vs solid (n = {len(comp_solid):,}): Delta = {comp_norm.mean() - comp_solid.mean():+.4f}, MWU p = {p_mwu_solid:.4f}")
    print(f"  * Composite 4-gene diploid vs SMARCA4-dep solid (n = {len(comp_smarca4_dep)}): Delta = {comp_norm.mean() - comp_smarca4_dep.mean():+.4f}, MWU p = {p_mwu_dep:.2e}")
    print(f"  * SMARCA4 single-gene diploid vs solid: Delta = {df_norm['SMARCA4'].mean() - df_solid['SMARCA4'].mean():+.4f}, MWU p = {p_mwu_smarca4:.4f}")

    # 4. Normoxic 21% O2 expression uncoupling & VHL-deficient Chronos restoration
    df_eh = pd.read_csv(expr_fp, nrows=1)
    e_cols = ["ModelID", "IsDefaultEntryForModel"]
    e_rename = {}
    for g in ["HIF1A", "SLC7A1"]:
        m = [c for c in df_eh.columns if c.split(" ")[0].strip("()") == g]
        e_cols.append(m[0])
        e_rename[m[0]] = g
    df_expr_all = pd.read_csv(expr_fp, usecols=e_cols).rename(columns=e_rename).dropna(subset=["HIF1A", "SLC7A1"])
    df_expr_u = df_expr_all[df_expr_all["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").copy()
    r_vitro_raw, p_vitro_raw = stats.pearsonr(df_expr_all["HIF1A"], df_expr_all["SLC7A1"])
    r_vitro_u, p_vitro_u = stats.pearsonr(df_expr_u["HIF1A"], df_expr_u["SLC7A1"])

    lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()
    df_expr_sol = df_expr_u[df_expr_u["ModelID"].isin(solid_ids)].copy()
    df_expr_sol["Lineage"] = df_expr_sol["ModelID"].map(lin_map)
    df_expr_sol = df_expr_sol.dropna(subset=["Lineage"]).groupby("Lineage").filter(lambda g: len(g) >= 2).copy()
    df_expr_sol["Hz"] = df_expr_sol.groupby("Lineage")["HIF1A"].transform(lambda x: (x - x.mean()) / x.std())
    df_expr_sol["Sz"] = df_expr_sol.groupby("Lineage")["SLC7A1"].transform(lambda x: (x - x.mean()) / x.std())
    df_expr_sol = df_expr_sol.dropna(subset=["Hz", "Sz"])
    r_linz, p_linz = stats.pearsonr(df_expr_sol["Hz"], df_expr_sol["Sz"])

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

    sub_map = df_m.drop_duplicates("ModelID").set_index("ModelID")[["OncotreeLineage", "OncotreeSubtype"]]
    vhl_scope = df_lin.dropna(subset=["EPAS1", "HIF1A", "SMARCA4", "SLC1A5", "SLC7A1"]).copy()
    vhl_scope["Lineage"] = vhl_scope["ModelID"].map(sub_map["OncotreeLineage"])
    vhl_scope["Subtype"] = vhl_scope["ModelID"].map(sub_map["OncotreeSubtype"]).fillna("")
    vhl_scope["VHL_LoF"] = vhl_scope["ModelID"].isin(vhl_mut_ids)

    df_vhl = vhl_scope[vhl_scope["VHL_LoF"]]
    df_wt = vhl_scope[~vhl_scope["VHL_LoF"]]
    r_vhl, p_vhl = stats.pearsonr(df_vhl["HIF1A"], df_vhl["SLC7A1"])
    rho_vhl, p_rho_vhl = stats.spearmanr(df_vhl["HIF1A"], df_vhl["SLC7A1"])
    loo_records = []
    for mid in df_vhl["ModelID"].tolist():
        sub_loo = df_vhl[df_vhl["ModelID"] != mid]
        r_loo, p_loo = stats.pearsonr(sub_loo["HIF1A"], sub_loo["SLC7A1"])
        loo_records.append((r_loo, p_loo, mid))
    loo_min = min(loo_records, key=lambda x: x[0])
    loo_max = max(loo_records, key=lambda x: x[0])
    r_vhl_wt, p_vhl_wt = stats.pearsonr(df_wt["HIF1A"], df_wt["SLC7A1"])
    d_epas1 = cohens_d(df_vhl["EPAS1"].values, df_wt["EPAS1"].values)
    _, p_epas1 = stats.ttest_ind(df_vhl["EPAS1"].values, df_wt["EPAS1"].values, equal_var=False)

    # Lineage-controlled & non-ccRCC sensitivity checks for EPAS1
    df_kid_wt = vhl_scope[(vhl_scope["Lineage"] == "Kidney") & (~vhl_scope["VHL_LoF"])]
    d_epas1_kid = cohens_d(df_vhl["EPAS1"].values, df_kid_wt["EPAS1"].values)
    _, p_epas1_kid = stats.ttest_ind(df_vhl["EPAS1"].values, df_kid_wt["EPAS1"].values, equal_var=False)
    df_nonkid = vhl_scope[vhl_scope["Lineage"] != "Kidney"]
    df_noncc_wt = vhl_scope[~((vhl_scope["Lineage"] == "Kidney") & (vhl_scope["VHL_LoF"] | vhl_scope["Subtype"].str.contains("Clear Cell", case=False)))]
    d_epas1_noncc = cohens_d(df_vhl["EPAS1"].values, df_noncc_wt["EPAS1"].values)
    _, p_epas1_noncc = stats.ttest_ind(df_vhl["EPAS1"].values, df_noncc_wt["EPAS1"].values, equal_var=False)
    _, p_kidwt_vs_nonkid = stats.ttest_ind(df_kid_wt["EPAS1"].values, df_nonkid["EPAS1"].values, equal_var=False)

    df_expr_u["Lineage"] = df_expr_u["ModelID"].map(lin_map)
    df_pcb_all = df_expr_u[df_expr_u["Lineage"].isin(["Pancreas", "CNS/Brain", "Breast"])].copy()
    df_pcb_all["Hz"] = df_pcb_all.groupby("Lineage")["HIF1A"].transform(lambda x: (x - x.mean()) / x.std())
    df_pcb_all["Sz"] = df_pcb_all.groupby("Lineage")["SLC7A1"].transform(lambda x: (x - x.mean()) / x.std())
    r_pcb_all, p_pcb_all = stats.pearsonr(df_pcb_all["Hz"], df_pcb_all["Sz"])
    r_pcb_all_raw, p_pcb_all_raw = stats.pearsonr(df_pcb_all["HIF1A"], df_pcb_all["SLC7A1"])

    df_pcb = df_expr_sol[df_expr_sol["Lineage"].isin(["Pancreas", "CNS/Brain", "Breast"])].copy()
    r_pcb_linz, p_pcb_linz = stats.pearsonr(df_pcb["Hz"], df_pcb["Sz"])
    r_pcb_raw, p_pcb_raw = stats.pearsonr(df_pcb["HIF1A"], df_pcb["SLC7A1"])

    print("\n[4] NORMOXIC 21% O2 UNCOUPLING & VHL-DEFICIENT RESTORATION")
    print(f"  * DepMap 21% O2 Expression: {len(df_expr_all):,} total rows ({len(df_expr_u):,} unique ModelIDs): raw r = {r_vitro_raw:+.4f} (p = {p_vitro_raw:.3f}) | unique raw r = {r_vitro_u:+.4f} (p = {p_vitro_u:.3f})")
    print(f"  * DepMap Within-Lineage Z-Scored Solid (n = {len(df_expr_sol):,} unique lines): r = {r_linz:+.4f}, p = {p_linz:.3f}")
    print(f"  * Lineage-Matched Pancreas/CNS/Breast (all unique n = {len(df_pcb_all)}): within-lineage r = {r_pcb_all:+.4f} (p = {p_pcb_all:.4f}), raw r = {r_pcb_all_raw:+.4f} (p = {p_pcb_all_raw:.2e})")
    print(f"  * Lineage-Matched Pancreas/CNS/Breast (solid-filtered n = {len(df_pcb)}): within-lineage r = {r_pcb_linz:+.4f} (p = {p_pcb_linz:.4f}), raw r = {r_pcb_raw:+.4f} (p = {p_pcb_raw:.2e})")
    print(f"  * VHL LoF/Damaging-Mutant Chronos (n = {len(df_vhl)} vs WT n = {len(df_wt):,}): Mutant HIF1A vs SLC7A1 Pearson r = {r_vhl:+.4f} (p = {p_vhl:.4f}), Spearman rho = {rho_vhl:+.4f} (p = {p_rho_vhl:.4f}), LOO Pearson r = [{loo_min[0]:+.4f} (p = {loo_min[1]:.4f}, excl. {loo_min[2]}), {loo_max[0]:+.4f} (p = {loo_max[1]:.4f}, excl. {loo_max[2]})] | WT Baseline r = {r_vhl_wt:+.4f} (p = {p_vhl_wt:.3f})")
    print(f"  * VHL-LoF EPAS1 vs Pan-Solid WT: mean = {df_vhl['EPAS1'].mean():+.4f} (SD = {df_vhl['EPAS1'].std(ddof=1):.4f}) vs WT mean = {df_wt['EPAS1'].mean():+.4f} (SD = {df_wt['EPAS1'].std(ddof=1):.4f}) | Delta = {df_vhl['EPAS1'].mean() - df_wt['EPAS1'].mean():+.4f}, d = {d_epas1:+.3f}, Welch p = {p_epas1:.2e}")
    print(f"  * EPAS1 Within-Kidney Sensitivity (Kidney VHL-LoF n = {len(df_vhl)} vs Kidney VHL-WT n = {len(df_kid_wt)}): WT mean = {df_kid_wt['EPAS1'].mean():+.4f} (SD = {df_kid_wt['EPAS1'].std(ddof=1):.4f}) | Delta = {df_vhl['EPAS1'].mean() - df_kid_wt['EPAS1'].mean():+.4f}, d = {d_epas1_kid:+.3f}, Welch p = {p_epas1_kid:.2e}")
    print(f"  * EPAS1 Non-ccRCC Lineage Check: Non-Kidney WT (n = {len(df_nonkid):,}) mean = {df_nonkid['EPAS1'].mean():+.4f} (SD = {df_nonkid['EPAS1'].std(ddof=1):.4f}; Kidney WT vs Non-Kidney Welch p = {p_kidwt_vs_nonkid:.3f}) | VHL-LoF vs Non-ccRCC WT (n = {len(df_noncc_wt):,}, mean = {df_noncc_wt['EPAS1'].mean():+.4f}): d = {d_epas1_noncc:+.3f}, Welch p = {p_epas1_noncc:.2e}")


if __name__ == "__main__":
    main()
