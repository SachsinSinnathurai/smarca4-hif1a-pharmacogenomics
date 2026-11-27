"""
Step 3: Rebuilt Pan-Solid TCGA Survival, BH-FDR Across Cohorts, Buffa-14 Benchmark,
GDSC2 Post-MKI67 BH-FDR (m = 286), and Multi-Omics Boundaries.
Writes exclusively inside results/.
"""

from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = repo_root
    data_proc = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(repo_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    df_sens = pd.read_csv(os.path.join(data_proc, "survival_full_tcga_and_mki67_sensitivity.csv"))
    df_t1 = pd.read_csv(os.path.join(data_proc, "tables", "Table_1_Clinical_Cohorts.csv"))
    df_brca = pd.read_csv(os.path.join(data_proc, "brca_subtype_receptor_sensitivity.csv"))
    df_gdsc = pd.read_csv(os.path.join(data_proc, "gdsc2_lineage_controlled_anova.csv"))
    df_sc = pd.read_csv(os.path.join(data_proc, "pdac_scrna_celltype_specificity.csv"))
    df_gsea = pd.read_csv(os.path.join(data_proc, "hallmark_gsea_transcriptome_wide_enrichment.csv"))
    df_rppa = pd.read_csv(os.path.join(data_proc, "tcga_rppa_proteomic_validation.csv"))
    df_cin = pd.read_csv(os.path.join(data_proc, "genomic_scars_hrd_aneuploidy_orthogonality.csv"))

    def row_dict(sub: pd.DataFrame) -> dict:
        r = sub.iloc[0]
        return {k: (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else str(v)))) for k, v in r.items()}

    # 1. 19-Cohort Solid TCGA Pan-Cancer Benchmark + BH-FDR (Base & +MKI67)
    base_19_cohorts = df_sens[(df_sens["Model"] == "SMARCA4_HIF1A_Base") & (df_sens["Cohort"].str.startswith("TCGA-"))].copy()
    base_19_cohorts["BH_FDR_q_19"] = multipletests(base_19_cohorts["P_Value"].astype(float).values, method="fdr_bh")[1]

    mk_19_cohorts = df_sens[
        (df_sens["Model"].isin(["Plus_MKI67", "PanTCGA_Plus_MKI67"]))
        & (df_sens["Cohort"].str.startswith("TCGA-"))
    ].drop_duplicates(subset=["Cohort"]).copy()
    mk_19_cohorts["BH_FDR_q_19"] = multipletests(mk_19_cohorts["P_Value"].astype(float).values, method="fdr_bh")[1]

    df_19_fdr = base_19_cohorts[["Cohort", "N", "Events", "HR", "CI_Low", "CI_High", "P_Value", "BH_FDR_q_19", "C_Index"]].rename(
        columns={"HR": "Base_HR", "CI_Low": "Base_CI_Low", "CI_High": "Base_CI_High", "P_Value": "Base_P", "BH_FDR_q_19": "Base_FDR_q_19", "C_Index": "Base_C_Index"}
    ).merge(
        mk_19_cohorts[["Cohort", "HR", "CI_Low", "CI_High", "P_Value", "BH_FDR_q_19", "C_Index"]].rename(
            columns={"HR": "MKI67_HR", "CI_Low": "MKI67_CI_Low", "CI_High": "MKI67_CI_High", "P_Value": "MKI67_P", "BH_FDR_q_19": "MKI67_FDR_q_19", "C_Index": "MKI67_C_Index"}
        ),
        on="Cohort", how="inner"
    ).sort_values("MKI67_P").reset_index(drop=True)
    df_19_fdr.to_csv(os.path.join(out_dir, "survival_19tcga_fdr.csv"), index=False)

    # 2. Secondary 10-Cohort Panel BH-FDR (Base & +MKI67)
    base_10 = df_sens[(df_sens["Model"] == "Base_No_MKI67") & (~df_sens["Cohort"].str.contains("Meta", na=False))].copy()
    base_10["Base_FDR_q_10"] = multipletests(base_10["P_Value"].astype(float).values, method="fdr_bh")[1]
    mk_10 = df_sens[(df_sens["Model"] == "Plus_MKI67") & (~df_sens["Cohort"].str.contains("Meta", na=False))].copy()
    mk_10["MKI67_FDR_q_10"] = multipletests(mk_10["P_Value"].astype(float).values, method="fdr_bh")[1]
    df_10_fdr = base_10[["Cohort", "N", "Events", "HR", "CI_Low", "CI_High", "P_Value", "Base_FDR_q_10"]].rename(
        columns={"HR": "Base_HR", "CI_Low": "Base_CI_Low", "CI_High": "Base_CI_High", "P_Value": "Base_P"}
    ).merge(
        mk_10[["Cohort", "HR", "CI_Low", "CI_High", "P_Value", "MKI67_FDR_q_10"]].rename(
            columns={"HR": "MKI67_HR", "CI_Low": "MKI67_CI_Low", "CI_High": "MKI67_CI_High", "P_Value": "MKI67_P"}
        ),
        on="Cohort", how="inner"
    ).sort_values("MKI67_P").reset_index(drop=True)
    df_10_fdr.to_csv(os.path.join(out_dir, "survival_10primary_fdr.csv"), index=False)

    # Extract meta summaries
    meta_19_base = row_dict(df_sens[(df_sens["Model"] == "PanTCGA_Base") & (df_sens["Cohort"] == "19-All-Available-Solid-TCGA Meta")])
    meta_19_mk = row_dict(df_sens[(df_sens["Model"] == "PanTCGA_Plus_MKI67") & (df_sens["Cohort"] == "19-All-Available-Solid-TCGA Meta")])
    meta_18_base = row_dict(df_sens[(df_sens["Model"] == "PanTCGA_Base") & (df_sens["Cohort"] == "18-Solid-TCGA (w/ Stage or Grade) Meta")])
    meta_11_base = row_dict(df_sens[(df_sens["Model"] == "PanTCGA_Base") & (df_sens["Cohort"] == "11-Remaining-TCGA Meta")])

    buffa_19_base = row_dict(df_sens[(df_sens["Model"] == "Buffa14_Alone_Base") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")])
    buffa_19_mk = row_dict(df_sens[(df_sens["Model"] == "Buffa14_Alone_PlusMKI67") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")])
    sbuffa_19_base = row_dict(df_sens[(df_sens["Model"] == "SMARCA4_Buffa14_Base") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")])
    sbuffa_19_mk = row_dict(df_sens[(df_sens["Model"] == "SMARCA4_Buffa14_PlusMKI67") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")])
    buffa_11_base = row_dict(df_sens[(df_sens["Model"] == "Buffa14_Alone_Base") & (df_sens["Cohort"] == "11-Remaining-TCGA Meta")])
    buffa_8_base = row_dict(df_sens[(df_sens["Model"] == "Buffa14_Alone_Base") & (df_sens["Cohort"] == "8-Primary-TCGA Meta")])
    sh_19_base_c = row_dict(df_sens[(df_sens["Model"] == "SMARCA4_HIF1A_Base") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")])

    # Parse reverse LRT from Buffa-14 +MKI67 note
    cov_note = str(buffa_19_mk.get("Covariates", ""))
    m_rev = re.search(r"Add SH chi2=([0-9.]+) p=([0-9eE.+-]+)", cov_note)
    rev_lrt = {"chi2": float(m_rev.group(1)), "p": float(m_rev.group(2))} if m_rev else None

    meta_10_base = row_dict(df_sens[(df_sens["Model"] == "Composite_SMARCA4_HIF1A") & (df_sens["Cohort"] == "10-Cohort DerSimonian-Laird Meta")])
    meta_10_mk = row_dict(df_sens[(df_sens["Model"] == "Plus_MKI67") & (df_sens["Cohort"] == "10-Cohort Full +MKI67 Meta")])
    meta_8nb = row_dict(df_sens[(df_sens["Model"] == "Composite_SMARCA4_HIF1A") & (df_sens["Cohort"] == "8-Cohort Non-Breast Primary Meta")])
    meta_10_smarca4 = row_dict(df_sens[(df_sens["Model"] == "SMARCA4_z_Alone") & (df_sens["Cohort"] == "10-Cohort DerSimonian-Laird Meta")])
    meta_10_hif1a = row_dict(df_sens[(df_sens["Model"] == "HIF1A_z_Alone") & (df_sens["Cohort"] == "10-Cohort DerSimonian-Laird Meta")])

    # Purity, CGGA, METABRIC, PH audit
    cgga_lin = row_dict(df_sens[(df_sens["Analysis"] == "CGGA-325 Full-Transcriptome Sensitivity") & (df_sens["Model"] == "Plus_MKI67_Linear_RSEM")])
    cgga_top2a = row_dict(df_sens[(df_sens["Analysis"] == "CGGA-325 Full-Transcriptome Sensitivity") & (df_sens["Model"] == "Plus_TOP2A_Log2p1")])
    kirc_pur = row_dict(df_sens[(df_sens["Analysis"] == "TCGA-KIRC & BLCA ABSOLUTE Purity Sensitivity") & (df_sens["Cohort"] == "TCGA-KIRC") & (df_sens["Model"] == "Plus_MKI67_and_ABSOLUTE_Purity")])
    blca_pur = row_dict(df_sens[(df_sens["Analysis"] == "TCGA-KIRC & BLCA ABSOLUTE Purity Sensitivity") & (df_sens["Cohort"] == "TCGA-BLCA") & (df_sens["Model"] == "Plus_MKI67_and_ABSOLUTE_Purity")])
    ph_all = df_sens[df_sens["Analysis"] == "Schoenfeld Proportional Hazards Audit"]
    ph_pass_n = int((ph_all["PH_Risk_P"].astype(float) >= 0.05).sum())

    # 3. GDSC2 Post-MKI67 BH-FDR across all m = 286 drugs
    df_gdsc["MKI67_Adj_FDR_Q"] = multipletests(df_gdsc["MKI67_Adj_P_Value"].astype(float).values, method="fdr_bh")[1]
    df_gdsc_sorted = df_gdsc.sort_values("MKI67_Adj_P_Value").reset_index(drop=True)
    df_gdsc_sorted.to_csv(os.path.join(out_dir, "gdsc2_post_mki67_fdr.csv"), index=False)

    gdsc_summary = {
        "n_compounds": int(len(df_gdsc)),
        "pre_mki67_fdr05": int((df_gdsc["FDR_Q_Value"] < 0.05).sum()),
        "pre_mki67_fdr10": int((df_gdsc["FDR_Q_Value"] < 0.10).sum()),
        "pre_mki67_nominal_p05": int((df_gdsc["P_Value"] < 0.05).sum()),
        "pre_mki67_lnic50_fdr10": int((df_gdsc["LN_IC50_FDR_Q_Value"] < 0.10).sum()),
        "pre_mki67_lnic50_min_q": float(df_gdsc["LN_IC50_FDR_Q_Value"].min()),
        "pre_mki67_partial_r2_median": float(df_gdsc["Partial_R2"].median()),
        "pre_mki67_partial_r2_max": float(df_gdsc["Partial_R2"].max()),
        "navitoclax_pre": row_dict(df_gdsc[df_gdsc["Compound_Name"] == "Navitoclax"]),
        "post_mki67_fdr05": int((df_gdsc["MKI67_Adj_FDR_Q"] < 0.05).sum()),
        "post_mki67_fdr10": int((df_gdsc["MKI67_Adj_FDR_Q"] < 0.10).sum()),
        "post_mki67_min_q": float(df_gdsc["MKI67_Adj_FDR_Q"].min()),
        "post_mki67_top5": df_gdsc_sorted.head(5)[
            ["Compound_Name", "Putative_Target", "N_Solid_Lines", "MKI67_Adj_ANOVA_F", "MKI67_Adj_Partial_R2", "MKI67_Adj_P_Value", "MKI67_Adj_FDR_Q"]
        ].to_dict(orient="records"),
    }

    # 4. Multi-omics summary (scRNA-seq, CAMERA/GSEA, RPPA, CIN)
    sc_summary = {}
    for g in ["SMARCA4", "HIF1A", "SLC7A1", "SLC1A5", "ABCB1"]:
        sub = df_sc[df_sc["Target_Gene"] == g]
        mal = sub[sub["Cell_Type"] == "Malignant"].iloc[0]
        caf = sub[sub["Cell_Type"] == "Fibroblasts"].iloc[0]
        sc_summary[g] = {
            "log2fc_mal_vs_caf": float(mal["Pseudobulk_Log2FC_Malignant_vs_CAF"]),
            "paired_wilcoxon_p": float(mal["Paired_Wilcoxon_P_vs_CAF"]),
            "paired_fdr_q": float(mal["Paired_Wilcoxon_FDR_q"]),
            "mal_pct_expr": float(mal["Pct_Expressing"]),
            "caf_pct_expr": float(caf["Pct_Expressing"]),
        }

    paad_g = df_gsea[df_gsea["Cohort"].str.contains("PAAD")]
    meta_g = df_gsea[df_gsea["Cohort"].str.contains("METABRIC")]
    merged_g = paad_g[["Hallmark_Short", "Normalized_ES_NES"]].merge(
        meta_g[["Hallmark_Short", "Normalized_ES_NES"]], on="Hallmark_Short", suffixes=("_PAAD", "_METABRIC")
    )
    rho_gsea, p_gsea = stats.spearmanr(merged_g["Normalized_ES_NES_PAAD"], merged_g["Normalized_ES_NES_METABRIC"])

    rppa_sig = int((df_rppa["Spearman_FDR_q"] < 0.05).sum())
    cin_7 = df_cin[(df_cin["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_cin["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]
    cin_ext_hrd = df_cin[(df_cin["Cohort"].str.contains("Extended")) & (df_cin["Genomic_Scar_Metric"] == "HRD_Sum")].iloc[0]
    cin_ext_aneu = df_cin[(df_cin["Cohort"].str.contains("Extended")) & (df_cin["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]

    payload = {
        "pan_solid_tcga_19": {
            "meta_19_base": meta_19_base,
            "meta_19_plus_mki67": meta_19_mk,
            "meta_18_stage_grade_base": meta_18_base,
            "meta_11_remaining_base": meta_11_base,
            "base_fdr05_count_19": int((df_19_fdr["Base_FDR_q_19"] < 0.05).sum()),
            "base_fdr10_count_19": int((df_19_fdr["Base_FDR_q_19"] < 0.10).sum()),
            "plus_mki67_fdr05_count_19": int((df_19_fdr["MKI67_FDR_q_19"] < 0.05).sum()),
            "plus_mki67_fdr10_count_19": int((df_19_fdr["MKI67_FDR_q_19"] < 0.10).sum()),
            "top_post_mki67_19": df_19_fdr.head(4).to_dict(orient="records"),
        },
        "buffa14_benchmark": {
            "buffa14_19_base": buffa_19_base,
            "buffa14_19_plus_mki67": buffa_19_mk,
            "smarca4_buffa14_19_base": sbuffa_19_base,
            "smarca4_buffa14_19_plus_mki67": sbuffa_19_mk,
            "buffa14_8_primary_base": buffa_8_base,
            "buffa14_11_remaining_base": buffa_11_base,
            "smarca4_hif1a_19_rsem_c_baseline": sh_19_base_c,
            "reverse_lrt_add_sh_over_buffa_mki67": rev_lrt,
        },
        "secondary_10_cohort_panel": {
            "table1_rows": df_t1.to_dict(orient="records"),
            "meta_10_base": meta_10_base,
            "meta_10_plus_mki67": meta_10_mk,
            "meta_8_non_breast_base": meta_8nb,
            "single_gene_smarca4_10": meta_10_smarca4,
            "single_gene_hif1a_10": meta_10_hif1a,
            "plus_mki67_fdr05_count_10": int((df_10_fdr["MKI67_FDR_q_10"] < 0.05).sum()),
            "plus_mki67_fdr10_count_10": int((df_10_fdr["MKI67_FDR_q_10"] < 0.10).sum()),
            "top_post_mki67_10": df_10_fdr.head(4).to_dict(orient="records"),
            "cgga_mki67_linear": cgga_lin,
            "cgga_top2a": cgga_top2a,
            "kirc_mki67_purity": kirc_pur,
            "blca_mki67_purity": blca_pur,
            "brca_subtype_sensitivity": df_brca.to_dict(orient="records"),
            "schoenfeld_ph_tested": int(len(ph_all)),
            "schoenfeld_ph_pass_ge05": ph_pass_n,
        },
        "gdsc2_monotherapy": gdsc_summary,
        "multiomics": {
            "pdac_scrna_cra001160": sc_summary,
            "hallmark_paad_camera_fdr05": int((paad_g["CAMERA_FDR_q"] < 0.05).sum()),
            "hallmark_paad_camera_min_q": float(paad_g["CAMERA_FDR_q"].min()),
            "hallmark_paad_gsea_fdr05": int((paad_g["GSEA_FDR_q"] < 0.05).sum()),
            "hallmark_metabric_camera_fdr05": int((meta_g["CAMERA_FDR_q"] < 0.05).sum()),
            "hallmark_metabric_gsea_fdr05": int((meta_g["GSEA_FDR_q"] < 0.05).sum()),
            "hallmark_nes_spearman_rho": float(rho_gsea),
            "hallmark_nes_spearman_p": float(p_gsea),
            "rppa_n_antibodies": int(len(df_rppa)),
            "rppa_n_fdr05": rppa_sig,
            "rppa_max_abs_rho": float(df_rppa["Within_Cohort_Spearman_rho"].abs().max()),
            "cin_7cohort_n_aneu": int(cin_7["N_Patients"]),
            "cin_7cohort_delta_r2_pct": float(cin_7["OLS_Incremental_Delta_R2"]) * 100.0,
            "cin_ext_hrd_n": int(cin_ext_hrd["N_Patients"]),
            "cin_ext_aneu_n": int(cin_ext_aneu["N_Patients"]),
            "cin_ext_delta_r2_pct": float(cin_ext_aneu["OLS_Incremental_Delta_R2"]) * 100.0,
        },
    }

    out_fp = os.path.join(out_dir, "survival_and_gdsc2_summary.json")
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== SURVIVAL BH-FDR & GDSC2 REANALYSIS COMPLETE ===")
    print(f"19-Cohort Solid TCGA: Base HR*={meta_19_base['HR']:.3f} (p={meta_19_base['P_Value']:.3f}, I2={meta_19_base['I2_Pct']:.1f}%) | +MKI67 HR*={meta_19_mk['HR']:.3f} (p={meta_19_mk['P_Value']:.3f}, I2={meta_19_mk['I2_Pct']:.1f}%)")
    print(f"19-Cohort +MKI67 BH-FDR: {payload['pan_solid_tcga_19']['plus_mki67_fdr10_count_19']}/19 at q<0.10 (KIRC q={df_19_fdr.iloc[0]['MKI67_FDR_q_19']:.3f}, BLCA q={df_19_fdr.iloc[1]['MKI67_FDR_q_19']:.3f})")
    print(f"10-Cohort +MKI67 BH-FDR: {payload['secondary_10_cohort_panel']['plus_mki67_fdr05_count_10']}/10 at q<0.05 (KIRC q={df_10_fdr.iloc[0]['MKI67_FDR_q_10']:.3f}, BLCA q={df_10_fdr.iloc[1]['MKI67_FDR_q_10']:.3f}, METABRIC q={df_10_fdr.iloc[2]['MKI67_FDR_q_10']:.3f})")
    print(f"GDSC2 post-MKI67 BH-FDR (m={len(df_gdsc)}): {gdsc_summary['post_mki67_fdr10']}/{len(df_gdsc)} at q<0.10 (min q={gdsc_summary['post_mki67_min_q']:.4f})")
    print(f"Saved JSON -> {out_fp}")


if __name__ == "__main__":
    main()
