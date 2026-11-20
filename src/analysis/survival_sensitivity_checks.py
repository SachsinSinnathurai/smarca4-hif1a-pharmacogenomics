#!/usr/bin/env python3
"""Read-Only Secondary Survival Sensitivity Checks:
1. Single-Gene (SMARCA4 vs HIF1A) 10-Cohort DerSimonian-Laird Meta-Analyses
2. Complete Pan-Solid TCGA Sensitivity Meta-Analysis (8 Primary + 11 Remaining cBioPortal Cohorts)
3. MKI67 Proliferation-Adjusted Meta-Analysis (8 Primary TCGA + BRCA-METABRIC; CGGA-325 audited for MKI67 availability)
4. GSE300765 Exact Platform Audit
"""

import gzip
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.survival_lasso_meta_analysis import (
    dersimonian_laird,
    load_rsem_genes,
    parse_extra_covs,
    run_cox,
    zscore,
)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(DATA_PROC_DIR := os.path.join(PROJECT_ROOT, "data", "processed"))
CURATED_CSV = os.path.join(DATA_PROC, "curated_10cohort_clinical_survival.csv")
HE_REG_CSV = os.path.join(DATA_PROC, "High_Event_10Cohort_Treatment_Registry.csv")
CGGA_RSEM_FP = os.path.join(DATA_RAW, "external", "CGGA.mRNAseq_325.RSEM-genes.20200506.txt")
CGGA_CLIN_FP = os.path.join(DATA_RAW, "external", "CGGA.mRNAseq_325_clinical.20200506.txt")
ABS_PURITY_FP = os.path.join(DATA_RAW, "TCGA_mastercalls.abs_tables_JSedit.fixed.txt")
OUT_CSV_LEGACY = os.path.join(DATA_PROC, "survival_gene_and_mki67_sensitivity.csv")
OUT_CSV_FULL = os.path.join(DATA_PROC, "survival_full_tcga_and_mki67_sensitivity.csv")

# 14 non-cell-cycle Buffa et al. (2010) hypoxia core genes (excluding CDKN3)
BUFFA_14_GENES = [
    "ACOT7", "ADM", "ALDOA", "ENO1", "LDHA", "MIF", "MRPS17",
    "NDRG1", "P4HA1", "PGAM1", "SLC2A1", "TPI1", "TUBB6", "VEGFA",
]

COHORT_SPECS_10 = [
    ("TCGA-PAAD", ["Age", "Stage", "Tumor_Purity"]),
    ("TCGA-GBM", ["Age"]),
    ("TCGA-BRCA", ["Age", "Stage"]),
    ("CGGA-325", ["Age", "Stage", "IDH_Mutant", "MGMT_Methylated"]),
    ("TCGA-BLCA", ["Age", "Stage"]),
    ("BRCA-METABRIC", ["Age", "Stage"]),
    ("TCGA-SKCM", ["Age", "Stage"]),
    ("TCGA-KIRC", ["Age", "Stage"]),
    ("TCGA-SARC", ["Age"]),
    ("TCGA-LIHC", ["Age", "Stage"]),
]

PRIMARY_TCGA_RSEM = {
    "TCGA-PAAD": os.path.join(DATA_RAW, "cbioportal_pancan", "paad", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-BRCA": os.path.join(DATA_RAW, "cbioportal_pancan", "brca", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-SARC": os.path.join(DATA_RAW, "cbioportal_pancan", "sarc", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-BLCA": os.path.join(DATA_RAW, "cbioportal_high_mortality", "blca_tcga", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-GBM": os.path.join(DATA_RAW, "cbioportal_high_mortality", "gbm_tcga", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-KIRC": os.path.join(DATA_RAW, "cbioportal_high_mortality", "kirc_tcga", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-LIHC": os.path.join(DATA_RAW, "cbioportal_high_mortality", "lihc_tcga", "data_mrna_seq_v2_rsem.txt"),
    "TCGA-SKCM": os.path.join(DATA_RAW, "cbioportal_high_mortality", "skcm_tcga", "data_mrna_seq_v2_rsem.txt"),
}

# All remaining adult solid TCGA cohorts with real cBioPortal PanCanAtlas/TCGA clinical + full RSEM files in data/raw/
REMAINING_TCGA_SPECS = [
    ("TCGA-LUAD", os.path.join(DATA_RAW, "cbioportal_pancan", "luad"), "stage"),
    ("TCGA-LUSC", os.path.join(DATA_RAW, "cbioportal_pancan", "lusc"), "stage"),
    ("TCGA-COADREAD", os.path.join(DATA_RAW, "cbioportal_pancan", "coadread"), "stage"),
    ("TCGA-HNSC", os.path.join(DATA_RAW, "cbioportal_pancan", "hnsc"), "stage"),
    ("TCGA-STAD", os.path.join(DATA_RAW, "cbioportal_pancan", "stad"), "stage"),
    ("TCGA-OV", os.path.join(DATA_RAW, "cbioportal_high_mortality", "ov_tcga"), "clin_stage"),
    ("TCGA-UCEC", os.path.join(DATA_RAW, "cbioportal_pancan", "ucec"), "grade"),
    ("TCGA-CESC", os.path.join(DATA_RAW, "cbioportal_pancan", "cesc"), "grade"),
    ("TCGA-ESCA", os.path.join(DATA_RAW, "cbioportal_pancan", "esca"), "stage"),
    ("TCGA-THCA", os.path.join(DATA_RAW, "cbioportal_pancan", "thca"), "stage"),
    ("TCGA-PRAD", os.path.join(DATA_RAW, "cbioportal_pancan", "prad"), "age_only"),
]

ABSENT_PANCAN_COHORTS = ["KIRP", "KICH", "MESO", "UCS", "UVM", "ACC", "CHOL", "PCPG", "TGCT"]


def _parse_stage_num(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if any(x in s for x in ("NOT APPLICABLE", "NOT AVAILABLE", "UNKNOWN", "DISCREPANCY", "STAGE X")):
        return np.nan
    s = re.sub(r"^STAGE\s*", "", s)
    if s.startswith("IV") or s.startswith("4"):
        return 4.0
    if s.startswith("III") or s.startswith("3"):
        return 3.0
    if s.startswith("II") or s.startswith("2"):
        return 2.0
    if s.startswith("I") or s.startswith("1"):
        return 1.0
    return np.nan


def _parse_grade_num(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("G1", "GRADE 1", "1"):
        return 1.0
    if s in ("G2", "GRADE 2", "2"):
        return 2.0
    if s in ("G3", "GRADE 3", "3", "G4", "HIGH GRADE"):
        return 3.0
    return np.nan


def compute_ph_pvals(d_c: pd.DataFrame, duration_col: str, event_col: str, target_col: str, covars: list[str]) -> tuple[float, float, float, int]:
    cols = [duration_col, event_col, target_col] + [c for c in covars if c in d_c.columns and d_c[c].dropna().nunique() > 1]
    sub = d_c[cols].dropna().copy()
    sub = sub[sub[duration_col] > 0].reset_index(drop=True)
    cph = CoxPHFitter(penalizer=0.0).fit(sub, duration_col=duration_col, event_col=event_col)
    ph = proportional_hazard_test(cph, sub, time_transform="rank")
    risk_p = float(ph.summary.loc[target_col, "p"])
    chi2_sum = float(ph.test_statistic.sum())
    df_tot = int(len(ph.summary))
    glob_p = float(stats.chi2.sf(chi2_sum, df_tot))
    return risk_p, chi2_sum, glob_p, df_tot


def _row(
    analysis: str, model: str, cohort: str, res: dict,
    ph_risk_p: str = "", ph_global_p: str = "",
    delta_c: str | float = "", lrt_chi2: str | float = "", lrt_p: str = "",
) -> dict:
    return {
        "Analysis": analysis,
        "Model": model,
        "Cohort": cohort,
        "N": res["N"],
        "Events": res["Events"],
        "Beta": round(res["Beta"], 8) if pd.notna(res.get("Beta")) else "",
        "SE": round(res["SE"], 8) if pd.notna(res.get("SE")) else "",
        "HR": round(res["HR"], 6) if pd.notna(res.get("HR")) else "",
        "CI_Low": round(res["CI_Low"], 6) if pd.notna(res.get("CI_Low")) else "",
        "CI_High": round(res["CI_High"], 6) if pd.notna(res.get("CI_High")) else "",
        "P_Value": f"{res['P_Value']:.3e}" if pd.notna(res.get("P_Value")) else "",
        "Z_Value": "",
        "I2_Pct": "",
        "C_Index": round(res["C_Index"], 4),
        "Log_Likelihood": round(res["LL"], 4) if pd.notna(res.get("LL")) else "",
        "Delta_C_Index": round(float(delta_c), 5) if delta_c != "" else "",
        "Nested_LRT_Chi2": round(float(lrt_chi2), 3) if lrt_chi2 != "" else "",
        "Nested_LRT_P": lrt_p,
        "Covariates": res["Covariates"],
        "PH_Risk_P": ph_risk_p,
        "PH_Global_P": ph_global_p,
    }


def _meta_row(
    analysis: str, model: str, label: str, meta: dict, n_tot: int, e_tot: int,
    note: str = "DerSimonian-Laird Random-Effects Pool",
    mean_c: str | float = "", delta_c: str | float = "", lrt_chi2: str | float = "", lrt_p: str = "",
) -> dict:
    return {
        "Analysis": analysis,
        "Model": model,
        "Cohort": label,
        "N": n_tot,
        "Events": e_tot,
        "Beta": round(meta["Beta_RE"], 8),
        "SE": round(meta["SE_RE"], 8),
        "HR": round(meta["HR_RE"], 6),
        "CI_Low": round(meta["CI_Low"], 6),
        "CI_High": round(meta["CI_High"], 6),
        "P_Value": f"{meta['P_RE']:.3e}",
        "Z_Value": round(meta["Z_RE"], 4),
        "I2_Pct": round(meta["I2"], 2),
        "C_Index": round(float(mean_c), 4) if mean_c != "" else "",
        "Log_Likelihood": "",
        "Delta_C_Index": round(float(delta_c), 5) if delta_c != "" else "",
        "Nested_LRT_Chi2": round(float(lrt_chi2), 3) if lrt_chi2 != "" else "",
        "Nested_LRT_P": lrt_p,
        "Covariates": note,
        "PH_Risk_P": "",
        "PH_Global_P": "",
    }


def load_metabric_mki67() -> dict[str, float]:
    """Read real MKI67 log2 microarray intensity from BRCA-METABRIC expression matrix."""
    fp = os.path.join(DATA_RAW, "cbioportal_high_mortality", "brca_metabric", "data_mrna_illumina_microarray.txt")
    out: dict[str, float] = {}
    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
        samps = fh.readline().strip().split("\t")[2:]
        for line in fh:
            if line.startswith("MKI67\t"):
                vals = line.strip().split("\t")[2:]
                for s, v in zip(samps, vals):
                    try:
                        out[s] = float(v)
                    except ValueError:
                        pass
                break
    return out


def load_cgga_full_rsem(genes: tuple[str, ...] = ("SMARCA4", "HIF1A", "MKI67", "TOP2A")) -> pd.DataFrame:
    """Read target genes from the full 24,326-gene public CGGA.mRNAseq_325 RSEM matrix and clinical table."""
    if not os.path.exists(CGGA_RSEM_FP):
        raise FileNotFoundError(f"Full CGGA-325 RSEM matrix missing: {CGGA_RSEM_FP}")
    if not os.path.exists(CGGA_CLIN_FP):
        raise FileNotFoundError(f"Full CGGA-325 clinical file missing: {CGGA_CLIN_FP}")
    clin = pd.read_csv(CGGA_CLIN_FP, sep="\t")
    rsem = pd.read_csv(CGGA_RSEM_FP, sep="\t", index_col=0)
    missing_g = [g for g in genes if g not in rsem.index]
    if missing_g:
        raise ValueError(f"Missing genes in CGGA-325 RSEM matrix: {missing_g}")
    sub = rsem.loc[list(genes)].T.apply(pd.to_numeric, errors="coerce")
    sub.index.name = "Sample_ID"
    out = sub.reset_index()
    out = out.merge(clin[["CGGA_ID", "Grade", "IDH_mutation_status", "MGMTp_methylation_status", "OS", "Censor (alive=0; dead=1)", "Age"]],
                    left_on="Sample_ID", right_on="CGGA_ID", how="left")
    print(f"  * Loaded full CGGA.mRNAseq_325 RSEM ({rsem.shape[0]:,} genes x {rsem.shape[1]} samples; clinical N={len(clin)})")
    return out


def load_absolute_purity_maps() -> tuple[dict[str, float], dict[str, float]]:
    """Read PanCanAtlas ABSOLUTE purity calls keyed by 15-char sample barcode and 12-char patient barcode."""
    abs_df = pd.read_csv(ABS_PURITY_FP, sep="\t", usecols=["array", "purity"]).dropna(subset=["array", "purity"])
    abs_df["purity"] = pd.to_numeric(abs_df["purity"], errors="coerce")
    abs_df = abs_df.dropna(subset=["purity"]).copy()
    abs_df["s15"] = abs_df["array"].astype(str).str.slice(0, 15)
    abs_df["p12"] = abs_df["array"].astype(str).str.slice(0, 12)
    s15_map = abs_df.drop_duplicates("s15").set_index("s15")["purity"].to_dict()
    p12_map = abs_df.drop_duplicates("p12").set_index("p12")["purity"].to_dict()
    return s15_map, p12_map


def audit_knijnenburg_cin_overlap() -> None:
    """Print exact complete-case sample overlap between expression registries, Knijnenburg HRD, and PanCan aneuploidy."""
    hrd_fp = os.path.join(DATA_RAW, "TCGA.HRD_withSampleID.txt")
    f1_fp = os.path.join(DATA_PROC, "Full_TCGA_PanCan_Patient_Treatment_Registry.csv")
    f2_fp = os.path.join(DATA_PROC, "Batch_Indication_Expansion_Patient_Registry.csv")
    f3_fp = os.path.join(DATA_PROC, "High_Event_10Cohort_Treatment_Registry.csv")
    if not all(os.path.exists(p) for p in (hrd_fp, f1_fp, f2_fp, f3_fp)):
        return

    hrd_raw = pd.read_csv(hrd_fp, sep="\t", index_col=0).T.apply(pd.to_numeric, errors="coerce").dropna()
    pancan_dirs = {
        "BRCA": "brca", "COADREAD": "coadread", "LUAD": "luad", "LUSC": "lusc",
        "PRAD": "prad", "THCA": "thca", "UCEC": "ucec",
        "CESC": "cesc/cesc_tcga_pan_can_atlas_2018", "ESCA": "esca/esca_tcga_pan_can_atlas_2018",
        "HNSC": "hnsc/hnsc_tcga_pan_can_atlas_2018", "SARC": "sarc/sarc_tcga_pan_can_atlas_2018",
        "STAD": "stad/stad_tcga_pan_can_atlas_2018",
    }
    aneu_map: dict[str, float] = {}
    for sub_d in pancan_dirs.values():
        clin = pd.read_csv(os.path.join(DATA_RAW, "cbioportal_pancan", sub_d, "data_clinical_sample.txt"), sep="\t", comment="#")
        sub_c = clin[["SAMPLE_ID", "ANEUPLOIDY_SCORE"]].dropna()
        aneu_map.update(dict(zip(sub_c["SAMPLE_ID"].astype(str), sub_c["ANEUPLOIDY_SCORE"].astype(float))))

    hm_lacking = []
    for hm_d in ("blca_tcga", "gbm_tcga", "kirc_tcga", "lihc_tcga", "ov_tcga", "skcm_tcga"):
        hm_fp = os.path.join(DATA_RAW, "cbioportal_high_mortality", hm_d, "data_clinical_sample.txt")
        if os.path.exists(hm_fp):
            cols = pd.read_csv(hm_fp, sep="\t", comment="#", nrows=1).columns
            if "ANEUPLOIDY_SCORE" not in cols:
                hm_lacking.append(hm_d.replace("_tcga", "").upper())

    f1 = pd.read_csv(f1_fp, usecols=["Sample_ID", "Cohort"])
    f2 = pd.read_csv(f2_fp, usecols=["Sample_ID", "Cohort"])
    f3 = pd.read_csv(f3_fp, usecols=["Sample_ID", "Cohort"])
    f3 = f3[f3["Cohort"].str.endswith("_TCGA")].copy()
    f3["Cohort"] = f3["Cohort"].str.replace("_TCGA", "", regex=False)
    f_ext = pd.concat([f1, f2, f3], ignore_index=True).drop_duplicates("Sample_ID").copy()

    print("\n[5] KNIJNENBURG HRD & ANEUPLOIDY COMPLETE-CASE OVERLAP AUDIT")
    n_ext_c = int(f_ext["Cohort"].nunique())
    for lbl, frame in [("7-Cohort PanCan", f1.copy()), (f"Extended TCGA ({n_ext_c} cohorts)", f_ext)]:
        frame["HRD"] = frame["Sample_ID"].map(hrd_raw["HRD"])
        frame["Aneuploidy"] = frame["Sample_ID"].map(aneu_map)
        n_tot = len(frame)
        n_hrd = int(frame["HRD"].notna().sum())
        n_aneu = int(frame["Aneuploidy"].notna().sum())
        n_joint = int(frame[["HRD", "Aneuploidy"]].dropna().shape[0])
        c_hrd = int(frame.loc[frame["HRD"].notna(), "Cohort"].nunique())
        c_joint = int(frame.dropna(subset=["HRD", "Aneuploidy"])["Cohort"].nunique())
        print(f"  * {lbl:<26}: expr={n_tot:,} -> HRD={n_hrd:,} ({c_hrd} cohorts), Aneuploidy={n_aneu:,}, Joint HRD+Aneu={n_joint:,} ({c_joint} cohorts)")
    print(f"  * Legacy cBioPortal cohorts lacking ANEUPLOIDY_SCORE in sample archive ({len(hm_lacking)}): {', '.join(hm_lacking)}")


def inspect_gse300765_platform() -> dict[str, str]:
    """Read GSE300765 platform and processing metadata directly from the cached GEO series matrix."""
    fp = os.path.join(DATA_RAW, "geo_hypoxia", "GSE300765_series_matrix.txt.gz")
    info = {}
    if not os.path.exists(fp):
        return info
    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            ls = line.strip()
            if ls.startswith("!Series_platform_id"):
                info["platform_id"] = ls.split("\t", 1)[1].strip('"')
            elif ls.startswith("!Series_overall_design"):
                info["overall_design"] = ls.split("\t", 1)[1].strip('"')
            elif ls.startswith("!Sample_data_processing"):
                info["data_processing"] = ls.split("\t")[1].strip('"')
            elif ls.startswith("!series_matrix_table_begin"):
                break
    print("\n[4] GSE300765 EXACT PLATFORM AUDIT")
    print(f"  * Series_platform_id: {info.get('platform_id', '')}")
    return info


def main() -> None:
    print("READ-ONLY COMPLETE PAN-SOLID TCGA & MKI67 SURVIVAL SENSITIVITY CHECKS")
    df = pd.read_csv(CURATED_CSV)
    ex = df["Extra_Covariates"].apply(parse_extra_covs)
    df["Tumor_Purity"] = [d.get("Tumor_Purity", np.nan) for d in ex]
    df["IDH_Mutant"] = [d.get("IDH_Mutant", np.nan) for d in ex]
    df["MGMT_Methylated"] = [d.get("MGMT_Methylated", np.nan) for d in ex]
    df["Is_Male"] = [d.get("Is_Male", np.nan) for d in ex]

    rows: list[dict] = []

    # 1. Single-gene (SMARCA4_z vs HIF1A_z) vs composite across the 10 primary cohorts (N = 5,102) + 8-cohort non-breast + Schoenfeld PH
    print("\n[1] 10-COHORT PRIMARY SINGLE-GENE VS COMPOSITE META-ANALYSIS (N = 5,102)")
    predictor_specs = [
        ("SMARCA4_z_Alone", lambda d: zscore(d["SMARCA4_z"])),
        ("HIF1A_z_Alone", lambda d: zscore(d["HIF1A_z"])),
        ("Composite_SMARCA4_HIF1A", lambda d: zscore(d["SMARCA4_z"] + d["HIF1A_z"])),
    ]
    for model_name, pred_fn in predictor_specs:
        betas, ses, n_tot, e_tot = [], [], 0, 0
        comp_records = []
        for cname, covars in COHORT_SPECS_10:
            d_c = df[df["Cohort"] == cname].copy()
            d_c["Predictor"] = pred_fn(d_c)
            res = run_cox(d_c, "OS_Days", "Event", "Predictor", covars)
            ph_risk_s, ph_glob_s = "", ""
            if model_name == "Composite_SMARCA4_HIF1A":
                rp, csum, gp, dft = compute_ph_pvals(d_c, "OS_Days", "Event", "Predictor", covars)
                ph_risk_s = f"{rp:.4f}"
                ph_glob_s = f"{gp:.4f} (chi2={csum:.2f}, df={dft})"
                comp_records.append((cname, res, rp, csum, gp, dft))
            betas.append(res["Beta"]); ses.append(res["SE"])
            n_tot += res["N"]; e_tot += res["Events"]
            rows.append(_row("10-Cohort Single-Gene vs Composite", model_name, cname, res, ph_risk_p=ph_risk_s, ph_global_p=ph_glob_s))
        meta = dersimonian_laird(np.array(betas), np.array(ses))
        print(
            f"  * {model_name:<25} | k=10, N={n_tot:,} ({e_tot:,} deaths) "
            f"| HR* = {meta['HR_RE']:.3f} [{meta['CI_Low']:.3f}-{meta['CI_High']:.3f}], "
            f"Z = {meta['Z_RE']:.2f}, p = {meta['P_RE']:.2e}, I^2 = {meta['I2']:.1f}%"
        )
        rows.append(_meta_row("10-Cohort Single-Gene vs Composite", model_name, "10-Cohort DerSimonian-Laird Meta", meta, n_tot, e_tot))

        if model_name == "Composite_SMARCA4_HIF1A":
            nb_items = [x for x in comp_records if x[0] not in ("TCGA-BRCA", "BRCA-METABRIC")]
            m_nb8 = dersimonian_laird(np.array([x[1]["Beta"] for x in nb_items]), np.array([x[1]["SE"] for x in nb_items]))
            n_nb8 = sum(x[1]["N"] for x in nb_items)
            e_nb8 = sum(x[1]["Events"] for x in nb_items)
            print(
                f"  * 8-Cohort Non-Breast Primary Meta (k=8, excl. TCGA-BRCA, METABRIC, ICGC-PAAD; N={n_nb8:,}, {e_nb8:,} deaths) "
                f"| HR* = {m_nb8['HR_RE']:.3f} [{m_nb8['CI_Low']:.3f}-{m_nb8['CI_High']:.3f}], "
                f"Z = {m_nb8['Z_RE']:.2f}, p = {m_nb8['P_RE']:.2e}, I^2 = {m_nb8['I2']:.1f}%"
            )
            rows.append(_meta_row(
                "10-Cohort Single-Gene vs Composite",
                model_name,
                "8-Cohort Non-Breast Primary Meta",
                m_nb8,
                n_nb8,
                e_nb8,
                "Excl. TCGA-BRCA, BRCA-METABRIC & unadjusted ICGC/PCAWG-PAAD",
            ))

            d_icgc = df[df["Cohort"] == "ICGC/PCAWG-PAAD"].copy()
            d_icgc["Predictor"] = zscore(d_icgc["SMARCA4_z"] + d_icgc["HIF1A_z"])
            r_icgc = run_cox(d_icgc, "OS_Days", "Event", "Predictor", ["Age", "Is_Male"])
            rp_i, cs_i, gp_i, dft_i = compute_ph_pvals(d_icgc, "OS_Days", "Event", "Predictor", ["Age", "Is_Male"])
            comp_records.append(("ICGC/PCAWG-PAAD", r_icgc, rp_i, cs_i, gp_i, dft_i))
            print("\n[1B] SCHOENFELD PROPORTIONAL HAZARDS AUDIT (RANK TIME TRANSFORM)")
            for cname_ph, r_ph, rp_v, cs_v, gp_v, df_v in comp_records:
                print(f"  * {cname_ph:<16} (N={r_ph['N']:4d}) | Risk_Z PH p = {rp_v:.4f} | Global chi2({df_v}) = {cs_v:6.2f}, p = {gp_v:.4f}")
                rows.append(_row(
                    "Schoenfeld Proportional Hazards Audit",
                    "Composite_SMARCA4_HIF1A",
                    cname_ph,
                    r_ph,
                    ph_risk_p=f"{rp_v:.4f}",
                    ph_global_p=f"{gp_v:.4f} (chi2={cs_v:.2f}, df={df_v})",
                ))

    # 2. Complete 10-Cohort MKI67 Proliferation Adjustment (8 TCGA + BRCA-METABRIC + Full-Transcriptome CGGA-325)
    print("\n[2] FULL 10-COHORT MKI67 PROLIFERATION SENSITIVITY (INCLUDING 24,326-GENE CGGA-325 RSEM)")
    cgga_full = load_cgga_full_rsem(("SMARCA4", "HIF1A", "MKI67", "TOP2A")).set_index("Sample_ID")
    mb_mki_map = load_metabric_mki67()

    primary_base_res: dict[str, dict] = {}
    primary_mki_res: dict[str, dict] = {}
    primary_rsem_dfs: dict[str, pd.DataFrame] = {}

    for cname, covars in COHORT_SPECS_10:
        d_c = df[df["Cohort"] == cname].copy()
        d_c["Risk_Z"] = zscore(d_c["SMARCA4_z"] + d_c["HIF1A_z"])
        r_base = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars)
        primary_base_res[cname] = r_base

        if cname in PRIMARY_TCGA_RSEM:
            rsem_cols = ["SMARCA4", "HIF1A", "MKI67"] + BUFFA_14_GENES
            m_df = load_rsem_genes(PRIMARY_TCGA_RSEM[cname], rsem_cols, primary_only=(cname == "TCGA-PAAD"))
            for g in rsem_cols:
                sid_map = m_df.set_index("Sample_ID")[g].to_dict()
                pid_map = m_df.drop_duplicates("PID").set_index("PID")[g].to_dict()
                d_c[f"RSEM_{g}"] = d_c["Sample_ID"].map(sid_map).fillna(d_c["Sample_ID"].astype(str).str.slice(0, 12).map(pid_map))
            d_c["MKI67_z"] = zscore(d_c["RSEM_MKI67"])
            r_mki = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars + ["MKI67_z"])
            primary_mki_res[cname] = r_mki
            primary_rsem_dfs[cname] = d_c
        elif cname == "BRCA-METABRIC":
            d_c["MKI67_raw"] = d_c["Sample_ID"].map(mb_mki_map)
            d_c["MKI67_z"] = zscore(d_c["MKI67_raw"])
            r_mki = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars + ["MKI67_z"])
            primary_mki_res[cname] = r_mki
        elif cname == "CGGA-325":
            d_c["MKI67_raw"] = d_c["Sample_ID"].map(cgga_full["MKI67"].to_dict())
            d_c["TOP2A_raw"] = d_c["Sample_ID"].map(cgga_full["TOP2A"].to_dict())
            d_c["MKI67_z"] = zscore(np.log2(d_c["MKI67_raw"] + 1.0))
            d_c["MKI67_lin_z"] = zscore(d_c["MKI67_raw"])
            d_c["TOP2A_z"] = zscore(np.log2(d_c["TOP2A_raw"] + 1.0))
            r_mki = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars + ["MKI67_z"])
            r_mki_lin = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars + ["MKI67_lin_z"])
            r_top2a = run_cox(d_c, "OS_Days", "Event", "Risk_Z", covars + ["TOP2A_z"])
            primary_mki_res[cname] = r_mki
            print(
                f"  * CGGA-325 sensitivity (N={r_mki['N']}): "
                f"+MKI67(log2p1) HR={r_mki['HR']:.3f} [{r_mki['CI_Low']:.3f}-{r_mki['CI_High']:.3f}] p={r_mki['P_Value']:.2e} | "
                f"+MKI67(linear) HR={r_mki_lin['HR']:.3f} p={r_mki_lin['P_Value']:.2e} | "
                f"+TOP2A(log2p1) HR={r_top2a['HR']:.3f} [{r_top2a['CI_Low']:.3f}-{r_top2a['CI_High']:.3f}] p={r_top2a['P_Value']:.2e}"
            )
            rows.append(_row("CGGA-325 Full-Transcriptome Sensitivity", "Plus_MKI67_Linear_RSEM", "CGGA-325", r_mki_lin))
            rows.append(_row("CGGA-325 Full-Transcriptome Sensitivity", "Plus_TOP2A_Log2p1", "CGGA-325", r_top2a))

    cohorts_10_names = list(PRIMARY_TCGA_RSEM.keys()) + ["BRCA-METABRIC", "CGGA-325"]
    for cname in cohorts_10_names:
        rb, rm = primary_base_res[cname], primary_mki_res[cname]
        print(
            f"  * {cname:<14} | N={rb['N']}->{rm['N']} ({rm['Events']} deaths) "
            f"| Base HR={rb['HR']:.3f} (p={rb['P_Value']:.2e}) "
            f"| +MKI67 HR={rm['HR']:.3f} [{rm['CI_Low']:.3f}-{rm['CI_High']:.3f}] (p={rm['P_Value']:.2e})"
        )
        rows.append(_row("Primary Cohorts +MKI67 Sensitivity", "Base_No_MKI67", cname, rb))
        rows.append(_row("Primary Cohorts +MKI67 Sensitivity", "Plus_MKI67", cname, rm))

    # Pool 8 TCGA, 9 (8 TCGA + METABRIC), and full 10 primary cohorts (+MKI67 across all N=5,102)
    b0 = np.array([primary_base_res[c]["Beta"] for c in PRIMARY_TCGA_RSEM])
    s0 = np.array([primary_base_res[c]["SE"] for c in PRIMARY_TCGA_RSEM])
    b1 = np.array([primary_mki_res[c]["Beta"] for c in PRIMARY_TCGA_RSEM])
    s1 = np.array([primary_mki_res[c]["SE"] for c in PRIMARY_TCGA_RSEM])
    n_tcga8 = sum(primary_mki_res[c]["N"] for c in PRIMARY_TCGA_RSEM)
    e_tcga8 = sum(primary_mki_res[c]["Events"] for c in PRIMARY_TCGA_RSEM)
    m_tcga8_b = dersimonian_laird(b0, s0)
    m_tcga8_m = dersimonian_laird(b1, s1)
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Base_No_MKI67", "8-TCGA Base Meta", m_tcga8_b, n_tcga8, e_tcga8))
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Plus_MKI67", "8-TCGA +MKI67 Meta", m_tcga8_m, n_tcga8, e_tcga8))

    cohorts_9_mki = list(PRIMARY_TCGA_RSEM.keys()) + ["BRCA-METABRIC"]
    m9_b = dersimonian_laird(
        np.array([primary_base_res[c]["Beta"] for c in cohorts_9_mki]),
        np.array([primary_base_res[c]["SE"] for c in cohorts_9_mki]),
    )
    m9_m = dersimonian_laird(
        np.array([primary_mki_res[c]["Beta"] for c in cohorts_9_mki]),
        np.array([primary_mki_res[c]["SE"] for c in cohorts_9_mki]),
    )
    n9 = sum(primary_mki_res[c]["N"] for c in cohorts_9_mki)
    e9 = sum(primary_mki_res[c]["Events"] for c in cohorts_9_mki)
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Base_No_MKI67", "9-Cohort (8 TCGA + METABRIC) Base Meta", m9_b, n9, e9))
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Plus_MKI67", "9-Cohort (8 TCGA + METABRIC) +MKI67 Meta", m9_m, n9, e9))

    b10_0 = np.array([primary_base_res[c]["Beta"] for c in cohorts_10_names])
    s10_0 = np.array([primary_base_res[c]["SE"] for c in cohorts_10_names])
    b10_1 = np.array([primary_mki_res[c]["Beta"] for c in cohorts_10_names])
    s10_1 = np.array([primary_mki_res[c]["SE"] for c in cohorts_10_names])
    n10 = sum(primary_mki_res[c]["N"] for c in cohorts_10_names)
    e10 = sum(primary_mki_res[c]["Events"] for c in cohorts_10_names)
    m10_b = dersimonian_laird(b10_0, s10_0)
    m10_m = dersimonian_laird(b10_1, s10_1)
    print(f"  -> 8-TCGA +MKI67 Meta (N={n_tcga8:,}, {e_tcga8:,} deaths): HR* = {m_tcga8_m['HR_RE']:.3f} [{m_tcga8_m['CI_Low']:.3f}-{m_tcga8_m['CI_High']:.3f}], Z={m_tcga8_m['Z_RE']:.2f}, p={m_tcga8_m['P_RE']:.2e}, I^2={m_tcga8_m['I2']:.1f}%")
    print(f"  -> Full 10-Cohort +MKI67 Meta (N={n10:,}, {e10:,} deaths): HR* = {m10_m['HR_RE']:.3f} [{m10_m['CI_Low']:.3f}-{m10_m['CI_High']:.3f}], Z={m10_m['Z_RE']:.2f}, p={m10_m['P_RE']:.2e}, I^2={m10_m['I2']:.1f}% (vs Base HR*={m10_b['HR_RE']:.3f}, p={m10_b['P_RE']:.2e}, I^2={m10_b['I2']:.1f}%)")
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Base_No_MKI67", "10-Cohort Full Base Meta", m10_b, n10, e10))
    rows.append(_meta_row("Primary Cohorts +MKI67 Sensitivity", "Plus_MKI67", "10-Cohort Full +MKI67 Meta", m10_m, n10, e10, "True MKI67 across all 10 primary cohorts (8 TCGA + METABRIC + CGGA-325)"))

    # 2B. METABRIC Missingness Bias Check (N = 1,979 vs N = 1,466)
    print("\n[2B] METABRIC STAGE-MISSINGNESS SENSITIVITY (N = 1,979 VS. N = 1,466)")
    he_reg = pd.read_csv(HE_REG_CSV)
    mb_all = he_reg[he_reg["Cohort"] == "BRCA_METABRIC"].copy()
    mb_1979 = mb_all[mb_all["OS_Days"] > 0].dropna(subset=["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
    mb_1979["Risk_Z"] = zscore(mb_1979["SMARCA4_zScore"] + mb_1979["HIF1A_zScore"])
    mb_1466 = mb_1979.dropna(subset=["Stage_Numeric"]).copy()
    mb_1466["Risk_Z"] = zscore(mb_1466["SMARCA4_zScore"] + mb_1466["HIF1A_zScore"])
    for m_lbl, m_df, m_cov in [
        ("METABRIC_All_OS_AgeOnly", mb_1979, ["Age"]),
        ("METABRIC_StageComplete_AgeOnly", mb_1466, ["Age"]),
        ("METABRIC_StageComplete_AgePlusStage", mb_1466, ["Age", "Stage_Numeric"]),
    ]:
        r_mb = run_cox(m_df, "OS_Days", "Event", "Risk_Z", m_cov)
        print(f"  * {m_lbl:<36} | N={r_mb['N']} ({r_mb['Events']} deaths) | HR={r_mb['HR']:.3f} [{r_mb['CI_Low']:.3f}-{r_mb['CI_High']:.3f}], p={r_mb['P_Value']:.2e}, C={r_mb['C_Index']:.3f}")
        rows.append(_row("METABRIC Stage-Missingness Sensitivity", m_lbl, "BRCA-METABRIC", r_mb))

    # 2C. PanCanAtlas ABSOLUTE Purity Adjustment in TCGA-KIRC & TCGA-BLCA
    print("\n[2C] PANCANATLAS ABSOLUTE PURITY SENSITIVITY IN TCGA-KIRC & TCGA-BLCA")
    abs_s15, abs_p12 = load_absolute_purity_maps()
    for cn_p in ["TCGA-KIRC", "TCGA-BLCA"]:
        dp_c = primary_rsem_dfs[cn_p].copy()
        dp_c["Purity"] = dp_c["Sample_ID"].astype(str).str.slice(0, 15).map(abs_s15).fillna(
            dp_c["Sample_ID"].astype(str).str.slice(0, 12).map(abs_p12)
        )
        r_pur = run_cox(dp_c, "OS_Days", "Event", "Risk_Z", ["Age", "Stage", "Purity"])
        r_pmk = run_cox(dp_c, "OS_Days", "Event", "Risk_Z", ["Age", "Stage", "MKI67_z", "Purity"])
        print(
            f"  * {cn_p:<10} (N={r_pur['N']}/{len(dp_c)}, {r_pur['Events']} deaths) "
            f"| +Purity HR={r_pur['HR']:.3f} [{r_pur['CI_Low']:.3f}-{r_pur['CI_High']:.3f}] p={r_pur['P_Value']:.2e} "
            f"| +MKI67+Purity HR={r_pmk['HR']:.3f} [{r_pmk['CI_Low']:.3f}-{r_pmk['CI_High']:.3f}] p={r_pmk['P_Value']:.2e}"
        )
        rows.append(_row("TCGA-KIRC & BLCA ABSOLUTE Purity Sensitivity", "Plus_ABSOLUTE_Purity", cn_p, r_pur))
        rows.append(_row("TCGA-KIRC & BLCA ABSOLUTE Purity Sensitivity", "Plus_MKI67_and_ABSOLUTE_Purity", cn_p, r_pmk))

    # 3. Remaining Adult Solid TCGA PanCanAtlas Cohorts (+ Complete 22-Cohort Schoenfeld PH Audit + Buffa-14 Cache)
    print("\n[3] COMPLETE PAN-SOLID TCGA SENSITIVITY & REMAINING 11 SCHOENFELD PH TESTS")
    rem_base_res: dict[str, dict] = {}
    rem_mki_res: dict[str, dict] = {}
    rem_rsem_dfs: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for cname, dp, mode in REMAINING_TCGA_SPECS:
        cl = pd.read_csv(os.path.join(dp, "data_clinical_patient.txt"), sep="\t", comment="#", low_memory=False)
        sp = pd.read_csv(os.path.join(dp, "data_clinical_sample.txt"), sep="\t", comment="#", low_memory=False)
        m = cl.merge(sp, on="PATIENT_ID", how="inner", suffixes=("", "_SAMP"))
        m["OS_Days"] = pd.to_numeric(m["OS_MONTHS"], errors="coerce") * 30.4375
        os_s = m["OS_STATUS"].astype(str)
        m["Event"] = np.where(os_s.str.startswith("1") | os_s.str.contains("DECEASED", case=False), 1,
                     np.where(os_s.str.startswith("0") | os_s.str.contains("LIVING", case=False), 0, np.nan))
        m["Age"] = pd.to_numeric(m["AGE"], errors="coerce")
        covs = ["Age"]
        if mode == "stage":
            m["Stage_Num"] = m["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(_parse_stage_num)
            covs.append("Stage_Num")
        elif mode == "clin_stage":
            m["Stage_Num"] = m["CLINICAL_STAGE"].apply(_parse_stage_num)
            covs.append("Stage_Num")
        elif mode == "grade":
            gcol = "GRADE" if "GRADE" in m.columns else "GRADE_SAMP"
            m["Grade_Num"] = m[gcol].apply(_parse_grade_num)
            covs.append("Grade_Num")

        rsem_cols = ["SMARCA4", "HIF1A", "MKI67"] + BUFFA_14_GENES
        rsem = load_rsem_genes(os.path.join(dp, "data_mrna_seq_v2_rsem.txt"), rsem_cols, primary_only=False)
        m = m.merge(rsem[["Sample_ID"] + rsem_cols], left_on="SAMPLE_ID", right_on="Sample_ID", how="inner")
        m = m[m["Sample_ID"].astype(str).str.slice(13, 15) == "01"].copy()
        m = m[m["OS_Days"] > 0].dropna(subset=["OS_Days", "Event"] + rsem_cols + covs).copy()
        m["Risk_Z"] = zscore(zscore(m["SMARCA4"]) + zscore(m["HIF1A"]))
        m["MKI67_z"] = zscore(m["MKI67"])

        rb = run_cox(m, "OS_Days", "Event", "Risk_Z", covs)
        rm = run_cox(m, "OS_Days", "Event", "Risk_Z", covs + ["MKI67_z"])
        rp_r, cs_r, gp_r, df_r = compute_ph_pvals(m, "OS_Days", "Event", "Risk_Z", covs)
        rem_base_res[cname] = rb
        rem_mki_res[cname] = rm
        rem_rsem_dfs[cname] = (m, covs)
        print(
            f"  * {cname:<13} | N={rb['N']:4d} ({rb['Events']:3d} deaths) | cov={covs} "
            f"| Base HR={rb['HR']:.3f} [{rb['CI_Low']:.3f}-{rb['CI_High']:.3f}] p={rb['P_Value']:.2e} "
            f"| +MKI67 HR={rm['HR']:.3f} p={rm['P_Value']:.2e} | PH risk p={rp_r:.4f} (glob p={gp_r:.4f})"
        )
        rows.append(_row("Pan-Solid TCGA Sensitivity", "PanTCGA_Base", cname, rb, ph_risk_p=f"{rp_r:.4f}", ph_global_p=f"{gp_r:.4f} (chi2={cs_r:.2f}, df={df_r})"))
        rows.append(_row("Pan-Solid TCGA Sensitivity", "PanTCGA_Plus_MKI67", cname, rm))
        rows.append(_row("Schoenfeld Proportional Hazards Audit", "Composite_SMARCA4_HIF1A", cname, rb, ph_risk_p=f"{rp_r:.4f}", ph_global_p=f"{gp_r:.4f} (chi2={cs_r:.2f}, df={df_r})"))

    # Pool: 11 remaining, 18 (with Stage/Grade), 19 (all available solid TCGA)
    all_19 = list(PRIMARY_TCGA_RSEM.keys()) + [c for c, _, _ in REMAINING_TCGA_SPECS]
    all_18_stg = [c for c in all_19 if c != "TCGA-PRAD"]
    rem_11 = [c for c, _, _ in REMAINING_TCGA_SPECS]

    merged_base = {**{c: primary_base_res[c] for c in PRIMARY_TCGA_RSEM}, **rem_base_res}
    merged_mki = {**{c: primary_mki_res[c] for c in PRIMARY_TCGA_RSEM}, **rem_mki_res}

    for grp_label, clist in [
        ("11-Remaining-TCGA Meta", rem_11),
        ("18-Solid-TCGA (w/ Stage or Grade) Meta", all_18_stg),
        ("19-All-Available-Solid-TCGA Meta", all_19),
    ]:
        bb = np.array([merged_base[c]["Beta"] for c in clist])
        sb = np.array([merged_base[c]["SE"] for c in clist])
        bm = np.array([merged_mki[c]["Beta"] for c in clist])
        sm = np.array([merged_mki[c]["SE"] for c in clist])
        nt = sum(merged_base[c]["N"] for c in clist)
        et = sum(merged_base[c]["Events"] for c in clist)
        mb = dersimonian_laird(bb, sb)
        mm = dersimonian_laird(bm, sm)
        print(
            f"  -> {grp_label:<38} (N={nt:,}, {et:,} deaths) | "
            f"Base HR*={mb['HR_RE']:.3f} [{mb['CI_Low']:.3f}-{mb['CI_High']:.3f}] p={mb['P_RE']:.2e} I^2={mb['I2']:.1f}% | "
            f"+MKI67 HR*={mm['HR_RE']:.3f} [{mm['CI_Low']:.3f}-{mm['CI_High']:.3f}] p={mm['P_RE']:.2e} I^2={mm['I2']:.1f}%"
        )
        rows.append(_meta_row("Pan-Solid TCGA Sensitivity", "PanTCGA_Base", grp_label, mb, nt, et))
        rows.append(_meta_row("Pan-Solid TCGA Sensitivity", "PanTCGA_Plus_MKI67", grp_label, mm, nt, et))

    # 4. Buffa-14 Non-Cell-Cycle Hypoxia Activity Benchmark across 19 Solid TCGA Cohorts (N = 7,875)
    print("\n[4] BUFFA-14 HYPOXIA ACTIVITY BENCHMARK VS HIF1A ACROSS 19 SOLID TCGA COHORTS")
    cov_map_10 = dict(COHORT_SPECS_10)
    buffa_models: dict[str, dict[str, dict]] = {
        "Buffa14_Alone_Base": {},
        "Buffa14_Alone_PlusMKI67": {},
        "SMARCA4_Buffa14_Base": {},
        "SMARCA4_Buffa14_PlusMKI67": {},
        "SMARCA4_HIF1A_Base": {},
        "SMARCA4_HIF1A_PlusMKI67": {},
    }
    nested_stats: dict[str, dict[str, dict]] = {"Base": {}, "PlusMKI67": {}}

    def _fit_joint_ll_c(d_in: pd.DataFrame, preds: list[str], extra_covs: list[str]) -> tuple[float, float]:
        cols = ["OS_Days", "Event"] + preds + [c for c in extra_covs if c in d_in.columns and d_in[c].dropna().nunique() > 1]
        sub = d_in[cols].dropna().copy()
        sub = sub[sub["OS_Days"] > 0]
        cph = CoxPHFitter(penalizer=0.0).fit(sub, duration_col="OS_Days", event_col="Event")
        return float(cph.log_likelihood_), float(cph.concordance_index_)

    for cname in all_19:
        if cname in PRIMARY_TCGA_RSEM:
            df_b = primary_rsem_dfs[cname].copy()
            covs_b = cov_map_10[cname]
            sm_col, hf_col = "RSEM_SMARCA4", "RSEM_HIF1A"
            bf_cols = [f"RSEM_{g}" for g in BUFFA_14_GENES]
        else:
            df_raw_r, covs_b = rem_rsem_dfs[cname]
            df_b = df_raw_r.copy()
            sm_col, hf_col = "SMARCA4", "HIF1A"
            bf_cols = BUFFA_14_GENES

        df_b["SM_z"] = zscore(df_b[sm_col])
        df_b["HF_z"] = zscore(df_b[hf_col])
        bf_z_df = pd.DataFrame({g: zscore(df_b[g]) for g in bf_cols}, index=df_b.index)
        df_b["Buffa14_z"] = zscore(bf_z_df.mean(axis=1))
        df_b["SMARCA4_Buffa14"] = zscore(df_b["SM_z"] + df_b["Buffa14_z"])
        df_b["SMARCA4_HIF1A_RSEM"] = zscore(df_b["SM_z"] + df_b["HF_z"])

        for suffix, extra_c in [("", []), ("PlusMKI67", ["MKI67_z"])]:
            tag = "Base" if suffix == "" else "PlusMKI67"
            k_sh = f"SMARCA4_HIF1A_{tag}"
            k_buf = f"Buffa14_Alone_{tag}"
            k_sbuf = f"SMARCA4_Buffa14_{tag}"
            r_sh = run_cox(df_b, "OS_Days", "Event", "SMARCA4_HIF1A_RSEM", covs_b + extra_c)
            r_buf = run_cox(df_b, "OS_Days", "Event", "Buffa14_z", covs_b + extra_c)
            r_sbuf = run_cox(df_b, "OS_Days", "Event", "SMARCA4_Buffa14", covs_b + extra_c)
            ll_joint, c_joint = _fit_joint_ll_c(df_b, ["SMARCA4_HIF1A_RSEM", "Buffa14_z"], covs_b + extra_c)
            chi2_add_buf = max(0.0, 2.0 * (ll_joint - r_sh["LL"]))
            chi2_add_sh = max(0.0, 2.0 * (ll_joint - r_buf["LL"]))
            p_add_buf = float(stats.chi2.sf(chi2_add_buf, df=1))
            buffa_models[k_sh][cname] = r_sh
            buffa_models[k_buf][cname] = r_buf
            buffa_models[k_sbuf][cname] = r_sbuf
            nested_stats[tag][cname] = {
                "N": r_sh["N"],
                "C_SH": r_sh["C_Index"],
                "C_Buf": r_buf["C_Index"],
                "C_SBuf": r_sbuf["C_Index"],
                "C_Joint": c_joint,
                "dC_Buf_vs_SH": r_buf["C_Index"] - r_sh["C_Index"],
                "dC_SBuf_vs_SH": r_sbuf["C_Index"] - r_sh["C_Index"],
                "dC_Joint_vs_SH": c_joint - r_sh["C_Index"],
                "Chi2_Add_Buf_Over_SH": chi2_add_buf,
                "Chi2_Add_SH_Over_Buf": chi2_add_sh,
            }
            rows.append(_row("Buffa-14 Hypoxia Activity Benchmark", k_sh, cname, r_sh))
            rows.append(_row(
                "Buffa-14 Hypoxia Activity Benchmark", k_buf, cname, r_buf,
                delta_c=r_buf["C_Index"] - r_sh["C_Index"],
                lrt_chi2=chi2_add_buf,
                lrt_p=f"{p_add_buf:.3e}",
            ))
            rows.append(_row(
                "Buffa-14 Hypoxia Activity Benchmark", k_sbuf, cname, r_sbuf,
                delta_c=r_sbuf["C_Index"] - r_sh["C_Index"],
            ))

    prim_8 = list(PRIMARY_TCGA_RSEM.keys())
    for grp_lbl, clist in [
        ("8-Primary-TCGA", prim_8),
        ("11-Remaining-TCGA", rem_11),
        ("19-All-Solid-TCGA", all_19),
    ]:
        print(f"  -- {grp_lbl} --")
        for tag in ["Base", "PlusMKI67"]:
            nt = sum(buffa_models[f"SMARCA4_HIF1A_{tag}"][c]["N"] for c in clist)
            et = sum(buffa_models[f"SMARCA4_HIF1A_{tag}"][c]["Events"] for c in clist)
            weights_n = np.array([nested_stats[tag][c]["N"] for c in clist], dtype=float)
            mean_c_sh = float(np.average([nested_stats[tag][c]["C_SH"] for c in clist], weights=weights_n))
            mean_c_buf = float(np.average([nested_stats[tag][c]["C_Buf"] for c in clist], weights=weights_n))
            mean_c_sbuf = float(np.average([nested_stats[tag][c]["C_SBuf"] for c in clist], weights=weights_n))
            mean_c_joint = float(np.average([nested_stats[tag][c]["C_Joint"] for c in clist], weights=weights_n))
            chi2_sum_add_buf = float(sum(nested_stats[tag][c]["Chi2_Add_Buf_Over_SH"] for c in clist))
            p_sum_add_buf = float(stats.chi2.sf(chi2_sum_add_buf, df=len(clist)))
            chi2_sum_add_sh = float(sum(nested_stats[tag][c]["Chi2_Add_SH_Over_Buf"] for c in clist))
            p_sum_add_sh = float(stats.chi2.sf(chi2_sum_add_sh, df=len(clist)))

            for mkey, mc, dc, lchi, lp in [
                (f"SMARCA4_HIF1A_{tag}", mean_c_sh, "", "", ""),
                (f"SMARCA4_Buffa14_{tag}", mean_c_sbuf, mean_c_sbuf - mean_c_sh, "", ""),
                (f"Buffa14_Alone_{tag}", mean_c_buf, mean_c_buf - mean_c_sh, chi2_sum_add_buf, f"{p_sum_add_buf:.3e}"),
            ]:
                bb = np.array([buffa_models[mkey][c]["Beta"] for c in clist])
                sb = np.array([buffa_models[mkey][c]["SE"] for c in clist])
                dm = dersimonian_laird(bb, sb)
                print(
                    f"     * {mkey:<26} (N={nt:,}, {et:,} deaths): HR*={dm['HR_RE']:.3f} [{dm['CI_Low']:.3f}-{dm['CI_High']:.3f}], "
                    f"Z={dm['Z_RE']:+.2f}, p={dm['P_RE']:.2e}, I^2={dm['I2']:.1f}%, N-wt C={mc:.4f}"
                    + (f" (dC={dc:+.4f})" if dc != "" else "")
                )
                rows.append(_meta_row(
                    "Buffa-14 Hypoxia Activity Benchmark", mkey, f"{grp_lbl} Meta", dm, nt, et,
                    note=f"DerSimonian-Laird 14-gene Buffa vs HIF1A ({len(clist)}-cohort nested LRT Add Buffa chi2({len(clist)})={chi2_sum_add_buf:.2f} p={p_sum_add_buf:.2e}; Add SH chi2={chi2_sum_add_sh:.2f} p={p_sum_add_sh:.2e})",
                    mean_c=mc, delta_c=dc, lrt_chi2=lchi, lrt_p=lp,
                ))
            print(
                f"     -> [{grp_lbl} {tag}] Summed Nested LRT (+Buffa14 over SMARCA4+HIF1A): chi2({len(clist)})={chi2_sum_add_buf:.2f}, p={p_sum_add_buf:.2e}, dC_joint={mean_c_joint - mean_c_sh:+.4f}, dC_Buffa_vs_SH={mean_c_buf - mean_c_sh:+.4f}, dC_SBuffa_vs_SH={mean_c_sbuf - mean_c_sh:+.4f} | (+SMARCA4+HIF1A over Buffa14): chi2({len(clist)})={chi2_sum_add_sh:.2f}, p={p_sum_add_sh:.2e}"
            )

    inspect_gse300765_platform()
    audit_knijnenburg_cin_overlap()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV_FULL, index=False)
    df_out.to_csv(OUT_CSV_LEGACY, index=False)
    print(f"\nSaved ({len(df_out)} rows) -> {OUT_CSV_FULL}")
    print(f"Synced -> {OUT_CSV_LEGACY}")


if __name__ == "__main__":
    main()
