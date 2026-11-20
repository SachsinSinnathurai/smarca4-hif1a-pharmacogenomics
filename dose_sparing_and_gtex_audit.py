#!/usr/bin/env python3
"""Translational Pharmacology & Harmonized UCSC Xena Toil Healthy-Organ Counter-Screen Audit (Strict Zero-Fabrication):
1. Harmonized UCSC Xena Toil RNA-seq recompute (https://toil.xenahubs.net, TcgaTargetGtex_rsem_gene_tpm,
   log2(tpm + 0.001) back-transformed via 2**y - 0.001) comparing TCGA Primary Solid Tumors against
   GTEx Heart - Left Ventricle, Heart - Atrial Appendage, Liver, and Whole Blood for MCL1, BAK1, BRD4, SMARCA4.
2. Empirical GDSC combination concentration grid bounds (D1-D7) and monotherapy IC50
   distributions across unique solid-tumor cell lines from gdsc-007--010 and sandpiper-01.
3. GDSC2 monotherapy empirical 1 - AUC (bounded [0, 1]) and screening ceiling audit.
4. Verifiable Phase I total plasma Cmax benchmarks from Hamilton et al. (2023, PMC10544002)
   alongside explicit documentation of unpublished clinical PK and 7x7 well-level gaps.
"""

import gzip
import io
import json
import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_CSV = os.path.join(DATA_PROC, "dose_sparing_dri_pk_gtex.csv")
TOIL_CSV = os.path.join(DATA_PROC, "toil_tcga_vs_gtex_tpm.csv")

GDSC_MATRIX_FILES = [
    "gdsc-007_matrix_results.csv",
    "gdsc-008_matrix_results.csv",
    "gdsc-009_matrix_results.csv",
    "gdsc-010_matrix_results.csv",
    "sandpiper-01_matrix_results.csv",
]

XENA_HUB_DATA_URL = "https://toil.xenahubs.net/data/"
XENA_PHENO_URL = "https://toil.xenahubs.net/download/TcgaTargetGTEX_phenotype.txt.gz"
TOIL_DATASET = "TcgaTargetGtex_rsem_gene_tpm"

# Verified GENCODE v23 IDs from probeMap/gencode.v23.annotation.gene.probemap
TOIL_PROBES = [
    ("MCL1", "ENSG00000143384.12"),
    ("BAK1", "ENSG00000030110.12"),
    ("BRD4", "ENSG00000141867.17"),
    ("SMARCA4", "ENSG00000127616.17"),
]

GTEX_ORGANS = [
    "Heart - Left Ventricle",
    "Heart - Atrial Appendage",
    "Liver",
    "Whole Blood",
]


def _xena_fetch_matrix(sample_ids: list[str], probe_ids: list[str], chunk_size: int = 800) -> dict[str, np.ndarray]:
    """Query UCSC Xena Toil hub via HTTP POST in chunks and return per-probe log2(TPM+0.001) arrays."""
    probe_str = " ".join(f'"{p}"' for p in probe_ids)
    accum = {p: [] for p in probe_ids}
    for start in range(0, len(sample_ids), chunk_size):
        chunk = sample_ids[start : start + chunk_size]
        samp_str = " ".join(f'"{s}"' for s in chunk)
        query = f'(fetch [{{:table "{TOIL_DATASET}" :columns [{probe_str}] :samples [{samp_str}]}}])'
        req = urllib.request.Request(
            XENA_HUB_DATA_URL,
            data=query.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        mat = json.loads(raw)
        if not isinstance(mat, list) or len(mat) != len(probe_ids):
            raise RuntimeError(f"Unexpected Xena fetch payload shape for chunk at {start}: len={len(mat) if isinstance(mat, list) else type(mat)}")
        for pid, row_vals in zip(probe_ids, mat):
            accum[pid].extend([np.nan if (v is None or v == "NaN") else float(v) for v in row_vals])
    return {p: np.array(vals, dtype=float) for p, vals in accum.items()}


def fetch_toil_tcga_vs_gtex() -> pd.DataFrame:
    """Fetch harmonized UCSC Xena Toil log2(TPM+0.001) across TCGA primary solid tumors vs GTEx organs."""
    ph_gz = urllib.request.urlopen(XENA_PHENO_URL, timeout=60).read()
    ph = pd.read_csv(io.BytesIO(gzip.decompress(ph_gz)), sep="\t", encoding="latin1")
    tcga_solid_ids = ph[
        (ph["_study"] == "TCGA")
        & (ph["_sample_type"] == "Primary Tumor")
        & (ph["detailed_category"] != "Diffuse Large B-Cell Lymphoma")
    ]["sample"].dropna().astype(str).tolist()

    group_specs = [("TCGA_Primary_Solid", "TCGA", "Primary Tumor (excl. DLBCL/LAML)", tcga_solid_ids)]
    for org in GTEX_ORGANS:
        g_ids = ph[
            (ph["_study"] == "GTEX")
            & (ph["_sample_type"] == "Normal Tissue")
            & (ph["detailed_category"] == org)
        ]["sample"].dropna().astype(str).tolist()
        group_specs.append((org, "GTEX", "Normal Tissue", g_ids))

    probe_ids = [pid for _, pid in TOIL_PROBES]
    group_data: dict[str, dict[str, np.ndarray]] = {}
    for g_name, _, _, s_ids in group_specs:
        fetched = _xena_fetch_matrix(s_ids, probe_ids)
        group_data[g_name] = fetched

    rows = []
    for gene, pid in TOIL_PROBES:
        log_tum = group_data["TCGA_Primary_Solid"][pid]
        log_tum = log_tum[~np.isnan(log_tum)]
        tpm_tum = np.maximum(0.0, np.power(2.0, log_tum) - 0.001)
        tum_med = float(np.median(tpm_tum))

        for g_name, study, stype, _ in group_specs:
            vals_log = group_data[g_name][pid]
            vals_log = vals_log[~np.isnan(vals_log)]
            vals_tpm = np.maximum(0.0, np.power(2.0, vals_log) - 0.001)
            med_log = float(np.median(vals_log))
            q25_log, q75_log = float(np.percentile(vals_log, 25)), float(np.percentile(vals_log, 75))
            med_tpm = float(np.median(vals_tpm))
            q25_tpm, q75_tpm = float(np.percentile(vals_tpm, 25)), float(np.percentile(vals_tpm, 75))
            mean_tpm = float(np.mean(vals_tpm))
            if g_name == "TCGA_Primary_Solid":
                ratio = 1.0
                mwu_p = np.nan
            else:
                ratio = tum_med / med_tpm if med_tpm > 0 else np.nan
                _, mwu_p = stats.mannwhitneyu(vals_tpm, tpm_tum, alternative="two-sided")
            rows.append({
                "Gene": gene,
                "Gencode_ID": pid,
                "Group": g_name,
                "Study": study,
                "Sample_Type": stype,
                "N_Samples": int(len(vals_tpm)),
                "Median_Log2_TPMp001": round(med_log, 4),
                "IQR_Log2_Low": round(q25_log, 4),
                "IQR_Log2_High": round(q75_log, 4),
                "Median_Linear_TPM": round(med_tpm, 4),
                "IQR_TPM_Low": round(q25_tpm, 4),
                "IQR_TPM_High": round(q75_tpm, 4),
                "Mean_Linear_TPM": round(mean_tpm, 4),
                "TCGA_Solid_to_GTEx_Ratio": round(float(ratio), 4),
                "Wilcoxon_MWU_P_vs_TCGA_Solid": float(mwu_p) if pd.notna(mwu_p) else np.nan,
                "Source": "UCSC Xena Toil Recompute (https://toil.xenahubs.net, TcgaTargetGtex_rsem_gene_tpm)",
            })
    df_toil = pd.DataFrame(rows)
    df_toil.to_csv(TOIL_CSV, index=False)
    print(f"  Saved harmonized Toil table ({len(df_toil)} rows) -> {TOIL_CSV}")
    return df_toil


def resolve_gdsc_path(filename: str) -> str:
    p = os.path.join(DATA_RAW, "gdsc", filename)
    return p if os.path.exists(p) else os.path.join(PROJECT_ROOT, filename)


def load_solid_depmap():
    df_m = pd.read_csv(os.path.join(DATA_RAW, "Model.csv"))
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
    s2m = {
        str(r["SangerModelID"]): str(r["ModelID"])
        for _, r in df_m.iterrows()
        if pd.notna(r.get("SangerModelID")) and pd.notna(r.get("ModelID"))
    }

    expr_fp = os.path.join(DATA_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    eh = pd.read_csv(expr_fp, nrows=1).columns
    genes = ["MCL1", "BAK1", "BRD4", "SMARCA4"]
    col_map = {"ModelID": "ModelID", "IsDefaultEntryForModel": "IsDefaultEntryForModel"}
    use_cols = ["ModelID", "IsDefaultEntryForModel"]
    for g in genes:
        col = next(c for c in eh if c.split(" ")[0] == g)
        use_cols.append(col)
        col_map[col] = g
    df_e = pd.read_csv(expr_fp, usecols=use_cols).rename(columns=col_map)
    df_e = df_e[df_e["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").dropna(subset=genes).copy()
    df_e["ModelID"] = df_e["ModelID"].astype(str)
    df_e = df_e[df_e["ModelID"].isin(solid_models)].copy()
    for g in genes:
        df_e[f"{g}_TPM"] = np.power(2.0, df_e[g].astype(float)) - 1.0
    return solid_models, c2m, s2m, df_e


def main():
    os.makedirs(DATA_PROC, exist_ok=True)
    solid_models, c2m, s2m, df_expr = load_solid_depmap()
    rows = []

    print("1. HARMONIZED UCSC XENA TOIL: TCGA PRIMARY SOLID TUMORS VS. GTEX HEALTHY ORGANS")
    df_toil = fetch_toil_tcga_vs_gtex()
    for gene in ["MCL1", "BAK1", "BRD4", "SMARCA4"]:
        sub_g = df_toil[df_toil["Gene"] == gene]
        tum_row = sub_g[sub_g["Group"] == "TCGA_Primary_Solid"].iloc[0]
        tum_med = float(tum_row["Median_Linear_TPM"])
        tum_log_med = float(tum_row["Median_Log2_TPMp001"])
        n_tum = int(tum_row["N_Samples"])
        org_map = sub_g.set_index("Group")["Median_Linear_TPM"].to_dict()
        print(
            f"  * {gene:<8} ({tum_row['Gencode_ID']}): Toil TCGA Primary Solid median = {tum_med:6.2f} TPM "
            f"(IQR {tum_row['IQR_TPM_Low']:.2f}-{tum_row['IQR_TPM_High']:.2f}, n={n_tum:,}) | "
            f"Heart LV={org_map['Heart - Left Ventricle']:.2f}, Heart AA={org_map['Heart - Atrial Appendage']:.2f}, "
            f"Liver={org_map['Liver']:.2f}, Blood={org_map['Whole Blood']:.2f}"
        )
        for tissue_label in GTEX_ORGANS:
            org_row = sub_g[sub_g["Group"] == tissue_label].iloc[0]
            gtex_tpm = float(org_row["Median_Linear_TPM"])
            ratio = float(org_row["TCGA_Solid_to_GTEx_Ratio"])
            rows.append({
                "Audit_Section": "Toil_TCGA_Solid_vs_GTEx",
                "Gene_or_Drug": gene,
                "Subgroup_or_Tissue": tissue_label,
                "Metric": "Median_Linear_TPM",
                "Value": round(gtex_tpm, 4),
                "Comparator_DepMap_Solid_Median_TPM": round(tum_med, 4),
                "Comparator_DepMap_Solid_IQR": f"TCGA IQR {tum_row['IQR_TPM_Low']:.2f}-{tum_row['IQR_TPM_High']:.2f} (n={n_tum:,}); GTEx n={int(org_row['N_Samples'])}; MWU p={org_row['Wilcoxon_MWU_P_vs_TCGA_Solid']:.2e}",
                "Comparator_DepMap_Solid_Log2p1": round(tum_log_med, 4),
                "Tumor_to_Organ_Ratio": round(ratio, 4),
                "N": int(org_row["N_Samples"]),
                "Source": f"UCSC Xena Toil Recompute ({tum_row['Gencode_ID']})",
            })

    print("\n2. GDSC COMBINATION D1 CONCENTRATION BOUNDS & MONOTHERAPY IC50s (SOLID LINES)")
    dfs = []
    for fn in GDSC_MATRIX_FILES:
        p = resolve_gdsc_path(fn)
        if os.path.exists(p):
            dfs.append(
                pd.read_csv(
                    p,
                    usecols=[
                        "COSMIC_ID", "lib1_name", "lib1_minc", "lib1_maxc", "lib1_IC50_ln",
                        "lib2_name", "lib2_minc", "lib2_maxc", "lib2_IC50_ln", "Bliss_window",
                    ],
                )
            )
    df_gdsc = pd.concat(dfs, ignore_index=True)
    df_gdsc["ModelID"] = [c2m.get(int(x)) if pd.notna(x) else None for x in df_gdsc["COSMIC_ID"]]
    df_solid = df_gdsc[df_gdsc["ModelID"].isin(solid_models)].copy()

    # Extract per-drug monotherapy IC50 (nM) and D1/D7 concentration bounds across all solid combination matrices
    drug_records = {}
    for drug in ["AZD5153", "AZD5991", "Olaparib", "SN-38"]:
        s1 = df_solid[df_solid["lib1_name"] == drug][["ModelID", "lib1_minc", "lib1_maxc", "lib1_IC50_ln"]].rename(
            columns={"lib1_minc": "minc_uM", "lib1_maxc": "maxc_uM", "lib1_IC50_ln": "IC50_ln"}
        )
        s2 = df_solid[df_solid["lib2_name"] == drug][["ModelID", "lib2_minc", "lib2_maxc", "lib2_IC50_ln"]].rename(
            columns={"lib2_minc": "minc_uM", "lib2_maxc": "maxc_uM", "lib2_IC50_ln": "IC50_ln"}
        )
        comb = pd.concat([s1, s2], ignore_index=True).dropna(subset=["ModelID", "IC50_ln"])
        per_line = comb.groupby("ModelID", as_index=False).agg(
            minc_uM=("minc_uM", "min"),
            maxc_uM=("maxc_uM", "max"),
            IC50_ln=("IC50_ln", "mean"),
        )
        per_line["IC50_nM"] = np.exp(per_line["IC50_ln"]) * 1000.0
        d1_nM = float(per_line["minc_uM"].median() * 1000.0)
        d7_nM = float(per_line["maxc_uM"].median() * 1000.0)
        med_ic50 = float(np.median(per_line["IC50_nM"]))
        q25_ic50 = float(np.percentile(per_line["IC50_nM"], 25))
        q75_ic50 = float(np.percentile(per_line["IC50_nM"], 75))
        pct_below_d1 = float((per_line["IC50_nM"] <= d1_nM).mean() * 100.0)
        if drug == "Olaparib":
            d1_pair = float(
                df_solid[
                    ((df_solid["lib1_name"] == "Olaparib") & (df_solid["lib2_name"] == "SN-38"))
                    | ((df_solid["lib2_name"] == "Olaparib") & (df_solid["lib1_name"] == "SN-38"))
                ].apply(lambda r: r["lib1_minc"] if r["lib1_name"] == "Olaparib" else r["lib2_minc"], axis=1).median()
                * 1000.0
            )
            d1_nM = d1_pair
        drug_records[drug] = per_line
        print(
            f"  * {drug:<10} (n={len(per_line):3d} solid lines): Grid D1={d1_nM:.1f} nM, D7={d7_nM:.1f} nM | "
            f"Monotherapy extrapolated IC50 median={med_ic50:.1f} nM (IQR {q25_ic50:.1f}-{q75_ic50:.1f} nM)"
        )
        rows.append({
            "Audit_Section": "GDSC_Grid_and_Monotherapy_IC50",
            "Gene_or_Drug": drug,
            "Subgroup_or_Tissue": "All_Solid_Lines",
            "Metric": "Monotherapy_Extrapolated_IC50_Median_nM",
            "Value": round(med_ic50, 2),
            "Comparator_DepMap_Solid_Median_TPM": np.nan,
            "Comparator_DepMap_Solid_IQR": f"IQR {q25_ic50:.1f}-{q75_ic50:.1f} nM; D1={d1_nM:.1f} nM, D7={d7_nM:.1f} nM; <=D1: {pct_below_d1:.1f}%",
            "Comparator_DepMap_Solid_Log2p1": np.nan,
            "Tumor_to_Organ_Ratio": round(med_ic50 / d1_nM, 2),
            "N": len(per_line),
            "Source": "Local Sanger GDSC combination summary matrices (gdsc-007--010, sandpiper-01)",
        })

    # AZD5991 monotherapy IC50 by BAK1 median split on matched solid lines
    az_m = drug_records["AZD5991"].merge(df_expr[["ModelID", "BAK1"]], on="ModelID", how="inner")
    bak_med = float(np.median(az_m["BAK1"]))
    az_hi = az_m[az_m["BAK1"] >= bak_med]
    az_lo = az_m[az_m["BAK1"] < bak_med]
    _, p_mwu_bak = stats.mannwhitneyu(az_hi["IC50_nM"], az_lo["IC50_nM"], alternative="two-sided")
    print(
        f"  * AZD5991 by BAK1 split (n={len(az_m)}, cut={bak_med:.2f} log2p1): "
        f"BAK1-high median IC50={np.median(az_hi['IC50_nM']):.1f} nM (n={len(az_hi)}) vs "
        f"BAK1-low={np.median(az_lo['IC50_nM']):.1f} nM (n={len(az_lo)}, MWU p={p_mwu_bak:.2e})"
    )
    for grp_lbl, sub_df in [("BAK1_High_Solid", az_hi), ("BAK1_Low_Solid", az_lo)]:
        rows.append({
            "Audit_Section": "GDSC_Grid_and_Monotherapy_IC50",
            "Gene_or_Drug": "AZD5991",
            "Subgroup_or_Tissue": grp_lbl,
            "Metric": "Monotherapy_Extrapolated_IC50_Median_nM",
            "Value": round(float(np.median(sub_df["IC50_nM"])), 2),
            "Comparator_DepMap_Solid_Median_TPM": np.nan,
            "Comparator_DepMap_Solid_IQR": f"IQR {np.percentile(sub_df['IC50_nM'], 25):.1f}-{np.percentile(sub_df['IC50_nM'], 75):.1f} nM; MWU p={p_mwu_bak:.2e}",
            "Comparator_DepMap_Solid_Log2p1": round(bak_med, 4),
            "Tumor_to_Organ_Ratio": np.nan,
            "N": len(sub_df),
            "Source": "GDSC combination monotherapy IC50_ln + DepMap 24Q2 BAK1 median split",
        })

    # GDSC2 monotherapy empirical 1 - AUC & maximum screened concentration audit
    gdsc2_fp = os.path.join(DATA_RAW, "gdsc", "GDSC2_fitted_dose_response_27Oct23.xlsx")
    if os.path.exists(gdsc2_fp):
        g2 = pd.read_excel(
            gdsc2_fp,
            usecols=["COSMIC_ID", "SANGER_MODEL_ID", "DRUG_NAME", "MIN_CONC", "MAX_CONC", "LN_IC50", "AUC"],
        )
        g2 = g2[g2["DRUG_NAME"].isin(["AZD5153", "AZD5991", "Olaparib", "SN-38"])].copy()
        g2["ModelID"] = g2["SANGER_MODEL_ID"].map(s2m)
        miss = g2["ModelID"].isna() & g2["COSMIC_ID"].notna()
        g2.loc[miss, "ModelID"] = [c2m.get(int(c)) for c in g2.loc[miss, "COSMIC_ID"]]
        g2 = g2[g2["ModelID"].isin(solid_models)].copy()
        print("\n2B. GDSC2 MONOTHERAPY 1 - AUC & MAXIMUM SCREENED CONCENTRATION AUDIT (UNIQUE SOLID LINES)")
        for drg in ["AZD5153", "AZD5991", "Olaparib", "SN-38"]:
            sub_g2 = g2[g2["DRUG_NAME"] == drg].groupby("ModelID", as_index=False).agg(
                MIN_CONC=("MIN_CONC", "min"),
                MAX_CONC=("MAX_CONC", "max"),
                LN_IC50=("LN_IC50", "mean"),
                AUC=("AUC", "mean"),
            )
            sub_g2["IC50_nM"] = np.exp(sub_g2["LN_IC50"]) * 1000.0
            sub_g2["OneMinusAUC"] = 1.0 - sub_g2["AUC"]
            min_nM = float(sub_g2["MIN_CONC"].median() * 1000.0)
            max_nM = float(sub_g2["MAX_CONC"].median() * 1000.0)
            auc_med = float(sub_g2["AUC"].median())
            oma_med = float(sub_g2["OneMinusAUC"].median())
            oma_q25, oma_q75 = float(np.percentile(sub_g2["OneMinusAUC"], 25)), float(np.percentile(sub_g2["OneMinusAUC"], 75))
            ic50_med = float(sub_g2["IC50_nM"].median())
            print(
                f"  * {drg:<10} (n={len(sub_g2):3d}): Screen={min_nM:.2f}-{max_nM:.1f} nM | "
                f"AUC={auc_med:.4f} | 1-AUC={oma_med:.4f} (IQR {oma_q25:.4f}-{oma_q75:.4f}) | "
                f"GDSC2 extrap. IC50 median={ic50_med:,.1f} nM"
            )
            rows.append({
                "Audit_Section": "GDSC2_Monotherapy_AUC_and_Ceiling",
                "Gene_or_Drug": drg,
                "Subgroup_or_Tissue": "All_Solid_Lines",
                "Metric": "GDSC2_1_Minus_AUC_Median",
                "Value": round(oma_med, 4),
                "Comparator_DepMap_Solid_Median_TPM": round(max_nM, 1),
                "Comparator_DepMap_Solid_IQR": f"IQR {oma_q25:.4f}-{oma_q75:.4f}; AUC={auc_med:.4f}; Screen={min_nM:.2f}-{max_nM:.1f} nM; GDSC2 IC50={ic50_med:,.1f} nM",
                "Comparator_DepMap_Solid_Log2p1": round(auc_med, 4),
                "Tumor_to_Organ_Ratio": round(ic50_med / max_nM, 2) if max_nM > 0 else np.nan,
                "N": len(sub_g2),
                "Source": "GDSC2_fitted_dose_response_27Oct23.xlsx (unique solid ModelIDs)",
            })

    print("\n3. VERIFIED PHASE I PK BENCHMARKS & PUBLIC DATA AVAILABILITY")
    pk_entries = [
        ("AZD5153", "30 mg QD RP2D (Single Dose)", "Total_Plasma_Cmax_nM", 245.4, "Geometric mean %CV=59.22 (N=3); D1=1.0 nM is 245.4x below total Cmax", "Hamilton et al. 2023 (PMC10544002 Table 3)"),
        ("AZD5153", "30 mg QD RP2D (Multiple Dose)", "Total_Plasma_Cmax_nM", 272.4, "Geometric mean %CV=40.23 (N=3); D1=1.0 nM is 272.4x below total Cmax", "Hamilton et al. 2023 (PMC10544002 Table 3)"),
        ("AZD5153", "15 mg BID (Single / Multiple)", "Total_Plasma_Cmax_nM", 113.8, "Single=113.8 nM (%CV=46.18); Multiple=106.3 nM (%CV=41.06)", "Hamilton et al. 2023 (PMC10544002 Table 3)"),
        ("Olaparib", "Combo Cohort (+ AZD5153)", "Total_Plasma_Cmax_ug_mL", np.nan, "Geometric mean 4.9-11.2 ug/mL reported verbatim", "Hamilton et al. 2023 (PMC10544002)"),
        ("AZD5991", "Phase I Escalation (>=100 mg IV)", "Total_Plasma_Cmax_and_Human_fu", np.nan, "MISSING/UNPUBLISHED: numerical Cmax and human fu not tabulated in Desai et al. 2024 (graphical AUC only); Tron et al. 2018 reports 0.1% free only in fetal calf serum", "Desai et al. 2024 (PMC11528199); Tron et al. 2018 (PMC6297231)"),
        ("AZD5153", "Human Plasma Protein Binding", "Human_Plasma_fu", np.nan, "MISSING/UNPUBLISHED: human fu not tabulated in Hamilton et al. 2023, precluding unbound Cmax,u", "Hamilton et al. 2023 (PMC10544002)"),
        ("GDSC_7x7_Wells", "AZD5153+AZD5991 / Olaparib+SN-38", "Hill_Combo_IC50_at_D1_and_DRI", np.nan, "MISSING/UNAVAILABLE: local 59-column summary tables lack 7x7 per-well viability values needed for 4PL Hill IC50,combo@D1 and DRI", "gdsc-007--010 & sandpiper-01 matrix summary schema"),
    ]
    for drug, sub, metric, val, note, src in pk_entries:
        print(f"  * {drug:<14} | {sub:<32} | {metric}={val if pd.notna(val) else 'N/A'} | {note}")
        rows.append({
            "Audit_Section": "Phase_I_PK_and_FailSafe_Gaps",
            "Gene_or_Drug": drug,
            "Subgroup_or_Tissue": sub,
            "Metric": metric,
            "Value": val,
            "Comparator_DepMap_Solid_Median_TPM": np.nan,
            "Comparator_DepMap_Solid_IQR": note,
            "Comparator_DepMap_Solid_Log2p1": np.nan,
            "Tumor_to_Organ_Ratio": np.nan,
            "N": np.nan,
            "Source": src,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved ({len(df_out)} rows) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
