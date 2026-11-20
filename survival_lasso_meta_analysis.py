#!/usr/bin/env python3
"""Multi-Cohort Clinical Survival Analysis, Pure PDAC Sensitivity, Nested LRT, and DerSimonian-Laird Meta-Analysis."""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / std


def run_cox(df: pd.DataFrame, duration_col: str, event_col: str, target_col: str, covars: list[str], penalizer: float = 0.0) -> dict:
    cols = [duration_col, event_col, target_col] + [c for c in covars if c in df.columns and df[c].dropna().nunique() > 1]
    sub = df[cols].dropna().copy()
    sub = sub[sub[duration_col] > 0]
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sub, duration_col=duration_col, event_col=event_col)
    s = cph.summary.loc[target_col]
    return {
        "N": int(len(sub)),
        "Events": int(sub[event_col].sum()),
        "Beta": float(s["coef"]),
        "SE": float(s["se(coef)"]),
        "HR": float(s["exp(coef)"]),
        "CI_Low": float(s["exp(coef) lower 95%"]),
        "CI_High": float(s["exp(coef) upper 95%"]),
        "P_Value": float(s["p"]),
        "C_Index": float(cph.concordance_index_),
        "LL": float(cph.log_likelihood_),
        "Covariates": ", ".join(cols[3:])
    }


def dersimonian_laird(betas: np.ndarray, ses: np.ndarray) -> dict:
    k = len(betas)
    w = 1.0 / (ses ** 2)
    beta_fe = float(np.sum(w * betas) / np.sum(w))
    Q = float(np.sum(w * ((betas - beta_fe) ** 2)))
    df_q = k - 1
    p_het = float(stats.chi2.sf(Q, df_q))
    c_val = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w)))
    tau2 = max(0.0, (Q - df_q) / c_val)
    I2 = max(0.0, (Q - df_q) / Q) * 100.0 if Q > 0 else 0.0
    w_star = 1.0 / (ses ** 2 + tau2)
    beta_re = float(np.sum(w_star * betas) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    z_re = beta_re / se_re
    p_re = float(2.0 * stats.norm.sf(abs(z_re)))
    z_crit = float(stats.norm.ppf(0.975))
    return {
        "k": k,
        "Beta_RE": beta_re,
        "SE_RE": se_re,
        "HR_RE": float(np.exp(beta_re)),
        "CI_Low": float(np.exp(beta_re - z_crit * se_re)),
        "CI_High": float(np.exp(beta_re + z_crit * se_re)),
        "Z_RE": z_re,
        "P_RE": p_re,
        "I2": I2,
        "Q": Q,
        "P_Het": p_het,
        "Tau2": tau2,
        "HR_FE": float(np.exp(beta_fe))
    }


def load_rsem_genes(fp: str, genes: list[str], primary_only: bool = True) -> pd.DataFrame:
    out = {g: {} for g in genes}
    with open(fp, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split("\t")
        samps = hdr[2:]
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in out:
                for s, v in zip(samps, parts[2:]):
                    if primary_only and (len(s) < 15 or s[13:15] != "01"):
                        continue
                    try:
                        out[parts[0]][s] = float(np.log2(float(v) + 1.0))
                    except Exception:
                        pass
    df_out = pd.DataFrame(out)
    df_out["Sample_ID"] = df_out.index
    df_out["PID"] = [s[:12] for s in df_out.index]
    return df_out


def parse_extra_covs(s: str) -> dict:
    out = {}
    if pd.isna(s) or not str(s).strip():
        return out
    for item in str(s).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def _expr_col(df: pd.DataFrame, gene: str) -> str:
    c_rsem = f"{gene}_log2RSEM"
    return c_rsem if c_rsem in df.columns else f"{gene}_log2TPM"


def run_lasso_and_ph_diagnostics(df_paad: pd.DataFrame, df_pdac_raphael: pd.DataFrame, df_mb: pd.DataFrame):
    """Run univariate Cox, 4-gene multivariable Cox, 5-fold CV L1-LASSO refit, and PH diagnostics."""
    print("\n[4] TCGA-PAAD UNIVARIATE, 4-GENE COX, 5-FOLD CV L1-LASSO REFIT & PH DIAGNOSTICS")

    mki_col = _expr_col(df_paad, "MKI67")
    if "MKI67_z" not in df_paad.columns and mki_col in df_paad.columns:
        df_paad["MKI67_z"] = zscore(df_paad[mki_col])

    for g in ["SMARCA4", "HIF1A", "SLC7A1", "ABCB1"]:
        zcol = f"{g}_z"
        lcol = _expr_col(df_paad, g)
        if zcol not in df_paad.columns and lcol in df_paad.columns:
            df_paad[zcol] = zscore(df_paad[lcol])
        if zcol not in df_pdac_raphael.columns and lcol in df_pdac_raphael.columns:
            df_pdac_raphael[zcol] = zscore(df_pdac_raphael[lcol])

    print("  Univariate candidate Cox models in TCGA-PAAD:")
    for gcol in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z", "MKI67_z"]:
        if gcol in df_paad.columns and df_paad[gcol].notna().any():
            u = run_cox(df_paad, "OS_Days", "OS_Status", gcol, [])
            print(f"    - {gcol.replace('_z', ''):<8} (N={u['N']}, {u['Events']} deaths): HR = {u['HR']:.3f} [{u['CI_Low']:.3f}-{u['CI_High']:.3f}], p = {u['P_Value']:.2e}, C = {u['C_Index']:.3f}")

    feats = ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z", "Age", "Stage_Advanced", "Tumor_Purity"]
    sub_all = df_paad[["OS_Days", "OS_Status"] + feats].dropna().copy()
    sub_all = sub_all[sub_all["OS_Days"] > 0].reset_index(drop=True)

    cph_4g = CoxPHFitter(penalizer=0.0).fit(sub_all, duration_col="OS_Days", event_col="OS_Status")
    print(f"  * Unpenalized 4-gene PAAD Cox (N = {len(sub_all)}, {int(sub_all['OS_Status'].sum())} deaths): " +
          ", ".join(f"{g.replace('_z', '')}: beta={cph_4g.params_[g]:+.4f} (p={cph_4g.summary.loc[g, 'p']:.2e})"
                    for g in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z"]))

    # Deterministic 5-fold cross-validated L1-LASSO refit over alpha grid on real TCGA-PAAD
    alpha_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(sub_all))
    folds = np.array_split(perm, 5)
    cv_scores = []
    for alpha in alpha_grid:
        fold_c = []
        for f_idx in range(5):
            val_idx = folds[f_idx]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != f_idx])
            tr_df = sub_all.iloc[tr_idx]
            va_df = sub_all.iloc[val_idx]
            try:
                cph_f = CoxPHFitter(penalizer=alpha, l1_ratio=1.0).fit(
                    tr_df, duration_col="OS_Days", event_col="OS_Status", fit_options={"step_size": 0.5}
                )
                pred = -cph_f.predict_partial_hazard(va_df).values.ravel()
                fold_c.append(float(concordance_index(va_df["OS_Days"].values, pred, va_df["OS_Status"].values)))
            except Exception:
                fold_c.append(0.5)
        cv_scores.append((alpha, float(np.mean(fold_c))))

    best_alpha, best_cv_c = max(cv_scores, key=lambda x: x[1])
    cph_cv = CoxPHFitter(penalizer=best_alpha, l1_ratio=1.0).fit(
        sub_all, duration_col="OS_Days", event_col="OS_Status", fit_options={"step_size": 0.5}
    )
    print(f"  * Fresh 5-fold CV L1-LASSO refit (CV-optimal alpha = {best_alpha:.3f}, mean OOF C = {best_cv_c:.3f}, N = {len(sub_all)}): " +
          ", ".join(f"{g.replace('_z', '')}={cph_cv.params_[g]:+.4f}" for g in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z"]))

    cph_l1 = CoxPHFitter(penalizer=0.10, l1_ratio=1.0).fit(
        sub_all, duration_col="OS_Days", event_col="OS_Status", fit_options={"step_size": 0.5}
    )
    print(f"  * L1-LASSO at fixed alpha = 0.10 (N = {len(sub_all)}): " +
          ", ".join(f"{g.replace('_z', '')}={cph_l1.params_[g]:+.4f}" for g in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z"]))

    pen_unpen_cov = np.array([0.10, 0.10, 0.10, 0.10, 0.0, 0.0, 0.0])
    cph_ucov = CoxPHFitter(penalizer=pen_unpen_cov, l1_ratio=1.0).fit(
        sub_all, duration_col="OS_Days", event_col="OS_Status", fit_options={"step_size": 0.5}
    )
    print(f"  * Unpenalized-covariate LASSO (TCGA-PAAD N = {len(sub_all)}, alpha=0.10): " +
          ", ".join(f"{g.replace('_z', '')}={cph_ucov.params_[g]:+.4f}" for g in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z"]))

    sub_pdac = df_pdac_raphael[["OS_Days", "OS_Status"] + feats].dropna()
    sub_pdac = sub_pdac[sub_pdac["OS_Days"] > 0]
    cph_ucov_pdac = CoxPHFitter(penalizer=pen_unpen_cov, l1_ratio=1.0).fit(
        sub_pdac, duration_col="OS_Days", event_col="OS_Status", fit_options={"step_size": 0.5}
    )
    print(f"  * Unpenalized-covariate LASSO (Raphael 2017 Pure Ductal PDAC N = {len(sub_pdac)}, alpha=0.10): " +
          ", ".join(f"{g.replace('_z', '')}={cph_ucov_pdac.params_[g]:+.4f}" for g in ["SMARCA4_z", "HIF1A_z", "SLC7A1_z", "ABCB1_z"]))

    # Schoenfeld PH tests in TCGA-PAAD and METABRIC (full, 5y, 10y)
    paad_ph_df = df_paad[["OS_Days", "OS_Status", "Risk_ZScore", "Age", "Stage_Advanced", "Tumor_Purity"]].dropna()
    paad_ph_df = paad_ph_df[paad_ph_df["OS_Days"] > 0]
    cph_paad_ph = CoxPHFitter(penalizer=0.0).fit(paad_ph_df, duration_col="OS_Days", event_col="OS_Status")
    ph_paad = proportional_hazard_test(cph_paad_ph, paad_ph_df, time_transform="rank")
    print(f"  * Schoenfeld PH test (TCGA-PAAD Risk_ZScore, N = {len(paad_ph_df)}): p = {ph_paad.summary.loc['Risk_ZScore', 'p']:.4f}")

    sm_c = "SMARCA4_zScore" if "SMARCA4_zScore" in df_mb.columns else "SMARCA4_z"
    hf_c = "HIF1A_zScore" if "HIF1A_zScore" in df_mb.columns else "HIF1A_z"
    mb_ph_df = df_mb[["OS_Days", "Event", sm_c, hf_c, "Age", "Stage_Numeric"]].dropna().copy()
    mb_ph_df = mb_ph_df[mb_ph_df["OS_Days"] > 0]
    mb_ph_df["Risk_ZScore"] = zscore(mb_ph_df[sm_c] + mb_ph_df[hf_c])
    mb_ph_fit_df = mb_ph_df[["OS_Days", "Event", "Risk_ZScore", "Age", "Stage_Numeric"]]
    cph_mb_25 = CoxPHFitter(penalizer=0.0).fit(mb_ph_fit_df, duration_col="OS_Days", event_col="Event")
    ph_mb_25 = proportional_hazard_test(cph_mb_25, mb_ph_fit_df, time_transform="rank")
    print(f"  * Schoenfeld PH test (METABRIC 25y full, N = {len(mb_ph_fit_df)}, {int(mb_ph_fit_df['Event'].sum())} deaths): p = {ph_mb_25.summary.loc['Risk_ZScore', 'p']:.4f}")
    for yrs in (5, 10):
        lim = yrs * 365.25
        d_tr = mb_ph_fit_df.copy()
        d_tr["Event"] = np.where(d_tr["OS_Days"] > lim, 0, d_tr["Event"])
        d_tr["OS_Days"] = np.minimum(d_tr["OS_Days"], lim)
        cph_tr = CoxPHFitter(penalizer=0.0).fit(d_tr, duration_col="OS_Days", event_col="Event")
        ph_tr = proportional_hazard_test(cph_tr, d_tr, time_transform="rank")
        s_tr = cph_tr.summary.loc["Risk_ZScore"]
        print(f"  * METABRIC {yrs}-year censored ({int(d_tr['Event'].sum())} deaths): HR = {s_tr['exp(coef)']:.3f} [{s_tr['exp(coef) lower 95%']:.3f}-{s_tr['exp(coef) upper 95%']:.3f}], p = {s_tr['p']:.2e}, PH p = {ph_tr.summary.loc['Risk_ZScore', 'p']:.4f}")


def build_meta_and_table1(
    df_paad, df_pdac_raphael, df_icgc_pcawg, df_cgga, df_kirc, df_sarc, df_skcm, df_brca, df_lihc, df_blca, df_mb, df_gbm,
    n_gsea_paad, paad_h_rsem_n=None, n_paad_dropped=None
):
    """Run multivariable Cox across all cohorts, DerSimonian-Laird pooling, diagnostics, and export Table 1."""
    paad_cont = run_cox(df_paad, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"], penalizer=0.0)
    paad_nopur_cont = run_cox(df_paad, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced"], penalizer=0.0)
    pdac_raph_res = run_cox(df_pdac_raphael, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])
    icgc_pcawg_cont = run_cox(df_icgc_pcawg, "OS_Days", "Event", "Risk_ZScore", ["Age", "Is_Male"])
    gbm_cont = run_cox(df_gbm, "OS_Days", "Event", "Risk_ZScore", ["Age"])
    brca_cont = run_cox(df_brca, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])
    cgga_cont = run_cox(df_cgga, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated"])
    blca_cont = run_cox(df_blca, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])
    mb_cont = run_cox(df_mb, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])
    skcm_cont = run_cox(df_skcm, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])
    skcm_spec_cont = run_cox(df_skcm, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric", "Is_Metastatic"])
    kirc_cont = run_cox(df_kirc, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])
    sarc_cont = run_cox(df_sarc, "OS_Days", "Event", "Risk_ZScore", ["Age"])
    lihc_cont = run_cox(df_lihc, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"])

    primary_10 = [
        ("TCGA-PAAD", paad_cont, "Age, Stage, ABSOLUTE Purity"),
        ("TCGA-GBM (RNA-Seq RSEM)", gbm_cont, "Age (Grade IV Ceiling)"),
        ("TCGA-BRCA (PanCan Atlas)", brca_cont, "Age, Stage"),
        ("CGGA-325 (Glioma II-IV)", cgga_cont, "Age, Grade, IDH, MGMT"),
        ("TCGA-BLCA (Bladder)", blca_cont, "Age, Stage"),
        ("BRCA-METABRIC (Breast OS)", mb_cont, "Age, Stage"),
        ("TCGA-SKCM (Melanoma)", skcm_cont, "Age, Stage"),
        ("TCGA-KIRC (Renal Clear Cell)", kirc_cont, "Age, Stage"),
        ("TCGA-SARC (Sarcoma)", sarc_cont, "Age"),
        ("TCGA-LIHC (Hepatocellular)", lihc_cont, "Age, Stage"),
    ]

    betas_10 = np.array([m["Beta"] for _, m, _ in primary_10])
    ses_10 = np.array([m["SE"] for _, m, _ in primary_10])
    dl_meta_10 = dersimonian_laird(betas_10, ses_10)

    for cname, m, cov_str in primary_10:
        print(f"  * {cname:<28} | N = {m['N']:4d} ({m['Events']:3d} deaths) | Cont. per-SD HR = {m['HR']:.3f} [{m['CI_Low']:.3f}-{m['CI_High']:.3f}] | p = {m['P_Value']:.2e} | C = {m['C_Index']:.3f} | {cov_str}")

    print(f"  10-Cohort Primary DerSimonian-Laird Meta-Analysis (N = {sum(m['N'] for _, m, _ in primary_10):,}, {sum(m['Events'] for _, m, _ in primary_10):,} deaths):")
    print(f"     Continuous per-SD HR* = {dl_meta_10['HR_RE']:.3f} (95% CI [{dl_meta_10['CI_Low']:.3f}-{dl_meta_10['CI_High']:.3f}]), Z = {dl_meta_10['Z_RE']:.2f}, p = {dl_meta_10['P_RE']:.2e}, I^2 = {dl_meta_10['I2']:.1f}%")
    cohorts_11 = primary_10 + [("ICGC/PCAWG-PAAD (N = 71)", icgc_pcawg_cont, "Age, Is_Male")]
    dl_meta_11 = dersimonian_laird(np.array([m["Beta"] for _, m, _ in cohorts_11]), np.array([m["SE"] for _, m, _ in cohorts_11]))
    print(f"  11-Cohort Unified DerSimonian-Laird Meta-Analysis (+ ICGC/PCAWG-PAAD, N = {sum(m['N'] for _, m, _ in cohorts_11):,}, {sum(m['Events'] for _, m, _ in cohorts_11):,} deaths):")
    print(f"     Continuous per-SD HR* = {dl_meta_11['HR_RE']:.3f} (95% CI [{dl_meta_11['CI_Low']:.3f}-{dl_meta_11['CI_High']:.3f}]), Z = {dl_meta_11['Z_RE']:.2f}, p = {dl_meta_11['P_RE']:.2e}, I^2 = {dl_meta_11['I2']:.1f}%")
    non_breast_9 = [(c, m, s) for (c, m, s) in cohorts_11 if "BRCA" not in c]
    dl_meta_9_nb = dersimonian_laird(np.array([m["Beta"] for _, m, _ in non_breast_9]), np.array([m["SE"] for _, m, _ in non_breast_9]))
    print(f"  9-Cohort Non-Breast Sensitivity Meta-Analysis (Excl. TCGA-BRCA & METABRIC, N = {sum(m['N'] for _, m, _ in non_breast_9):,}, {sum(m['Events'] for _, m, _ in non_breast_9):,} deaths):")
    print(f"     Continuous per-SD HR* = {dl_meta_9_nb['HR_RE']:.3f} (95% CI [{dl_meta_9_nb['CI_Low']:.3f}-{dl_meta_9_nb['CI_High']:.3f}]), Z = {dl_meta_9_nb['Z_RE']:.2f}, p = {dl_meta_9_nb['P_RE']:.2e}, I^2 = {dl_meta_9_nb['I2']:.1f}%")

    print("\n  [TCGA-PAAD SUBCOHORT SENSITIVITIES, ICGC/PCAWG-PAAD & SKCM SITE SENSITIVITY]")
    print(f"  * TCGA-PAAD Survival (Age, Stage, ABSOLUTE Purity): N = {paad_cont['N']} ({paad_cont['Events']} deaths) | HR = {paad_cont['HR']:.3f} [{paad_cont['CI_Low']:.3f}-{paad_cont['CI_High']:.3f}], p = {paad_cont['P_Value']:.4f}, C = {paad_cont['C_Index']:.3f}")
    print(f"  * TCGA-PAAD Survival (Age, Stage only, no Purity): N = {paad_nopur_cont['N']} ({paad_nopur_cont['Events']} deaths) | HR = {paad_nopur_cont['HR']:.3f} [{paad_nopur_cont['CI_Low']:.3f}-{paad_nopur_cont['CI_High']:.3f}], p = {paad_nopur_cont['P_Value']:.4f}, C = {paad_nopur_cont['C_Index']:.3f}")
    print(f"  * TCGA-PAAD GSEA: N = {n_gsea_paad}")
    if paad_h_rsem_n is not None:
        print(f"  * TCGA-PAAD RSEM complete-case: n = {paad_h_rsem_n} ({n_paad_dropped} dropped)")
    print(f"  * TCGA-PAAD Raphael 2017 Pure Ductal PDAC: N = {pdac_raph_res['N']} ({pdac_raph_res['Events']} deaths) | HR = {pdac_raph_res['HR']:.3f} [{pdac_raph_res['CI_Low']:.3f}-{pdac_raph_res['CI_High']:.3f}], p = {pdac_raph_res['P_Value']:.4f}, C = {pdac_raph_res['C_Index']:.3f}")
    print(f"  * ICGC/PCAWG-PAAD (N = {icgc_pcawg_cont['N']}) ({icgc_pcawg_cont['Covariates']}): N = {icgc_pcawg_cont['N']} ({icgc_pcawg_cont['Events']} deaths) | Cont. per-SD HR = {icgc_pcawg_cont['HR']:.3f} [{icgc_pcawg_cont['CI_Low']:.3f}-{icgc_pcawg_cont['CI_High']:.3f}], p = {icgc_pcawg_cont['P_Value']:.4f}, C = {icgc_pcawg_cont['C_Index']:.3f}")
    print(f"  * TCGA-SKCM Specimen-Type Sensitivity (+ Is_Metastatic): N = {skcm_spec_cont['N']} ({skcm_spec_cont['Events']} deaths) | Cont. per-SD HR = {skcm_spec_cont['HR']:.3f} [{skcm_spec_cont['CI_Low']:.3f}-{skcm_spec_cont['CI_High']:.3f}], p = {skcm_spec_cont['P_Value']:.4f}, C = {skcm_spec_cont['C_Index']:.3f}")

    run_lasso_and_ph_diagnostics(df_paad, df_pdac_raphael, df_mb)

    t1_dir = os.path.join(DATA_PROC, "tables")
    t1_fp = os.path.join(t1_dir, "Table_1_Clinical_Cohorts.csv")
    os.makedirs(t1_dir, exist_ok=True)

    def run_median_split(df_sub, dur_col, ev_col, score_col, covs):
        d = df_sub.dropna(subset=[dur_col, ev_col, score_col]).copy()
        d = d[d[dur_col] > 0]
        d["High_Risk"] = (d[score_col] >= d[score_col].median()).astype(int)
        return run_cox(d, dur_col, ev_col, "High_Risk", covs)

    specs = [
        ("TCGA-PAAD", df_paad, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"], paad_cont, "Age, Stage_Advanced, Tumor_Purity"),
        ("TCGA-PAAD (Pure Ductal PDAC)", df_pdac_raphael, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"], pdac_raph_res, "Age, Stage_Advanced, Tumor_Purity"),
        ("ICGC/PCAWG-PAAD (N = 71)", df_icgc_pcawg, "OS_Days", "Event", "Risk_ZScore", ["Age", "Is_Male"], icgc_pcawg_cont, "Age, Is_Male"),
        ("CGGA-325 (Diffuse Glioma II-IV)", df_cgga, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated"], cgga_cont, "Age, Stage_Advanced, IDH_Mut, MGMT_Meth"),
        ("TCGA-KIRC (Clear Cell Renal)", df_kirc, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], kirc_cont, "Age, Stage_Numeric"),
        ("TCGA-SARC (Soft-Tissue Sarcoma)", df_sarc, "OS_Days", "Event", "Risk_ZScore", ["Age"], sarc_cont, "Age"),
        ("TCGA-SKCM (Cutaneous Melanoma)", df_skcm, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], skcm_cont, "Age, Stage_Numeric"),
        ("TCGA-BRCA (Breast Carcinoma)", df_brca, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], brca_cont, "Age, Stage_Numeric"),
        ("TCGA-LIHC (Hepatocellular)", df_lihc, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], lihc_cont, "Age, Stage_Numeric"),
        ("TCGA-BLCA (Urothelial Bladder)", df_blca, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], blca_cont, "Age, Stage_Numeric"),
        ("BRCA-METABRIC (Breast OS)", df_mb, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"], mb_cont, "Age, Stage_Numeric"),
        ("TCGA-GBM (WHO Grade IV)", df_gbm, "OS_Days", "Event", "Risk_ZScore", ["Age"], gbm_cont, "Age"),
    ]
    t1_rows = []
    for cname, d_sub, dcol, ecol, scol, covs, c_res, cov_lbl in specs:
        m_res = run_median_split(d_sub, dcol, ecol, scol, covs)
        t1_rows.append({
            "Cohort": cname,
            "N_Patients": c_res["N"],
            "N_Deaths": c_res["Events"],
            "Continuous_PerSD_HR_95CI": f"{c_res['HR']:.3f} [{c_res['CI_Low']:.3f}-{c_res['CI_High']:.3f}]",
            "Continuous_Wald_P": f"{c_res['P_Value']:.2e}",
            "Continuous_Beta": round(c_res["Beta"], 8),
            "Continuous_SE": round(c_res["SE"], 8),
            "MedianSplit_HR_95CI": f"{m_res['HR']:.3f} [{m_res['CI_Low']:.3f}-{m_res['CI_High']:.3f}]",
            "MedianSplit_Wald_P": f"{m_res['P_Value']:.2e}",
            "C_Index": f"{c_res['C_Index']:.3f}",
            "Adjusted_Covariates": cov_lbl,
        })
    for label, meta_d, items, cov_note in [
        (f"9-Cohort Non-Breast Meta (I^2 = {dl_meta_9_nb['I2']:.1f}%)", dl_meta_9_nb, non_breast_9, "Excl. TCGA-BRCA & METABRIC"),
        (f"10-Cohort Primary Meta (I^2 = {dl_meta_10['I2']:.1f}%)", dl_meta_10, primary_10, "Primary 10-Cohort Pooled"),
        (f"11-Cohort Unified Meta (I^2 = {dl_meta_11['I2']:.1f}%)", dl_meta_11, cohorts_11, "10 Primary + ICGC/PCAWG-PAAD"),
    ]:
        t1_rows.append({
            "Cohort": label,
            "N_Patients": int(sum(m["N"] for _, m, _ in items)),
            "N_Deaths": int(sum(m["Events"] for _, m, _ in items)),
            "Continuous_PerSD_HR_95CI": f"{meta_d['HR_RE']:.3f} [{meta_d['CI_Low']:.3f}-{meta_d['CI_High']:.3f}]",
            "Continuous_Wald_P": f"{meta_d['P_RE']:.2e}",
            "Continuous_Beta": round(meta_d["Beta_RE"], 8),
            "Continuous_SE": round(meta_d["SE_RE"], 8),
            "MedianSplit_HR_95CI": "--",
            "MedianSplit_Wald_P": "--",
            "C_Index": "--",
            "Adjusted_Covariates": cov_note,
        })
    pd.DataFrame(t1_rows).to_csv(t1_fp, index=False)
    run_breast_subtype_sensitivity(df_brca, df_mb)


def attach_breast_clinical_subtypes(df_brca: pd.DataFrame, df_mb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach PAM50/Claudin subtypes and ER/PR/HER2 receptor status from local raw cBioPortal clinical files when available."""
    df_brca = df_brca.copy()
    df_mb = df_mb.copy()

    brca_clin_fp = os.path.join(DATA_RAW, "cbioportal_pancan", "brca", "data_clinical_patient.txt")
    if "PAM50_Subtype" not in df_brca.columns and os.path.exists(brca_clin_fp):
        bc = pd.read_csv(brca_clin_fp, sep="\t", comment="#", usecols=["PATIENT_ID", "SUBTYPE"]).dropna(subset=["PATIENT_ID"])
        bc_map = bc.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")["SUBTYPE"].to_dict()
        df_brca["PAM50_Subtype"] = df_brca["Sample_ID"].astype(str).str.slice(0, 12).map(bc_map)

    mb_pat_fp = os.path.join(DATA_RAW, "cbioportal_high_mortality", "brca_metabric", "data_clinical_patient.txt")
    mb_samp_fp = os.path.join(DATA_RAW, "cbioportal_high_mortality", "brca_metabric", "data_clinical_sample.txt")
    if "CLAUDIN_SUBTYPE" not in df_mb.columns and os.path.exists(mb_pat_fp):
        mp = pd.read_csv(mb_pat_fp, sep="\t", comment="#", usecols=["PATIENT_ID", "CLAUDIN_SUBTYPE"]).dropna(subset=["PATIENT_ID"])
        mp_map = mp.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")["CLAUDIN_SUBTYPE"].to_dict()
        df_mb["CLAUDIN_SUBTYPE"] = df_mb["Sample_ID"].astype(str).map(mp_map)
    if ("ER_STATUS" not in df_mb.columns or "PR_STATUS" not in df_mb.columns or "HER2_STATUS" not in df_mb.columns) and os.path.exists(mb_samp_fp):
        ms = pd.read_csv(mb_samp_fp, sep="\t", comment="#", usecols=["SAMPLE_ID", "ER_STATUS", "PR_STATUS", "HER2_STATUS"]).dropna(subset=["SAMPLE_ID"])
        ms_dedup = ms.drop_duplicates("SAMPLE_ID").set_index("SAMPLE_ID")
        for col in ["ER_STATUS", "PR_STATUS", "HER2_STATUS"]:
            df_mb[col] = df_mb["Sample_ID"].astype(str).map(ms_dedup[col].to_dict())
    return df_brca, df_mb


def run_breast_subtype_sensitivity(df_brca: pd.DataFrame, df_mb: pd.DataFrame) -> pd.DataFrame:
    """Fit strict complete-case PAM50/receptor-adjusted Cox sensitivity models in TCGA-BRCA and BRCA-METABRIC."""
    df_brca, df_mb = attach_breast_clinical_subtypes(df_brca, df_mb)
    out_fp = os.path.join(DATA_PROC, "brca_subtype_receptor_sensitivity.csv")

    print("\n[5] BREAST COHORT RECEPTOR & PAM50 SUBTYPE COMPLETE-CASE SENSITIVITY MODELS")
    rows = []

    def fit_categorical_cox(df_src, cohort_name, model_label, cat_specs, censor_years=None):
        base_cols = ["OS_Days", "Event", "Risk_ZScore", "Age", "Stage_Numeric"]
        req_cats = [c for c, _ in cat_specs]
        if any(c not in df_src.columns for c in req_cats):
            return
        sub = df_src[df_src["OS_Days"] > 0].copy()
        n_before = len(sub)
        sub = sub.dropna(subset=base_cols + req_cats).copy()
        n_after_na = len(sub)
        n_nc = 0
        for col, excl_vals in cat_specs:
            if excl_vals:
                mask_excl = sub[col].astype(str).isin(excl_vals)
                n_nc += int(mask_excl.sum())
                sub = sub[~mask_excl].copy()
        n_after = len(sub)
        n_dropped = n_before - n_after
        if n_after == 0:
            return
        if censor_years is not None:
            lim = float(censor_years) * 365.25
            sub["Event"] = np.where(sub["OS_Days"] > lim, 0, sub["Event"])
            sub["OS_Days"] = np.minimum(sub["OS_Days"], lim)

        dummy_cols = []
        for col, _ in cat_specs:
            vals = sorted(sub[col].astype(str).unique())
            ref = "BRCA_LumA" if "BRCA_LumA" in vals else ("LumA" if "LumA" in vals else ("Positive" if "Positive" in vals else vals[0]))
            for v in vals:
                if v == ref:
                    continue
                dcol = f"{col}_{v}".replace("-", "_").replace(" ", "_")
                sub[dcol] = (sub[col].astype(str) == v).astype(int)
                dummy_cols.append(dcol)

        res = run_cox(sub, "OS_Days", "Event", "Risk_ZScore", ["Age", "Stage_Numeric"] + dummy_cols)
        drop_detail = f"{n_dropped} dropped ({n_before - n_after_na} missing, {n_nc} 'NC' unclassifiable)" if n_nc else f"{n_dropped} missing dropped"
        print(
            f"  * {cohort_name} | {model_label}: N = {n_before:,} -> {n_after:,} ({drop_detail}), {res['Events']} deaths "
            f"| HR = {res['HR']:.3f} [{res['CI_Low']:.3f}-{res['CI_High']:.3f}], p = {res['P_Value']:.2e}, C = {res['C_Index']:.3f}"
        )
        rows.append({
            "Cohort": cohort_name,
            "Model": model_label,
            "N_Before": n_before,
            "N_Dropped": n_dropped,
            "N_Complete": res["N"],
            "Events": res["Events"],
            "Continuous_PerSD_HR": round(res["HR"], 6),
            "CI_Low": round(res["CI_Low"], 6),
            "CI_High": round(res["CI_High"], 6),
            "P_Value": f"{res['P_Value']:.2e}",
            "C_Index": round(res["C_Index"], 4),
            "Covariates": "Age, Stage_Numeric, " + ", ".join(req_cats) + (f" ({censor_years}y censored)" if censor_years else ""),
        })

    if "PAM50_Subtype" in df_brca.columns and df_brca["PAM50_Subtype"].notna().any():
        fit_categorical_cox(df_brca, "TCGA-BRCA", "Age + Stage + PAM50 (SUBTYPE)", [("PAM50_Subtype", None)])
    if "CLAUDIN_SUBTYPE" in df_mb.columns and df_mb["CLAUDIN_SUBTYPE"].notna().any():
        fit_categorical_cox(df_mb, "BRCA-METABRIC", "Age + Stage + PAM50/Claudin-Low", [("CLAUDIN_SUBTYPE", {"NC"})])
    if all(c in df_mb.columns and df_mb[c].notna().any() for c in ["ER_STATUS", "PR_STATUS", "HER2_STATUS"]):
        fit_categorical_cox(df_mb, "BRCA-METABRIC", "Age + Stage + ER/PR/HER2 Receptors", [("ER_STATUS", None), ("PR_STATUS", None), ("HER2_STATUS", None)])
        fit_categorical_cox(df_mb, "BRCA-METABRIC", "Age + Stage + ER/PR/HER2 Receptors (5y)", [("ER_STATUS", None), ("PR_STATUS", None), ("HER2_STATUS", None)], censor_years=5)
    if all(c in df_mb.columns and df_mb[c].notna().any() for c in ["CLAUDIN_SUBTYPE", "ER_STATUS", "PR_STATUS", "HER2_STATUS"]):
        fit_categorical_cox(df_mb, "BRCA-METABRIC", "Age + Stage + PAM50/Claudin-Low + ER/PR/HER2", [("CLAUDIN_SUBTYPE", {"NC"}), ("ER_STATUS", None), ("PR_STATUS", None), ("HER2_STATUS", None)])

    if rows:
        df_sens = pd.DataFrame(rows)
        df_sens.to_csv(out_fp, index=False)
        print(f"  Saved breast subtype/receptor sensitivity table -> {out_fp}")
        return df_sens
    if os.path.exists(out_fp):
        df_cached = pd.read_csv(out_fp)
        print(f"  Raw breast subtype/receptor archives absent; loaded cached results -> {out_fp}")
        return df_cached
    return pd.DataFrame()


def export_curated_10cohort_table(
    df_paad, df_gbm, df_brca, df_cgga, df_blca, df_mb, df_skcm, df_kirc, df_sarc, df_lihc, df_icgc_pcawg
):
    """Export open-access de-identified 10-cohort clinical-expression survival table + ICGC/PCAWG-PAAD."""
    out_fp = os.path.join(DATA_PROC, "curated_10cohort_clinical_survival.csv")
    rows = []

    def add_paad(df):
        mki_col = _expr_col(df, "MKI67")
        for _, r in df[df["OS_Days"] > 0].dropna(subset=["OS_Days", "OS_Status", "SMARCA4_z", "HIF1A_z", "Age"]).iterrows():
            extras = [
                f"Pure_Ductal_Raphael2017={int(r.get('Pure_Ductal_Raphael2017', 0))}",
                f"Pure_Ductal_cBioPortal={int(r.get('Pure_Ductal_cBioPortal', 0))}",
                f"Non_PNET={int(r.get('Non_PNET', 1))}",
                f"SLC7A1_z={float(r['SLC7A1_z']):.6f}",
                f"SLC1A5_z={float(r['SLC1A5_z']):.6f}",
                f"ABCB1_z={float(r['ABCB1_z']):.6f}",
            ]
            if pd.notna(r.get("Tumor_Purity")):
                extras.insert(0, f"Tumor_Purity={float(r['Tumor_Purity']):.6f}")
            if pd.notna(r.get(mki_col)):
                extras.append(f"MKI67_log2RSEM={float(r[mki_col]):.6f}")
            rows.append({
                "Cohort": "TCGA-PAAD",
                "Sample_ID": str(r["Sample_Barcode"]),
                "OS_Days": float(r["OS_Days"]),
                "Event": int(r["OS_Status"]),
                "Age": float(r["Age"]),
                "Stage": float(r["Stage_Advanced"]) if pd.notna(r.get("Stage_Advanced")) else np.nan,
                "Extra_Covariates": ";".join(extras),
                "SMARCA4_z": float(r["SMARCA4_z"]),
                "HIF1A_z": float(r["HIF1A_z"]),
            })

    add_paad(df_paad)

    def brca_extra(r):
        parts = [f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"]
        if pd.notna(r.get("PAM50_Subtype")):
            parts.append(f"PAM50_Subtype={r['PAM50_Subtype']}")
        return ";".join(parts)

    def mb_extra(r):
        parts = [f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"]
        for col in ["CLAUDIN_SUBTYPE", "ER_STATUS", "PR_STATUS", "HER2_STATUS"]:
            if pd.notna(r.get(col)):
                parts.append(f"{col}={r[col]}")
        return ";".join(parts)

    for cname, df, req_cols, stage_col, extra_fn in [
        ("TCGA-GBM", df_gbm, ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"], None, lambda r: f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
        ("TCGA-BRCA", df_brca, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", brca_extra),
        ("TCGA-BLCA", df_blca, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", lambda r: f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
        ("BRCA-METABRIC", df_mb, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", mb_extra),
        ("TCGA-SKCM", df_skcm, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", lambda r: f"Is_Metastatic={int(r['Is_Metastatic'])};Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
        ("TCGA-KIRC", df_kirc, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", lambda r: f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
        ("TCGA-SARC", df_sarc, ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"], None, lambda r: f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
        ("TCGA-LIHC", df_lihc, ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", lambda r: f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"),
    ]:
        sub = df[df["OS_Days"] > 0].dropna(subset=req_cols)
        for _, r in sub.iterrows():
            rows.append({
                "Cohort": cname,
                "Sample_ID": str(r["Sample_ID"]),
                "OS_Days": float(r["OS_Days"]),
                "Event": int(r["Event"]),
                "Age": float(r["Age"]),
                "Stage": float(r[stage_col]) if stage_col else np.nan,
                "Extra_Covariates": extra_fn(r),
                "SMARCA4_z": float(r["SMARCA4_zScore"]),
                "HIF1A_z": float(r["HIF1A_zScore"]),
            })

    # CGGA-325 (unweighted a priori Z(SMARCA4_z + HIF1A_z))
    sub_c = df_cgga[df_cgga["OS_Days"] > 0].dropna(subset=["OS_Days", "OS_Status", "Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated", "SMARCA4_z", "HIF1A_z"]).copy()
    sub_c["Risk_ZScore_Cohort"] = zscore(sub_c["SMARCA4_z"] + sub_c["HIF1A_z"])
    for idx, r in sub_c.iterrows():
        rows.append({
            "Cohort": "CGGA-325",
            "Sample_ID": str(r.get("CGGA_ID", r.get("Sample_ID", f"CGGA_{idx}"))),
            "OS_Days": float(r["OS_Days"]),
            "Event": int(r["OS_Status"]),
            "Age": float(r["Age"]),
            "Stage": float(r["Stage_Advanced"]),
            "Extra_Covariates": f"IDH_Mutant={int(r['IDH_Mutant'])};MGMT_Methylated={int(r['MGMT_Methylated'])};Risk_ZScore={float(r['Risk_ZScore_Cohort']):.6f}",
            "SMARCA4_z": float(r["SMARCA4_z"]),
            "HIF1A_z": float(r["HIF1A_z"]),
        })

    # External ICGC/PCAWG-PAAD (N = 71) validation cohort
    for _, r in df_icgc_pcawg[df_icgc_pcawg["OS_Days"] > 0].dropna(subset=["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"]).iterrows():
        rows.append({
            "Cohort": "ICGC/PCAWG-PAAD",
            "Sample_ID": str(r["Sample_ID"]),
            "OS_Days": float(r["OS_Days"]),
            "Event": int(r["Event"]),
            "Age": float(r["Age"]),
            "Stage": np.nan,
            "Extra_Covariates": f"Is_Male={int(r['Is_Male'])};Risk_ZScore={float(r['Risk_ZScore']):.6f}",
            "SMARCA4_z": float(r["SMARCA4_zScore"]),
            "HIF1A_z": float(r["HIF1A_zScore"]),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_fp, index=False)
    n_primary = int((df_out["Cohort"] != "ICGC/PCAWG-PAAD").sum())
    print(f"  Exported open-access survival table ({n_primary:,} primary 10-cohort cases + {int((df_out['Cohort'] == 'ICGC/PCAWG-PAAD').sum())} ICGC/PCAWG-PAAD) -> {out_fp}")


def run_from_curated_table(curated_fp: str):
    """Run all clinical Cox models, pure-PDAC sensitivity, LRT, LASSO refit, PH tests, and meta-analysis from curated CSV."""
    print(f"Raw cohort archives absent; running full dynamic survival pipeline from {curated_fp}...")
    df_cur = pd.read_csv(curated_fp)
    extra_dicts = df_cur["Extra_Covariates"].apply(parse_extra_covs)
    df_cur["Risk_2Gene"] = df_cur["SMARCA4_z"] + df_cur["HIF1A_z"]

    def cohort_df(name, stage_col="Stage_Numeric"):
        d = df_cur[df_cur["Cohort"] == name].copy()
        d["OS_Status"] = d["Event"].astype(int)
        d["Risk_Score"] = zscore(d["Risk_2Gene"])
        if stage_col != "Stage":
            d[stage_col] = d["Stage"]
        ex = extra_dicts.loc[d.index]
        for k in set().union(*(x.keys() for x in ex)):
            d[k] = [x.get(k, np.nan) for x in ex]
        d["Risk_ZScore"] = zscore(d["Risk_2Gene"])
        return d

    df_paad = cohort_df("TCGA-PAAD", stage_col="Stage_Advanced")
    raph_flag = "Pure_Ductal_Raphael2017" if "Pure_Ductal_Raphael2017" in df_paad.columns else "Pure_Ductal_PDAC"
    df_pdac_raph = df_paad[df_paad[raph_flag] == 1].copy()
    df_pdac_raph["Risk_ZScore"] = zscore(df_pdac_raph["Risk_2Gene"])
    df_pdac_cbio = df_paad[df_paad["Pure_Ductal_cBioPortal"] == 1].copy() if "Pure_Ductal_cBioPortal" in df_paad.columns else df_pdac_raph
    df_pdac_cbio["Risk_ZScore"] = zscore(df_pdac_cbio["Risk_2Gene"])
    df_pdac_nonpnet = df_paad[df_paad["Non_PNET"] == 1].copy()
    df_pdac_nonpnet["Risk_ZScore"] = zscore(df_pdac_nonpnet["Risk_2Gene"])

    print("\n[2] PURE PDAC HISTOLOGICAL FILTER & NESTED LIKELIHOOD-RATIO TEST (LRT) VS MKI67")
    res_r_pur = run_cox(df_pdac_raph, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])
    res_r_np = run_cox(df_pdac_raph, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced"])
    res_cb = run_cox(df_pdac_cbio, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])
    res_npnet = run_cox(df_pdac_nonpnet, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])
    for label, res in [
        ("Pure Ductal PDAC (Raphael et al. 2017 whitelist + Purity)", res_r_pur),
        ("Pure Ductal PDAC (Raphael et al. 2017 whitelist, Age+Stage)", res_r_np),
        ("cBioPortal 'Pancreas Adenocarcinoma, Ductal Type' (+ Purity)", res_cb),
        ("Non-PNET TCGA-PAAD (Excluding 8246/3 Neuroendocrine)", res_npnet),
    ]:
        print(f"  * {label:<62}: N = {res['N']} ({res['Events']} deaths) -> HR = {res['HR']:.3f} [{res['CI_Low']:.3f}-{res['CI_High']:.3f}], p = {res['P_Value']:.2e}, C = {res['C_Index']:.3f}")

    mki_col = _expr_col(df_paad, "MKI67")
    sub_lrt = df_paad[["OS_Days", "OS_Status", mki_col, "Risk_2Gene", "Age", "Stage_Advanced"]].dropna().copy()
    sub_lrt = sub_lrt[sub_lrt["OS_Days"] > 0]
    sub_lrt["MKI67_z"] = zscore(sub_lrt[mki_col])
    sub_lrt["Risk_ZScore"] = zscore(sub_lrt["Risk_2Gene"])
    cph_null = CoxPHFitter(penalizer=0.0).fit(sub_lrt[["OS_Days", "OS_Status", "MKI67_z", "Age", "Stage_Advanced"]], duration_col="OS_Days", event_col="OS_Status")
    cph_full = CoxPHFitter(penalizer=0.0).fit(sub_lrt[["OS_Days", "OS_Status", "MKI67_z", "Risk_ZScore", "Age", "Stage_Advanced"]], duration_col="OS_Days", event_col="OS_Status")
    lrt_chi2 = 2.0 * (float(cph_full.log_likelihood_) - float(cph_null.log_likelihood_))
    lrt_p = float(stats.chi2.sf(lrt_chi2, df=1))
    print(f"  * Complete-Case Primary Nested LRT vs MKI67 (N = {len(sub_lrt)}, {int(sub_lrt['OS_Status'].sum())} deaths): chi2(1) = {lrt_chi2:.2f}, p = {lrt_p:.2e} (C: {cph_null.concordance_index_:.3f} -> {cph_full.concordance_index_:.3f})")

    print("\n[3] UNIFORM CONTINUOUS PER-SD Z-SCORE COX MODELS ACROSS ALL 10 CLINICAL COHORTS")
    gsea_fp = os.path.join(DATA_PROC, "hallmark_gsea_transcriptome_wide_enrichment.csv")
    n_gsea_paad = int(pd.read_csv(gsea_fp).loc[lambda d: d["Cohort"].str.contains("PAAD"), "N_Patients"].iloc[0]) if os.path.exists(gsea_fp) else 178
    build_meta_and_table1(
        df_paad, df_pdac_raph, cohort_df("ICGC/PCAWG-PAAD"), cohort_df("CGGA-325", stage_col="Stage_Advanced"),
        cohort_df("TCGA-KIRC"), cohort_df("TCGA-SARC"), cohort_df("TCGA-SKCM"), cohort_df("TCGA-BRCA"),
        cohort_df("TCGA-LIHC"), cohort_df("TCGA-BLCA"), cohort_df("BRCA-METABRIC"), cohort_df("TCGA-GBM"),
        n_gsea_paad
    )


def resolve_registry_path(filename: str) -> str:
    """Resolve clinical registry CSV from data/raw/clinical_registries/ with fallback to data/processed/."""
    raw_reg = os.path.join(DATA_RAW, "clinical_registries", filename)
    return raw_reg if os.path.exists(raw_reg) else os.path.join(DATA_PROC, filename)


def main():
    print("MULTI-COHORT CLINICAL SURVIVAL ANALYSIS & META-ANALYSIS")

    paad_raw_fp = os.path.join(DATA_RAW, "tcga", "TCGA_PAAD_expression_survival.csv")
    he_fp = resolve_registry_path("High_Event_10Cohort_Treatment_Registry.csv")
    curated_fp = os.path.join(DATA_PROC, "curated_10cohort_clinical_survival.csv")
    if not (os.path.exists(paad_raw_fp) and os.path.exists(he_fp)):
        if os.path.exists(curated_fp):
            run_from_curated_table(curated_fp)
            return
        t1_fp = os.path.join(DATA_PROC, "tables", "Table_1_Clinical_Cohorts.csv")
        print("Raw cohort archives and curated table not found; displaying pre-computed Table 1 summary:\n")
        print(pd.read_csv(t1_fp).to_string(index=False))
        return

    he_reg = pd.read_csv(he_fp)
    pc_reg = pd.read_csv(resolve_registry_path("Full_TCGA_PanCan_Patient_Treatment_Registry.csv"))
    bi_reg = pd.read_csv(resolve_registry_path("Batch_Indication_Expansion_Patient_Registry.csv"))
    pcawg_reg = pd.read_csv(resolve_registry_path("PCAWG_1210_Personalized_Treatment_Registry.csv"))
    df_paad = pd.read_csv(paad_raw_fp)
    df_cgga = pd.read_csv(os.path.join(DATA_RAW, "external", "CGGA_mRNAseq_325_expression_survival.csv"))

    # Complete-case TCGA in vivo (PAAD + GBM + BRCA) vs DepMap in vitro Pearson r and Fisher Z
    print("\n[1] COMPLETE-CASE TCGA IN VIVO (PAAD + GBM + BRCA) PEARSON r & FISHER Z")

    paad_rsem_fp = os.path.join(DATA_RAW, "paad_tcga_pan_can_atlas_2018", "data_mrna_seq_v2_rsem.txt")
    if not os.path.exists(paad_rsem_fp):
        paad_rsem_fp = os.path.join(DATA_RAW, "cbioportal_pancan", "paad", "data_mrna_seq_v2_rsem.txt")
    paad_rsem = load_rsem_genes(paad_rsem_fp, ["HIF1A", "SLC7A1", "SMARCA4", "MKI67"], primary_only=True)
    gbm_rsem = load_rsem_genes(os.path.join(DATA_RAW, "cbioportal_high_mortality", "gbm_tcga", "data_mrna_seq_v2_rsem.txt"), ["HIF1A", "SLC7A1", "SMARCA4"], primary_only=False)
    brca_rsem = load_rsem_genes(os.path.join(DATA_RAW, "cbioportal_pancan", "brca", "data_mrna_seq_v2_rsem.txt"), ["HIF1A", "SLC7A1", "SMARCA4"], primary_only=False)

    df_gbm_158 = he_reg[he_reg["Cohort"] == "GBM_TCGA"].dropna(subset=["OS_Days", "Event", "Age"]).copy()
    df_gbm_158 = df_gbm_158[df_gbm_158["OS_Days"] > 0]
    df_brca_1050 = pc_reg[pc_reg["Cohort"] == "BRCA"].dropna(subset=["OS_Days", "Event", "Age", "Stage_Numeric"]).copy()
    df_brca_1050 = df_brca_1050[df_brca_1050["OS_Days"] > 0]

    paad_rsem_dedup = paad_rsem.drop_duplicates(subset=["PID"]).set_index("PID")
    paad_h_rsem, paad_s_rsem, paad_in_rsem = [], [], []
    for _, r in df_paad.iterrows():
        bc = r["Sample_Barcode"]
        if bc in paad_rsem_dedup.index:
            paad_h_rsem.append(float(paad_rsem_dedup.loc[bc, "HIF1A"]))
            paad_s_rsem.append(float(paad_rsem_dedup.loc[bc, "SLC7A1"]))
            paad_in_rsem.append(True)
        else:
            paad_in_rsem.append(False)
    paad_in_rsem = np.array(paad_in_rsem)
    n_paad_dropped = int((~paad_in_rsem).sum())

    gbm_158_exp = gbm_rsem.loc[df_gbm_158["Sample_ID"]]
    brca_1050_exp = brca_rsem.loc[df_brca_1050["Sample_ID"]]

    h_vivo = np.concatenate([np.array(paad_h_rsem), gbm_158_exp["HIF1A"].values, brca_1050_exp["HIF1A"].values])
    s_vivo = np.concatenate([np.array(paad_s_rsem), gbm_158_exp["SLC7A1"].values, brca_1050_exp["SLC7A1"].values])
    n_vivo = len(h_vivo)
    r_vivo, p_vivo = stats.pearsonr(h_vivo, s_vivo)

    h_z_vivo = np.concatenate([
        zscore(pd.Series(paad_h_rsem)).values,
        zscore(gbm_158_exp["HIF1A"]).values,
        zscore(brca_1050_exp["HIF1A"]).values,
    ])
    s_z_vivo = np.concatenate([
        zscore(pd.Series(paad_s_rsem)).values,
        zscore(gbm_158_exp["SLC7A1"]).values,
        zscore(brca_1050_exp["SLC7A1"]).values,
    ])
    r_z_vivo, p_z_vivo = stats.pearsonr(h_z_vivo, s_z_vivo)

    expr_fp = os.path.join(DATA_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    df_eh = pd.read_csv(expr_fp, nrows=1)
    e_cols = [df_eh.columns[0]]
    rename_e = {}
    for g in ["HIF1A", "SLC7A1"]:
        m = [c for c in df_eh.columns if c.split(" ")[0].strip("()") == g]
        if m:
            e_cols.append(m[0])
            rename_e[m[0]] = g
    df_depmap_e = pd.read_csv(expr_fp, usecols=e_cols).rename(columns=rename_e).dropna(subset=["HIF1A", "SLC7A1"])
    n_vitro = len(df_depmap_e)
    r_vitro, p_vitro = stats.pearsonr(df_depmap_e["HIF1A"], df_depmap_e["SLC7A1"])

    se_fz = np.sqrt(1.0 / (n_vivo - 3) + 1.0 / (n_vitro - 3))
    fisher_z_rsem = float((np.arctanh(r_vivo) - np.arctanh(r_vitro)) / se_fz)
    p_fisher_rsem = float(2.0 * stats.norm.sf(abs(fisher_z_rsem)))

    fisher_z_zscore = float((np.arctanh(r_z_vivo) - np.arctanh(r_vitro)) / se_fz)
    p_fisher_zscore = float(2.0 * stats.norm.sf(abs(fisher_z_zscore)))

    assert n_paad_dropped == 0, f"Expected 0 PAAD RSEM drops on clean PAAD build, got {n_paad_dropped}"
    assert 1378 <= n_vivo <= 1386, f"Unexpected pooled in vivo count: {n_vivo}"
    assert r_vivo > 0.30, f"Expected pooled PAAD+GBM+BRCA RSEM r > 0.30, got {r_vivo:.4f}"
    print(f"  * Complete-case TCGA In Vivo (PAAD n={len(paad_h_rsem)} [{n_paad_dropped} dropped] + GBM n=158 + BRCA n=1,050; Total N = {n_vivo:,}):")
    print(f"      Uniform log2(RSEM+1): Pearson r = +{r_vivo:.4f}, p = {p_vivo:.2e} | vs DepMap (N={n_vitro:,}, r=+{r_vitro:.4f}): Fisher Z = {fisher_z_rsem:.2f} (p = {p_fisher_rsem:.2e})")
    print(f"      Within-Cohort Z-Score: Pearson r = +{r_z_vivo:.4f}, p = {p_z_vivo:.2e} | vs DepMap (N={n_vitro:,}, r=+{r_vitro:.4f}): Fisher Z = {fisher_z_zscore:.2f} (p = {p_fisher_zscore:.2e})")

    # Pure PDAC histological filter and nested likelihood-ratio test (LRT) vs MKI67
    print("\n[2] PURE PDAC HISTOLOGICAL FILTER & NESTED LIKELIHOOD-RATIO TEST (LRT) VS MKI67")

    for g in ["SMARCA4", "HIF1A", "SLC7A1", "SLC1A5", "ABCB1", "MKI67"]:
        zcol = f"{g}_z"
        lcol = _expr_col(df_paad, g)
        if zcol not in df_paad.columns and lcol in df_paad.columns:
            df_paad[zcol] = zscore(df_paad[lcol])

    df_paad["Risk_2Gene"] = df_paad["SMARCA4_z"] + df_paad["HIF1A_z"]
    df_paad["Risk_ZScore"] = zscore(df_paad["Risk_2Gene"])

    hist_out_fp = os.path.join(DATA_PROC, "tcga_paad_histology_mki67_results.csv")
    df_paad.to_csv(hist_out_fp, index=False)

    raph_col = "Pure_Ductal_Raphael2017" if "Pure_Ductal_Raphael2017" in df_paad.columns else "Pure_Ductal_cBioPortal"
    df_pdac_raphael = df_paad[df_paad[raph_col] == 1].copy()
    df_pdac_raphael["Risk_ZScore"] = zscore(df_pdac_raphael["Risk_2Gene"])
    pdac_raph_pur_res = run_cox(df_pdac_raphael, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])
    pdac_raph_np_res = run_cox(df_pdac_raphael, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced"])

    df_pdac_cbio = df_paad[df_paad["Pure_Ductal_cBioPortal"] == 1].copy()
    df_pdac_cbio["Risk_ZScore"] = zscore(df_pdac_cbio["Risk_2Gene"])
    pdac_cbio_res = run_cox(df_pdac_cbio, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])

    df_pdac_nonpnet = df_paad[df_paad["Non_PNET"] == 1].copy()
    df_pdac_nonpnet["Risk_ZScore"] = zscore(df_pdac_nonpnet["Risk_2Gene"])
    pdac_nonpnet_res = run_cox(df_pdac_nonpnet, "OS_Days", "OS_Status", "Risk_ZScore", ["Age", "Stage_Advanced", "Tumor_Purity"])

    print(f"  * Pure Ductal PDAC (Raphael et al. 2017 whitelist + Purity)    : N = {pdac_raph_pur_res['N']} ({pdac_raph_pur_res['Events']} deaths) -> HR = {pdac_raph_pur_res['HR']:.3f} [{pdac_raph_pur_res['CI_Low']:.3f}-{pdac_raph_pur_res['CI_High']:.3f}], p = {pdac_raph_pur_res['P_Value']:.2e}, C = {pdac_raph_pur_res['C_Index']:.3f}")
    print(f"  * Pure Ductal PDAC (Raphael et al. 2017 whitelist, Age+Stage)  : N = {pdac_raph_np_res['N']} ({pdac_raph_np_res['Events']} deaths) -> HR = {pdac_raph_np_res['HR']:.3f} [{pdac_raph_np_res['CI_Low']:.3f}-{pdac_raph_np_res['CI_High']:.3f}], p = {pdac_raph_np_res['P_Value']:.2e}, C = {pdac_raph_np_res['C_Index']:.3f}")
    print(f"  * cBioPortal 'Pancreas Adenocarcinoma, Ductal Type' (+ Purity) : N = {pdac_cbio_res['N']} ({pdac_cbio_res['Events']} deaths) -> HR = {pdac_cbio_res['HR']:.3f} [{pdac_cbio_res['CI_Low']:.3f}-{pdac_cbio_res['CI_High']:.3f}], p = {pdac_cbio_res['P_Value']:.2e}, C = {pdac_cbio_res['C_Index']:.3f}")
    print(f"  * Non-PNET TCGA-PAAD (Excluding 8246/3 Neuroendocrine)         : N = {pdac_nonpnet_res['N']} ({pdac_nonpnet_res['Events']} deaths) -> HR = {pdac_nonpnet_res['HR']:.3f} [{pdac_nonpnet_res['CI_Low']:.3f}-{pdac_nonpnet_res['CI_High']:.3f}], p = {pdac_nonpnet_res['P_Value']:.2e}, C = {pdac_nonpnet_res['C_Index']:.3f}")

    # Nested Likelihood-Ratio Test (LRT) vs MKI67 on strict complete-case (-01) primary tumor overlap
    mki_col = _expr_col(df_paad, "MKI67")
    sub_lrt = df_paad[["OS_Days", "OS_Status", mki_col, "Risk_2Gene", "Age", "Stage_Advanced"]].dropna().copy()
    sub_lrt = sub_lrt[sub_lrt["OS_Days"] > 0]
    sub_lrt["MKI67_z"] = zscore(sub_lrt[mki_col])
    sub_lrt["Risk_ZScore"] = zscore(sub_lrt["Risk_2Gene"])
    cph_null = CoxPHFitter(penalizer=0.0)
    cph_null.fit(sub_lrt[["OS_Days", "OS_Status", "MKI67_z", "Age", "Stage_Advanced"]], duration_col="OS_Days", event_col="OS_Status")
    ll_null = float(cph_null.log_likelihood_)

    cph_full = CoxPHFitter(penalizer=0.0)
    cph_full.fit(sub_lrt[["OS_Days", "OS_Status", "MKI67_z", "Risk_ZScore", "Age", "Stage_Advanced"]], duration_col="OS_Days", event_col="OS_Status")
    ll_full = float(cph_full.log_likelihood_)

    lrt_chi2 = 2.0 * (ll_full - ll_null)
    lrt_p = float(stats.chi2.sf(lrt_chi2, df=1))
    print(f"  * Complete-Case Primary Nested LRT vs MKI67 (Null: MKI67 + Age + Stage vs Full: + Risk_ZScore, N = {len(sub_lrt)}, {int(sub_lrt['OS_Status'].sum())} deaths):")
    print(f"      LL_null = {ll_null:.3f}, LL_full = {ll_full:.3f} -> LRT chi2(1) = {lrt_chi2:.2f}, p = {lrt_p:.2e} (C-index: {cph_null.concordance_index_:.3f} -> {cph_full.concordance_index_:.3f})")

    # Uniform continuous per-SD Z-score Cox models across all 10 clinical cohorts
    # Using unweighted a priori two-gene score: Risk_Score = Z(SMARCA4_z + HIF1A_z)
    print("\n[3] UNIFORM CONTINUOUS PER-SD Z-SCORE COX MODELS ACROSS ALL 10 CLINICAL COHORTS")

    def prep_external_cohort(df_src, req_cols, is_skcm=False):
        d = df_src[df_src["OS_Days"] > 0].dropna(subset=req_cols).copy()
        d["Risk_ZScore"] = zscore(d["SMARCA4_zScore"] + d["HIF1A_zScore"])
        if is_skcm:
            d["Is_Metastatic"] = d["Sample_ID"].astype(str).str.slice(13, 15).eq("06").astype(int)
        return d

    df_icgc_pcawg = pcawg_reg[pcawg_reg["Cancer_Type"] == "Pancreatic Cancer"].copy()
    df_icgc_pcawg = df_icgc_pcawg[df_icgc_pcawg["OS_Days"] > 0].dropna(subset=["OS_Days", "Age", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
    df_icgc_pcawg["Event"] = (df_icgc_pcawg["Vital_Status"] == "Deceased").astype(int)
    df_icgc_pcawg["Is_Male"] = (df_icgc_pcawg["Sex"] == "Male").astype(int)
    df_icgc_pcawg["Risk_ZScore"] = zscore(df_icgc_pcawg["SMARCA4_zScore"] + df_icgc_pcawg["HIF1A_zScore"])

    df_gbm = prep_external_cohort(he_reg[he_reg["Cohort"] == "GBM_TCGA"], ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"])
    df_brca = prep_external_cohort(pc_reg[pc_reg["Cohort"] == "BRCA"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"])

    df_cgga["IDH_Mutant"] = (df_cgga["IDH_mutation_status"] == "Mutant").astype(int)
    df_cgga["MGMT_Methylated"] = (df_cgga["MGMTp_methylation_status"] == "methylated").astype(int)
    df_cgga = df_cgga[df_cgga["OS_Days"] > 0].dropna(subset=["OS_Days", "OS_Status", "Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated", "SMARCA4", "HIF1A"]).copy()
    df_cgga["SMARCA4_z"] = zscore(df_cgga["SMARCA4"])
    df_cgga["HIF1A_z"] = zscore(df_cgga["HIF1A"])
    df_cgga["Risk_2Gene"] = df_cgga["SMARCA4_z"] + df_cgga["HIF1A_z"]
    df_cgga["Risk_ZScore"] = zscore(df_cgga["Risk_2Gene"])

    df_blca = prep_external_cohort(he_reg[he_reg["Cohort"] == "BLCA_TCGA"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"])
    df_mb = prep_external_cohort(he_reg[he_reg["Cohort"] == "BRCA_METABRIC"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"])
    df_skcm = prep_external_cohort(he_reg[he_reg["Cohort"] == "SKCM_TCGA"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], is_skcm=True)
    df_kirc = prep_external_cohort(he_reg[he_reg["Cohort"] == "KIRC_TCGA"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"])
    df_sarc = prep_external_cohort(bi_reg[bi_reg["Cohort"] == "SARC"], ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"])
    df_lihc = prep_external_cohort(bi_reg[bi_reg["Cohort"] == "LIHC"], ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"])

    gsea_fp = os.path.join(DATA_PROC, "hallmark_gsea_transcriptome_wide_enrichment.csv")
    n_gsea_paad = int(pd.read_csv(gsea_fp).loc[lambda d: d["Cohort"].str.contains("PAAD"), "N_Patients"].iloc[0]) if os.path.exists(gsea_fp) else 178

    df_brca, df_mb = attach_breast_clinical_subtypes(df_brca, df_mb)

    build_meta_and_table1(
        df_paad, df_pdac_raphael, df_icgc_pcawg, df_cgga, df_kirc, df_sarc, df_skcm, df_brca, df_lihc, df_blca, df_mb, df_gbm,
        n_gsea_paad, paad_h_rsem_n=len(paad_h_rsem), n_paad_dropped=n_paad_dropped
    )
    export_curated_10cohort_table(
        df_paad, df_gbm, df_brca, df_cgga, df_blca, df_mb, df_skcm, df_kirc, df_sarc, df_lihc, df_icgc_pcawg
    )


if __name__ == "__main__":
    main()
