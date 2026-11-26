"""
Step 4: Compute primary 18-cohort stage/grade TCGA survival & Buffa-14 meta-analyses
(excluding TCGA-PRAD, N = 7,382, 2,352 deaths) alongside 19-cohort sensitivity (N = 7,875),
export GDSC2 post-MKI67 BH step-up table (i = 1..6), verify every number and tag across
manuscript/revised_manuscript.md and manuscript/revised_manuscript.tex, and emit
audit/final_sync_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def dersimonian_laird(betas: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    w = 1.0 / (ses ** 2)
    mu_fe = float(np.sum(w * betas) / np.sum(w))
    q = float(np.sum(w * ((betas - mu_fe) ** 2)))
    df = len(betas) - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q * 100.0) if q > df and q > 0 else 0.0
    w_re = 1.0 / (ses ** 2 + tau2)
    mu_re = float(np.sum(w_re * betas) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    z_re = float(mu_re / se_re)
    p_re = float(2.0 * stats.norm.sf(abs(z_re)))
    return {
        "beta": mu_re,
        "se": se_re,
        "hr": float(math.exp(mu_re)),
        "ci_low": float(math.exp(mu_re - 1.96 * se_re)),
        "ci_high": float(math.exp(mu_re + 1.96 * se_re)),
        "z": z_re,
        "p": p_re,
        "i2_pct": i2,
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_latex_balance(tex: str) -> tuple[bool, list[str]]:
    errs: list[str] = []
    # Check braces (ignoring escaped \{ and \})
    clean = re.sub(r"\\[{}]", "", tex)
    depth = 0
    for idx, ch in enumerate(clean):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errs.append(f"Unmatched closing brace near char {idx}")
                break
    if depth != 0:
        errs.append(f"Unbalanced braces: net depth={depth}")

    # Check \begin{env} / \end{env}
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                errs.append(f"Mismatched \\end{{{env}}} (stack={stack})")
                break
            stack.pop()
    if stack:
        errs.append(f"Unclosed environments: {stack}")
    return (len(errs) == 0), errs


def export_tables(project_root: str, repo_root: str) -> dict:
    data_proc = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(repo_root, "results")

    df_sens = pd.read_csv(os.path.join(data_proc, "survival_full_tcga_and_mki67_sensitivity.csv"))
    df_gdsc = pd.read_csv(os.path.join(out_dir, "gdsc2_post_mki67_fdr.csv"))
    df_geo = pd.read_csv(os.path.join(out_dir, "geo_hypoxia_clean.csv"))

    # 1. Primary 18 Stage/Grade-Adjusted Solid TCGA Cohorts (excluding TCGA-PRAD)
    b18 = df_sens[(df_sens["Model"] == "SMARCA4_HIF1A_Base") & (df_sens["Cohort"].str.startswith("TCGA-")) & (df_sens["Cohort"] != "TCGA-PRAD")].copy()
    m18 = df_sens[(df_sens["Model"].isin(["Plus_MKI67", "PanTCGA_Plus_MKI67"])) & (df_sens["Cohort"].str.startswith("TCGA-")) & (df_sens["Cohort"] != "TCGA-PRAD")].drop_duplicates("Cohort").copy()
    b18["Base_FDR_q_18"] = multipletests(b18["P_Value"].astype(float).values, method="fdr_bh")[1]
    m18["MKI67_FDR_q_18"] = multipletests(m18["P_Value"].astype(float).values, method="fdr_bh")[1]

    df_18_fdr = b18[["Cohort", "N", "Events", "HR", "CI_Low", "CI_High", "P_Value", "Base_FDR_q_18", "C_Index"]].rename(
        columns={"HR": "Base_HR", "CI_Low": "Base_CI_Low", "CI_High": "Base_CI_High", "P_Value": "Base_P", "C_Index": "Base_C_Index"}
    ).merge(
        m18[["Cohort", "HR", "CI_Low", "CI_High", "P_Value", "MKI67_FDR_q_18", "C_Index"]].rename(
            columns={"HR": "MKI67_HR", "CI_Low": "MKI67_CI_Low", "CI_High": "MKI67_CI_High", "P_Value": "MKI67_P", "C_Index": "MKI67_C_Index"}
        ),
        on="Cohort", how="inner"
    ).sort_values("MKI67_P").reset_index(drop=True)
    df_18_fdr.to_csv(os.path.join(out_dir, "survival_18tcga_fdr.csv"), index=False)

    # Compute 18-cohort DerSimonian-Laird & Buffa-14 nested LRTs directly from cohort rows
    cohorts_18 = list(df_18_fdr["Cohort"])
    buffa_18_stats = {}
    for mod in ["SMARCA4_HIF1A_Base", "SMARCA4_HIF1A_PlusMKI67", "Buffa14_Alone_Base", "Buffa14_Alone_PlusMKI67", "SMARCA4_Buffa14_Base", "SMARCA4_Buffa14_PlusMKI67"]:
        sub = df_sens[(df_sens["Model"] == mod) & (df_sens["Cohort"].isin(cohorts_18))].copy()
        dl = dersimonian_laird(sub["Beta"].astype(float).values, sub["SE"].astype(float).values)
        n_tot = int(sub["N"].sum())
        e_tot = int(sub["Events"].sum())
        wt_c = float(np.average(sub["C_Index"].astype(float).values, weights=sub["N"].astype(float).values))
        lrt_chi2 = float(sub["Nested_LRT_Chi2"].dropna().sum()) if sub["Nested_LRT_Chi2"].notna().any() else None
        lrt_p = float(stats.chi2.sf(lrt_chi2, df=len(sub))) if lrt_chi2 is not None else None
        buffa_18_stats[mod] = {
            "n_cohorts": int(len(sub)), "N": n_tot, "Events": e_tot,
            **dl, "n_weighted_c_index": wt_c, "nested_lrt_chi2": lrt_chi2, "nested_lrt_p": lrt_p,
        }

    # 2. GDSC2 post-MKI67 BH Step-Up top 6
    df_g_sorted = df_gdsc.sort_values("MKI67_Adj_P_Value").reset_index(drop=True)
    m_drugs = len(df_g_sorted)
    step_rows = []
    for idx in range(6):
        r = df_g_sorted.iloc[idx]
        rank_i = idx + 1
        raw_p = float(r["MKI67_Adj_P_Value"])
        unclamped = raw_p * m_drugs / rank_i
        clamped_q = float(r["MKI67_Adj_FDR_Q"])
        step_rows.append({
            "Rank_i": rank_i,
            "Compound_Name": str(r["Compound_Name"]),
            "Putative_Target": str(r["Putative_Target"]),
            "N_Solid_Lines": int(r["N_Solid_Lines"]),
            "MKI67_Adj_F": float(r["MKI67_Adj_ANOVA_F"]),
            "MKI67_Adj_Partial_R2": float(r["MKI67_Adj_Partial_R2"]),
            "Raw_P_i": raw_p,
            "Unclamped_BH": unclamped,
            "Clamped_BH_Q_i": clamped_q,
        })
    pd.DataFrame(step_rows).to_csv(os.path.join(out_dir, "gdsc2_stepup_top6.csv"), index=False)

    # 3. Clean Table 2 GEO rows (24 powered single-lineage t-tests with BH-FDR) & CRA001160 5-gene BH-FDR
    df_geo_pow = df_geo[(~df_geo["Descriptive_Only"]) & (df_geo["GEO_Accession"] != "GSE145108")].copy()
    df_geo_pow["BH_FDR_q_24"] = multipletests(df_geo_pow["P_Value"].astype(float).values, method="fdr_bh")[1]
    geo_rows = []
    for (acc, comp), grp in df_geo_pow.groupby(["GEO_Accession", "Comparison"], sort=False):
        gmap = grp.set_index("Gene")
        geo_rows.append({
            "GEO_Accession": acc,
            "Comparison": comp,
            "Cell_Line": str(grp["Cell_Line"].iloc[0]),
            "N_Norm": int(grp["N_Normoxia"].iloc[0]),
            "N_Hyp": int(grp["N_Hypoxia"].iloc[0]),
            **{
                f"{g}_{fld}": float(gmap.loc[g, col])
                for g in ["CA9", "VEGFA", "HIF1A", "SLC7A1", "SLC1A5", "SMARCA4"]
                for fld, col in [("log2fc", "Log2FC"), ("p", "P_Value"), ("q24", "BH_FDR_q_24")]
            },
        })

    df_sc = pd.read_csv(os.path.join(data_proc, "pdac_scrna_celltype_specificity.csv"))
    sc_mal = df_sc[df_sc["Cell_Type"] == "Malignant"].set_index("Target_Gene")
    sc_genes = ["SLC7A1", "SMARCA4", "ABCB1", "HIF1A", "SLC1A5"]
    sc_ps = [float(sc_mal.loc[g, "Paired_Wilcoxon_P_vs_CAF"]) for g in sc_genes]
    sc_qs = multipletests(sc_ps, method="fdr_bh")[1]
    cra_fdr = {
        g: {
            "log2fc": float(sc_mal.loc[g, "Pseudobulk_Log2FC_Malignant_vs_CAF"]),
            "raw_p": float(p),
            "bh_fdr_q": float(q),
            "bonferroni_p": float(min(1.0, p * len(sc_genes))),
        }
        for g, p, q in zip(sc_genes, sc_ps, sc_qs)
    }

    metrics = {
        "primary_18_stage_grade_tcga": {
            "fdr05_base_count": int((df_18_fdr["Base_FDR_q_18"] < 0.05).sum()),
            "fdr05_mki67_count": int((df_18_fdr["MKI67_FDR_q_18"] < 0.05).sum()),
            "fdr10_mki67_count": int((df_18_fdr["MKI67_FDR_q_18"] < 0.10).sum()),
            "top_post_mki67_18": df_18_fdr.head(4).to_dict(orient="records"),
            "models": buffa_18_stats,
        },
        "gdsc2_stepup_top6": step_rows,
        "geo_table2": geo_rows,
        "cra001160_5gene_fdr": cra_fdr,
    }
    with open(os.path.join(out_dir, "final_verified_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def verify_manuscripts(project_root: str, repo_root: str, metrics: dict) -> int:
    out_dir = os.path.join(repo_root, "results")
    with open(os.path.join(repo_root, "audit", "input_hashes.json"), "r", encoding="utf-8") as f:
        hash_doc = json.load(f)
    with open(os.path.join(out_dir, "depmap_and_hypoxia_summary.json"), "r", encoding="utf-8") as f:
        dep_sum = json.load(f)
    with open(os.path.join(out_dir, "survival_and_gdsc2_summary.json"), "r", encoding="utf-8") as f:
        surv_sum = json.load(f)

    md_fp = os.path.join(repo_root, "manuscript", "Proliferation_Confounding_and_Bulk_Transcriptomic_Limitations_of_the_SMARCA4_HIF1A_Axis_Across_Solid_Tumors.md")
    tex_fp = os.path.join(repo_root, "manuscript", "Proliferation_Confounding_and_Bulk_Transcriptomic_Limitations_of_the_SMARCA4_HIF1A_Axis_Across_Solid_Tumors.tex")
    readme_fp = os.path.join(repo_root, "README.md")

    with open(md_fp, "r", encoding="utf-8") as f:
        md_txt = f.read()
    with open(tex_fp, "r", encoding="utf-8") as f:
        tex_txt = f.read()

    checks: list[dict] = []

    def rec(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # 1. Input SHA-256 integrity
    for rel, info in hash_doc["files"].items():
        fp = os.path.join(project_root, rel)
        rec(f"SHA256 intact: {rel}", os.path.exists(fp) and sha256_file(fp) == info["sha256"])

    # 2. Zero stubs / placeholders / internal filenames in BOTH manuscripts
    banned_tokens = [
        "[VERIFY_DOI_PMID]",
        "[OMITTED_MISSING_DATA]",
        ".py",
        ".csv",
        ".json",
        "systematic negative evaluation",
        "reworked study",
        "clear proliferation and lineage boundaries",
        "Remarkably",
        "Meanwhile",
        "underscoring",
        "a priori",
        "AZD5991",
        "AZD5153",
        "troponin",
        "tron2018",
        "SMARCAL",
    ]
    for doc_lbl, txt in [("MD", md_txt), ("TEX", tex_txt)]:
        for bt in banned_tokens:
            found = bt.lower() in txt.lower()
            rec(f"{doc_lbl} zero forbidden/internal token: '{bt}'", not found)

    # 3. Verify key numbers across BOTH MD and TEX
    m18_sh_b = metrics["primary_18_stage_grade_tcga"]["models"]["SMARCA4_HIF1A_Base"]
    m18_sh_m = metrics["primary_18_stage_grade_tcga"]["models"]["SMARCA4_HIF1A_PlusMKI67"]
    m18_bf_b = metrics["primary_18_stage_grade_tcga"]["models"]["Buffa14_Alone_Base"]
    m18_bf_m = metrics["primary_18_stage_grade_tcga"]["models"]["Buffa14_Alone_PlusMKI67"]
    top18 = metrics["primary_18_stage_grade_tcga"]["top_post_mki67_18"]
    m19b = surv_sum["pan_solid_tcga_19"]["meta_19_base"]
    m19m = surv_sum["pan_solid_tcga_19"]["meta_19_plus_mki67"]
    step6 = metrics["gdsc2_stepup_top6"]
    f1d = dep_sum["fig1d_lineage_matched_expression"]
    vhl = dep_sum["vhl_lof_biology"]

    num_tokens = [
        ("18-cohort N", "7,382"),
        ("18-cohort Base HR*", f"{m18_sh_b['hr']:.3f}"),
        ("18-cohort Base p", f"{m18_sh_b['p']:.3f}"),
        ("18-cohort +MKI67 HR*", f"{m18_sh_m['hr']:.3f}"),
        ("18-cohort +MKI67 p", f"{m18_sh_m['p']:.3f}"),
        ("18-cohort KIRC q18", f"{top18[0]['MKI67_FDR_q_18']:.3f}"),
        ("18-cohort BLCA q18", f"{top18[1]['MKI67_FDR_q_18']:.3f}"),
        ("18-cohort Buffa-14 Base HR*", f"{m18_bf_b['hr']:.3f}"),
        ("18-cohort Buffa-14 +MKI67 HR*", f"{m18_bf_m['hr']:.3f}"),
        ("18-cohort Buffa-14 Base LRT chi2", f"{m18_bf_b['nested_lrt_chi2']:.2f}"),
        ("18-cohort Buffa-14 +MKI67 LRT chi2", f"{m18_bf_m['nested_lrt_chi2']:.2f}"),
        ("19-cohort Sensitivity Base HR*", f"{m19b['HR']:.3f}"),
        ("19-cohort Sensitivity +MKI67 HR*", f"{m19m['HR']:.3f}"),
        ("GDSC2 post-MKI67 0/286", "0/286"),
        ("GDSC2 stepup plateau q", f"{step6[0]['Clamped_BH_Q_i']:.4f}"),
        ("GDSC2 rank 5 ABT737", "ABT737"),
        ("GEO GSE283995 24h SLC7A1 p=4.78", "4.78"),
        ("GEO GSE283995 24h SLC1A5 +0.128", "0.128"),
        ("Fig 1D matched all n=229", str(f1d["depmap_matched_3lin_all_unique_n"])),
        ("Fig 1D matched all Fisher Z=1.02", f"{f1d['fisher_z_matched_all_unique']:.2f}"),
        ("VHL HIF1A +0.3690", f"{vhl['genes']['HIF1A']['vhl_mean']:.4f}"),
        ("VHL EPAS1 -0.3464", f"{vhl['genes']['EPAS1']['vhl_mean']:.4f}"),
    ]
    for lbl, tok in num_tokens:
        rec(f"MD matches [{lbl}]", tok in md_txt, tok)
        rec(f"TEX matches [{lbl}]", tok in tex_txt, tok)

    # 4. LaTeX brace/environment balance & README check
    bal_ok, bal_errs = check_latex_balance(tex_txt)
    rec("LaTeX brace/environment balance", bal_ok, "; ".join(bal_errs))
    rec("README.md exists and non-empty", os.path.exists(readme_fp) and os.path.getsize(readme_fp) > 0)

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    sync_doc = {
        "status": "PASS" if n_fail == 0 else "FAIL",
        "total_checks": len(checks),
        "passed": len(checks) - n_fail,
        "failed": n_fail,
        "checks": checks,
    }
    sync_fp = os.path.join(repo_root, "audit", "final_sync_check.json")
    with open(sync_fp, "w", encoding="utf-8") as f:
        json.dump(sync_doc, f, indent=2)

    print(f"FINAL SYNC AUDIT: {sync_doc['status']} ({sync_doc['passed']}/{sync_doc['total_checks']} PASS, {n_fail} FAIL)")
    for c in checks:
        if c["status"] == "FAIL":
            print(f"  [FAIL] {c['check']} | {c['detail']}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = repo_root
    metrics = export_tables(project_root, repo_root)
    print("Exported 18-cohort primary metrics, GDSC2 step-up table, and GEO Table 2 metrics.")
    if args.export_only:
        m18b = metrics["primary_18_stage_grade_tcga"]["models"]["SMARCA4_HIF1A_Base"]
        m18m = metrics["primary_18_stage_grade_tcga"]["models"]["SMARCA4_HIF1A_PlusMKI67"]
        b18b = metrics["primary_18_stage_grade_tcga"]["models"]["Buffa14_Alone_Base"]
        b18m = metrics["primary_18_stage_grade_tcga"]["models"]["Buffa14_Alone_PlusMKI67"]
        print(f"  18-Cohort SH Base: HR*={m18b['hr']:.3f} [{m18b['ci_low']:.3f}-{m18b['ci_high']:.3f}], p={m18b['p']:.3f}, I2={m18b['i2_pct']:.1f}%")
        print(f"  18-Cohort SH +MKI67: HR*={m18m['hr']:.3f} [{m18m['ci_low']:.3f}-{m18m['ci_high']:.3f}], p={m18m['p']:.3f}, I2={m18m['i2_pct']:.1f}%")
        print(f"  18-Cohort Buffa Base: HR*={b18b['hr']:.3f} [{b18b['ci_low']:.3f}-{b18b['ci_high']:.3f}], p={b18b['p']:.2e}, I2={b18b['i2_pct']:.1f}%, LRT={b18b['nested_lrt_chi2']:.2f} (p={b18b['nested_lrt_p']:.2e})")
        print(f"  18-Cohort Buffa +MKI67: HR*={b18m['hr']:.3f} [{b18m['ci_low']:.3f}-{b18m['ci_high']:.3f}], p={b18m['p']:.2e}, I2={b18m['i2_pct']:.1f}%, LRT={b18m['nested_lrt_chi2']:.2f} (p={b18m['nested_lrt_p']:.2e})")
        print("  Top 18-cohort +MKI67 BH-FDR:", [(x["Cohort"], round(x["MKI67_P"], 4), round(x["MKI67_FDR_q_18"], 4)) for x in metrics["primary_18_stage_grade_tcga"]["top_post_mki67_18"]])
        return 0
    return verify_manuscripts(project_root, repo_root, metrics)


if __name__ == "__main__":
    sys.exit(main())
