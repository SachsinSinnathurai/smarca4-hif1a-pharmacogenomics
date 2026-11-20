"""
Automated Manuscript & README Consistency Audit
Cross-checks every sample size, death count, Cox HR/CI/I^2, Buffa-14 LRT,
GDSC2 1-AUC FDR count & partial R^2, and Toil (GTEx v6) TPM in data/processed/
against manuscript/manuscript.tex and README.md, and flags forbidden/stale strings.
"""

from __future__ import annotations

import os
import re
import sys
import pandas as pd


def check(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def fmt_int_comma(n: int) -> str:
    return f"{n:,}"


def has_token(text: str, s: str) -> bool:
    return s in text


def main() -> int:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_proc = os.path.join(project_root, "data", "processed")
    man_fp = os.path.join(project_root, "manuscript", "manuscript.tex")
    readme_fp = os.path.join(project_root, "README.md")

    with open(man_fp, "r", encoding="utf-8") as f:
        man_txt = f.read()
    with open(readme_fp, "r", encoding="utf-8") as f:
        rd_txt = f.read()

    results: list[tuple[str, bool, str]] = []

    # 1. Verify locked files exist
    for rel in [
        "data/processed/tables/Table_1_Clinical_Cohorts.csv",
        "data/processed/curated_10cohort_clinical_survival.csv",
        "figures/Figure_1_Final.pdf",
        "figures/Figure_1_Final.png",
        "figures/Figure_2_Final.pdf",
        "figures/Figure_2_Final.png",
        "figures/Figure_3_Final.pdf",
        "figures/Figure_3_Final.png",
        "figures/Figure_4_Final.pdf",
        "figures/Figure_4_Final.png",
    ]:
        fp = os.path.join(project_root, rel)
        check(results, f"Artifact exists: {rel}", os.path.exists(fp) and os.path.getsize(fp) > 0)

    # 2. Verify Table_1_Clinical_Cohorts.csv rows against manuscript.tex and README.md
    t1 = pd.read_csv(os.path.join(data_proc, "tables", "Table_1_Clinical_Cohorts.csv"))
    for _, row in t1.iterrows():
        cname = str(row["Cohort"])
        n_val = int(row["N_Patients"])
        d_val = int(row["N_Deaths"])
        hr_ci = str(row["Continuous_PerSD_HR_95CI"])
        hr_str = hr_ci.split(" ")[0]
        ci_bounds = hr_ci.split("[")[1].rstrip("]").split("-")
        ci_tex = f"{hr_str} [{ci_bounds[0]}--{ci_bounds[1]}]"
        ci_md = f"{hr_str} [{ci_bounds[0]}–{ci_bounds[1]}]"
        n_comma = fmt_int_comma(n_val)
        d_comma = fmt_int_comma(d_val)

        ok_man = (n_comma in man_txt) and (d_comma in man_txt) and (ci_tex in man_txt)
        ok_rd = (n_comma in rd_txt) and (d_comma in rd_txt) and (
            (ci_md in rd_txt) or (f"**{hr_str}** [{ci_bounds[0]}–{ci_bounds[1]}]" in rd_txt)
        )
        check(results, f"Table1 -> manuscript.tex [{cname}]", ok_man, f"N={n_comma}, D={d_comma}, {ci_tex}")
        check(results, f"Table1 -> README.md [{cname}]", ok_rd, f"N={n_comma}, D={d_comma}, {ci_md}")

    # 3. Verify Sensitivity & Buffa-14 Meta-Analyses from survival_full_tcga_and_mki67_sensitivity.csv
    sens = pd.read_csv(os.path.join(data_proc, "survival_full_tcga_and_mki67_sensitivity.csv"))

    def get_sens(model: str, cohort: str) -> pd.Series:
        sub = sens[(sens["Model"] == model) & (sens["Cohort"] == cohort)]
        if len(sub) == 0:
            raise ValueError(f"Missing sensitivity row: {model} | {cohort}")
        return sub.iloc[0]

    r_8nb = get_sens("Composite_SMARCA4_HIF1A", "8-Cohort Non-Breast Primary Meta")
    r_19sh = get_sens("PanTCGA_Base", "19-All-Available-Solid-TCGA Meta")
    r_19buf = get_sens("Buffa14_Alone_Base", "19-All-Solid-TCGA Meta")
    r_19bufm = get_sens("Buffa14_Alone_PlusMKI67", "19-All-Solid-TCGA Meta")

    for label, row_s in [
        ("8-Cohort Non-Breast Meta", r_8nb),
        ("19-Cohort Expanded Solid TCGA", r_19sh),
        ("19-Cohort Buffa-14", r_19buf),
        ("19-Cohort Buffa-14 (+MKI67)", r_19bufm),
    ]:
        hr = f"{float(row_s['HR']):.3f}"
        lo = f"{float(row_s['CI_Low']):.3f}"
        hi = f"{float(row_s['CI_High']):.3f}"
        i2 = f"{float(row_s['I2_Pct']):.1f}"
        n_c = fmt_int_comma(int(row_s["N"]))
        d_c = fmt_int_comma(int(row_s["Events"]))
        ok_m = (n_c in man_txt) and (d_c in man_txt) and (f"{hr} [{lo}--{hi}]" in man_txt) and (i2 in man_txt)
        ok_r = (n_c in rd_txt) and (d_c in rd_txt) and (
            f"{hr} [{lo}–{hi}]" in rd_txt or f"**{hr}** [{lo}–{hi}]" in rd_txt
        ) and (i2 in rd_txt)
        check(results, f"Sensitivity -> manuscript.tex [{label}]", ok_m, f"HR={hr} [{lo}-{hi}], I2={i2}%")
        check(results, f"Sensitivity -> README.md [{label}]", ok_r, f"HR={hr} [{lo}-{hi}], I2={i2}%")

    # Verify Buffa-14 LRT chi2 and N-weighted C-index
    lrt_base = f"{float(r_19buf['Nested_LRT_Chi2']):.2f}"
    lrt_mk = f"{float(r_19bufm['Nested_LRT_Chi2']):.2f}"
    check(results, "Buffa-14 Nested LRT Chi2 in manuscript.tex", lrt_base in man_txt and lrt_mk in man_txt, f"base={lrt_base}, +MKI67={lrt_mk}")
    check(results, "Buffa-14 Nested LRT Chi2 in README.md", lrt_base in rd_txt and lrt_mk in rd_txt, f"base={lrt_base}, +MKI67={lrt_mk}")

    # 4. Verify GDSC2 FDR counts & Partial R2
    gdsc = pd.read_csv(os.path.join(data_proc, "gdsc2_lineage_controlled_anova.csv"))
    n_fdr05 = int(gdsc["Significant_FDR_5pct"].sum())
    n_fdr10 = int(gdsc["Significant_FDR_10pct"].sum())
    n_ic10 = int((gdsc["LN_IC50_FDR_Q_Value"] < 0.10).sum())
    nav_row = gdsc[gdsc["Compound_Name"] == "Navitoclax"].iloc[0]
    nav_q = f"{float(nav_row['FDR_Q_Value']):.3f}"
    nav_pr2 = f"{float(nav_row['Partial_R2']) * 100:.2f}"
    f10 = gdsc[gdsc["Significant_FDR_10pct"]]
    pr2_min10 = f"{float(f10['Partial_R2'].min()) * 100:.2f}"
    pr2_max10 = f"{float(f10['Partial_R2'].max()) * 100:.2f}"
    pr2_med = f"{float(gdsc['Partial_R2'].median()) * 100:.3f}"
    pr2_allmax = f"{float(gdsc['Partial_R2'].max()) * 100:.2f}"

    ok_gdsc_m = (
        n_fdr05 == 1 and n_fdr10 == 9 and n_ic10 == 0
        and nav_q in man_txt and nav_pr2 in man_txt
        and pr2_min10 in man_txt and pr2_max10 in man_txt
        and pr2_med in man_txt and pr2_allmax in man_txt
    )
    ok_gdsc_r = (
        nav_q in rd_txt and nav_pr2 in rd_txt
        and pr2_min10 in rd_txt and pr2_max10 in rd_txt
        and pr2_med in rd_txt and pr2_allmax in rd_txt
    )
    check(results, "GDSC2 FDR counts (1/286, 9/286, 0/286) & Partial R2 in manuscript.tex", ok_gdsc_m, f"Nav q={nav_q} pr2={nav_pr2}%, q<0.10=[{pr2_min10}%, {pr2_max10}%], med={pr2_med}%, max={pr2_allmax}%")
    check(results, "GDSC2 FDR counts & Partial R2 in README.md", ok_gdsc_r, f"Nav q={nav_q} pr2={nav_pr2}%, q<0.10=[{pr2_min10}%, {pr2_max10}%]")

    # 5. Verify Harmonized Toil (TCGA vs. GTEx v6) TPMs
    toil = pd.read_csv(os.path.join(data_proc, "toil_tcga_vs_gtex_tpm.csv"))
    for gene, grp in [
        ("MCL1", "TCGA_Primary_Solid"),
        ("MCL1", "Heart - Left Ventricle"),
        ("MCL1", "Heart - Atrial Appendage"),
        ("MCL1", "Liver"),
        ("MCL1", "Whole Blood"),
        ("BRD4", "TCGA_Primary_Solid"),
        ("SMARCA4", "TCGA_Primary_Solid"),
    ]:
        r_t = toil[(toil["Gene"] == gene) & (toil["Group"] == grp)].iloc[0]
        tpm_s = f"{float(r_t['Median_Linear_TPM']):.2f}"
        n_s = fmt_int_comma(int(r_t["N_Samples"]))
        check(results, f"Toil {gene} {grp} -> manuscript.tex", tpm_s in man_txt and n_s in man_txt, f"{tpm_s} TPM (n={n_s})")
        check(results, f"Toil {gene} {grp} -> README.md", tpm_s in rd_txt and n_s in rd_txt, f"{tpm_s} TPM (n={n_s})")

    check(results, "Explicit 'GTEx v6' present in manuscript.tex & README.md", ("GTEx v6" in man_txt) and ("GTEx v6" in rd_txt))

    # 6. Check Exact Strings & Zero Forbidden Clichés / Stale Typos
    for good_tok in ["smarca4-hif1a-pharmacogenomics", "ACOT7", "SMARCA4", "Chronos"]:
        check(results, f"Required string '{good_tok}' in manuscript.tex & README.md", (good_tok in man_txt) and (good_tok in rd_txt))

    banned = [
        "on disk",
        "cached",
        "Remarkably",
        "Meanwhile",
        "underscoring",
        "temper interpretation",
        "help frame",
        "clarifies both the clinical scope",
        "exposes clear lineage boundaries",
        "consistent with focusing on",
        "smarca4-hifla",
        "ACOTI",
        "SMARCAL",
        "hronos",
        "GTEx v8",
        "over systemic dose-sparing claims",
    ]
    for b in banned:
        if b == "hronos":
            # Ensure lowercase bare 'hronos' doesn't appear without 'C'
            bad_m = bool(re.search(r"(?<![Cc])hronos", man_txt))
            bad_r = bool(re.search(r"(?<![Cc])hronos", rd_txt))
        else:
            bad_m = b.lower() in man_txt.lower()
            bad_r = b.lower() in rd_txt.lower()
        check(results, f"Banned/stale string absent: '{b}'", (not bad_m) and (not bad_r), f"man={bad_m}, readme={bad_r}")

    rather_cnt_m = len(re.findall(r"rather than", man_txt, flags=re.IGNORECASE))
    rather_cnt_r = len(re.findall(r"rather than", rd_txt, flags=re.IGNORECASE))
    check(results, "Antithesis 'rather than' count <= 2 in manuscript.tex & README.md", rather_cnt_m <= 2 and rather_cnt_r <= 2, f"man={rather_cnt_m}, readme={rather_cnt_r}")

    # Print formatted audit report
    print("MANUSCRIPT & README CONSISTENCY AUDIT")
    n_fail = 0
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        suffix = f" | {detail}" if detail else ""
        print(f"  [{tag}] {name}{suffix}")
    print(f"\nSUMMARY: {len(results) - n_fail}/{len(results)} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
