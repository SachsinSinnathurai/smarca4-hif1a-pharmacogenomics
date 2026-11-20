import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)


GDSC_MATRIX_FILES = [
    "gdsc-007_matrix_results.csv",
    "gdsc-008_matrix_results.csv",
    "gdsc-009_matrix_results.csv",
    "gdsc-010_matrix_results.csv",
    "sandpiper-01_matrix_results.csv",
]


def resolve_data_path(primary_dir, fallback_dir, filename):
    primary = os.path.join(primary_dir, filename)
    return primary if os.path.exists(primary) else os.path.join(fallback_dir, filename)


def verify_required_inputs(project_root, data_raw, data_proc):
    if not os.path.isdir(data_raw):
        return [data_raw]
    required = [
        os.path.join(data_raw, "Model.csv"),
        os.path.join(data_raw, "CRISPRGeneEffect.csv"),
        os.path.join(data_raw, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"),
        os.path.join(data_raw, "tcga", "TCGA_PAAD_expression_survival.csv"),
        os.path.join(data_raw, "cbioportal_pancan", "paad", "data_mrna_seq_v2_rsem.txt"),
        os.path.join(data_raw, "cbioportal_high_mortality", "gbm_tcga", "data_mrna_seq_v2_rsem.txt"),
        os.path.join(data_raw, "cbioportal_pancan", "brca", "data_mrna_seq_v2_rsem.txt"),
        os.path.join(data_raw, "external", "CGGA_mRNAseq_325_expression_survival.csv"),
        os.path.join(data_raw, "gdsc", "GDSC2_fitted_dose_response_27Oct23.xlsx"),
        resolve_data_path(os.path.join(data_raw, "intermediate"), data_proc, "smarca4_significant_synthetic_lethal_genes.csv"),
        resolve_data_path(os.path.join(data_raw, "intermediate"), data_proc, "smarca4_coessentiality_genomewide.csv"),
        resolve_data_path(os.path.join(data_raw, "clinical_registries"), data_proc, "High_Event_10Cohort_Treatment_Registry.csv"),
        resolve_data_path(os.path.join(data_raw, "clinical_registries"), data_proc, "Full_TCGA_PanCan_Patient_Treatment_Registry.csv"),
        os.path.join(data_proc, "gdsc2_lineage_controlled_anova.csv"),
        os.path.join(data_proc, "pdac_scrna_celltype_specificity.csv"),
        os.path.join(data_proc, "hallmark_gsea_transcriptome_wide_enrichment.csv"),
        os.path.join(data_proc, "tcga_rppa_proteomic_validation.csv"),
        os.path.join(data_proc, "genomic_scars_hrd_aneuploidy_orthogonality.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    gdsc_found = any(
        os.path.exists(resolve_data_path(os.path.join(data_raw, "gdsc"), project_root, fn))
        for fn in GDSC_MATRIX_FILES
    )
    if not gdsc_found:
        missing.append("gdsc-007..010 / sandpiper-01_matrix_results.csv")
    return missing


def fmt_p_tex(p: float) -> str:
    """Format p-values consistently: 3 decimals for p >= 0.001, clean scientific notation otherwise."""
    p = float(p)
    if p == 0.0:
        return r"$< 10^{-300}$"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.2e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def get_km_curve(time_arr, status_arr):
    order = np.argsort(time_arr)
    t_sorted = time_arr[order]
    s_sorted = status_arr[order]
    unique_times = np.unique(t_sorted)
    surv = 1.0
    curve_t = [0.0]
    curve_s = [1.0]
    for ut in unique_times:
        at_risk = np.sum(t_sorted >= ut)
        events = np.sum((t_sorted == ut) & (s_sorted == 1))
        if at_risk > 0:
            surv *= (1.0 - events / at_risk)
        curve_t.append(ut)
        curve_s.append(surv)
    ct = np.array(curve_t)
    cs = np.array(curve_s)
    cens_t = t_sorted[s_sorted == 0]
    cens_s = cs[np.searchsorted(ct, cens_t, side="right") - 1]
    return ct, cs, cens_t, cens_s


def compute_5x5_peak_window_grid(df_gdsc, d1, d2):
    sub = df_gdsc[
        ((df_gdsc['lib1_name'] == d1) & (df_gdsc['lib2_name'] == d2)) |
        ((df_gdsc['lib1_name'] == d2) & (df_gdsc['lib2_name'] == d1))
    ].copy()
    mask_swap = sub['lib1_name'] == d2
    if mask_swap.any():
        b1 = sub.loc[mask_swap, 'Bliss_window_dose1'].values.copy()
        b2 = sub.loc[mask_swap, 'Bliss_window_dose2'].values.copy()
        sub.loc[mask_swap, 'Bliss_window_dose1'] = b2
        sub.loc[mask_swap, 'Bliss_window_dose2'] = b1

    # Collapse replicate matrices per (ModelID, dose1, dose2), then assign each unique ModelID to its single modal/highest-mean peak 3x3 window
    per_line_win = sub.groupby(['ModelID', 'Bliss_window_dose1', 'Bliss_window_dose2'], as_index=False).agg(
        n_mat=('Bliss_window', 'count'),
        Bliss_window=('Bliss_window', 'mean'),
    )
    per_line_dose = (
        per_line_win.sort_values(['ModelID', 'n_mat', 'Bliss_window'], ascending=[True, False, False])
        .drop_duplicates(subset=['ModelID'], keep='first')
    )
    doses = ['D1', 'D2', 'D3', 'D4', 'D5']
    piv = per_line_dose.pivot_table(
        index='Bliss_window_dose1', columns='Bliss_window_dose2',
        values='Bliss_window', aggfunc='mean'
    ).reindex(index=doses, columns=doses)
    cnt_piv = per_line_dose.pivot_table(
        index='Bliss_window_dose1', columns='Bliss_window_dose2',
        values='Bliss_window', aggfunc='count'
    ).reindex(index=doses, columns=doses)
    return piv.values, cnt_piv.values, len(sub), int(sub['ModelID'].nunique())


def setup_rcparams():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9.0,
        'axes.labelsize': 9.2,
        'axes.titlesize': 10.2,
        'xtick.labelsize': 8.2,
        'ytick.labelsize': 8.2,
        'legend.fontsize': 7.6,
        'axes.linewidth': 0.85,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })


def build_figure_1(project_root, data_raw, data_proc, fig_dir):
    log("Building Figure 1...")
    df_m = pd.read_csv(os.path.join(data_raw, "Model.csv"))
    non_solid = ["Lymphoid", "Myeloid", "Normal", "Fibroblast"]
    norm_ids = set(df_m[
        (df_m["OncotreePrimaryDisease"] == "Non-Cancerous") |
        (df_m["OncotreeLineage"].isin(["Normal", "Fibroblast"]))
    ]["ModelID"].dropna().astype(str))
    solid_ids = set(df_m[
        (~df_m["OncotreeLineage"].isin(non_solid)) &
        (~df_m["ModelID"].astype(str).isin(norm_ids))
    ]["ModelID"].dropna().astype(str))

    crispr_file = os.path.join(data_raw, "CRISPRGeneEffect.csv")
    df_head = pd.read_csv(crispr_file, nrows=2)
    id_col = "ModelID" if "ModelID" in df_head.columns else df_head.columns[0]
    target_genes = ["SMARCA4", "HIF1A", "SLC1A5", "SLC7A1"]
    extra_b_genes = ["SMARCB1", "SMARCA2"]
    use_cols = [id_col] + [c for c in df_head.columns if c.split(" ")[0].strip("()") in target_genes + extra_b_genes]
    df_c = pd.read_csv(crispr_file, usecols=use_cols).rename(columns={id_col: "ModelID"})
    df_c["ModelID"] = df_c["ModelID"].astype(str)
    for c in use_cols[1:]:
        df_c = df_c.rename(columns={c: c.split(" ")[0].strip("()")})

    df_solid = df_c[df_c["ModelID"].isin(solid_ids)].dropna(subset=target_genes).copy()
    df_norm = df_c[df_c["ModelID"].isin(norm_ids)].dropna(subset=target_genes).copy()
    df_solid_b = df_solid[["SMARCA4", "SMARCB1", "SMARCA2"]].dropna()
    r_smcb1, p_smcb1 = stats.pearsonr(df_solid_b["SMARCA4"], df_solid_b["SMARCB1"])
    rho_smcb1, _ = stats.spearmanr(df_solid_b["SMARCA4"], df_solid_b["SMARCB1"])
    r_smca2, _ = stats.pearsonr(df_solid_b["SMARCA4"], df_solid_b["SMARCA2"])
    rho_smca2, _ = stats.spearmanr(df_solid_b["SMARCA4"], df_solid_b["SMARCA2"])

    inter_dir = os.path.join(data_raw, "intermediate")
    reg_dir = os.path.join(data_raw, "clinical_registries")
    df_sl = pd.read_csv(resolve_data_path(data_proc, inter_dir, "smarca4_significant_synthetic_lethal_genes.csv"))
    df_volcano = pd.read_csv(resolve_data_path(data_proc, inter_dir, "smarca4_coessentiality_genomewide.csv"))

    def load_rsem_pair(fp, primary_only=False):
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

    df_paad_t = pd.read_csv(os.path.join(data_raw, "tcga", "TCGA_PAAD_expression_survival.csv"))
    he_reg = pd.read_csv(resolve_data_path(reg_dir, data_proc, "High_Event_10Cohort_Treatment_Registry.csv"))
    pc_reg = pd.read_csv(resolve_data_path(reg_dir, data_proc, "Full_TCGA_PanCan_Patient_Treatment_Registry.csv"))
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
    r_vivo, p_vivo = stats.pearsonr(df_tcga["HIF1A"], df_tcga["SLC7A1"])

    # Compute within-cohort Z-scored in vivo correlation across N = 1,385
    df_tcga["H_z"] = df_tcga.groupby("Cohort")["HIF1A"].transform(lambda s: (s - s.mean()) / s.std())
    df_tcga["S_z"] = df_tcga.groupby("Cohort")["SLC7A1"].transform(lambda s: (s - s.mean()) / s.std())
    r_vivo_z, p_vivo_z = stats.pearsonr(df_tcga["H_z"], df_tcga["S_z"])

    expr_fp = os.path.join(data_raw, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    df_eh = pd.read_csv(expr_fp, nrows=1)
    e_cols = ["ModelID", "IsDefaultEntryForModel"]
    rename_e = {}
    for g in ["HIF1A", "SLC7A1"]:
        m = [c for c in df_eh.columns if c.split(" ")[0] == g][0]
        e_cols.append(m); rename_e[m] = g
    df_depmap_all = pd.read_csv(expr_fp, usecols=e_cols).rename(columns=rename_e).dropna(subset=["HIF1A", "SLC7A1"])
    df_depmap_e = df_depmap_all[df_depmap_all["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").copy()
    df_depmap_e["ModelID"] = df_depmap_e["ModelID"].astype(str)
    lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()
    df_depmap_sol = df_depmap_e[df_depmap_e["ModelID"].isin(solid_ids)].copy()
    df_depmap_sol["Lineage"] = df_depmap_sol["ModelID"].map(lin_map)
    df_depmap_sol = df_depmap_sol.dropna(subset=["Lineage"]).groupby("Lineage").filter(lambda g: len(g) >= 2).copy()
    df_depmap_sol["H_z"] = df_depmap_sol.groupby("Lineage")["HIF1A"].transform(lambda s: (s - s.mean()) / s.std())
    df_depmap_sol["S_z"] = df_depmap_sol.groupby("Lineage")["SLC7A1"].transform(lambda s: (s - s.mean()) / s.std())
    df_depmap_sol = df_depmap_sol.dropna(subset=["H_z", "S_z"])
    r_vitro_raw, p_vitro_raw = stats.pearsonr(df_depmap_all["HIF1A"], df_depmap_all["SLC7A1"])
    r_vitro_linz, p_vitro_linz = stats.pearsonr(df_depmap_sol["H_z"], df_depmap_sol["S_z"])

    fig = plt.figure(figsize=(17.5, 12.2), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25, left=0.06, right=0.97, top=0.94, bottom=0.07)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    x, y = df_solid["SLC1A5"].values, df_solid["SLC7A1"].values
    r_val, p_val = stats.pearsonr(x, y)
    ax1.scatter(x, y, color='#2b5c8f', alpha=0.35, s=16, edgecolors='none', label=f"Solid tumor lines (n = {len(df_solid):,})", rasterized=True)
    slope, intercept, _, _, stderr = stats.linregress(x, y)
    xg = np.linspace(x.min(), x.max(), 200)
    yp = intercept + slope * xg
    ci = 1.96 * stderr * np.sqrt(1/len(x) + (xg - x.mean())**2 / np.sum((x - x.mean())**2))
    ax1.plot(xg, yp, color='#d95f02', linewidth=2.0, label=f"OLS fit (r = +{r_val:.3f}, p = {fmt_p_tex(p_val)})")
    ax1.fill_between(xg, yp - ci, yp + ci, color='#d95f02', alpha=0.18)
    ax1.axhline(0, color='#888888', linestyle='--', linewidth=0.7, alpha=0.6)
    ax1.axvline(0, color='#888888', linestyle='--', linewidth=0.7, alpha=0.6)
    ax1.set_title("A   SLC1A5 and SLC7A1 CRISPR Co-Dependency (DepMap 24Q2)", loc='left', fontweight='bold')
    ax1.set_xlabel("SLC1A5 Chronos Score")
    ax1.set_ylabel("SLC7A1 Chronos Score")
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    df_volcano["neg_log10_p"] = -np.log10(np.clip(df_volcano["P_Value"], 1e-25, 1.0))
    p_bh_cutoff = float(df_sl["P_Value"].max())
    bh_neg_log10 = -np.log10(p_bh_cutoff)
    sig_mask = (df_volcano["FDR_Q"] < 0.05) & (df_volcano["Delta"] < -0.05)
    ax2.scatter(df_volcano.loc[~sig_mask, "Delta"], df_volcano.loc[~sig_mask, "neg_log10_p"],
                color='#cccccc', s=10, alpha=0.45, edgecolors='none', rasterized=True)
    ax2.scatter(df_volcano.loc[sig_mask, "Delta"], df_volcano.loc[sig_mask, "neg_log10_p"],
                color='#7294b8', s=15, alpha=0.65, edgecolors='none')
    targets_b = {
        "SMARCB1": ('#e41a1c', (-8, 14), 'center'),
        "SMARCC1": ('#e41a1c', (14, -2), 'left'),
        "SMARCD1": ('#e41a1c', (-14, 4), 'right'),
        "ARID1A":  ('#e41a1c', (-14, 0), 'right'),
        "SMARCE1": ('#e41a1c', (-14, -7), 'right'),
        "MLLT1":   ('#984ea3', (14, 5), 'left'),
        "SMARCA2": ('#d95f02', (-12, 12), 'right'),
    }
    for gene, (col, offset, ha) in targets_b.items():
        sub_r = df_sl[df_sl["Gene"] == gene]
        if len(sub_r) > 0:
            row = sub_r.iloc[0]
            dx, dy = row["Delta_Chronos"], -np.log10(row["P_Value"])
            ax2.scatter([dx], [dy], color=col, s=58, edgecolors='#111111', linewidth=1.0, zorder=5)
            lbl_g = f"{gene} ({dx:+.3f})" if gene == "SMARCA2" else gene
            ax2.annotate(lbl_g, xy=(dx, dy), xytext=offset, textcoords='offset points',
                         ha=ha, va='center', fontsize=8.0, fontweight='bold', color='#111111',
                         arrowprops=dict(arrowstyle='->', color='#333333', lw=0.85, shrinkB=3))
    ax2.axhline(bh_neg_log10, color='#b71c1c', linestyle='--', linewidth=1.0, zorder=2)
    ax2.axvline(0.0, color='#888888', linestyle='--', linewidth=0.7)
    ax2.text(
        0.98, 0.66,
        f"Continuous (N = {len(df_solid_b):,} solid):\n"
        + rf"SMARCB1: $r = {r_smcb1:+.3f}$, $\rho = {rho_smcb1:+.3f}$" + "\n"
        + rf"SMARCA2: $r = {r_smca2:+.3f}$, $\rho = {rho_smca2:+.3f}$",
        transform=ax2.transAxes, ha='right', va='top', fontsize=7.0, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.88),
    )
    ax2.set_title("B   SMARCA4-Dependent Co-Essentiality Scan (18,434 Genes)", loc='left', fontweight='bold')
    ax2.set_xlabel(r"Chronos Shift in SMARCA4-Dependent Lines ($\Delta$ Chronos)")
    ax2.set_ylabel(r"$-\log_{10}$(Welch's $t$-test $P$-value)")
    ax2.set_xlim(-0.44, 0.25); ax2.set_ylim(-0.5, 21.0)
    leg_b = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e41a1c', markersize=7, markeredgecolor='k', label='Core BAF Subunits'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#984ea3', markersize=7, markeredgecolor='k', label='MLLT1 (ENL)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d95f02', markersize=7, markeredgecolor='k', label='SMARCA2 Paralog Buffering'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7294b8', markersize=6, label=r'Co-essential (FDR $q < 0.05$, $\Delta < -0.05$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=5.5, label='Other tested genes'),
        Line2D([0], [0], color='#b71c1c', linestyle='--', lw=1.0, label=r'BH FDR $q = 0.05$ cutoff'),
    ]
    ax2.legend(handles=leg_b, loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.0)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel C: Median ± IQR Box-and-Strip Plots Across Individual Target Genes (single-gene SMARCA4 sparing)
    ax3 = fig.add_subplot(gs[1, 0])
    m_list = ["SMARCA4", "SLC7A1", "SLC1A5", "HIF1A"]
    m_lbls = ["SMARCA4", "SLC7A1", "SLC1A5", "HIF1A"]
    xp = np.arange(len(m_list)); w = 0.34
    rng = np.random.default_rng(42)
    bp_n = ax3.boxplot([df_norm[m].values for m in m_list], positions=xp - w/2, widths=0.28,
                       patch_artist=True, showfliers=False, zorder=3)
    bp_s = ax3.boxplot([df_solid[m].values for m in m_list], positions=xp + w/2, widths=0.28,
                       patch_artist=True, showfliers=False, zorder=3)
    for box in bp_n['boxes']:
        box.set(facecolor='#1b9e77', edgecolor='#111111', linewidth=0.9, alpha=0.65)
    for box in bp_s['boxes']:
        box.set(facecolor='#e41a1c', edgecolor='#111111', linewidth=0.9, alpha=0.60)
    for med in bp_n['medians'] + bp_s['medians']:
        med.set(color='#111111', linewidth=1.6)
    for elem in bp_n['whiskers'] + bp_n['caps'] + bp_s['whiskers'] + bp_s['caps']:
        elem.set(color='#333333', linewidth=0.8)
    for i, m in enumerate(m_list):
        yp_n = df_norm[m].values
        jit_n = rng.uniform(-0.07, 0.07, size=len(yp_n))
        ax3.scatter(xp[i] - w/2 + jit_n, yp_n, color='#004d40', s=14, alpha=0.80, zorder=5, edgecolors='none')
        yp_s = df_solid[m].values
        jit_s = rng.uniform(-0.08, 0.08, size=len(yp_s))
        ax3.scatter(xp[i] + w/2 + jit_s, yp_s, color='#991b1b', s=4, alpha=0.14, zorder=2, edgecolors='none', rasterized=True)
    ax3.axhline(0.0, color='#222222', linewidth=0.8)
    ax3.axhline(-0.5, color='#d95f02', linestyle='--', linewidth=1.1)
    delta_mean_sm = float(df_norm["SMARCA4"].mean() - df_solid["SMARCA4"].mean())
    delta_med_sm = float(df_norm["SMARCA4"].median() - df_solid["SMARCA4"].median())
    _, p_mwu_sm = stats.mannwhitneyu(df_norm["SMARCA4"], df_solid["SMARCA4"], alternative="two-sided")
    ax3.annotate(
        r"$\mathbf{SMARCA4}$" + " diploid sparing:\n"
        + rf"$\Delta_{{\mathrm{{mean}}}} = {delta_mean_sm:+.3f},\ \Delta_{{\mathrm{{med}}}} = {delta_med_sm:+.3f}$"
        + f"\n(p = {fmt_p_tex(p_mwu_sm)})",
        xy=(0 - w / 2, float(df_norm["SMARCA4"].median())),
        xytext=(0.12, -0.92),
        fontsize=7.3,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#1b9e77', lw=1.2),
    )
    ax3.set_xticks(xp); ax3.set_xticklabels(m_lbls, fontsize=7.4, fontweight='bold')
    ax3.set_ylim(-1.35, 0.60)
    ax3.set_title("C   Non-Malignant Diploid Sparing vs. Solid Tumor Dependency", loc='left', fontweight='bold')
    ax3.set_ylabel("CRISPR Chronos Score (Median & IQR)")
    leg_c = [
        mpatches.Patch(facecolor='#1b9e77', edgecolor='#111111', alpha=0.65, label=f"Non-malignant diploid (n = {len(df_norm)})"),
        mpatches.Patch(facecolor='#e41a1c', edgecolor='#111111', alpha=0.60, label=f"Solid tumor lines (n = {len(df_solid):,})"),
        Line2D([0], [0], color='#d95f02', linestyle='--', lw=1.1, label="Essentiality cutoff (-0.50)"),
    ]
    ax3.legend(handles=leg_c, loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.2)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # Panel D: Within-Cohort Z-Scored In Vivo vs Within-Lineage Z-Scored Normoxic Culture
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.32)
    ax4a = fig.add_subplot(gs_d[0, 0])
    ax4b = fig.add_subplot(gs_d[0, 1])

    for cname, col in [("TCGA-PAAD", "#e41a1c"), ("TCGA-GBM", "#2b5c8f"), ("TCGA-BRCA", "#1b9e77")]:
        sub_c = df_tcga[df_tcga["Cohort"] == cname]
        ax4a.scatter(sub_c["H_z"], sub_c["S_z"], color=col, alpha=0.38, s=12, edgecolors='none', label=cname, rasterized=True)
    mv, bv, _, _, _ = stats.linregress(df_tcga["H_z"], df_tcga["S_z"])
    xv = np.linspace(df_tcga["H_z"].min(), df_tcga["H_z"].max(), 100)
    ax4a.plot(xv, bv + mv * xv, color='#111111', linewidth=1.9)
    ax4a.set_title(f"D1  In Vivo Tumors (N = {len(df_tcga):,})", loc='left', fontweight='bold', fontsize=9.2)
    ax4a.set_xlabel("HIF1A Within-Cohort Z-Score")
    ax4a.set_ylabel("SLC7A1 Within-Cohort Z-Score")
    ax4a.text(0.05, 0.94, f"r = +{r_vivo_z:.4f} (raw +{r_vivo:.4f})\np = {fmt_p_tex(p_vivo_z)}",
              transform=ax4a.transAxes, va='top', fontsize=7.5, fontweight='bold')
    ax4a.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=6.8)
    ax4a.spines['top'].set_visible(False); ax4a.spines['right'].set_visible(False)

    ax4b.scatter(df_depmap_sol["H_z"], df_depmap_sol["S_z"], color='#78909c', alpha=0.34, s=12, edgecolors='none', rasterized=True)
    mi, bi, _, _, _ = stats.linregress(df_depmap_sol["H_z"], df_depmap_sol["S_z"])
    xi = np.linspace(df_depmap_sol["H_z"].min(), df_depmap_sol["H_z"].max(), 100)
    ax4b.plot(xi, bi + mi * xi, color='#e41a1c', linestyle='--', linewidth=1.9)
    se_fz = np.sqrt(1.0 / (len(df_tcga) - 3) + 1.0 / (len(df_depmap_sol) - 3))
    fz_val = float((np.arctanh(r_vivo_z) - np.arctanh(r_vitro_linz)) / se_fz)
    p_fz = float(2.0 * stats.norm.sf(abs(fz_val)))
    ax4b.set_title(rf"D2  21% $\mathrm{{O}}_2$ Solid (n = {len(df_depmap_sol):,})", loc='left', fontweight='bold', fontsize=9.2)
    ax4b.set_xlabel("HIF1A Within-Lineage Z-Score")
    ax4b.set_ylabel("SLC7A1 Within-Lineage Z-Score")
    ax4b.text(0.04, 0.95,
              f"Within-lin. (n={len(df_depmap_sol):,}): r = +{r_vitro_linz:.4f} (p = {fmt_p_tex(p_vitro_linz)})\n"
              f"Pan-cancer raw (N={len(df_depmap_all):,}): r = +{r_vitro_raw:.4f}\n"
              f"Fisher Z = {fz_val:.2f} (p = {fmt_p_tex(p_fz)})",
              transform=ax4b.transAxes, va='top', fontsize=7.0, fontweight='bold')
    ax4b.spines['top'].set_visible(False); ax4b.spines['right'].set_visible(False)

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_dir, f"Figure_1_Final.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log("[OK] Saved Figure_1_Final.{pdf,png}")


def _parse_extra_covs(s):
    out = {}
    if pd.isna(s) or not str(s).strip():
        return out
    for item in str(s).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def build_figure_2(project_root, data_raw, data_proc, fig_dir):
    log("Building Figure 2...")
    paad_fp = os.path.join(data_raw, "tcga", "TCGA_PAAD_expression_survival.csv")
    cgga_fp = os.path.join(data_raw, "external", "CGGA_mRNAseq_325_expression_survival.csv")
    reg_dir = os.path.join(data_raw, "clinical_registries")
    he_fp = resolve_data_path(reg_dir, data_proc, "High_Event_10Cohort_Treatment_Registry.csv")
    pcawg_fp = resolve_data_path(reg_dir, data_proc, "PCAWG_1210_Personalized_Treatment_Registry.csv")
    curated_fp = os.path.join(data_proc, "curated_10cohort_clinical_survival.csv")

    def _z(s):
        vals = pd.to_numeric(s, errors="coerce").astype(float)
        sd = vals.std(ddof=1)
        return (vals - vals.mean()) / sd if (sd and not np.isnan(sd)) else pd.Series(0.0, index=vals.index)

    genes = ["SMARCA4", "HIF1A", "SLC7A1", "ABCB1"]
    if os.path.exists(paad_fp) and os.path.exists(cgga_fp) and os.path.exists(he_fp) and os.path.exists(pcawg_fp):
        df_paad = pd.read_csv(paad_fp)
        for g in genes:
            col = f"{g}_log2RSEM" if f"{g}_log2RSEM" in df_paad.columns else f"{g}_log2TPM"
            if f"{g}_z" not in df_paad.columns:
                df_paad[f"{g}_z"] = _z(df_paad[col])
        df_paad["Risk"] = _z(df_paad["SMARCA4_z"] + df_paad["HIF1A_z"])
        df_paad_v = df_paad[df_paad["OS_Days"] > 0].dropna(
            subset=["OS_Days", "OS_Status", "Risk", "Age", "Stage_Advanced", "Tumor_Purity"]
        ).copy()
        df_paad_v["OS_Months"] = df_paad_v["OS_Days"] / 30.4375
        df_paad_v["High"] = (df_paad_v["Risk"] >= df_paad_v["Risk"].median()).astype(int)

        pcawg_reg = pd.read_csv(pcawg_fp)
        df_icgc = pcawg_reg[pcawg_reg["Cancer_Type"] == "Pancreatic Cancer"].copy()
        df_icgc = df_icgc[df_icgc["OS_Days"] > 0].dropna(subset=["OS_Days", "Age", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
        df_icgc["Event"] = (df_icgc["Vital_Status"] == "Deceased").astype(int)
        df_icgc["Risk"] = _z(df_icgc["SMARCA4_zScore"] + df_icgc["HIF1A_zScore"])
        df_icgc["OS_Months"] = df_icgc["OS_Days"] / 30.4375
        df_icgc["High"] = (df_icgc["Risk"] >= df_icgc["Risk"].median()).astype(int)

        df_cgga = pd.read_csv(cgga_fp)
        df_cgga["IDH_Mutant"] = (df_cgga["IDH_mutation_status"] == "Mutant").astype(int)
        df_cgga["MGMT_Methylated"] = (df_cgga["MGMTp_methylation_status"] == "methylated").astype(int)
        df_cgga_v = df_cgga[df_cgga["OS_Days"] > 0].dropna(
            subset=["OS_Days", "OS_Status", "Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated", "SMARCA4", "HIF1A"]
        ).copy()
        df_cgga_v["Risk"] = _z(_z(df_cgga_v["SMARCA4"]) + _z(df_cgga_v["HIF1A"]))
        df_cgga_v["OS_Months"] = df_cgga_v["OS_Days"] / 30.4375
        df_cgga_v["High"] = (df_cgga_v["Risk"] >= df_cgga_v["Risk"].median()).astype(int)

        he_reg = pd.read_csv(he_fp)
        df_kirc = he_reg[he_reg["Cohort"] == "KIRC_TCGA"].dropna(subset=["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
        df_kirc = df_kirc[df_kirc["OS_Days"] > 0]
        df_kirc["Risk"] = _z(df_kirc["SMARCA4_zScore"] + df_kirc["HIF1A_zScore"])
        df_kirc["OS_Months"] = df_kirc["OS_Days"] / 30.4375
        df_kirc["High"] = (df_kirc["Risk"] >= df_kirc["Risk"].median()).astype(int)

        df_blca = he_reg[he_reg["Cohort"] == "BLCA_TCGA"].dropna(subset=["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
        df_blca = df_blca[df_blca["OS_Days"] > 0]
        df_blca["Risk"] = _z(df_blca["SMARCA4_zScore"] + df_blca["HIF1A_zScore"])
        df_blca["OS_Months"] = df_blca["OS_Days"] / 30.4375
        df_blca["High"] = (df_blca["Risk"] >= df_blca["Risk"].median()).astype(int)

        df_mb = he_reg[he_reg["Cohort"] == "BRCA_METABRIC"].dropna(subset=["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
        df_mb = df_mb[df_mb["OS_Days"] > 0]
        df_mb["Risk"] = _z(df_mb["SMARCA4_zScore"] + df_mb["HIF1A_zScore"])
        df_mb["OS_Months"] = df_mb["OS_Days"] / 30.4375
        df_mb["High"] = (df_mb["Risk"] >= df_mb["Risk"].median()).astype(int)
    else:
        df_cur = pd.read_csv(curated_fp)
        df_cur["Risk_Sum"] = df_cur["SMARCA4_z"] + df_cur["HIF1A_z"]
        df_cur["Risk"] = df_cur.groupby("Cohort")["Risk_Sum"].transform(_z)
        df_cur["OS_Months"] = df_cur["OS_Days"] / 30.4375

        df_paad = df_cur[df_cur["Cohort"] == "TCGA-PAAD"].copy()
        ex = df_paad["Extra_Covariates"].apply(_parse_extra_covs)
        for k in ["Tumor_Purity"]:
            df_paad[k] = [d.get(k, np.nan) for d in ex]
        df_paad["Stage_Advanced"] = df_paad["Stage"]
        df_paad["OS_Status"] = df_paad["Event"]
        df_paad_v = df_paad[df_paad["OS_Days"] > 0].dropna(
            subset=["OS_Days", "OS_Status", "Risk", "Age", "Stage_Advanced", "Tumor_Purity"]
        ).copy()
        df_paad_v["High"] = (df_paad_v["Risk"] >= df_paad_v["Risk"].median()).astype(int)

        df_icgc = df_cur[df_cur["Cohort"] == "ICGC/PCAWG-PAAD"].copy()
        df_icgc["High"] = (df_icgc["Risk"] >= df_icgc["Risk"].median()).astype(int)

        df_cgga_v = df_cur[df_cur["Cohort"] == "CGGA-325"].copy()
        df_cgga_v["OS_Status"] = df_cgga_v["Event"]
        df_cgga_v["High"] = (df_cgga_v["Risk"] >= df_cgga_v["Risk"].median()).astype(int)

        df_kirc = df_cur[df_cur["Cohort"] == "TCGA-KIRC"].copy()
        df_kirc["High"] = (df_kirc["Risk"] >= df_kirc["Risk"].median()).astype(int)

        df_blca = df_cur[df_cur["Cohort"] == "TCGA-BLCA"].copy()
        df_blca["High"] = (df_blca["Risk"] >= df_blca["Risk"].median()).astype(int)

        df_mb = df_cur[df_cur["Cohort"] == "BRCA-METABRIC"].copy()
        df_mb["High"] = (df_mb["Risk"] >= df_mb["Risk"].median()).astype(int)

    fig = plt.figure(figsize=(18.5, 13.8), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.38, left=0.055, right=0.975, top=0.94, bottom=0.13)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.94, bottom=0.13)

    t1_df = pd.read_csv(os.path.join(data_proc, "tables", "Table_1_Clinical_Cohorts.csv"))

    def plot_km_panel(sub_df, ax_k, title_k, xmax_k, t1_key, ev_col, xticks_custom=None):
        hi = sub_df[sub_df["High"] == 1]
        lo = sub_df[sub_df["High"] == 0]
        th, sh, ch_t, ch_s = get_km_curve(hi["OS_Months"].values, hi[ev_col].values)
        tl, sl, cl_t, cl_s = get_km_curve(lo["OS_Months"].values, lo[ev_col].values)
        ax_k.step(th, sh, where='post', color='#e41a1c', linewidth=2.0, label=f"High (n={len(hi)})")
        ax_k.step(tl, sl, where='post', color='#2b5c8f', linewidth=2.0, label=f"Low (n={len(lo)})")
        ax_k.plot(ch_t, ch_s, marker='|', linestyle='None', color='#e41a1c', markersize=4.2, markeredgewidth=0.85, alpha=0.75)
        ax_k.plot(cl_t, cl_s, marker='|', linestyle='None', color='#2b5c8f', markersize=4.2, markeredgewidth=0.85, alpha=0.75)

        lr = logrank_test(hi["OS_Months"].values, lo["OS_Months"].values, event_observed_A=hi[ev_col].values, event_observed_B=lo[ev_col].values)
        row_t1 = t1_df[t1_df["Cohort"] == t1_key].iloc[0]
        med_hr = float(str(row_t1["MedianSplit_HR_95CI"]).split()[0])
        med_wp = float(row_t1["MedianSplit_Wald_P"])
        stat_str = f"Adj. HR = {med_hr:.3f} (p = {fmt_p_tex(med_wp)})\nLog-rank p = {fmt_p_tex(lr.p_value)}"

        ax_k.set_title(title_k, loc='left', fontweight='bold', fontsize=9.1)
        ax_k.set_xlabel("Overall Survival (Months)", fontsize=8.3, labelpad=2)
        ax_k.set_ylabel("Survival Probability", fontsize=8.3)
        ax_k.set_xlim(0, xmax_k); ax_k.set_ylim(0, 1.05)
        if xticks_custom is not None:
            ax_k.set_xticks(xticks_custom)
        ax_k.text(0.04, 0.14, stat_str, transform=ax_k.transAxes, fontsize=7.3, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.78))
        ax_k.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.0)
        ax_k.spines['top'].set_visible(False); ax_k.spines['right'].set_visible(False)

        # Compact number-at-risk table strictly anchored to explicit data x-tick coordinates
        ticks_x = xticks_custom if xticks_custom is not None else [t for t in ax_k.get_xticks() if -1e-9 <= t <= xmax_k + 1e-9]
        ax_k.set_xticks(ticks_x)
        ax_k.text(-0.07, -0.23, "High", transform=ax_k.transAxes, ha='right', va='center', fontsize=6.8, fontweight='bold', color='#e41a1c', clip_on=False)
        ax_k.text(-0.07, -0.30, "Low", transform=ax_k.transAxes, ha='right', va='center', fontsize=6.8, fontweight='bold', color='#2b5c8f', clip_on=False)
        for t_tick in ticks_x:
            nh_r = int((hi["OS_Months"].values >= t_tick).sum())
            nl_r = int((lo["OS_Months"].values >= t_tick).sum())
            ax_k.text(t_tick, -0.23, f"{nh_r}", transform=ax_k.get_xaxis_transform(), ha='center', va='center', fontsize=6.7, color='#333333', clip_on=False)
            ax_k.text(t_tick, -0.30, f"{nl_r}", transform=ax_k.get_xaxis_transform(), ha='center', va='center', fontsize=6.7, color='#333333', clip_on=False)

    # Panel A: Pancreatic Cohorts (TCGA-PAAD vs. ICGC/PCAWG-PAAD)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.34)
    ax1a = fig.add_subplot(gs_a[0, 0])
    ax1b = fig.add_subplot(gs_a[0, 1])
    plot_km_panel(df_paad_v, ax1a, f"A1  TCGA-PAAD (N = {len(df_paad_v)})", 90, "TCGA-PAAD", "OS_Status", xticks_custom=[0, 15, 30, 45, 60, 75, 90])
    plot_km_panel(df_icgc, ax1b, f"A2  ICGC/PCAWG-PAAD (N = {len(df_icgc)})", 80, "ICGC/PCAWG-PAAD (N = 71)", "Event", xticks_custom=[0, 20, 40, 60, 80])

    # Panel B: Diffuse Glioma & Clear Cell Renal Cohorts
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.34)
    ax2a = fig.add_subplot(gs_b[0, 0])
    ax2b = fig.add_subplot(gs_b[0, 1])
    plot_km_panel(df_cgga_v, ax2a, f"B1  CGGA-325 (N = {len(df_cgga_v)})", 160, "CGGA-325 (Diffuse Glioma II-IV)", "OS_Status")
    plot_km_panel(df_kirc, ax2b, f"B2  TCGA-KIRC (N = {len(df_kirc)})", 140, "TCGA-KIRC (Clear Cell Renal)", "Event")

    # Panel C: Breast & Urothelial Bladder Cohorts
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.34)
    ax3a = fig.add_subplot(gs_c[0, 0])
    ax3b = fig.add_subplot(gs_c[0, 1])
    plot_km_panel(df_mb, ax3a, f"C1  METABRIC OS (N = {len(df_mb):,})", 300, "BRCA-METABRIC (Breast OS)", "Event")
    plot_km_panel(df_blca, ax3b, f"C2  TCGA-BLCA (N = {len(df_blca)})", 160, "TCGA-BLCA (Urothelial Bladder)", "Event", xticks_custom=[0, 40, 80, 120, 160])

    # Panel D: Unified 10-Cohort Primary & 11-Cohort Overall (+ ICGC/PCAWG-PAAD) Meta-Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    t1_map = {
        "TCGA-GBM (WHO Grade IV)": "TCGA-GBM",
        "BRCA-METABRIC (Breast OS)": "BRCA-METABRIC",
        "TCGA-BLCA (Urothelial Bladder)": "TCGA-BLCA",
        "TCGA-LIHC (Hepatocellular)": "TCGA-LIHC",
        "TCGA-BRCA (Breast Carcinoma)": "TCGA-BRCA",
        "TCGA-SKCM (Cutaneous Melanoma)": "TCGA-SKCM",
        "CGGA-325 (Diffuse Glioma II-IV)": "CGGA-325",
        "TCGA-SARC (Soft-Tissue Sarcoma)": "TCGA-SARC",
        "TCGA-KIRC (Clear Cell Renal)": "TCGA-KIRC",
        "TCGA-PAAD": "TCGA-PAAD",
        "ICGC/PCAWG-PAAD (N = 71)": "ICGC/PCAWG-PAAD",
    }
    z_crit = float(stats.norm.ppf(0.975))

    def dl_from_rows(rows_sub):
        betas = np.array([x[8] for x in rows_sub], dtype=float)
        ses = np.array([x[9] for x in rows_sub], dtype=float)
        w = 1.0 / (ses ** 2)
        bfe = float(np.sum(w * betas) / np.sum(w))
        Q = float(np.sum(w * ((betas - bfe) ** 2)))
        df_q = len(betas) - 1
        tau2 = max(0.0, (Q - df_q) / float(np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        i2 = max(0.0, (Q - df_q) / Q) * 100.0 if Q > 0 else 0.0
        ws = 1.0 / (ses ** 2 + tau2)
        bre = float(np.sum(ws * betas) / np.sum(ws))
        sere = float(np.sqrt(1.0 / np.sum(ws)))
        pre = float(2.0 * stats.norm.sf(abs(bre / sere)))
        return float(np.exp(bre)), float(np.exp(bre - z_crit * sere)), float(np.exp(bre + z_crit * sere)), pre, i2

    cohort_rows = []
    for key, short_nm in t1_map.items():
        row = t1_df[t1_df["Cohort"] == key].iloc[0]
        parts = str(row["Continuous_PerSD_HR_95CI"]).replace("[", " ").replace("]", " ").replace("-", " ").split()
        hr, lo, hi = float(parts[0]), float(parts[1]), float(parts[2])
        pv = float(row["Continuous_Wald_P"])
        disp = f"{short_nm} (N = {int(row['N_Patients']):,}, {int(row['N_Deaths']):,} deaths)"
        col = "#2b5c8f" if pv < 0.05 else "#78909c"
        mk = "o" if pv < 0.05 else "s"
        beta_exact = float(row["Continuous_Beta"])
        se_exact = float(row["Continuous_SE"])
        cohort_rows.append((disp, hr, lo, hi, f"p = {fmt_p_tex(pv)}", col, mk, key, beta_exact, se_exact))

    rows_10 = [r for r in cohort_rows if r[7] != "ICGC/PCAWG-PAAD (N = 71)"]
    hr10, lo10, hi10, p10, i2_10 = dl_from_rows(rows_10)
    hr11, lo11, hi11, p11, i2_11 = dl_from_rows(cohort_rows)
    sens_fp = os.path.join(data_proc, "survival_full_tcga_and_mki67_sensitivity.csv")
    df_sens = pd.read_csv(sens_fp)
    r19 = df_sens[(df_sens["Model"] == "PanTCGA_Base") & (df_sens["Cohort"] == "19-All-Available-Solid-TCGA Meta")].iloc[0]
    hr19, lo19, hi19, p19, i2_19, n_19 = (
        float(r19["HR"]), float(r19["CI_Low"]), float(r19["CI_High"]),
        float(r19["P_Value"]), float(r19["I2_Pct"]), int(r19["N"]),
    )
    rbuf = df_sens[(df_sens["Model"] == "Buffa14_Alone_Base") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")].iloc[0]
    hr_b, lo_b, hi_b, p_b, i2_b, n_b = (
        float(rbuf["HR"]), float(rbuf["CI_Low"]), float(rbuf["CI_High"]),
        float(rbuf["P_Value"]), float(rbuf["I2_Pct"]), int(rbuf["N"]),
    )
    rbuf_m = df_sens[(df_sens["Model"] == "Buffa14_Alone_PlusMKI67") & (df_sens["Cohort"] == "19-All-Solid-TCGA Meta")].iloc[0]
    hr_bm, lo_bm, hi_bm, p_bm, i2_bm, n_bm = (
        float(rbuf_m["HR"]), float(rbuf_m["CI_Low"]), float(rbuf_m["CI_High"]),
        float(rbuf_m["P_Value"]), float(rbuf_m["I2_Pct"]), int(rbuf_m["N"]),
    )
    cohort_rows_disp = [r[:7] for r in sorted(cohort_rows, key=lambda x: x[1])]
    n_10 = int(t1_df[t1_df["Cohort"].isin([k for k in t1_map if k != "ICGC/PCAWG-PAAD (N = 71)"])]["N_Patients"].sum())
    n_11 = int(t1_df[t1_df["Cohort"].isin(t1_map.keys())]["N_Patients"].sum())
    forest_data = [
        (f"19-Cohort Solid TCGA: Buffa-14 (+MKI67)", hr_bm, lo_bm, hi_bm, f"p = {fmt_p_tex(p_bm)}", "#9467bd", "D"),
        (f"19-Cohort Solid TCGA: Buffa-14", hr_b, lo_b, hi_b, f"p = {fmt_p_tex(p_b)}", "#6a3d9a", "D"),
        (f"19-Cohort Solid TCGA (N = {n_19:,})", hr19, lo19, hi19, f"p = {fmt_p_tex(p19)}", "#475569", "D"),
        (f"11-Cohort Unified Meta (N = {n_11:,})", hr11, lo11, hi11, f"p = {fmt_p_tex(p11)}", "#d95f02", "D"),
        (f"10-Cohort Primary Meta (N = {n_10:,})", hr10, lo10, hi10, f"p = {fmt_p_tex(p10)}", "#1b9e77", "D"),
    ] + cohort_rows_disp
    y_pos = np.arange(len(forest_data))
    text_x = max(1.80, max(x[3] for x in forest_data) + 0.14)
    for i, (lbl, hr, lo, hi, pstr, col, mk) in enumerate(forest_data):
        ax4.errorbar([hr], [i], xerr=[[hr - lo], [hi - hr]], fmt=mk, color=col,
                     ecolor=col, elinewidth=1.8, capsize=3.2, markersize=5.8)
        ax4.text(text_x, i, f"{hr:.3f} [{lo:.3f}\u2013{hi:.3f}] ({pstr})",
                 va='center', ha='left', fontsize=6.9, fontweight=('bold' if i <= 4 else 'normal'), color='#111111')

    ax4.axvline(1.0, color='#111111', linestyle='--', linewidth=1.0)
    ax4.axhline(4.5, color='#cccccc', linestyle='-', linewidth=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([x[0] for x in forest_data], fontsize=7.0, fontweight='bold')
    ax4.set_xlim(0.70, text_x + 1.45); ax4.set_ylim(-0.7, len(forest_data) - 0.3)
    ax4.set_xlabel("Continuous Per-SD Hazard Ratio (95% CI)")
    ax4.set_title(rf"D   Multi-Cohort Meta & Buffa-14 Benchmark ($I^2 = {i2_10:.1f}\%$ / ${i2_11:.1f}\%$; 19-TCGA $I^2 = {i2_19:.1f}\%$ / Buffa $I^2 = {i2_b:.1f}\%$)", loc='left', fontsize=9.2, fontweight='bold')
    leg_d2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2b5c8f', markersize=6.2, label=r'Cohort ($p < 0.05$)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#78909c', markersize=5.8, label=r'Cohort ($p \geq 0.05$)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#1b9e77', markersize=5.8, label='10-cohort primary meta'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#d95f02', markersize=5.8, label='11-cohort unified meta'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#475569', markersize=5.8, label='19-cohort SMARCA4+HIF1A'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#6a3d9a', markersize=5.8, label='19-cohort Buffa-14 (+MKI67 in light violet)'),
    ]
    ax4.legend(handles=leg_d2, loc='upper center', bbox_to_anchor=(0.48, -0.14), ncol=3, frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=6.5)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_dir, f"Figure_2_Final.{ext}"), dpi=300, bbox_inches='tight', pad_inches=0.28)
    plt.close(fig)
    log("[OK] Saved Figure_2_Final.{pdf,png}")


def build_figure_3(project_root, data_raw, data_proc, fig_dir):
    log("Building Figure 3...")
    df_anova = pd.read_csv(os.path.join(data_proc, "gdsc2_lineage_controlled_anova.csv"))
    focus_drugs = [
        "Navitoclax", "Cisplatin", "ABT737", "Elephantin", "WEHI-539",
        "Entinostat", "Uprosertib", "GSK591", "Oxaliplatin", "Irinotecan",
        "GSK2110183B", "Niraparib", "Ipatasertib", "Rucaparib",
        "Olaparib", "Talazoparib", "Paclitaxel", "AZD5991", "AZD5153", "SN-38",
    ]
    sub_aov = df_anova[df_anova["Compound_Name"].isin(focus_drugs)].sort_values("Type_II_ANOVA_F", ascending=True).reset_index(drop=True)
    short_tgt = {
        "ABT737": "BCL2/XL/W/B",
        "Navitoclax": "BCL2/XL/W",
        "GSK2110183B": "AKT1/2/3",
        "Ipatasertib": "AKT1/2/3",
        "Uprosertib": "AKT1/2/3",
    }

    df_m = pd.read_csv(os.path.join(data_raw, "Model.csv"))
    non_solid = ["Lymphoid", "Myeloid", "Normal", "Fibroblast"]
    norm_ids_g3 = set(
        df_m[
            (df_m["OncotreePrimaryDisease"] == "Non-Cancerous")
            | (df_m["OncotreeLineage"].isin(["Normal", "Fibroblast"]))
        ]["ModelID"].dropna().astype(str)
    )
    solid_models = set(
        df_m[
            (~df_m["OncotreeLineage"].isin(non_solid))
            & (~df_m["ModelID"].astype(str).isin(norm_ids_g3))
        ]["ModelID"].dropna().astype(str)
    )
    c2m = {int(float(r["COSMICID"])): str(r["ModelID"]) for _, r in df_m.iterrows() if pd.notna(r.get("COSMICID")) and pd.notna(r.get("ModelID"))}
    s2m = {str(r["SangerModelID"]): str(r["ModelID"]) for _, r in df_m.iterrows() if pd.notna(r.get("SangerModelID")) and pd.notna(r.get("ModelID"))}
    lin_map = df_m.drop_duplicates("ModelID").set_index("ModelID")["OncotreeLineage"].to_dict()

    gdsc_dfs = []
    gdsc_dir = os.path.join(data_raw, "gdsc")
    for fn in GDSC_MATRIX_FILES:
        p = resolve_data_path(gdsc_dir, project_root, fn)
        if os.path.exists(p):
            gdsc_dfs.append(pd.read_csv(p, usecols=[
                "COSMIC_ID", "lib1_name", "lib2_name", "Bliss_window",
                "Bliss_window_dose1", "Bliss_window_dose2"
            ]))
    df_gdsc = pd.concat(gdsc_dfs, ignore_index=True)
    df_gdsc["ModelID"] = [c2m.get(int(x)) if pd.notna(x) else None for x in df_gdsc["COSMIC_ID"]]
    df_gdsc_solid = df_gdsc[df_gdsc["ModelID"].isin(solid_models)].copy()

    mat_hub1, cnt_hub1, n_hub1, n_lines1 = compute_5x5_peak_window_grid(df_gdsc_solid, "AZD5153", "AZD5991")

    expr_fp = os.path.join(data_raw, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    df_eh = pd.read_csv(expr_fp, nrows=1)
    target_genes = ["SLFN11", "BAX", "BAK1", "MCL1"]
    col_map = {"ModelID": "ModelID", "IsDefaultEntryForModel": "IsDefaultEntryForModel"}
    use_cols = ["ModelID", "IsDefaultEntryForModel"]
    for g in target_genes:
        m = [c for c in df_eh.columns if c.split(" ")[0] == g][0]
        use_cols.append(m); col_map[m] = g
    df_expr = pd.read_csv(expr_fp, usecols=use_cols).rename(columns=col_map)
    df_expr = df_expr[df_expr["IsDefaultEntryForModel"] == "Yes"].drop_duplicates("ModelID").dropna(subset=target_genes).copy()
    df_expr["ModelID"] = df_expr["ModelID"].astype(str)
    df_expr["Lineage"] = df_expr["ModelID"].map(lin_map)

    g2 = pd.read_excel(os.path.join(data_raw, "gdsc", "GDSC2_fitted_dose_response_27Oct23.xlsx"))
    g2["ModelID"] = [
        s2m.get(str(s), c2m.get(int(c)) if pd.notna(c) else None)
        for s, c in zip(g2.get("SANGER_MODEL_ID", pd.Series([None] * len(g2))), g2["COSMIC_ID"])
    ]
    g2 = g2[g2["ModelID"].isin(solid_models)].copy()
    g2["OneMinusAUC"] = 1.0 - g2["AUC"].astype(float)

    gtex_fp = os.path.join(data_proc, "dose_sparing_dri_pk_gtex.csv")
    df_gtex = pd.read_csv(gtex_fp) if os.path.exists(gtex_fp) else None
    toil_fp = os.path.join(data_proc, "toil_tcga_vs_gtex_tpm.csv")
    df_toil = pd.read_csv(toil_fp) if os.path.exists(toil_fp) else None

    fig = plt.figure(figsize=(18.5, 15.2), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.08, 0.95, 0.78], hspace=0.40, wspace=0.28, left=0.145, right=0.97, top=0.95, bottom=0.06)

    # Panel A: Full-screen GDSC2 Type II ANOVA (Primary 1 - AUC, m = 286 drugs)
    ax1 = fig.add_subplot(gs[0, 0])
    yp = np.arange(len(sub_aov))
    cols_a = ['#1e3a8a' if q < 0.10 else ('#5c88b5' if p < 0.05 else '#b0bec5') for p, q in zip(sub_aov["P_Value"], sub_aov["FDR_Q_Value"])]
    ax1.barh(yp, sub_aov["Type_II_ANOVA_F"], color=cols_a, edgecolor='#222222', linewidth=0.6, height=0.66)
    ax1.set_yticks(yp)
    ylabels_a = [
        f"{r['Compound_Name']} ({short_tgt.get(r['Compound_Name'], r['Putative_Target'])})"
        for _, r in sub_aov.iterrows()
    ]
    ax1.set_yticklabels(ylabels_a, fontsize=7.6, fontweight='bold')
    max_f = float(sub_aov["Type_II_ANOVA_F"].max())
    ax1.set_xlim(0, max(22.2, max_f * 1.56))
    for idx, r in sub_aov.iterrows():
        p_disp = float(r["P_Value"])
        res_tag = " [res.]" if float(r["OLS_Beta_Risk"]) < 0 else ""
        ax1.text(r["Type_II_ANOVA_F"] + 0.28, idx, f"F={r['Type_II_ANOVA_F']:.2f}, p={fmt_p_tex(p_disp)} (q={r['FDR_Q_Value']:.3f}){res_tag}",
                 va='center', fontsize=6.9, fontweight='bold', zorder=6,
                 bbox=dict(boxstyle='square,pad=0.04', facecolor='white', edgecolor='none', alpha=0.82))
    ax1.axvline(3.86, color='#e41a1c', linestyle='--', linewidth=1.0)
    n_fdr10_a = int((df_anova["FDR_Q_Value"] < 0.10).sum())
    leg_3a = [
        mpatches.Patch(facecolor='#1e3a8a', edgecolor='#222222', label=rf'FDR $q < 0.10$ ({n_fdr10_a}/{len(df_anova)} drugs)'),
        mpatches.Patch(facecolor='#5c88b5', edgecolor='#222222', label=r'Nominal $p < 0.05$ ($q \geq 0.10$)'),
        mpatches.Patch(facecolor='#b0bec5', edgecolor='#222222', label=r'Non-significant ($p \geq 0.05$)'),
        Line2D([0], [0], color='#e41a1c', linestyle='--', lw=1.0, label=r'$p = 0.05$ threshold'),
    ]
    ax1.set_title(f"A   GDSC2 1 \u2212 AUC Screen: Top Hits & Pathway Controls (m = {len(df_anova)})", loc='left', fontweight='bold')
    ax1.set_xlabel(r"Type II ANOVA $F$-Statistic ($1 - \mathrm{AUC}$ Risk Score Association)")
    ax1.legend(handles=leg_3a, loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.0)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel B: 5x5 Peak 3x3 Window Bliss Surface & Combination Bliss Distributions
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.54)
    ax2a = fig.add_subplot(gs_b[0, 0])
    ax2b = fig.add_subplot(gs_b[0, 1])

    cmap_3b1 = plt.get_cmap('YlOrRd').copy()
    cmap_3b1.set_bad('#f1f5f9')
    im1 = ax2a.imshow(mat_hub1, cmap=cmap_3b1, origin='lower', aspect='auto', vmin=0.0, vmax=0.18)
    cbar1 = fig.colorbar(im1, ax=ax2a, fraction=0.046, pad=0.03)
    cbar1.set_label("Mean Peak Bliss (by Window)", fontsize=7.2)
    cbar1.ax.tick_params(labelsize=6.8)
    win_lbls = ['D1', 'D2', 'D3', 'D4', 'D5']
    ax2a.set_xticks(np.arange(5)); ax2a.set_xticklabels(win_lbls, fontsize=7.4, fontweight='bold')
    ax2a.set_yticks(np.arange(5)); ax2a.set_yticklabels(win_lbls, fontsize=7.4, fontweight='bold')
    ax2a.set_xlabel(r"AZD5991 (MCL1i) Peak 3$\times$3 Window (D1$\rightarrow$D5)", fontsize=7.6)
    ax2a.set_ylabel(r"AZD5153 (BRD4i) Peak 3$\times$3 Window (D1$\rightarrow$D5)", fontsize=7.6)
    ax2a.set_title(f"B1  AZD5153 + AZD5991\nPeak 3\u00d73 Window (n = {n_lines1} lines, {n_hub1} matrices)", loc='left', fontweight='bold', fontsize=8.3)
    for r in range(5):
        for c in range(5):
            val = mat_hub1[r, c]
            cnt = cnt_hub1[r, c]
            if np.isnan(val):
                ax2a.text(c, r, "\u2014", ha='center', va='center', fontsize=7.0, fontweight='bold', color='#475569')
            else:
                n_disp = int(cnt) if not np.isnan(cnt) else 0
                ax2a.text(c, r, f"{val:.3f}\n(n={n_disp})", ha='center', va='center', fontsize=5.6, fontweight='bold', color=('#ffffff' if val > 0.105 else '#111111'))

    df_gdsc_solid["Pair_Key"] = np.where(
        df_gdsc_solid["lib1_name"].astype(str) <= df_gdsc_solid["lib2_name"].astype(str),
        df_gdsc_solid["lib1_name"].astype(str) + " + " + df_gdsc_solid["lib2_name"].astype(str),
        df_gdsc_solid["lib2_name"].astype(str) + " + " + df_gdsc_solid["lib1_name"].astype(str),
    )
    all_pl = df_gdsc_solid.groupby(["Pair_Key", "ModelID"], as_index=False)["Bliss_window"].mean().dropna()
    pair_stats = all_pl.groupby("Pair_Key").agg(n_lines=("ModelID", "nunique"), med_bliss=("Bliss_window", "median"))
    bg_pairs = pair_stats[pair_stats["n_lines"] >= 30]
    bg_median = float(bg_pairs["med_bliss"].median())

    pair_data, pair_lbls = [], []
    for d1, d2 in [("AZD5153", "AZD5991"), ("Olaparib", "SN-38"), ("Capivasertib", "Selumetinib")]:
        sub_p = df_gdsc_solid[((df_gdsc_solid["lib1_name"]==d1)&(df_gdsc_solid["lib2_name"]==d2)) | ((df_gdsc_solid["lib1_name"]==d2)&(df_gdsc_solid["lib2_name"]==d1))]
        per_line_vals = sub_p.groupby("ModelID")["Bliss_window"].mean().dropna().values
        pair_data.append(per_line_vals)
        pair_lbls.append(f"{d1} +\n{d2}\n(n={len(per_line_vals)})")
    bp = ax2b.boxplot(pair_data, patch_artist=True, widths=0.5, showfliers=False)
    for patch, col in zip(bp['boxes'], ['#e41a1c', '#2b5c8f', '#1b9e77']):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for med in bp['medians']:
        med.set_color('#111111'); med.set_linewidth(1.8)
    ax2b.axhline(0.0, color='#888888', linestyle=':', linewidth=0.8)
    ax2b.axhline(bg_median, color='#d95f02', linestyle='--', linewidth=1.2, label=f"{len(bg_pairs)}-pair bg ({bg_median:+.4f})")
    ax2b.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=6.6)
    ax2b.set_xticks([1, 2, 3]); ax2b.set_xticklabels(pair_lbls, fontsize=7.2, fontweight='bold')
    ax2b.set_ylabel("Cell-Line Mean Bliss Excess", fontsize=8.4)
    ax2b.set_title("B2  Unique-Line Synergy\nvs. Empirical Background", loc='left', fontweight='bold', fontsize=8.8)
    ax2b.spines['top'].set_visible(False); ax2b.spines['right'].set_visible(False)

    # Panel C: SLFN11 Gating of SN-38 Empirical Activity (1 - AUC, Solid-Only Unique ModelIDs in GDSC2)
    ax3 = fig.add_subplot(gs[1, 0])
    sub_sn = g2[g2["DRUG_NAME"] == "SN-38"].groupby("ModelID", as_index=False)["OneMinusAUC"].mean().merge(
        df_expr[["ModelID", "SLFN11", "Lineage"]], on="ModelID"
    ).dropna()
    r_sn, p_sn = stats.pearsonr(sub_sn["SLFN11"], sub_sn["OneMinusAUC"])
    res_sn_x = smf.ols("SLFN11 ~ C(Lineage)", data=sub_sn).fit().resid
    res_sn_y = smf.ols("OneMinusAUC ~ C(Lineage)", data=sub_sn).fit().resid
    r_sn_adj, _ = stats.pearsonr(res_sn_x, res_sn_y)
    df_sn = len(sub_sn) - sub_sn["Lineage"].nunique() - 1
    p_sn_adj = float(2.0 * stats.t.sf(abs(r_sn_adj * np.sqrt(df_sn / (1.0 - r_sn_adj ** 2))), df=df_sn))
    sil = sub_sn["SLFN11"] < 1.0
    d_sn = float(sub_sn.loc[~sil, "OneMinusAUC"].mean() - sub_sn.loc[sil, "OneMinusAUC"].mean())
    _, p_sn_mwu = stats.mannwhitneyu(sub_sn.loc[~sil, "OneMinusAUC"], sub_sn.loc[sil, "OneMinusAUC"], alternative='two-sided')
    ax3.scatter(sub_sn.loc[sil, "SLFN11"], sub_sn.loc[sil, "OneMinusAUC"], color='#e41a1c', alpha=0.55, s=18, edgecolors='none', label=f"Silenced (< 1.0; n = {sil.sum()})")
    ax3.scatter(sub_sn.loc[~sil, "SLFN11"], sub_sn.loc[~sil, "OneMinusAUC"], color='#2b5c8f', alpha=0.55, s=18, edgecolors='none', label=rf"Expressed ($\geq 1.0$; $n = {(~sil).sum()}$)")
    ms, bs, _, _, _ = stats.linregress(sub_sn["SLFN11"], sub_sn["OneMinusAUC"])
    xs = np.linspace(sub_sn["SLFN11"].min(), sub_sn["SLFN11"].max(), 100)
    ax3.plot(xs, bs + ms * xs, color='#111111', linewidth=1.9)
    ax3.axvline(1.0, color='#b71c1c', linestyle='--', linewidth=1.1)
    ax3.set_ylim(-0.04, max(0.86, float(sub_sn["OneMinusAUC"].max()) * 1.18))
    ax3.text(0.96, 0.96, f"Solid n = {len(sub_sn)}, r = {r_sn:+.4f} (p = {fmt_p_tex(p_sn)})\n" + rf"Lineage-adj. $r_{{\mathrm{{adj}}}} = {r_sn_adj:+.4f}$" + f" (df={df_sn}, p = {fmt_p_tex(p_sn_adj)})\n" + rf"$\Delta(1 - \mathrm{{AUC}}) = {d_sn:+.4f}$ (MWU p = {fmt_p_tex(p_sn_mwu)})",
             transform=ax3.transAxes, ha='right', va='top', fontsize=7.4, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.82))
    ax3.set_title("C   SLFN11 Expression Gates SN-38 Activity (1 \u2212 AUC, Solid GDSC2)", loc='left', fontweight='bold')
    ax3.set_xlabel(r"SLFN11 Expression $\log_2(\mathrm{TPM} + 1)$")
    ax3.set_ylabel(r"SN-38 Empirical Activity ($1 - \mathrm{AUC}$)")
    ax3.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.2)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # Panel D: BAK1 Expression Gates AZD5991 Empirical Activity (1 - AUC, Solid-Only Unique ModelIDs in GDSC2)
    ax4 = fig.add_subplot(gs[1, 1])
    sub_az = g2[g2["DRUG_NAME"] == "AZD5991"].groupby("ModelID", as_index=False)["OneMinusAUC"].mean().merge(
        df_expr[["ModelID", "BAK1", "BAX", "Lineage"]], on="ModelID"
    ).dropna()
    r_bk, p_bk = stats.pearsonr(sub_az["BAK1"], sub_az["OneMinusAUC"])
    res_bk_x = smf.ols("BAK1 ~ C(Lineage)", data=sub_az).fit().resid
    res_bk_y = smf.ols("OneMinusAUC ~ C(Lineage)", data=sub_az).fit().resid
    r_bk_adj, _ = stats.pearsonr(res_bk_x, res_bk_y)
    df_bk = len(sub_az) - sub_az["Lineage"].nunique() - 1
    p_bk_adj = float(2.0 * stats.t.sf(abs(r_bk_adj * np.sqrt(df_bk / (1.0 - r_bk_adj ** 2))), df=df_bk))
    res_bx_x = smf.ols("BAX ~ C(Lineage)", data=sub_az).fit().resid
    r_bx_adj, _ = stats.pearsonr(res_bx_x, res_bk_y)
    p_bx_adj = float(2.0 * stats.t.sf(abs(r_bx_adj * np.sqrt(df_bk / (1.0 - r_bx_adj ** 2))), df=df_bk))
    med_bk = float(np.median(sub_az["BAK1"]))
    low_bk = sub_az["BAK1"] < med_bk
    d_bk_med = float(sub_az.loc[~low_bk, "OneMinusAUC"].mean() - sub_az.loc[low_bk, "OneMinusAUC"].mean())
    _, p_bk_mwu = stats.mannwhitneyu(sub_az.loc[~low_bk, "OneMinusAUC"], sub_az.loc[low_bk, "OneMinusAUC"], alternative='two-sided')
    ax4.scatter(sub_az.loc[low_bk, "BAK1"], sub_az.loc[low_bk, "OneMinusAUC"], color='#e41a1c', alpha=0.55, s=18, edgecolors='none', label=f"Below median (< {med_bk:.2f}; n = {low_bk.sum()})")
    ax4.scatter(sub_az.loc[~low_bk, "BAK1"], sub_az.loc[~low_bk, "OneMinusAUC"], color='#2b5c8f', alpha=0.55, s=18, edgecolors='none', label=rf"Above median ($\geq {med_bk:.2f}$; $n = {(~low_bk).sum()}$)")
    mbk, bbk, _, _, _ = stats.linregress(sub_az["BAK1"], sub_az["OneMinusAUC"])
    xbk = np.linspace(sub_az["BAK1"].min(), sub_az["BAK1"].max(), 100)
    ax4.plot(xbk, bbk + mbk * xbk, color='#111111', linewidth=1.9)
    ax4.axvline(med_bk, color='#b71c1c', linestyle='--', linewidth=1.1)
    ax4.set_ylim(-0.04, max(0.75, float(sub_az["OneMinusAUC"].max()) * 1.18))
    ax4.text(0.96, 0.96,
             f"Solid n = {len(sub_az)}, r = {r_bk:+.4f} (p = {fmt_p_tex(p_bk)})\n"
             + rf"Lineage-adj. $r_{{\mathrm{{adj}}}} = {r_bk_adj:+.4f}$" + f" (df={df_bk}, p = {fmt_p_tex(p_bk_adj)})\n"
             + rf"Median split $\Delta(1 - \mathrm{{AUC}}) = {d_bk_med:+.4f}$ (MWU p = {fmt_p_tex(p_bk_mwu)})" + "\n"
             + rf"BAX control $r_{{\mathrm{{adj}}}} = {r_bx_adj:+.4f}$" + f" (p = {fmt_p_tex(p_bx_adj)})",
             transform=ax4.transAxes, ha='right', va='top', fontsize=7.2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.82))
    sub_ol = g2[g2["DRUG_NAME"] == "Olaparib"].groupby("ModelID", as_index=False)["OneMinusAUC"].mean().merge(
        df_expr[["ModelID", "SLFN11", "Lineage"]], on="ModelID"
    ).dropna()
    r_ol, p_ol = stats.pearsonr(sub_ol["SLFN11"], sub_ol["OneMinusAUC"])
    r_ol_adj, _ = stats.pearsonr(smf.ols("SLFN11 ~ C(Lineage)", data=sub_ol).fit().resid, smf.ols("OneMinusAUC ~ C(Lineage)", data=sub_ol).fit().resid)
    df_ol = len(sub_ol) - sub_ol["Lineage"].nunique() - 1
    p_ol_adj = float(2.0 * stats.t.sf(abs(r_ol_adj * np.sqrt(df_ol / (1.0 - r_ol_adj ** 2))), df=df_ol))
    sil_ol = sub_ol["SLFN11"] < 1.0
    d_ol = float(sub_ol.loc[~sil_ol, "OneMinusAUC"].mean() - sub_ol.loc[sil_ol, "OneMinusAUC"].mean())
    _, p_ol_mwu = stats.mannwhitneyu(sub_ol.loc[~sil_ol, "OneMinusAUC"], sub_ol.loc[sil_ol, "OneMinusAUC"], alternative='two-sided')
    log(f"  [GDSC2 Solid Gates 1-AUC] SN-38 vs SLFN11: n={len(sub_sn)}, df={df_sn}, raw r={r_sn:+.4f} (p={p_sn:.2e}), adj r={r_sn_adj:+.4f} (p_df={p_sn_adj:.2e}), d={d_sn:+.4f} (MWU p={p_sn_mwu:.2e})")
    log(f"  [GDSC2 Solid Gates 1-AUC] Olaparib vs SLFN11: n={len(sub_ol)}, df={df_ol}, raw r={r_ol:+.4f} (p={p_ol:.2e}), adj r={r_ol_adj:+.4f} (p_df={p_ol_adj:.2e}), d={d_ol:+.4f} (MWU p={p_ol_mwu:.2e})")
    log(f"  [GDSC2 Solid Gates 1-AUC] AZD5991 vs BAK1: n={len(sub_az)}, df={df_bk}, raw r={r_bk:+.4f} (p={p_bk:.2e}), adj r={r_bk_adj:+.4f} (p_df={p_bk_adj:.2e}), med d={d_bk_med:+.4f} (MWU p={p_bk_mwu:.2e}) | BAX adj r={r_bx_adj:+.4f} (p_df={p_bx_adj:.3f})")
    ax4.set_title("D   BAK1 Mitochondrial Pore Executioner Gates AZD5991 (1 \u2212 AUC)", loc='left', fontweight='bold')
    ax4.set_xlabel(r"BAK1 Expression $\log_2(\mathrm{TPM} + 1)$")
    ax4.set_ylabel(r"AZD5991 Empirical Activity ($1 - \mathrm{AUC}$)")
    ax4.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.2)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    # Panel E: Harmonized UCSC Xena Toil (TCGA Primary Solid Tumors vs. GTEx Healthy Organs)
    ax5 = fig.add_subplot(gs[2, 0])
    genes_gtex = ["MCL1", "BAK1", "BRD4", "SMARCA4"]
    tissues_gtex = ["Heart - Left Ventricle", "Heart - Atrial Appendage", "Liver", "Whole Blood"]
    tissue_short = ["Heart LV", "Heart AA", "Liver", "Whole Blood"]
    t_colors = ["#1f2937", "#e41a1c", "#f97316", "#8b5cf6", "#3b82f6"]
    xg = np.arange(len(genes_gtex))
    bw = 0.155
    if df_toil is not None:
        n_tcga_toil = int(df_toil[df_toil["Group"] == "TCGA_Primary_Solid"]["N_Samples"].iloc[0])
        tum_vals = [float(df_toil[(df_toil["Gene"] == g) & (df_toil["Group"] == "TCGA_Primary_Solid")]["Median_Linear_TPM"].iloc[0]) for g in genes_gtex]
        bars0 = ax5.bar(xg - 2 * bw, tum_vals, width=bw, color=t_colors[0], edgecolor="#111111", linewidth=0.5, label=f"TCGA Solid Tumors (n={n_tcga_toil:,})")
        for b, v in zip(bars0, tum_vals):
            ax5.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold")
        for ti, (tkey, tlbl) in enumerate(zip(tissues_gtex, tissue_short), start=1):
            tvals = [float(df_toil[(df_toil["Gene"] == g) & (df_toil["Group"] == tkey)]["Median_Linear_TPM"].iloc[0]) for g in genes_gtex]
            n_org = int(df_toil[df_toil["Group"] == tkey]["N_Samples"].iloc[0])
            bars_t = ax5.bar(xg + (ti - 2) * bw, tvals, width=bw, color=t_colors[ti], edgecolor="#111111", linewidth=0.5, label=f"GTEx v6 {tlbl} (n={n_org})")
            for b, v in zip(bars_t, tvals):
                ax5.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=5.6, color="#1e293b")
    ax5.set_yscale("log")
    ax5.set_ylim(2.0, 950)
    ax5.set_xticks(xg); ax5.set_xticklabels(genes_gtex, fontsize=9.0, fontweight="bold")
    ax5.set_ylabel("Median Expression (Linear TPM, log scale)")
    ax5.set_title("E   UCSC Xena Toil (TCGA vs. GTEx v6)", loc="left", fontweight="bold")
    ax5.text(
        0.02, 0.96,
        "UCSC Xena Toil (TCGA vs. GTEx v6, 2**y - 0.001);\nWhole Blood transcripts diluted by globin mRNA",
        transform=ax5.transAxes, ha="left", va="top", fontsize=6.1, color="#475569",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax5.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=6.3)
    ax5.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # Panel F: GDSC Matrix D1 Anchors, Monotherapy Extrapolated IC50s & Verified AZD5153 Total Cmax
    ax6 = fig.add_subplot(gs[2, 1])
    if df_gtex is not None:
        sub_gd = df_gtex[(df_gtex["Audit_Section"] == "GDSC_Grid_and_Monotherapy_IC50") & (df_gtex["Subgroup_or_Tissue"] == "All_Solid_Lines")].set_index("Gene_or_Drug")
        sub_auc = df_gtex[df_gtex["Audit_Section"] == "GDSC2_Monotherapy_AUC_and_Ceiling"].set_index("Gene_or_Drug")
        drugs_pk = ["SN-38", "AZD5153", "AZD5991", "Olaparib"]
        d1_bounds = {"SN-38": 0.3, "AZD5153": 1.0, "AZD5991": 1.0, "Olaparib": 10.0}
        yp_pk = np.arange(len(drugs_pk))
        ic50_meds = [float(sub_gd.loc[d, "Value"]) for d in drugs_pk]
        d1_vals = [d1_bounds[d] for d in drugs_pk]
        ax6.axvspan(10000.0, 1.5e6, color="#f1f5f9", alpha=0.70, zorder=1)
        ax6.axvline(10000.0, color="#6d28d9", linestyle=":", lw=1.4, zorder=4, label=r"Max GDSC2 screen conc. ($10\ \mu\mathrm{M} = 10,000$ nM)")
        ax6.scatter(d1_vals, yp_pk, s=75, marker="D", color="#16a34a", edgecolor="#111111", zorder=5, label=r"Matrix $D_1$ lowest anchor (nM)")
        ax6.scatter(ic50_meds, yp_pk, s=85, marker="o", color="#2b5c8f", edgecolor="#111111", zorder=5, label=r"Combo-screen extrap. $\mathrm{IC}_{50}$ med. (nM)")
        for yi, (d_v, i_v, dr) in enumerate(zip(d1_vals, ic50_meds, drugs_pk)):
            ax6.plot([d_v, i_v], [yi, yi], color="#64748b", lw=2.0, zorder=3)
            ax6.text(d_v * 0.72, yi, f"{d_v:g}", ha="right", va="center", fontsize=6.8, fontweight="bold", color="#15803d")
            ax6.text(i_v * 1.28, yi, f"{i_v:,.0f} nM", ha="left", va="center", fontsize=6.8, fontweight="bold", color="#1e3a8a",
                     bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.85))
        ax6.axvline(245.4, color="#e41a1c", linestyle="--", lw=1.4, zorder=4, label=r"AZD5153 30 mg QD $C_{\max}$ (245.4 nM)")
        ax6.text(245.4 * 1.18, 1.48, "AZD5153 single/multi\n$C_{\\max}$ = 245.4 / 272.4 nM\n(Hamilton et al. 2023)", fontsize=6.2, color="#b91c1c", fontweight="bold", va="center")
        ax6.text(13500.0, 1.42, r"Extrapolated above" + "\n" + r"$10\ \mu\mathrm{M}$ ceiling", fontsize=6.1, color="#5b21b6", fontweight="bold", va="center")
        ax6.text(
            0.98, 0.06,
            "AZD5991 Cmax, human plasma fu,\n& 7x7 well curves unpublished publicly",
            transform=ax6.transAxes, ha="right", va="bottom", fontsize=6.2, color="#475569",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#f8fafc", edgecolor="#cbd5e1"),
        )
        ax6.set_xscale("log")
        ax6.set_xlim(0.08, 1.8e6)
        ax6.set_ylim(-0.65, len(drugs_pk) + 0.65)
        yt_lbls = [
            f"{d}\n(n={int(sub_gd.loc[d, 'N'])}, 1-AUC={float(sub_auc.loc[d, 'Value']):.3f})"
            if d in sub_auc.index else f"{d}\n(n={int(sub_gd.loc[d, 'N'])})"
            for d in drugs_pk
        ]
        ax6.set_yticks(yp_pk); ax6.set_yticklabels(yt_lbls, fontsize=7.6, fontweight="bold")
        ax6.set_xlabel("Concentration (nM, log scale)")
        ax6.set_title("F   GDSC Matrix D1 Bounds, 10 uM Screen Ceiling & Extrap. IC50s", loc="left", fontweight="bold")
        ax6.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=6.4)
        ax6.grid(True, axis="x", linestyle=":", alpha=0.45)
        ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_dir, f"Figure_3_Final.{ext}"), dpi=300, bbox_inches='tight', pad_inches=0.22)
    plt.close(fig)
    log("[OK] Saved Figure_3_Final.{pdf,png}")


def build_figure_4(project_root, data_raw, data_proc, fig_dir):
    log("Building Figure 4...")
    df_sc = pd.read_csv(os.path.join(data_proc, "pdac_scrna_celltype_specificity.csv"))
    df_m1 = pd.read_csv(os.path.join(data_proc, "hallmark_gsea_transcriptome_wide_enrichment.csv"))
    df_m2 = pd.read_csv(os.path.join(data_proc, "tcga_rppa_proteomic_validation.csv"))
    df_m4 = pd.read_csv(os.path.join(data_proc, "genomic_scars_hrd_aneuploidy_orthogonality.csv"))

    fig = plt.figure(figsize=(18.5, 12.8), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.33, wspace=0.26, left=0.055, right=0.975, top=0.94, bottom=0.07)

    # Panel A: Patient-Level Pseudobulk scRNA-seq Specificity (CRA001160, 41,976 tumor-sample cells, n=23 paired patients)
    ax1 = fig.add_subplot(gs[0, 0])
    genes_c = ["SMARCA4", "HIF1A", "SLC7A1", "SLC1A5", "ABCB1"]
    cells_c = ["Malignant", "Ductal", "Acinar", "Endocrine", "Fibroblasts", "Stellate", "Endothelial", "CD8Tex", "Mono/Macro", "DC", "B", "Plasma"]
    cell_labels = ["Malignant", "Ductal", "Acinar", "Endocrine", "Fibroblasts\n(CAF)", "Stellate", "Endothelial", "CD8Tex", "Mono/\nMacro", "DC", "B", "Plasma"]
    dot_scale = 4.2
    xs, ys, sizes, cols = [], [], [], []
    for yi, g in enumerate(genes_c):
        for xi, ct in enumerate(cells_c):
            sub = df_sc[(df_sc["Target_Gene"] == g) & (df_sc["Cell_Type"] == ct)]
            if len(sub) > 0:
                r = sub.iloc[0]
                xs.append(xi); ys.append(yi)
                sizes.append(float(r["Pct_Expressing"]) * dot_scale)
                cols.append(float(r["Mean_LogNorm_Expr"]))
    sc = ax1.scatter(xs, ys, s=sizes, c=cols, cmap='YlOrRd', edgecolors='#222222', linewidth=0.75, vmin=0.0, vmax=1.0, zorder=4)
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.032, pad=0.02, aspect=22)
    cbar.set_label("Mean Log-Norm Expr.", fontsize=8.0)
    for yi, g in enumerate(genes_c):
        mal = df_sc[(df_sc["Target_Gene"] == g) & (df_sc["Cell_Type"] == "Malignant")].iloc[0]
        fc_caf = float(mal["Pseudobulk_Log2FC_Malignant_vs_CAF"])
        p_caf = float(mal["Paired_Wilcoxon_P_vs_CAF"])
        ax1.text(len(cells_c) - 0.52, yi, f"PB {fc_caf:+.3f} " + r"$\log_2$FC" + f"\np = {fmt_p_tex(p_caf)}", va='center', ha='left', fontsize=7.0, fontweight='bold')
    ax1.text(len(cells_c) - 0.52, len(genes_c) - 0.30, "Paired PB\n(Mal vs. CAF)", va='bottom', ha='left', fontsize=7.4, fontweight='bold', color='#111111')
    ax1.axvspan(-0.45, 0.45, color='#fff3e0', alpha=0.55, zorder=1)
    ax1.axvspan(3.55, 4.45, color='#e8f5e9', alpha=0.55, zorder=1)
    ax1.set_xticks(np.arange(len(cells_c))); ax1.set_xticklabels(cell_labels, fontsize=6.8, fontweight='bold', rotation=28, ha='right')
    ax1.set_yticks(np.arange(len(genes_c))); ax1.set_yticklabels(genes_c, fontsize=8.8, fontweight='bold')
    ax1.set_xlim(-0.65, len(cells_c) + 2.85); ax1.set_ylim(-0.6, len(genes_c) + 0.35)
    ax1.grid(True, axis='both', linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    ax1.set_title("A   PDAC scRNA-seq (41,976 Tumor Cells) & Pseudobulk (n = 23 Paired)", loc='left', fontweight='bold')
    for pct in [20, 50, 80]:
        ax1.scatter([], [], s=pct * dot_scale, c='#d95f02', edgecolors='#222222', label=f"{pct}% positive")
    ax1.legend(loc='upper left', ncol=3, frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=7.0)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel B: MSigDB Hallmark Concordance
    ax2 = fig.add_subplot(gs[0, 1])
    paad = df_m1[df_m1["Cohort"].str.contains("PAAD")].set_index("Hallmark_Short")
    meta = df_m1[df_m1["Cohort"].str.contains("METABRIC")].set_index("Hallmark_Short")
    common = paad.index.intersection(meta.index)
    x_nes = paad.loc[common, "Normalized_ES_NES"].astype(float)
    y_nes = meta.loc[common, "Normalized_ES_NES"].astype(float)
    q_paad = paad.loc[common, "GSEA_FDR_q"].astype(float)
    q_meta = meta.loc[common, "GSEA_FDR_q"].astype(float)
    rho_hm, p_hm = stats.spearmanr(x_nes, y_nes)

    colors_a, sizes_a = [], []
    for hm in common:
        both = (q_paad[hm] < 0.05) and (q_meta[hm] < 0.05)
        only_p = (q_paad[hm] < 0.05) and not both
        only_m = (q_meta[hm] < 0.05) and not both
        if both: colors_a.append("#e41a1c"); sizes_a.append(60)
        elif only_p: colors_a.append("#2b5c8f"); sizes_a.append(46)
        elif only_m: colors_a.append("#D97706"); sizes_a.append(46)
        else: colors_a.append("#94A3B8"); sizes_a.append(32)
    ax2.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.8)
    ax2.axvline(0.0, color="#64748B", linestyle="--", linewidth=0.8)
    ax2.scatter(x_nes, y_nes, s=sizes_a, c=colors_a, edgecolor="#0F172A", linewidth=0.5, alpha=0.88, zorder=3)
    for hm, sname, (dx, dy) in [
        ("G2M_CHECKPOINT", "G2M Checkpoint", (0.15, 0.16)),
        ("E2F_TARGETS", "E2F Targets", (-0.95, 0.22)),
        ("MYC_TARGETS_V1", "MYC Targets V1", (-1.15, 0.28)),
        ("MTORC1_SIGNALING", "mTORC1", (-0.82, -0.32)),
        ("GLYCOLYSIS", "Glycolysis", (0.45, 0.30)),
        ("HYPOXIA", "Hypoxia", (0.45, -0.35)),
        ("TNFA_SIGNALING_VIA_NFKB", r"TNF$\alpha$/NF-$\kappa$B", (-0.15, -0.60)),
        ("EPITHELIAL_MESENCHYMAL_TRANSITION", "EMT", (-0.15, -0.52)),
        ("COAGULATION", "Coagulation", (-1.02, 0.22)),
        ("ESTROGEN_RESPONSE_LATE", "Estrogen Resp. Late", (-1.40, -0.22)),
        ("ESTROGEN_RESPONSE_EARLY", "Estrogen Resp. Early", (-1.55, 0.02)),
        ("OXIDATIVE_PHOSPHORYLATION", "OXPHOS", (0.28, -0.06)),
        ("ADIPOGENESIS", "Adipogenesis", (0.22, 0.28)),
        ("FATTY_ACID_METABOLISM", "Fatty Acid Metab.", (0.40, -0.04)),
        ("BILE_ACID_METABOLISM", "Bile Acid Metab.", (0.35, -0.28)),
    ]:
        if hm in common:
            xv, yv = float(x_nes[hm]), float(y_nes[hm])
            ax2.annotate(sname, xy=(xv, yv), xytext=(xv + dx, yv + dy), fontsize=7.2, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6))
    ax2.text(0.04, 0.95, rf"Spearman $\rho = +{rho_hm:.4f}$ (p = {fmt_p_tex(p_hm)})", transform=ax2.transAxes, va="top", fontsize=8.0, fontweight="bold")
    ax2.text(
        0.04,
        0.88,
        "Colors: unadjusted GSEA FDR |\nCorrelation-adjusted CAMERA FDR < 0.05: 0/50 PAAD, 10/50 METABRIC",
        transform=ax2.transAxes,
        va="top",
        fontsize=6.6,
        color="#334155",
    )
    legend_a = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markeredgecolor="#0F172A", markersize=6.8, label="Both FDR q < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2b5c8f", markeredgecolor="#0F172A", markersize=6.2, label="PAAD only q < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D97706", markeredgecolor="#0F172A", markersize=6.2, label="METABRIC only q < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94A3B8", markeredgecolor="#0F172A", markersize=5.8, label=r"FDR $\geq 0.05$"),
    ]
    n_paad_gsea = int(paad["N_Patients"].iloc[0])
    n_meta_gsea = int(meta["N_Patients"].iloc[0])
    ax2.legend(handles=legend_a, loc="lower right", frameon=True, facecolor="white", fontsize=7.2)
    ax2.set_xlim(-2.6, 3.8); ax2.set_ylim(-2.8, 3.7)
    ax2.set_xlabel(f"TCGA-PAAD (N = {n_paad_gsea}) Normalized Enrichment Score (NES)")
    ax2.set_ylabel(f"METABRIC (N = {n_meta_gsea:,}) Normalized Enrichment Score (NES)")
    ax2.set_title("B   MSigDB v2024.1 Hallmark GSEA Concordance", loc="left", fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel C: Cohort-Standardized TCGA RPPA Proteomics (12 Cohorts, up to N=4,192)
    ax3 = fig.add_subplot(gs[1, 0])
    rho_p = df_m2["Within_Cohort_Spearman_rho"].astype(float)
    q_p = df_m2["Spearman_FDR_q"].astype(float).clip(lower=1e-30)
    neglogq = -np.log10(q_p)
    col_b = ["#CBD5E1" if q >= 0.05 else ("#e41a1c" if r > 0 else "#2b5c8f") for r, q in zip(rho_p, df_m2["Spearman_FDR_q"])]
    ax3.scatter(rho_p, neglogq, s=28, c=col_b, edgecolor="#0F172A", linewidth=0.4, alpha=0.85)
    ax3.axhline(-np.log10(0.05), color="#0F172A", linestyle="--", linewidth=0.9)
    ax3.axvline(0.0, color="#64748B", linestyle=":", linewidth=0.8)
    for tgt, lbl, (dx, dy) in [
        ("ASNS", "ASNS", (0.010, 0.5)),
        ("Cyclin_B1", "Cyclin B1", (0.014, 0.4)),
        ("eEF2", "eEF2", (0.018, 1.4)),
        ("4E-BP1", "4E-BP1", (-0.055, 0.9)),
        ("Fibronectin", "Fibronectin", (0.018, -1.3)),
        ("Chk2", "Chk2", (0.018, -0.8)),
        ("DJ-1", "DJ-1", (-0.032, 0.5)),
        ("Bcl-2", "Bcl-2", (-0.035, -0.5)),
        ("INPP4B", "INPP4B", (-0.042, 0.5)),
        ("Caveolin-1", "Caveolin-1", (-0.048, -0.7)),
    ]:
        sub = df_m2[df_m2["Protein_Target"] == tgt]
        if len(sub) > 0:
            rv = float(sub["Within_Cohort_Spearman_rho"].iloc[0])
            yq = float(-np.log10(max(float(sub["Spearman_FDR_q"].iloc[0]), 1e-30)))
            ax3.annotate(lbl, xy=(rv, yq), xytext=(rv + dx, yq + dy), fontsize=7.0, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6))
    n_sig = int((df_m2["Spearman_FDR_q"] < 0.05).sum())
    max_n_rppa = int(df_m2["N_Patients"].max())
    ax3.text(0.04, 0.94, f"{n_sig}/{len(df_m2)} antibodies FDR q < 0.05\n(12 cohorts, up to N = {max_n_rppa:,})", transform=ax3.transAxes, va="top", fontsize=7.8, fontweight="bold")
    legend_c4 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markeredgecolor="#0F172A", markersize=6.0, label=r"Positive $\rho$, FDR $q < 0.05$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2b5c8f", markeredgecolor="#0F172A", markersize=6.0, label=r"Negative $\rho$, FDR $q < 0.05$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CBD5E1", markeredgecolor="#0F172A", markersize=5.6, label=r"FDR $q \geq 0.05$"),
        Line2D([0], [0], color="#0F172A", linestyle="--", lw=0.9, label=r"FDR $q = 0.05$"),
    ]
    ax3.set_xlim(-0.20, 0.21); ax3.set_ylim(-1.0, 29.0)
    ax3.set_xlabel(r"Within-Cohort Standardized Spearman $\rho$")
    ax3.set_ylabel(r"$-\log_{10}$(Spearman FDR $q$-value)")
    ax3.set_title(f"C   12-Cohort TCPA Level-4 RPPA Proteomics (up to N = {max_n_rppa:,})", loc="left", fontweight="bold")
    ax3.legend(handles=legend_c4, loc="upper center", bbox_to_anchor=(0.48, 0.96), frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=6.8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel D: HRD & Aneuploidy Orthogonality
    ax4 = fig.add_subplot(gs[1, 1])
    n7_hrd = int(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "HRD_Sum"), "N_Patients"].iloc[0])
    n7_aneu = int(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "Aneuploidy_Score"), "N_Patients"].iloc[0])
    n17_hrd = int(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "HRD_Sum"), "N_Patients"].iloc[0])
    n17_aneu = int(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "Aneuploidy_Score"), "N_Patients"].iloc[0])
    dr2_7 = float(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Pooled (7 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "HRD_Sum"), "OLS_Incremental_Delta_R2"].iloc[0]) * 100.0
    dr2_17 = float(df_m4.loc[(df_m4["Cohort"] == "Pan-Cancer Extended TCGA (17 Cohorts)") & (df_m4["Genomic_Scar_Metric"] == "HRD_Sum"), "OLS_Incremental_Delta_R2"].iloc[0]) * 100.0
    cohort_order = [
        ("BRCA", "BRCA\n(1,041)"),
        ("COADREAD", "COADREAD\n(585)"),
        ("LUAD", "LUAD\n(497)"),
        ("PRAD", "PRAD\n(471)"),
        ("THCA", "THCA\n(461)"),
        ("UCEC", "UCEC\n(514)"),
        ("LUSC", "LUSC\n(479)"),
        ("Pan-Cancer Pooled (7 Cohorts)", f"7-Cohort\n({n7_hrd:,} /\n{n7_aneu:,})"),
        ("Pan-Cancer Extended TCGA (17 Cohorts)", f"12-Joint\n({n17_aneu:,}) /\n18-HRD ({n17_hrd:,})"),
    ]
    d_labels, abs_hrd, abs_aneu = [], [], []
    for ck, lbl in cohort_order:
        r_h = df_m4[(df_m4["Cohort"] == ck) & (df_m4["Genomic_Scar_Metric"] == "HRD_Sum")].iloc[0]
        r_a = df_m4[(df_m4["Cohort"] == ck) & (df_m4["Genomic_Scar_Metric"] == "Aneuploidy_Score")].iloc[0]
        d_labels.append(lbl)
        abs_hrd.append(abs(float(r_h["Spearman_Rho"])))
        abs_aneu.append(abs(float(r_a["Spearman_Rho"])))

    xd = np.arange(len(d_labels)); wd = 0.36
    cols_hrd = ["#e41a1c" if v >= 0.15 else "#2b5c8f" for v in abs_hrd]
    ax4.bar(xd - wd / 2, abs_hrd, width=wd, color=cols_hrd, edgecolor="#0F172A", linewidth=0.6)
    bars_aneu = ax4.bar(xd + wd / 2, abs_aneu, width=wd, color="#64748B", edgecolor="#0F172A", linewidth=0.6)
    for bar, v in zip(bars_aneu, abs_aneu):
        if v >= 0.15:
            bar.set_hatch("///")
            bar.set_edgecolor("#e41a1c")
            bar.set_linewidth(1.0)
    for i in range(len(d_labels)):
        ax4.text(xd[i] - wd / 2, abs_hrd[i] + 0.008, f"{abs_hrd[i]:.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
        ax4.text(xd[i] + wd / 2, abs_aneu[i] + 0.008, f"{abs_aneu[i]:.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=("#e41a1c" if abs_aneu[i] >= 0.15 else "#334155"))
    ax4.axhline(0.15, color="#e41a1c", linestyle="--", linewidth=1.0)
    ax4.text(0.97, 0.94, r"Cohort-FE partial $\Delta R^2$:" + f"\n7-Cohort (N={n7_aneu:,}): {dr2_7:.2f}% | 12-Cohort Joint (N={n17_aneu:,}) / 18-Cohort HRD (N={n17_hrd:,}): {dr2_17:.2f}%", transform=ax4.transAxes, va="top", ha="right", fontsize=7.2, fontweight="bold")
    legend_d = [
        mpatches.Patch(facecolor="#2b5c8f", edgecolor="#0F172A", linewidth=0.6, label=r"|Spearman $\rho$| with HRD (< 0.15)"),
        mpatches.Patch(facecolor="#e41a1c", edgecolor="#0F172A", linewidth=0.6, label=r"|Spearman $\rho$| with HRD ($\geq 0.15$)"),
        mpatches.Patch(facecolor="#64748B", edgecolor="#0F172A", linewidth=0.6, label=r"|Spearman $\rho$| with Aneuploidy (< 0.15)"),
        mpatches.Patch(facecolor="#64748B", edgecolor="#e41a1c", hatch="///", linewidth=1.0, label=r"|Spearman $\rho$| with Aneuploidy ($\geq 0.15$)"),
        Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=1.0, label=r"Orthogonality ceiling ($|\rho| = 0.15$)"),
    ]
    ax4.set_xticks(xd); ax4.set_xticklabels(d_labels, fontsize=7.4, fontweight="bold")
    ax4.set_ylim(0, 0.46)
    ax4.set_ylabel(r"$|\rho|$ vs. SMARCA4 + HIF1A Score")
    ax4.set_title("D   Chromosomal Instability Orthogonality (Knijnenburg et al.)", loc="left", fontweight="bold")
    ax4.legend(handles=legend_d, loc="upper right", bbox_to_anchor=(0.98, 0.82), frameon=True, facecolor="white", fontsize=7.0)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_dir, f"Figure_4_Final.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log("[OK] Saved Figure_4_Final.{pdf,png}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_raw = os.path.join(project_root, "data", "raw")
    data_proc = os.path.join(project_root, "data", "processed")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    missing = verify_required_inputs(project_root, data_raw, data_proc)
    if missing:
        curated_fp = os.path.join(data_proc, "curated_10cohort_clinical_survival.csv")
        t1_fp = os.path.join(data_proc, "tables", "Table_1_Clinical_Cohorts.csv")
        if os.path.exists(curated_fp) and os.path.exists(t1_fp):
            log("Raw archives absent; regenerating Figure 2 from curated_10cohort_clinical_survival.csv and preserving Figures 1, 3, and 4.")
            setup_rcparams()
            build_figure_2(project_root, data_raw, data_proc, fig_dir)
            return
        log("Required raw or intermediate datasets are absent; preserving existing files in figures/.")
        return

    setup_rcparams()
    build_figure_1(project_root, data_raw, data_proc, fig_dir)
    build_figure_2(project_root, data_raw, data_proc, fig_dir)
    build_figure_3(project_root, data_raw, data_proc, fig_dir)
    build_figure_4(project_root, data_raw, data_proc, fig_dir)

    remaining = sorted(os.listdir(fig_dir))
    log(f"Final files in figures/ ({len(remaining)}): {remaining}")


if __name__ == "__main__":
    main()
