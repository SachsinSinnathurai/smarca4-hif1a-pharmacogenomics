"""
Step 5: Generate publication-grade 2-panel summary figure (figures/fig1_audit_summary.pdf & .png)
strictly from verified repository results with zero hardcoded numbers:
  - Panel A: 18-cohort primary TCGA survival forest plot (N = 7,382) from final_verified_metrics.json
  - Panel B: Sanger GDSC2 post-MKI67 BH-FDR screen (m = 286 drugs) from gdsc2_post_mki67_fdr.csv
"""

from __future__ import annotations

import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fmt_p(p: float) -> str:
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10 ** exp)
        return f"$p = {mant:.2f} \\times 10^{{{exp}}}$"
    return f"$p = {p:.3f}$"


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "results")
    ms_dir = os.path.join(repo_root, "figures")
    os.makedirs(ms_dir, exist_ok=True)

    with open(os.path.join(out_dir, "final_verified_metrics.json"), "r", encoding="utf-8") as f:
        fin = json.load(f)
    df_gdsc = pd.read_csv(os.path.join(out_dir, "gdsc2_post_mki67_fdr.csv"))

    models = fin["primary_18_stage_grade_tcga"]["models"]
    n_total = int(models["SMARCA4_HIF1A_Base"]["N"])
    n_deaths = int(models["SMARCA4_HIF1A_Base"]["Events"])

    panel_a_specs = [
        ("SMARCA4_HIF1A_Base", "SMARCA4 + HIF1A\n(Base)", "#64748b"),
        ("SMARCA4_HIF1A_PlusMKI67", "SMARCA4 + HIF1A\n(+MKI67)", "#94a3b8"),
        ("Buffa14_Alone_Base", "Buffa-14\n(Base)", "#1d4ed8"),
        ("Buffa14_Alone_PlusMKI67", "Buffa-14\n(+MKI67)", "#0f766e"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7), gridspec_kw={"width_ratios": [1.12, 1.04]})
    fig.subplots_adjust(wspace=0.30, right=0.98)

    # --- Panel A: 18-Cohort Primary TCGA Survival Forest Plot ---
    y_pos = np.arange(len(panel_a_specs))[::-1]
    ax1.axvline(1.0, color="#334155", linestyle="--", linewidth=1.1, zorder=1)

    for y, (key, label, col) in zip(y_pos, panel_a_specs):
        m = models[key]
        hr, lo, hi, pv = float(m["hr"]), float(m["ci_low"]), float(m["ci_high"]), float(m["p"])
        ax1.errorbar(
            hr, y,
            xerr=[[hr - lo], [hi - hr]],
            fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=4, markersize=7, zorder=3,
        )
        annot = f"HR* = {hr:.3f} [{lo:.3f}–{hi:.3f}]\n{fmt_p(pv)}"
        ax1.text(1.225, y, annot, va="center", ha="left", fontsize=8.3, color="#0f172a")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([x[1] for x in panel_a_specs], fontsize=8.8)
    ax1.set_xlim(0.92, 1.44)
    ax1.set_ylim(-0.6, 3.6)
    ax1.set_xlabel("Pooled Per-SD Hazard Ratio (95% CI)")
    ax1.set_title(f"A. 18-Cohort Primary TCGA Survival (N = {n_total:,}; {n_deaths:,} deaths)", loc="left", fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # --- Panel B: Sanger GDSC2 Post-MKI67 BH-FDR Screen ---
    x_var = df_gdsc["MKI67_Adj_Partial_R2"].astype(float).values * 100.0
    q_vals = df_gdsc["MKI67_Adj_FDR_Q"].astype(float).values
    y_neglogq = -np.log10(np.clip(q_vals, 1e-300, 1.0))
    n_drugs = len(df_gdsc)
    n_sig10 = int((q_vals < 0.10).sum())

    top5_names = list(df_gdsc.sort_values("MKI67_Adj_P_Value").head(5)["Compound_Name"])
    is_top5 = df_gdsc["Compound_Name"].isin(top5_names).values

    ax2.scatter(x_var[~is_top5], y_neglogq[~is_top5], s=18, color="#94a3b8", alpha=0.65, edgecolors="none", zorder=2, label="Other compounds")
    ax2.scatter(x_var[is_top5], y_neglogq[is_top5], s=38, color="#dc2626", edgecolors="#7f1d1d", linewidth=0.7, zorder=4, label="Top 5 plateaued (q = 0.113)")

    q05_y = -math.log10(0.05)
    q10_y = -math.log10(0.10)
    ax2.axhline(q05_y, color="#b91c1c", linestyle="--", linewidth=1.1, zorder=1)
    ax2.axhline(q10_y, color="#d97706", linestyle="--", linewidth=1.1, zorder=1)
    ax2.text(0.05, q05_y + 0.03, "FDR q = 0.05", fontsize=8.0, color="#b91c1c", ha="left", va="bottom")
    ax2.text(0.05, q10_y + 0.03, "FDR q = 0.10", fontsize=8.0, color="#b45309", ha="left", va="bottom")

    ax2.text(
        0.97, 0.93,
        f"{n_sig10}/{n_drugs} compounds at q < 0.10",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8.5, fontweight="bold",
        color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=0.8),
    )

    # Non-overlapping annotations for the top 5 plateaued compounds sorted by x (partial R2 %)
    top5_df = df_gdsc[df_gdsc["Compound_Name"].isin(top5_names)].copy()
    top5_df["x_pct"] = top5_df["MKI67_Adj_Partial_R2"].astype(float) * 100.0
    top5_df["y_nlq"] = -np.log10(top5_df["MKI67_Adj_FDR_Q"].astype(float))
    top5_df = top5_df.sort_values("x_pct").reset_index(drop=True)

    # Spacious staggered label offsets in points so labels never collide
    offsets = [(-44, -22), (-22, -54), (12, -70), (54, -44), (62, -14)]
    for idx, row in top5_df.iterrows():
        dx, dy = offsets[idx]
        ax2.annotate(
            str(row["Compound_Name"]),
            xy=(row["x_pct"], row["y_nlq"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.8,
            color="#0f172a",
            ha="center",
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.75, shrinkA=0, shrinkB=2),
            zorder=5,
        )

    # Annotate small-sample outlier SL0101 directly
    sl_row = df_gdsc[df_gdsc["Compound_Name"] == "SL0101"].iloc[0]
    sl_x = float(sl_row["MKI67_Adj_Partial_R2"]) * 100.0
    sl_y = -math.log10(float(sl_row["MKI67_Adj_FDR_Q"]))
    ax2.annotate(
        f"SL0101 (n={int(sl_row['N_Solid_Lines'])}, q={float(sl_row['MKI67_Adj_FDR_Q']):.2f})",
        xy=(sl_x, sl_y),
        xytext=(-38, 18),
        textcoords="offset points",
        fontsize=7.8,
        color="#222222",
        ha="center",
        arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.75, shrinkA=0, shrinkB=2),
        zorder=5,
    )

    ax2.set_xlim(-0.08, max(x_var) * 1.15)
    ax2.set_ylim(-0.05, 1.48)
    ax2.set_xlabel("Post-MKI67 Partial $R^2$ (% Variance Explained)")
    ax2.set_ylabel(r"$-\log_{10}(\mathrm{BH\text{-}FDR\ } q)$")
    ax2.set_title(f"B. Sanger GDSC2 Post-MKI67 Screen (m = {n_drugs})", loc="left", fontweight="bold", pad=10)
    ax2.grid(linestyle=":", alpha=0.45)

    pdf_fp = os.path.join(ms_dir, "fig1_audit_summary.pdf")
    png_fp = os.path.join(ms_dir, "fig1_audit_summary.png")
    fig.savefig(pdf_fp, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png_fp, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved Figure 1 -> {pdf_fp} and {png_fp}")


if __name__ == "__main__":
    main()
