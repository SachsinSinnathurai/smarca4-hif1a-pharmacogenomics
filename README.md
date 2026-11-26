# Proliferation Confounding and Bulk Transcriptomic Limitations of the *SMARCA4*–*HIF1A* Axis Across Solid Tumors

Reproducible code, aggregated summary tables, summary figure, and manuscript source for the pan-solid-tumor multi-omic evaluation of bulk *SMARCA4* and *HIF1A* mRNA abundance.

## Study Summary

Using 18 stage/grade-adjusted adult solid TCGA cohorts ($N = 7,382$; 2,352 deaths), a 19-cohort sensitivity pool ($N = 7,875$), a 10-cohort clinical panel ($N = 5,102$), Broad DepMap 24Q2 ($n = 1,038$ solid cancer cell lines), Sanger GDSC2 ($m = 286$ compounds), four NCBI GEO $1\%$ vs. $21\%\ \text{O}_2$ hypoxia series, and PDAC single-cell RNA-seq (`CRA001160`, 57,443 cells), this study shows that:

1. **Prognostic confounding by proliferation:** Unadjusted *SMARCA4* + *HIF1A* shows no significant pooled survival association across the 18 primary TCGA cohorts ($\text{HR}^* = 1.044$, $p = 0.162$) and attenuates further after *MKI67* adjustment ($\text{HR}^* = 1.018$, $p = 0.545$), whereas the downstream *Buffa-14* hypoxia metagene retains independent prognostic signal ($\text{HR}^* = 1.115$, $p = 7.80 \times 10^{-7}$).
2. **Pharmacogenomic null after proliferation control:** Across $m = 286$ GDSC2 compounds controlling for lineage and *MKI67*, zero drugs pass $\text{FDR} < 0.10$ (minimum BH $q = 0.1126$).
3. **Cellular and post-transcriptional decoupling:** *SMARCA4* and *HIF1A* exhibit modest correlation in bulk solid tumors and cell lines, *VHL* loss-of-function selectively induces *EPAS1* rather than *HIF1A* dependency, acute hypoxia stabilizes HIF-1$\alpha$ protein without consistent *HIF1A* mRNA induction, and PDAC scRNA-seq localizes *HIF1A* mRNA predominantly to myeloid cells rather than malignant ductal cells.

## Repository Layout

- `manuscript/`: Compiled 11-page PDF, Markdown manuscript, and standalone LaTeX source (`Proliferation_Confounding_and_Bulk_Transcriptomic_Limitations_of_the_SMARCA4_HIF1A_Axis_Across_Solid_Tumors.pdf`, `.md`, `.tex`).
- `figures/`: Two-panel publication summary figure (`fig1_audit_summary.pdf`).
- `scripts/`: Reproducible Python scripts (`01_record_input_hashes.py`–`05_make_summary_figure.py`).
- `data/processed/`: Aggregated summary-level inputs used by `scripts/01`–`05` (zero patient-level clinical/expression rows from controlled-access cohorts).
- `results/`: Aggregated cohort-, drug-, cell-type-, and pathway-level `.csv` and `.json` summary tables backing Tables 1–2 and Figure 1.
- `data/README.md`: Data-access guide for downloading public DepMap/TCGA/GEO inputs and controlled-access cohorts (`data/raw/` is excluded via `.gitignore`).

## Reproduction

Install exact Python dependencies (`Python 3.14.7` tested):

```bash
pip install -r requirements.txt
```

`scripts/01`–`05` reproduce all meta-analyses, BH-FDR corrections, DepMap/VHL reanalyses (from cached summary metrics when `data/raw/` is absent, or from scratch when public DepMap/TCGA inputs are downloaded into `data/raw/`), figures, and manuscript verification checks directly from `data/processed/` plus public DepMap/TCGA inputs, while `data/raw/` controlled-access inputs are excluded:

```bash
python scripts/01_record_input_hashes.py
python scripts/02_depmap_and_hypoxia_reanalysis.py
python scripts/03_survival_fdr_and_gdsc2_reanalysis.py
python scripts/04_finalize_tables_and_verify.py
python scripts/05_make_summary_figure.py
```

## License

Released under the [MIT License](LICENSE) (Sachsin Sinnathurai, 2026).
