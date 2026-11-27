# Data Access Guide

Raw and patient-level datasets (`data/raw/`) are **not** redistributed in this repository to respect Data Use Agreements (DUAs), controlled-access governance, and repository size limits. `data/processed/` contains only aggregated summary-level tables required by `scripts/01`–`05`.

## 1. Publicly Accessible Datasets

| Resource | Accessions / Files | Official Source |
| :--- | :--- | :--- |
| **TCGA PanCancer Atlas** | Clinical CDR (`TCGA-CDR-SupplementalTableS1.xlsx`), RSEM normalized RNA-seq, ABSOLUTE purity, Knijnenburg CIN scars | [GDC PanCanAtlas](https://gdc.cancer.gov/about-data/publications/pancanatlas) / [cBioPortal](https://www.cbioportal.org/) |
| **Broad DepMap Public 24Q2** | `CRISPRGeneEffect.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutations.csv`, `Model.csv` | [depmap.org](https://depmap.org/portal/) |
| **Sanger GDSC2** | `GDSC2_fitted_dose_response_*.xlsx` ($m = 286$ compounds) | [cancerrxgene.org](https://www.cancerrxgene.org/) |
| **NCBI GEO Hypoxia Series** | `GSE139673`, `GSE145108`, `GSE283995`, `GSE300765` ($1\%$ vs. $21\%\ \text{O}_2$) | [ncbi.nlm.nih.gov/geo](https://www.ncbi.nlm.nih.gov/geo/) |
| **PDAC scRNA-seq (`CRA001160`)** | Peng et al. PDAC scRNA-seq (`57,443` cells, processed H5/metadata) | [GSA CRA001160](https://ngdc.cncb.ac.cn/gsa/browse/CRA001160) / [TISCH2](http://tisch.comp-genomics.org/) |
| **CGGA-325** | Primary/recurrent glioma RSEM expression & survival (`N = 313` complete) | [cgga.org.cn](http://www.cgga.org.cn/) |
| **MSigDB v2024.1** | Human Hallmark gene sets (`h.all.v2024.1.Hs.symbols.gmt`) | [gsea-msigdb.org](https://www.gsea-msigdb.org/gsea/msigdb/) |
| **TCPA Level-4 RPPA** | Pan-cancer Level-4 RPPA protein expression (v4) | [tcpaportal.org](https://tcpaportal.org/tcpa/) |

## 2. Controlled-Access / DUA-Restricted Datasets

- **`BRCA-METABRIC`** (Curtis et al., *Nature* 2012): Patient-level clinical and expression matrices are governed by EGA (`EGAS00000000083`) and cBioPortal redistribution terms. Request access via EGA or download authorized extracts from cBioPortal (`brca_metabric`) for local reproduction.
- **`ICGC/PCAWG-PAAD`** ($N = 71$): Controlled-access/ICGC-governed clinical and donor expression tables must be obtained directly through the [ICGC Data Portal](https://dcc.icgc.org/) / PCAWG data access channels.
