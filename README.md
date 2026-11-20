# Lineage-Restricted Transcriptomic Prognosis and Normoxic Pharmacogenomic Uncoupling of the *SMARCA4*–*HIF1A* Axis Across Solid Tumors

**Author:** Sachsin Sinnathurai  
**Affiliation:** Department of Anatomy & Cell Biology, McGill University, Montreal, QC, Canada  
**Correspondence:** sachsin.sinnathurai@mail.mcgill.ca  
**Repository:** [https://github.com/SachsinSinnathurai/smarca4-hif1a-pharmacogenomics](https://github.com/SachsinSinnathurai/smarca4-hif1a-pharmacogenomics)

***

## Overview

This repository examines the biological and clinical behavior of the apical **SMARCA4**–**HIF1A** transcriptional axis across Broad DepMap 24Q2 CRISPR-Cas9 Chronos screens ($1,038$ solid tumor lines across $1,719$ unique expression models), $5,102$ complete-case primary tumors ($2,162$ deaths across 10 cohorts), independent pancreatic validation in ICGC/PCAWG-PAAD ($N = 71$, $48$ deaths), and an expanded 19-cohort solid TCGA benchmark ($N = 7,875$).

We evaluated an unweighted *a priori* score standardized within each cohort ($\text{mean} = 0$, $\text{SD} = 1$):

$$\text{Risk Score} = Z\bigl(z_{\text{SMARCA4}} + z_{\text{HIF1A}}\bigr)$$

which focuses on upstream transcriptional regulators (*SMARCA4* and *HIF1A*) while excluding downstream effectors (*SLC7A1*, *SLC1A5*, *ABCB1*). In hypocellular bulk TCGA-PAAD ($N = 168$ with Age, Stage, and ABSOLUTE Purity; median purity $0.350$), the unweighted score is null ($\text{HR} = 1.143$, $p = 0.273$), and paired single-cell PDAC profiling (`CRA001160`) shows comparable *HIF1A* expression in malignant ductal cells and stromal fibroblasts ($\log_2\text{FC} = -0.217$, $p = 0.134$). Age- and sex-adjusted ICGC/PCAWG-PAAD reaches multivariable significance ($\text{HR} = 1.504$, $p = 0.009$), though unadjusted log-rank separation is borderline ($p = 0.060$). Across 10 primary cohorts ($N = 5,102$), DerSimonian-Laird random-effects meta-analysis yields $\text{HR}^* = 1.144$ ($p = 9.10 \times 10^{-8}$, $I^2 = 16.1\%$) and persists across eight non-breast primary cohorts ($N = 2,586$, $\text{HR}^* = 1.152$, $p = 7.27 \times 10^{-5}$). Single-gene decomposition shows that *SMARCA4* drives most of the hazard ($\text{HR}^* = 1.153$, $p = 2.82 \times 10^{-4}$), whereas *HIF1A* mRNA alone is null ($\text{HR}^* = 1.047$, $p = 0.222$). Complete 10-cohort *MKI67* proliferation adjustment using the full $24,326$-gene public CGGA-325 RSEM transcriptome lowers pooled risk to $\text{HR}^* = 1.084$ ($p = 0.004$) and abolishes glioma stratification ($\text{HR} = 1.006$, $p = 0.943$). Across all 19 solid TCGA cohorts ($N = 7,875$), the two-gene score becomes null ($\text{HR}^* = 1.048$, $p = 0.130$), while the 14-gene non-cell-cycle *Buffa-14* hypoxia signature stays strongly prognostic ($\text{HR}^* = 1.133$, $p = 2.17 \times 10^{-6}$; $+\textit{MKI67}$ $\text{HR}^* = 1.116$, $p = 5.54 \times 10^{-7}$). Independent *SMARCA4* + *HIF1A* risk after *MKI67* and purity control therefore localizes strictly to clear cell renal and urothelial bladder carcinomas. Under ambient $21\%\ \text{O}_2$ Sanger GDSC2 $1 - \text{AUC}$ screening ($m = 286$ drugs), only Navitoclax clears 5\% BH FDR ($q = 0.049$, partial $R^2 = 2.45\%$) and nine compounds clear 10\% FDR ($\eta_p^2 = 1.55\%\text{--}2.63\%$), whereas TOP1 and PARP1/2 trends weaken after *MKI67* control and $0/3$ combinations track baseline risk after lineage adjustment; lineage-independent *SLFN11* and *BAK1* gates stay intact. Harmonized UCSC Xena Toil comparisons against GTEx v6 show high normal cardiac ($44.08\text{--}59.22$ TPM) and hepatic ($67.54$ TPM) *MCL1* relative to solid tumors ($114.25$ TPM), favoring tumor-targeted delivery while systemic dose-sparing claims and public human AZD5991 pharmacokinetics remain unverified.

***

## Core Empirical Findings

### 1. Solute Carrier Co-Dependency, Core BAF Co-Essentiality & Normoxic Uncoupling (`Figure 1`)
- **Solute Carrier Co-Dependency:** Across 1,038 solid tumor lines in DepMap 24Q2, *SLC1A5* and *SLC7A1* share Chronos dependency ($r = +0.207$, $p = 1.51 \times 10^{-11}$, FDR $q = 1.51 \times 10^{-10}$), and *SLC7A1* couples with *SLC7A5* (solid $r = +0.3882$) while staying independent of *SLC7A2* ($r = +0.0110$).
- **BAF Co-Essentiality & Diploid Sparing:** *SMARCA4*-dependent solid lines ($n = 299$ vs. $251$) co-depend on core BAF subunits (*SMARCB1* $\Delta = -0.307$, $p = 3.04 \times 10^{-17}$; continuous $r = +0.295$, $\rho = +0.304$) and *MLLT1* alongside *SMARCA2* paralog buffering ($\Delta = +0.133$, $q = 1.93 \times 10^{-5}$; continuous $r = -0.197$). Non-malignant diploid lines ($n = 16$) show weaker single-gene *SMARCA4* dependency than solid tumor lines ($\Delta_{\text{mean}} = +0.260$, $\Delta_{\text{median}} = +0.205$, Mann-Whitney $p = 0.007$).
- **Normoxic 21% $\text{O}_2$ Uncoupling & Strict *VHL*-LoF Coupling:** In vivo *HIF1A*–*SLC7A1* mRNA coupling across 1,384 primary TCGA tumors (`PAAD`, `GBM`, `BRCA`; within-cohort $r = +0.2844$, $p = 3.72 \times 10^{-27}$) vanishes in 21% $\text{O}_2$ solid DepMap lines ($n = 1,349$, $r = +0.0249$, $p = 0.360$; Fisher $Z = 6.98$, $p = 2.87 \times 10^{-12}$). Truncating or splice loss-of-function *VHL* mutations ($n = 19$) preserve *HIF1A*–*SLC7A1* Chronos coupling ($r = +0.5412$, Spearman $\rho = +0.5439$, $p = 0.016$; 19-fold leave-one-out $r = +0.4706\text{--}+0.6274$) alongside selective *EPAS1* vulnerability compared with *VHL* wild-type kidney ($d = -1.229$, $p = 6.74 \times 10^{-4}$) and non-ccRCC WT lines ($d = -2.649$, $p = 1.15 \times 10^{-3}$).
- **Paired 1% vs. 21% $\text{O}_2$ GEO Hypoxia Validation & Solid-Tumor ChIP-Atlas Overlaps:** Across four NCBI GEO $1\%\ \text{O}_2$ vs. $21\%\ \text{O}_2$ series (`GSE283995`, `GSE139673`, `GSE145108`, `GSE300765` on Illumina HumanHT-12 v4 `GPL10558`), *CA9* and *VEGFA* rise strongly without elevating *HIF1A* mRNA ($\log_2\text{FC} = -1.111$ to $+0.107$). Acute 24–48 h hypoxia fails to induce *SLC7A1* in U87MG ($p = 0.517$), AsPC-1 ($-0.458$, $p = 0.001$), and pooled RKO/A549/H460 ($-0.779$, $p = 0.028$; `H460` $-1.367$, $p = 0.009$). In HepG2, 24-hour *SLC7A1* induction ($+0.299$) reverses by day 6 of chronic hypoxia ($-0.200$, $p = 0.121$, $n = 5$) despite sustained *CA9* ($+3.376$). In unpaired ChIP-Atlas tracks, *SMARCA4* binding in *HIF1A*-null `786-O` cells (`SRX8566545`, MACS2 score 493, $7.08\times$) indicates constitutive SWI/SNF bookmarking at an intronic *SLC7A1* enhancer that overlaps hypoxic *HIF1A* binding in `RCC4+VHL` (`SRX15691692`, $11.18\times$).

### 2. Unweighted Two-Gene Stratification & Unified Multi-Cohort Meta-Analysis (`Figure 2`)
- **Pancreatic Cohorts (`TCGA-PAAD` vs. `ICGC/PCAWG-PAAD`):** On PanCancer Atlas TCGA-PAAD ($N = 168$ with Stage and ABSOLUTE Purity), the unweighted *SMARCA4 + HIF1A* score is null ($\text{HR} = 1.143$, $p = 0.273$; Pure Ductal PDAC $N = 148$, $\text{HR} = 0.950$, $p = 0.669$; nested LRT vs. *MKI67* $\chi^2_{\text{LRT}}(1) = 0.00$, $p = 0.997$). Age- and sex-adjusted ICGC/PCAWG-PAAD ($N = 71$) validates in multivariable Cox modeling ($\text{HR} = 1.504$, $p = 0.009$, C-index $= 0.658$) with borderline log-rank separation ($p = 0.060$).
- **Unified Meta-Analysis, Full 10-Cohort *MKI67*, Purity Controls & *Buffa-14* Benchmark:** DerSimonian-Laird pooling yields $\text{HR}^* = 1.144$ ($p = 9.10 \times 10^{-8}$, $I^2 = 16.1\%$) across 10 primary cohorts ($N = 5,102$), $\text{HR}^* = 1.152$ ($p = 7.27 \times 10^{-5}$, $I^2 = 28.9\%$) across 8 non-breast primary cohorts ($N = 2,586$), $\text{HR}^* = 1.166$ ($p = 4.22 \times 10^{-5}$, $N = 2,657$) across 9 non-breast cohorts, and $\text{HR}^* = 1.155$ ($p = 1.96 \times 10^{-7}$, $N = 5,173$) across 11 cohorts. Single-gene *SMARCA4* carries most of the signal ($\text{HR}^* = 1.153$) while *HIF1A* is null ($\text{HR}^* = 1.047$). Adding full $24,326$-gene CGGA-325 *MKI67* completes 10-cohort proliferation control ($\text{HR}^* = 1.084$, $p = 0.004$, $I^2 = 20.3\%$, $N = 5,102$) and eliminates glioma stratification ($\text{HR} = 1.006$, $p = 0.943$; *TOP2A*: $\text{HR} = 0.949$). Excluding 513 stage-missing METABRIC cases introduces no selection bias ($N = 1,979$ age-only $\text{HR} = 1.111$, $p = 3.91 \times 10^{-4}$ vs. $N = 1,466$ $\text{HR} = 1.115$), and PAM50 or receptor controls attenuate breast hazard ($\text{HR} = 1.035\text{--}1.070$). ABSOLUTE purity control preserves signal in TCGA-KIRC ($N = 496$, $+\text{Purity}$ $\text{HR} = 1.314$, $p = 4.74 \times 10^{-4}$; $+\textit{MKI67}+\text{Purity}$ $\text{HR} = 1.282$, $p = 0.002$) and TCGA-BLCA ($N = 393$, $+\text{Purity}$ $\text{HR} = 1.195$, $p = 0.035$; $+\textit{MKI67}+\text{Purity}$ $\text{HR} = 1.214$, $p = 0.028$). Expanding across all 19 solid TCGA cohorts ($N = 7,875$) gives a null two-gene hazard ($\text{HR}^* = 1.048$, $p = 0.130$, $I^2 = 50.5\%$), whereas the non-cell-cycle *Buffa-14* hypoxia signature (*ACOT7*, *ADM*, *ALDOA*, *ENO1*, *LDHA*, *MIF*, *MRPS17*, *NDRG1*, *P4HA1*, *PGAM1*, *SLC2A1*, *TPI1*, *TUBB6*, *VEGFA*; excluding *CDKN3*) remains strongly prognostic ($\text{HR}^* = 1.133$, $95\%$ CI $[1.076\text{--}1.193]$, $p = 2.17 \times 10^{-6}$, $I^2 = 30.7\%$; $+\textit{MKI67}$ $\text{HR}^* = 1.116$, $95\%$ CI $[1.069\text{--}1.165]$, $p = 5.54 \times 10^{-7}$, $I^2 = 0.0\%$) and improves nested model fit over *SMARCA4 + HIF1A* ($\chi^2_{\text{LRT}}(19) = 56.56$, $p = 1.33 \times 10^{-5}$, joint $\Delta C = +0.0104$; $+\textit{MKI67}$ $\chi^2_{\text{LRT}}(19) = 43.03$, $p = 1.29 \times 10^{-3}$, joint $\Delta C = +0.0070$). Across 22 cohorts, 19 satisfy Schoenfeld proportional hazards ($p = 0.112\text{--}0.979$).

### 3. Lineage-Controlled $1 - \text{AUC}$ Pharmacology, Bliss Synergy & UCSC Xena Toil (`Figure 3`)
- **Full-Screen GDSC2 $1 - \text{AUC}$ Type II ANOVA ($m = 286$ Drugs):** Across all 286 compounds on empirical $1 - \text{AUC}$ (screen-wide median partial $R^2$ $\eta_p^2 = 0.176\%$, maximum $3.53\%$), only Navitoclax clears 5\% BH FDR ($F(1,570) = 14.32$, $\eta_p^2 = 2.45\%$, $p = 1.71 \times 10^{-4}$, $q = 0.049$) and nine compounds clear 10\% FDR ($\eta_p^2 = 1.55\%\text{--}2.63\%$) among 35 nominal associations: BH3-mimetics (Navitoclax $q = 0.049$; ABT737 $q = 0.059$; WEHI-539 $q = 0.059$), platinum agents (Cisplatin $q = 0.055$; Oxaliplatin $q = 0.091$), Elephantin ($q = 0.059$), Entinostat ($q = 0.086$), Uprosertib ($q = 0.091$), and PRMT5 inhibitor GSK591 ($q = 0.091$, resistance slope $\beta = -0.0046$). All nine persist after *MKI67* control ($p = 6.43 \times 10^{-4}$ to $0.011$), whereas TOP1 and PARP1/2 agents attenuate (Niraparib $p = 0.072$; Olaparib $p = 0.120$; Talazoparib $p = 0.136$) and secondary $\ln(\text{IC}_{50})$ testing yields $0/286$ at $q < 0.10$ (minimum $q = 0.110$).
- **Combination Synergy, $1 - \text{AUC}$ Gates & UCSC Xena Toil (TCGA vs. GTEx v6):** Against the 109-pair Bliss background ($+0.0662$), **Olaparib + SN-38** ($+0.1543$, 89.9th percentile, $p = 9.19 \times 10^{-67}$) and **Capivasertib + Selumetinib** ($+0.0872$) exceed background while **AZD5153 + AZD5991** ($+0.0626$) peaks at low-dose `(D1, D1)` ($+0.156$, $n = 199$). Zero of three combinations track baseline risk after lineage control ($p = 0.068\text{--}0.613$), whereas *SLFN11* (SN-38 $r_{\text{adj}} = +0.1775$, $p = 2.19 \times 10^{-5}$; Olaparib $r_{\text{adj}} = +0.2486$, $p = 1.66 \times 10^{-9}$) and *BAK1* (AZD5991 $r_{\text{adj}} = +0.1553$, $p = 0.001$; *BAX* null $p = 0.280$) predict empirical $1 - \text{AUC}$ across solid lines. In harmonized UCSC Xena Toil comparisons ($n = 9,138$ TCGA primary solid tumors vs. GTEx v6 organs), normal *MCL1* stays substantial in Heart Left Ventricle (44.08 TPM, $n = 202$), Heart Atrial Appendage (59.22 TPM, $n = 175$), Liver (67.54 TPM, $n = 110$), and Whole Blood (135.67 TPM, $n = 337$, vs. 114.25 TPM in tumors) alongside modest tumor margins for *BRD4* ($11.92\text{--}15.57$ TPM vs. $22.63$ TPM in tumors; $1.45\times\text{--}1.90\times$) and tumor-enriched *SMARCA4* ($73.08$ TPM vs. $6.90\text{--}19.67$ TPM). Flat GDSC2 $1 - \text{AUC}$ medians for AZD5991 (0.0309) and Olaparib (0.0407) explain $>10\ \mu\text{M}$ $\text{IC}_{50}$ curve extrapolations, and missing public human AZD5991 $C_{\max}$, plasma $f_u$, and per-well 7×7 viability data preclude verified unbound $C_{\max, u}$ or Chou-Talalay DRI models.

### 4. Paired scRNA-seq Specificity, Hallmark Pathways, RPPA & Chromosomal Orthogonality (`Figure 4`)
- **Single-Cell PDAC Pseudobulk (`CRA001160`):** Across $n = 23$ paired resections ($41,976$ tumor cells back-transformed via $\exp(y) - 1$), malignant ductal epithelium enriches for *SLC7A1* ($+1.564\ \log_2\text{FC}$, $p = 2.38 \times 10^{-7}$) and *SMARCA4* ($+0.430\ \log_2\text{FC}$, $p = 0.005$), whereas *HIF1A* ($-0.217$, $p = 0.134$) and *SLC1A5* ($+0.275$, $p = 0.377$) match CAFs and *ABCB1* is depleted ($-0.666$, $p = 0.011$).
- **Hallmark GSEA, CAMERA, TCPA RPPA & Orthogonality:** Pre-ranked Hallmark GSEA NES correlates across TCGA-PAAD ($N = 176$) and METABRIC ($N = 1,980$; $\rho = +0.4182$, $p = 0.003$), while competitive CAMERA yields `0/50` FDR $< 0.05$ in PAAD and `10/50` in METABRIC. Cohort-standardized 12-cohort TCPA RPPA ($N = 4,192$) identifies 98/194 antibodies at FDR $q < 0.05$. CIN partial $\Delta R^2$ is $0.98\%$ (7 cohorts: $N = 4,045$ joint) and $0.58\%$ (12 PanCan joint cohorts $N = 5,604$ out of $7,770$ HRD across 18 cohorts), and anti-PD-L1 OS in IMvigor210 ($N = 347$) is orthogonal ($\text{HR} = 1.013$, $p = 0.841$).

***

## Clinical Survival Summary (`Table 1`)

*Multivariable Cox proportional hazards models for the unweighted a priori score $Z(z_{\text{SMARCA4}} + z_{\text{HIF1A}})$ across 5,102 complete-case primary tumors (2,162 deaths), secondary Pure Ductal PDAC ($N = 148$), independent ICGC/PCAWG-PAAD ($N = 71$, 48 deaths), and expanded 19-cohort solid TCGA ($N = 7,875$) pools alongside the 19-cohort Buffa-14 benchmark.*

| Cohort | $N$ (Deaths) | Continuous Per-SD HR [95% CI] | Wald $p$ | Median Split HR [95% CI] | Median $p$ | C-Idx | Covariates / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCGA-PAAD** | 168 (89) | 1.143 [0.900–1.452] | 0.273 | 1.093 [0.717–1.666] | 0.679 | 0.579 | Age, Stage, ABSOLUTE Purity |
| **CGGA-325** (Diffuse Glioma II-IV) | 313 (218) | **1.193** [1.050–1.355] | **0.007** | **1.684** [1.264–2.245] | **$3.73 \times 10^{-4}$** | 0.717 | Age, Grade, IDH, MGMT |
| **TCGA-KIRC** (Clear Cell Renal) | 531 (175) | **1.289** [1.114–1.492] | **$6.61 \times 10^{-4}$** | **1.647** [1.212–2.238] | **0.001** | 0.764 | Age, Stage |
| **TCGA-SARC** (Soft-Tissue Sarcoma) | 253 (96) | 1.183 [0.963–1.453] | 0.110 | 1.491 [0.992–2.240] | 0.055 | 0.584 | Age |
| **TCGA-SKCM** (Cutaneous Melanoma) | 417 (196) | 1.108 [0.956–1.283] | 0.172 | 1.272 [0.952–1.698] | 0.103 | 0.648 | Age, Stage |
| **TCGA-BRCA** (Breast Carcinoma) | 1,050 (143) | **1.187** [1.012–1.393] | **0.035** | 1.385 [0.994–1.930] | 0.054 | 0.762 | Age, Stage |
| **TCGA-LIHC** (Hepatocellular) | 343 (118) | 1.172 [0.977–1.407] | 0.088 | 1.213 [0.838–1.755] | 0.306 | 0.643 | Age, Stage |
| **TCGA-BLCA** (Urothelial Bladder) | 403 (177) | **1.206** [1.024–1.419] | **0.024** | **1.530** [1.132–2.068] | **0.006** | 0.668 | Age, Stage |
| **BRCA-METABRIC** (Breast OS) | 1,466 (823) | **1.114** [1.040–1.192] | **0.002** | **1.238** [1.078–1.421] | **0.002** | 0.641 | Age, Stage |
| **TCGA-GBM** (WHO Grade IV) | 158 (127) | 0.922 [0.778–1.093] | 0.348 | 0.802 [0.565–1.140] | 0.219 | 0.618 | Age |
| **TCGA-PAAD** (Pure Ductal PDAC; Secondary) | 148 (82) | 0.950 [0.750–1.202] | 0.669 | 0.780 [0.504–1.209] | 0.267 | 0.578 | Age, Stage, ABSOLUTE Purity |
| **ICGC/PCAWG-PAAD** (Validation) | 71 (48) | **1.504** [1.109–2.041] | **0.009** | **1.821** [1.017–3.263] | **0.044** | 0.658 | Age, Sex (stage unavailable) |
| **8-Cohort Non-Breast Primary Meta** | 2,586 (1,196) | **1.152** [1.074–1.235] | **$7.27 \times 10^{-5}$** | -- | -- | -- | Excl. TCGA-BRCA, METABRIC & ICGC-PAAD ($I^2 = 28.9\%$) |
| **9-Cohort Non-Breast Meta** | 2,657 (1,244) | **1.166** [1.083–1.255] | **$4.22 \times 10^{-5}$** | -- | -- | -- | Excl. TCGA-BRCA & METABRIC ($I^2 = 36.7\%$) |
| **10-Cohort Primary Meta** | 5,102 (2,162) | **1.144** [1.089–1.203] | **$9.10 \times 10^{-8}$** | -- | -- | -- | 10 Primary Cohorts ($I^2 = 16.1\%$) |
| **11-Cohort Unified Meta** | 5,173 (2,210) | **1.155** [1.094–1.220] | **$1.96 \times 10^{-7}$** | -- | -- | -- | 10 Primary + ICGC/PCAWG-PAAD ($I^2 = 27.7\%$) |
| **19-Cohort Expanded Solid TCGA** | 7,875 (2,362) | 1.048 [0.986–1.114] | 0.130 | -- | -- | 0.688 | 8 Primary + 11 Additional TCGA ($I^2 = 50.5\%$) |
| **19-Cohort Solid TCGA: *Buffa-14*** | 7,875 (2,362) | **1.133** [1.076–1.193] | **$2.17 \times 10^{-6}$** | -- | -- | 0.690 | *Buffa-14* ($I^2 = 30.7\%$; LRT $\chi^2(19)=56.56$, $p=1.33\times 10^{-5}$) |
| **19-Cohort Solid TCGA: *Buffa-14* (+*MKI67*)** | 7,875 (2,362) | **1.116** [1.069–1.165] | **$5.54 \times 10^{-7}$** | -- | -- | 0.696 | *Buffa-14* +*MKI67* ($I^2 = 0.0\%$; LRT $\chi^2(19)=43.03$, $p=1.29\times 10^{-3}$) |

***

## Repository Structure

```
├── data/
│   ├── raw/                                                  # Primary DepMap 24Q2, TCGA, METABRIC, CGGA-325 (full RSEM + clinical), Sanger GDSC & TISCH2 archives
│   └── processed/
│       ├── tables/Table_1_Clinical_Cohorts.csv               # Primary survival summary; generated by src/analysis/survival_lasso_meta_analysis.py
│       ├── curated_10cohort_clinical_survival.csv            # Open-access 10-cohort survival table (5,110 primary + 71 ICGC/PCAWG-PAAD; Fig. 2 fallback)
│       ├── smarca4_coessentiality_genomewide.csv             # Genome-wide SMARCA4 co-essentiality scan (18,434 genes; Fig. 1B)
│       ├── smarca4_significant_synthetic_lethal_genes.csv    # Significant SMARCA4 co-essentiality & buffering hits (FDR q < 0.05)
│       ├── geo_hypoxia_perturbation.csv                      # Paired 1% vs 21% O2 GEO validation (GSE283995, GSE139673, GSE145108, GSE300765)
│       ├── chipseq_slc7a1_occupancy.csv                      # ChIP-Atlas hg38 HIF1A/SMARCA4 peaks at SLC7A1 & VEGFA (9 solid-tumor tracks)
│       ├── tcga_paad_histology_mki67_results.csv             # Generated by src/analysis/survival_lasso_meta_analysis.py
│       ├── survival_full_tcga_and_mki67_sensitivity.csv      # 10-cohort +MKI67 (incl. CGGA-325), METABRIC missingness, KIRC/BLCA purity, 22-cohort PH & Buffa-14 benchmark
│       ├── survival_gene_and_mki67_sensitivity.csv           # Synchronized sensitivity alias for downstream reproducibility
│       ├── brca_subtype_receptor_sensitivity.csv             # Complete-case TCGA-BRCA PAM50 & METABRIC subtype/ER/PR/HER2 Cox models
│       ├── gdsc2_lineage_controlled_anova.csv                # Generated by src/analysis/multiomics_and_gdsc2_anova.py (1-AUC primary, partial R^2 & ln(IC50); Fig. 3A)
│       ├── gdsc_combination_synergy_summary.csv                # Generated by src/analysis/gdsc_combination_synergy.py
│       ├── toil_tcga_vs_gtex_tpm.csv                         # Harmonized UCSC Xena Toil (TCGA primary solid vs. GTEx v6 organs; Fig. 3E)
│       ├── dose_sparing_dri_pk_gtex.csv                      # Harmonized Toil (GTEx v6) TPMs, GDSC D1/IC50, GDSC2 1-AUC ceiling & AZD5153 Phase I Cmax summary
│       ├── pdac_scrna_celltype_specificity.csv               # Generated by src/analysis/pdac_scrna_specificity.py
│       ├── hallmark_gsea_transcriptome_wide_enrichment.csv   # Generated by src/analysis/multiomics_and_gdsc2_anova.py (PAAD N=176, METABRIC N=1,980; Fig. 4B)
│       ├── tcga_rppa_proteomic_validation.csv                # Generated by src/analysis/multiomics_and_gdsc2_anova.py (12 TCPA cohorts, N=4,192; Fig. 4C)
│       └── genomic_scars_hrd_aneuploidy_orthogonality.csv    # Generated by src/analysis/multiomics_and_gdsc2_anova.py (7/12/18 TCGA cohorts; Fig. 4D)
├── figures/
│   ├── Figure_1_Final.{pdf,png}                              # DepMap co-dependency (n = 1,038), BAF co-essentiality, sparing & 21% O2 uncoupling
│   ├── Figure_2_Final.{pdf,png}                              # Uniform median-split KM curves (6 cohorts) & unified 10/11/19-cohort + Buffa-14 meta-analysis
│   ├── Figure_3_Final.{pdf,png}                              # GDSC2 1-AUC ANOVA, peak 3x3 Bliss surfaces, 1-AUC biomarker gates & Toil (GTEx v6)/PK counter-screens
│   └── Figure_4_Final.{pdf,png}                              # Paired PDAC scRNA-seq pseudobulk, Hallmark GSEA, RPPA proteomics & CIN orthogonality
├── manuscript/
│   ├── manuscript.pdf                                        # Compiled academic preprint with Figures 1-4 embedded
│   └── manuscript.tex                                        # LaTeX manuscript source with 24-reference bibliography
├── src/
│   ├── preprocessing/
│   │   └── build_tcga_paad.py                                # PanCanAtlas TCGA-PAAD builder (ABSOLUTE purity, Raphael 2017, curated fallback)
│   ├── analysis/
│   │   ├── depmap_coessentiality_and_sparing.py              # DepMap 24Q2 co-dependency (n = 1,038), BAF co-essentiality, sparing & VHL restoration
│   │   ├── geo_hypoxia_validation.py                         # Paired 1% vs 21% O2 validation across 4 NCBI GEO solid-tumor series
│   │   ├── chipseq_slc7a1_occupancy.py                       # Unpaired solid-tumor ChIP-Atlas HIF1A & SMARCA4 peak overlap at SLC7A1 & VEGFA
│   │   ├── survival_lasso_meta_analysis.py                   # Cohort sample size, pure PDAC, LRT, unified 10/11-cohort meta-analysis
│   │   ├── survival_sensitivity_checks.py                    # Full 10-cohort +MKI67, METABRIC missingness, KIRC/BLCA purity, 22-cohort PH & Buffa-14 benchmark
│   │   ├── pdac_scrna_specificity.py                         # Primary PDAC scRNA-seq (CRA001160) patient-level pseudobulk specificity
│   │   ├── gdsc_combination_synergy.py                       # Unique-cell-line Bliss synergy, Risk Score correlations & lineage-adjusted ANOVA
│   │   ├── dose_sparing_and_gtex_audit.py                    # Harmonized UCSC Xena Toil (TCGA vs. GTEx v6) TPMs, GDSC D1/IC50, GDSC2 1-AUC & AZD5153 Cmax audit
│   │   ├── multiomics_and_gdsc2_anova.py                     # GDSC2 1-AUC ANOVA, partial R^2, Hallmark GSEA, 12-cohort TCPA RPPA, CIN & IMvigor210
│   │   └── verify_manuscript_numbers.py                      # Automated consistency audit cross-checking CSVs against manuscript.tex and README.md
│   └── visualization/
│       └── generate_all_figures.py                           # High-resolution 4-figure generator (300 DPI PDF/PNG)
├── requirements.txt                                          # Python package dependencies
└── README.md                                                 # Repository documentation
```

***

## Data and Code Availability

Primary raw datasets analyzed in this study are publicly accessible from Broad DepMap 24Q2, the TCGA Pan-Cancer Atlas and cBioPortal, METABRIC, CGGA-325 (`cgga.org.cn`, full `CGGA.mRNAseq_325` RSEM and clinical release), Sanger GDSC2/Combinations, TCPA, TISCH2 (`PAAD_CRA001160`), `IMvigor210`, UCSC Xena Toil (`toil.xenahubs.net`, `TcgaTargetGtex_rsem_gene_tpm`, GTEx v6), NCBI GEO (`GSE283995`, `GSE139673`, `GSE145108`, `GSE300765`), and ChIP-Atlas (9 solid-tumor hg38 HIF1A and SMARCA4 BED tracks). All source code, processed summary tables, and figure scripts are archived at [https://github.com/SachsinSinnathurai/smarca4-hif1a-pharmacogenomics](https://github.com/SachsinSinnathurai/smarca4-hif1a-pharmacogenomics).

## Reproduction (`Python >= 3.10`)

Re-running the full pipeline from raw matrices requires placing public source archives from Broad DepMap 24Q2, TCGA PanCancer Atlas, METABRIC, CGGA-325, Sanger GDSC, TCPA, TISCH2 `PAAD_CRA001160`, IMvigor210, NCBI GEO (`GSE283995`, `GSE139673`, `GSE145108`, `GSE300765`), and ChIP-Atlas BED files inside `data/raw/`. On a fresh clone without raw archives, `survival_lasso_meta_analysis.py` executes dynamically from `data/processed/curated_10cohort_clinical_survival.csv` and `generate_all_figures.py` regenerates Figure 2 while preserving Figures 1, 3, and 4.

```bash
pip install -r requirements.txt
python src/analysis/depmap_coessentiality_and_sparing.py
python src/analysis/geo_hypoxia_validation.py
python src/analysis/chipseq_slc7a1_occupancy.py
python src/analysis/survival_lasso_meta_analysis.py
python src/analysis/survival_sensitivity_checks.py
python src/analysis/pdac_scrna_specificity.py
python src/analysis/gdsc_combination_synergy.py
python src/analysis/dose_sparing_and_gtex_audit.py
python src/analysis/multiomics_and_gdsc2_anova.py
python src/visualization/generate_all_figures.py
python src/analysis/verify_manuscript_numbers.py
```

***

## License & Citation

This work is released under the MIT License for academic and research use.

```bibtex
@misc{sinnathurai2026pharmacogenomics,
  title={Lineage-Restricted Transcriptomic Prognosis and Normoxic Pharmacogenomic Uncoupling of the SMARCA4-HIF1A Axis Across Solid Tumors},
  author={Sinnathurai, Sachsin},
  note={Department of Anatomy \& Cell Biology, McGill University},
  year={2026}
}
```
