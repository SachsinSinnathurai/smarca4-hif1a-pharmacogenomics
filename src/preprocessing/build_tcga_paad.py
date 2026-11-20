#!/usr/bin/env python3
"""Build clean, non-synthetic TCGA_PAAD_expression_survival.csv and rebuild curated_10cohort_clinical_survival.csv."""

import gzip
import os
import struct
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")

PAAD_DIR = os.path.join(DATA_RAW, "paad_tcga_pan_can_atlas_2018")
ARAN_XLSX = os.path.join(DATA_RAW, "41467_2015_BFncomms9971_MOESM1236_ESM.xlsx")
ABS_PANCAN_TXT = os.path.join(DATA_RAW, "TCGA_mastercalls.abs_tables_JSedit.fixed.txt")
RAPHAEL_RDS = os.path.join(DATA_RAW, "TCGA_PAAD.rds")
OUT_PAAD_CSV = os.path.join(DATA_RAW, "tcga", "TCGA_PAAD_expression_survival.csv")
OUT_CURATED_CSV = os.path.join(DATA_PROC, "curated_10cohort_clinical_survival.csv")

TARGET_GENES = ["SMARCA4", "HIF1A", "SLC7A1", "SLC1A5", "ABCB1", "MKI67"]


class RDSReader:
    """Minimal pure-Python XDR v2/v3 R .rds parser for extracting small metadata tables."""

    def __init__(self, buf: bytes):
        assert buf[:2] == b"X\n", "Expected XDR binary RDS stream"
        self.buf = buf
        fmt_ver = struct.unpack(">I", buf[2:6])[0]
        self.pos = 14
        if fmt_ver >= 3:
            enc_len = struct.unpack(">I", buf[self.pos:self.pos + 4])[0]
            self.pos += 4 + enc_len
        self.refs = []

    def u32(self) -> int:
        v = struct.unpack(">I", self.buf[self.pos:self.pos + 4])[0]
        self.pos += 4
        return v

    def i32(self) -> int:
        v = struct.unpack(">i", self.buf[self.pos:self.pos + 4])[0]
        self.pos += 4
        return v

    def read_char(self, flags: int):
        levs = (flags >> 12) & 0xFFFF
        n = self.i32()
        if n == -1:
            return None
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b.decode("latin1" if (levs & 4) else "utf-8", errors="replace")

    def read_obj(self, skip_big: bool = True):
        flags = self.u32()
        stype = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))

        if stype in (254, 253, 252, 251):
            return None
        if stype == 255:
            idx = flags >> 8
            if idx == 0:
                idx = self.u32()
            return self.refs[idx - 1]
        if stype == 1:
            pname = self.read_obj(skip_big=False)
            self.refs.append(pname)
            return pname
        if stype == 9:
            return self.read_char(flags)
        if stype in (2, 6, 17):
            _attr = self.read_attrs() if has_attr else None
            tag = self.read_obj(skip_big=False) if has_tag else None
            car = self.read_obj(skip_big=False)
            cdr = self.read_obj(skip_big=False)
            return (tag, car, cdr)
        if stype == 10:
            n = self.u32()
            arr = struct.unpack(f">{n}i", self.buf[self.pos:self.pos + 4 * n])
            self.pos += 4 * n
            vals = [None if v == -2147483648 else bool(v) for v in arr]
            if has_attr:
                self.read_attrs()
            return vals
        if stype == 13:
            n = self.u32()
            arr = struct.unpack(f">{n}i", self.buf[self.pos:self.pos + 4 * n])
            self.pos += 4 * n
            vals = [np.nan if v == -2147483648 else int(v) for v in arr]
            attr = self.read_attrs() if has_attr else {}
            if attr and "levels" in attr:
                lv = attr["levels"]
                return [None if (isinstance(v, float) and np.isnan(v)) else lv[v - 1] for v in vals]
            return vals
        if stype == 14:
            n = self.u32()
            if skip_big and n > 200000:
                self.pos += 8 * n
                if has_attr:
                    self.read_attrs()
                return None
            arr = list(struct.unpack(f">{n}d", self.buf[self.pos:self.pos + 8 * n]))
            self.pos += 8 * n
            if has_attr:
                self.read_attrs()
            return arr
        if stype == 16:
            n = self.u32()
            vals = [self.read_obj(skip_big=False) for _ in range(n)]
            if has_attr:
                self.read_attrs()
            return vals
        if stype == 19:
            n = self.u32()
            vals = [self.read_obj(skip_big=skip_big) for _ in range(n)]
            attr = self.read_attrs() if has_attr else {}
            if attr and "names" in attr:
                return dict(zip(attr["names"], vals))
            return vals
        raise ValueError(f"Unsupported R SEXPTYPE {stype} at byte {self.pos - 4}")

    def read_attrs(self) -> dict:
        d = {}
        cur = self.read_obj(skip_big=False)
        while isinstance(cur, tuple) and len(cur) == 3:
            tag, car, cdr = cur
            if isinstance(tag, str):
                d[tag] = car
            cur = cdr
        return d


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / std


def parse_stage_advanced(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if any(tok in s for tok in ("STAGE IV", "STAGE III")):
        return 1.0
    if any(tok in s for tok in ("STAGE II", "STAGE I")):
        return 0.0
    return np.nan


def parse_event_status(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith("1:"):
        return 1
    if s.startswith("0:"):
        return 0
    return np.nan


def load_raphael2017_metadata(rds_fp: str) -> pd.DataFrame:
    with gzip.open(rds_fp, "rb") as fh:
        raw = fh.read()
    obj = RDSReader(raw).read_obj(skip_big=True)
    si = pd.DataFrame(obj["sampInfo"])
    si["Sample_ID"] = si["Tumor.Sample.ID"].astype(str).str.slice(0, 15)
    si = si.drop_duplicates("Sample_ID").copy()
    keep = {
        "Sample_ID": si["Sample_ID"],
        "ABSOLUTE_Raphael2017": pd.to_numeric(si["ABSOLUTE.Purity"], errors="coerce"),
        "DNA_Meth_Purity_Raphael2017": pd.to_numeric(si["DNA.hypermethylation.mode.purity"], errors="coerce"),
        "Raphael2017_Filter_by": si["Filter_by"].astype(str),
        "Pure_Ductal_Raphael2017": (si["Filter_by"].astype(str) == "whitelist").astype(int),
        "Histological_Type_RHH": si["Histological.type.by.RHH"].astype(str).replace({"None": np.nan}),
    }
    return pd.DataFrame(keep)


def resolve_reg_path(filename: str) -> str:
    raw_fp = os.path.join(DATA_RAW, "clinical_registries", filename)
    return raw_fp if os.path.exists(raw_fp) else os.path.join(DATA_PROC, filename)


def build_tcga_paad() -> pd.DataFrame:
    rsem_fp = os.path.join(PAAD_DIR, "data_mrna_seq_v2_rsem.txt")
    pat_fp = os.path.join(PAAD_DIR, "data_clinical_patient.txt")
    smp_fp = os.path.join(PAAD_DIR, "data_clinical_sample.txt")

    # 1. Extract target gene RSEM strictly for primary -01 tumor samples
    expr_map = {g: {} for g in TARGET_GENES}
    with open(rsem_fp, "r", encoding="utf-8") as fh:
        hdr = fh.readline().strip().split("\t")
        samps = hdr[2:]
        for line in fh:
            parts = line.strip().split("\t")
            gene = parts[0]
            if gene in expr_map:
                for sid, val in zip(samps, parts[2:]):
                    if len(sid) >= 15 and sid[13:15] == "01":
                        expr_map[gene][sid] = float(val)

    df_expr = pd.DataFrame(expr_map)
    df_expr["Sample_ID"] = df_expr.index.astype(str)
    df_expr["Patient_ID"] = df_expr["Sample_ID"].str.slice(0, 12)
    df_expr = df_expr.drop_duplicates("Sample_ID").reset_index(drop=True)

    # 2. Merge clinical sample and patient records
    smp = pd.read_csv(smp_fp, sep="\t", comment="#")
    pat = pd.read_csv(pat_fp, sep="\t", comment="#")
    df = df_expr.merge(smp, left_on=["Sample_ID", "Patient_ID"], right_on=["SAMPLE_ID", "PATIENT_ID"], how="left")
    df = df.merge(pat, left_on="Patient_ID", right_on="PATIENT_ID", how="left", suffixes=("", "_pat"))

    # Filter strictly to primary -01 tumors with non-missing OS_MONTHS > 0 and OS_STATUS
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_Status"] = df["OS_STATUS"].apply(parse_event_status)
    n_pre = len(df)
    df = df[df["OS_MONTHS"].notna() & (df["OS_MONTHS"] > 0) & df["OS_Status"].notna()].copy()
    df["OS_Status"] = df["OS_Status"].astype(int)
    df["OS_Days"] = df["OS_MONTHS"] * 30.4375

    df["Sample_Barcode"] = df["Patient_ID"]
    df["Sample_ID_01A"] = df["Sample_ID"] + "A"
    df["Cohort"] = "TCGA-PAAD"
    df["Gender"] = df["SEX"].astype(str)
    df["Age"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["Vital_Status"] = np.where(df["OS_Status"] == 1, "Deceased", "Living")
    df["Clinical_Stage"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"]
    df["Stage_Advanced"] = df["Clinical_Stage"].apply(parse_stage_advanced)
    df["PFS_Days"] = pd.to_numeric(df["PFS_MONTHS"], errors="coerce") * 30.4375
    df["PFS_Status"] = df["PFS_STATUS"].apply(parse_event_status)

    # 3. Compute log2(RSEM + 1) and within-cohort Z-scores
    for g in TARGET_GENES:
        df[f"{g}_log2RSEM"] = np.log2(pd.to_numeric(df[g], errors="coerce") + 1.0)
        df[f"{g}_z"] = zscore(df[f"{g}_log2RSEM"])

    # 4. Merge Aran et al. (2015) MOESM1236 on -01A (noting PAAD coverage)
    aran = pd.read_excel(ARAN_XLSX, sheet_name="Supp Data 1", header=3)
    aran["Sample_ID_01A"] = aran["Sample ID"].astype(str).str.slice(0, 16)
    aran = aran.drop_duplicates("Sample_ID_01A")
    aran_sub = pd.DataFrame({
        "Sample_ID_01A": aran["Sample_ID_01A"],
        "ESTIMATE_Aran2015": pd.to_numeric(aran["ESTIMATE"], errors="coerce"),
        "ABSOLUTE_Aran2015": pd.to_numeric(aran["ABSOLUTE"], errors="coerce"),
        "CPE_Aran2015": pd.to_numeric(aran["CPE"], errors="coerce"),
    })
    df = df.merge(aran_sub, on="Sample_ID_01A", how="left")
    n_aran_match = int(df["CPE_Aran2015"].notna().sum())

    # 5. Merge TCGA PanCanAtlas ABSOLUTE purity and Raphael et al. (2017) whitelist + ABSOLUTE purity
    abs_pan = pd.read_csv(ABS_PANCAN_TXT, sep="\t")
    abs_pan["Sample_ID"] = abs_pan["array"].astype(str).str.slice(0, 15)
    abs_pan["ABSOLUTE_PanCanAtlas"] = pd.to_numeric(abs_pan["purity"], errors="coerce")
    abs_pan = abs_pan[["Sample_ID", "ABSOLUTE_PanCanAtlas"]].dropna().drop_duplicates("Sample_ID")
    df = df.merge(abs_pan, on="Sample_ID", how="left")

    raph = load_raphael2017_metadata(RAPHAEL_RDS)
    df = df.merge(raph, on="Sample_ID", how="left")
    df["Pure_Ductal_Raphael2017"] = df["Pure_Ductal_Raphael2017"].fillna(0).astype(int)
    df["Histological_Subtype"] = df["TUMOR_TYPE"].astype(str)
    df["Pure_Ductal_cBioPortal"] = (df["Histological_Subtype"] == "Pancreas Adenocarcinoma, Ductal Type").astype(int)
    df["Non_PNET"] = (df["ICD_O_3_HISTOLOGY"].astype(str) != "8246/3").astype(int)
    df["Tumor_Purity"] = df["ABSOLUTE_Raphael2017"].combine_first(df["ABSOLUTE_PanCanAtlas"])

    # Strict validation checks
    assert 170 <= len(df) <= 178, f"Expected 170..178 PAAD rows, got {len(df)}"
    assert 85 <= int(df["OS_Status"].sum()) <= 96, f"Expected 85..96 PAAD deaths, got {int(df['OS_Status'].sum())}"
    assert int(df["Stage_Advanced"].isna().sum()) >= 1, "Expected missing stages preserved as NaN, not imputed"
    med_s34 = float(df.loc[df["Stage_Advanced"] == 1, "OS_Days"].median())
    med_s12 = float(df.loc[df["Stage_Advanced"] == 0, "OS_Days"].median())
    assert med_s34 < med_s12, f"Expected Stage III/IV median OS ({med_s34:.1f}d) < Stage I/II ({med_s12:.1f}d)"
    med_pur = float(df["Tumor_Purity"].median())
    assert 0.25 <= med_pur <= 0.45, f"Expected median Tumor_Purity in [0.25, 0.45], got {med_pur:.4f}"
    r_hs, p_hs = stats.pearsonr(df["HIF1A_z"], df["SLC7A1_z"])
    assert r_hs > 0.05, f"Expected HIF1A_z vs SLC7A1_z > 0.05, got {r_hs:.4f}"
    assert int(df["Pure_Ductal_Raphael2017"].sum()) >= 140, f"Expected >= 140 Raphael whitelist cases, got {int(df['Pure_Ductal_Raphael2017'].sum())}"

    out_cols = [
        "Sample_Barcode", "Sample_ID", "Sample_ID_01A", "Cohort", "Gender", "Age", "Vital_Status",
        "OS_Days", "OS_Status", "Clinical_Stage", "Stage_Advanced", "ICD_O_3_HISTOLOGY",
        "Histological_Subtype", "Histological_Type_RHH", "Raphael2017_Filter_by",
        "Pure_Ductal_Raphael2017", "Pure_Ductal_cBioPortal", "Non_PNET",
        "Tumor_Purity", "ABSOLUTE_Raphael2017", "ABSOLUTE_PanCanAtlas", "DNA_Meth_Purity_Raphael2017",
        "ESTIMATE_Aran2015", "ABSOLUTE_Aran2015", "CPE_Aran2015",
        "PFS_Days", "PFS_Status",
        "SMARCA4_log2RSEM", "HIF1A_log2RSEM", "SLC7A1_log2RSEM", "SLC1A5_log2RSEM", "ABCB1_log2RSEM", "MKI67_log2RSEM",
        "SMARCA4_z", "HIF1A_z", "SLC7A1_z", "SLC1A5_z", "ABCB1_z", "MKI67_z",
    ]
    df_out = df[out_cols].copy()
    os.makedirs(os.path.dirname(OUT_PAAD_CSV), exist_ok=True)
    df_out.to_csv(OUT_PAAD_CSV, index=False)

    print(f"Built clean TCGA-PAAD dataset: {len(df_out)} primary -01 tumors ({n_pre - len(df_out)} with OS_MONTHS <= 0 excluded; {int(df_out['OS_Status'].sum())} deaths) -> {OUT_PAAD_CSV}")
    print(f"  Stage III/IV median OS = {med_s34:.1f}d vs Stage I/II = {med_s12:.1f}d (NaN stage = {int(df_out['Stage_Advanced'].isna().sum())})")
    print(f"  Purity: Aran 2015 MOESM1236 PAAD matches = {n_aran_match}/{len(df_out)}; combined ABSOLUTE non-null = {int(df_out['Tumor_Purity'].notna().sum())}/{len(df_out)} (median = {med_pur:.4f})")
    print(f"  Raphael et al. (2017) Pure Ductal whitelist = {int(df_out['Pure_Ductal_Raphael2017'].sum())}/{len(df_out)}; HIF1A_z vs SLC7A1_z r = {r_hs:+.4f} (p = {p_hs:.3e})")
    return df_out


def rebuild_curated_survival_table(df_paad: pd.DataFrame):
    he_reg = pd.read_csv(resolve_reg_path("High_Event_10Cohort_Treatment_Registry.csv"))
    pc_reg = pd.read_csv(resolve_reg_path("Full_TCGA_PanCan_Patient_Treatment_Registry.csv"))
    bi_reg = pd.read_csv(resolve_reg_path("Batch_Indication_Expansion_Patient_Registry.csv"))
    pcawg_reg = pd.read_csv(resolve_reg_path("PCAWG_1210_Personalized_Treatment_Registry.csv"))
    df_cgga = pd.read_csv(os.path.join(DATA_RAW, "external", "CGGA_mRNAseq_325_expression_survival.csv"))

    rows = []
    # 1. Clean real TCGA-PAAD (preserve Stage=NaN and omit Tumor_Purity if NaN)
    for _, r in df_paad.iterrows():
        extras = [
            f"Pure_Ductal_Raphael2017={int(r['Pure_Ductal_Raphael2017'])}",
            f"Pure_Ductal_cBioPortal={int(r['Pure_Ductal_cBioPortal'])}",
            f"Non_PNET={int(r['Non_PNET'])}",
            f"SLC7A1_z={float(r['SLC7A1_z']):.6f}",
            f"SLC1A5_z={float(r['SLC1A5_z']):.6f}",
            f"ABCB1_z={float(r['ABCB1_z']):.6f}",
            f"MKI67_log2RSEM={float(r['MKI67_log2RSEM']):.6f}",
        ]
        if pd.notna(r["Tumor_Purity"]):
            extras.insert(0, f"Tumor_Purity={float(r['Tumor_Purity']):.6f}")
        rows.append({
            "Cohort": "TCGA-PAAD",
            "Sample_ID": str(r["Sample_Barcode"]),
            "OS_Days": float(r["OS_Days"]),
            "Event": int(r["OS_Status"]),
            "Age": float(r["Age"]),
            "Stage": float(r["Stage_Advanced"]) if pd.notna(r["Stage_Advanced"]) else np.nan,
            "Extra_Covariates": ";".join(extras),
            "SMARCA4_z": float(r["SMARCA4_z"]),
            "HIF1A_z": float(r["HIF1A_z"]),
        })

    # 2. Eight real TCGA / METABRIC external cohorts
    cohort_specs = [
        ("TCGA-GBM", he_reg[he_reg["Cohort"] == "GBM_TCGA"].copy(), ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"], None, False),
        ("TCGA-BRCA", pc_reg[pc_reg["Cohort"] == "BRCA"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", False),
        ("TCGA-BLCA", he_reg[he_reg["Cohort"] == "BLCA_TCGA"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", False),
        ("BRCA-METABRIC", he_reg[he_reg["Cohort"] == "BRCA_METABRIC"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", False),
        ("TCGA-SKCM", he_reg[he_reg["Cohort"] == "SKCM_TCGA"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", True),
        ("TCGA-KIRC", he_reg[he_reg["Cohort"] == "KIRC_TCGA"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", False),
        ("TCGA-SARC", bi_reg[bi_reg["Cohort"] == "SARC"].copy(), ["OS_Days", "Event", "Age", "SMARCA4_zScore", "HIF1A_zScore"], None, False),
        ("TCGA-LIHC", bi_reg[bi_reg["Cohort"] == "LIHC"].copy(), ["OS_Days", "Event", "Age", "Stage_Numeric", "SMARCA4_zScore", "HIF1A_zScore"], "Stage_Numeric", False),
    ]
    for cname, df_c, req_cols, stage_col, is_skcm in cohort_specs:
        if is_skcm:
            df_c["Is_Metastatic"] = df_c["Sample_ID"].astype(str).str.slice(13, 15).eq("06").astype(int)
        sub = df_c[df_c["OS_Days"] > 0].dropna(subset=req_cols).copy()
        sub["Risk_ZScore"] = zscore(sub["SMARCA4_zScore"] + sub["HIF1A_zScore"])
        for _, r in sub.iterrows():
            ex = f"Is_Metastatic={int(r['Is_Metastatic'])};Risk_ZScore={float(r['Risk_ZScore']):.6f}" if is_skcm else f"Risk_ZScore={float(r['Risk_ZScore']):.6f}"
            rows.append({
                "Cohort": cname,
                "Sample_ID": str(r["Sample_ID"]),
                "OS_Days": float(r["OS_Days"]),
                "Event": int(r["Event"]),
                "Age": float(r["Age"]),
                "Stage": float(r[stage_col]) if stage_col else np.nan,
                "Extra_Covariates": ex,
                "SMARCA4_z": float(r["SMARCA4_zScore"]),
                "HIF1A_z": float(r["HIF1A_zScore"]),
            })

    # 3. Real CGGA-325 external glioma cohort (unweighted a priori Z(SMARCA4_z + HIF1A_z))
    df_cgga["IDH_Mutant"] = (df_cgga["IDH_mutation_status"] == "Mutant").astype(int)
    df_cgga["MGMT_Methylated"] = (df_cgga["MGMTp_methylation_status"] == "methylated").astype(int)
    sub_cgga = df_cgga[df_cgga["OS_Days"] > 0].dropna(
        subset=["OS_Days", "OS_Status", "Age", "Stage_Advanced", "IDH_Mutant", "MGMT_Methylated", "SMARCA4", "HIF1A"]
    ).copy()
    sub_cgga["SMARCA4_z"] = zscore(sub_cgga["SMARCA4"])
    sub_cgga["HIF1A_z"] = zscore(sub_cgga["HIF1A"])
    sub_cgga["Risk_ZScore"] = zscore(sub_cgga["SMARCA4_z"] + sub_cgga["HIF1A_z"])
    for idx, r in sub_cgga.iterrows():
        rows.append({
            "Cohort": "CGGA-325",
            "Sample_ID": str(r.get("CGGA_ID", r.get("Sample_ID", f"CGGA_{idx}"))),
            "OS_Days": float(r["OS_Days"]),
            "Event": int(r["OS_Status"]),
            "Age": float(r["Age"]),
            "Stage": float(r["Stage_Advanced"]),
            "Extra_Covariates": f"IDH_Mutant={int(r['IDH_Mutant'])};MGMT_Methylated={int(r['MGMT_Methylated'])};Risk_ZScore={float(r['Risk_ZScore']):.6f}",
            "SMARCA4_z": float(r["SMARCA4_z"]),
            "HIF1A_z": float(r["HIF1A_z"]),
        })

    # 4. External ICGC/PCAWG-PAAD (N = 71) validation cohort (truthfully labeled)
    df_icgc = pcawg_reg[pcawg_reg["Cancer_Type"] == "Pancreatic Cancer"].copy()
    df_icgc = df_icgc[df_icgc["OS_Days"] > 0].dropna(subset=["OS_Days", "Age", "SMARCA4_zScore", "HIF1A_zScore"]).copy()
    df_icgc["Event"] = (df_icgc["Vital_Status"] == "Deceased").astype(int)
    df_icgc["Is_Male"] = (df_icgc["Sex"] == "Male").astype(int)
    df_icgc["Risk_ZScore"] = zscore(df_icgc["SMARCA4_zScore"] + df_icgc["HIF1A_zScore"])
    for _, r in df_icgc.iterrows():
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

    df_cur = pd.DataFrame(rows)
    df_cur.to_csv(OUT_CURATED_CSV, index=False)
    n_primary = int((df_cur["Cohort"] != "ICGC/PCAWG-PAAD").sum())
    n_icgc = int((df_cur["Cohort"] == "ICGC/PCAWG-PAAD").sum())
    print(f"Rebuilt open-access survival table ({n_primary:,} primary 10-cohort cases + {n_icgc} ICGC/PCAWG-PAAD) -> {OUT_CURATED_CSV}")


def main():
    df_paad = build_tcga_paad()
    rebuild_curated_survival_table(df_paad)


if __name__ == "__main__":
    main()
