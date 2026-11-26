"""
Step 1: Initialize repository subdirectories and record SHA-256 hashes
of all read-only input files (raw and processed) before any downstream script runs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os


INPUT_FILES = [
    "data/processed/tables/Table_1_Clinical_Cohorts.csv",
    "data/processed/survival_full_tcga_and_mki67_sensitivity.csv",
    "data/processed/brca_subtype_receptor_sensitivity.csv",
    "data/processed/gdsc2_lineage_controlled_anova.csv",
    "data/processed/geo_hypoxia_perturbation.csv",
    "data/processed/pdac_scrna_celltype_specificity.csv",
    "data/processed/hallmark_gsea_transcriptome_wide_enrichment.csv",
    "data/processed/tcga_rppa_proteomic_validation.csv",
    "data/processed/genomic_scars_hrd_aneuploidy_orthogonality.csv",
    "manuscript/Proliferation_Confounding_and_Bulk_Transcriptomic_Limitations_of_the_SMARCA4_HIF1A_Axis_Across_Solid_Tumors.tex",
]


def sha256_file(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = repo_root

    for sub in ["audit", "scripts", "results", "manuscript", "figures"]:
        os.makedirs(os.path.join(repo_root, sub), exist_ok=True)

    records: dict[str, dict] = {}
    missing: list[str] = []
    for rel in INPUT_FILES:
        abs_fp = os.path.join(project_root, rel)
        if not os.path.exists(abs_fp):
            missing.append(rel)
            records[rel] = {"exists": False, "sha256": None, "size_bytes": 0}
        else:
            digest, sz = sha256_file(abs_fp)
            records[rel] = {
                "exists": True,
                "sha256": digest,
                "size_bytes": sz,
            }

    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root": project_root,
        "repo_root": repo_root,
        "missing_files": missing,
        "files": records,
    }
    out_fp = os.path.join(repo_root, "audit", "input_hashes.json")
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Recorded {len(records) - len(missing)}/{len(records)} input hashes -> {out_fp}")
    if missing:
        print(f"MISSING INPUTS: {missing}")


if __name__ == "__main__":
    main()
