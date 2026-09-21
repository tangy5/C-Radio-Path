#!/usr/bin/env python3
"""
Select 50 high-quality cases per tissue site from source C for pathologist annotation.

Uses metadata, QC progress data, and bad file records to score and select cases.
Outputs: source_c_annotation_selection.json + source_c_annotation_selection.csv
"""

import json
import csv
import os
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
SOURCE_C_DIR = HERE.parent / "source_c"
LOGS_DIR = HERE / "logs"
METADATA_PATH = SOURCE_C_DIR / "source C-metadata" / "metadata.json"
OUTPUT_DIR = HERE / "annotation_selection"
OUTPUT_DIR.mkdir(exist_ok=True)
SELECTION_OUTPUT = OUTPUT_DIR / "source_c_annotation_selection.json"
CSV_OUTPUT = OUTPUT_DIR / "source_c_annotation_selection.csv"

RANDOM_SEED = 42
TARGET_PER_SITE = 50

# Datasets confirmed on S3 (from s3_transfer_runbook.md)
S3_DATASETS = {
    "source C-breast", "source C-colorectal-b1", "source C-colorectal-b2",
    "source C-gastrointestinal", "source C-hematologic",
    "source C-skin-b1", "source C-skin-b2", "source C-thorax",
}

# ── Tissue site definitions ──────────────────────────────────────
# Direct dataset -> tissue site mapping
DATASET_TO_SITE = {
    "source C-skin-b1": "Skin",
    "source C-skin-b2": "Skin",
    "source C-breast": "Breast",
    "source C-colorectal-b1": "Colorectal",
    "source C-colorectal-b2": "Colorectal",
    "source C-thorax": "Thorax",
    "source C-gastrointestinal": "Gastrointestinal",
    "source C-hematologic": "Hematologic",
}

# Specialization -> tissue site
SPEC_TO_SITE = {
    "Skin": "Skin",
    "Breast": "Breast",
    "Gastrointestinal": "Gastrointestinal",
    "Thorax": "Thorax",
    "Hematologic": "Hematologic",
    "Genitourinary": "Genitourinary",
    "Gynecologic": "Gynecologic",
    "Head and Neck": "Head & Neck",
    "Soft tissue": "Other",
    "Endocrine": "Other",
    "Central nervous system": "Other",
    "Cancer of unknown primary origin": "Other",
    "Unknown": "Other",
}

# Diagnosis keywords for mixed classification (Tier 2)
DIAG_KEYWORDS = {
    "Skin": [
        "skin", "melanoma", "nevus", "nevi", "basal cell carcinoma",
        "squamous cell carcinoma of skin", "seborrheic keratosis",
        "actinic keratosis", "dermatofibroma", "cutaneous",
    ],
    "Breast": [
        "breast", "mammary", "ductal carcinoma", "lobular carcinoma",
        "phyllodes", "fibroadenoma",
    ],
    "Colorectal": [
        "colon", "colorectal", "rectum", "rectal", "sigmoid",
        "colonic", "cecum", "cecal", "colorectal",
    ],
    "Thorax": [
        "lung", "thorax", "pleura", "mediastinum", "thymus", "thymic",
        "bronchus", "bronchial", "pulmonary", "mesothelioma",
    ],
    "Gastrointestinal": [
        "stomach", "gastric", "esophagus", "esophageal", "duodenum",
        "duodenal", "pancreas", "pancreatic", "liver", "hepatic",
        "gallbladder", "bile duct", "cholangiocarcinoma",
        "small intestine", "ileum", "jejunum",
    ],
    "Hematologic": [
        "lymph node", "lymphoma", "leukemia", "myeloma",
        "spleen", "bone marrow", "hodgkin",
    ],
    "Genitourinary": [
        "prostate", "prostatic", "kidney", "renal",
        "bladder", "urothelial", "testis", "testicular",
        "penis", "urinary",
    ],
    "Gynecologic": [
        "ovary", "ovarian", "uterus", "uterine",
        "endometrium", "endometrial", "cervix", "cervical",
        "fallopian", "vulva", "vagina", "vulvar", "vaginal",
    ],
    "Head & Neck": [
        "thyroid", "larynx", "laryngeal", "oral", "tongue",
        "salivary", "parotid", "pharynx", "pharyngeal",
        "nasopharynx", "oropharynx", "tonsil",
    ],
}

# ICD-10 -> tissue site (Tier 3, using chapter prefixes)
ICD10_TO_SITE = {
    "C43": "Skin", "C44": "Skin",
    "C50": "Breast",
    "C18": "Colorectal", "C19": "Colorectal", "C20": "Colorectal",
    "C34": "Thorax",
    "C15": "Gastrointestinal", "C16": "Gastrointestinal",
    "C22": "Gastrointestinal", "C23": "Gastrointestinal",
    "C24": "Gastrointestinal", "C25": "Gastrointestinal",
    "C81": "Hematologic", "C82": "Hematologic", "C83": "Hematologic",
    "C84": "Hematologic", "C85": "Hematologic", "C88": "Hematologic",
    "C90": "Hematologic", "C91": "Hematologic", "C92": "Hematologic",
    "C93": "Hematologic", "C94": "Hematologic", "C95": "Hematologic",
    "C96": "Hematologic",
    "C61": "Genitourinary", "C62": "Genitourinary", "C64": "Genitourinary",
    "C65": "Genitourinary", "C66": "Genitourinary", "C67": "Genitourinary",
    "C68": "Genitourinary",
    "C51": "Gynecologic", "C52": "Gynecologic", "C53": "Gynecologic",
    "C54": "Gynecologic", "C55": "Gynecologic", "C56": "Gynecologic",
    "C57": "Gynecologic", "C58": "Gynecologic",
    "C00": "Head & Neck", "C01": "Head & Neck", "C02": "Head & Neck",
    "C03": "Head & Neck", "C04": "Head & Neck", "C05": "Head & Neck",
    "C06": "Head & Neck", "C07": "Head & Neck", "C08": "Head & Neck",
    "C09": "Head & Neck", "C10": "Head & Neck", "C11": "Head & Neck",
    "C12": "Head & Neck", "C13": "Head & Neck", "C14": "Head & Neck",
    "C30": "Head & Neck", "C31": "Head & Neck", "C32": "Head & Neck",
    "C73": "Head & Neck",
}

ALL_SITES = [
    "Skin", "Breast", "Colorectal", "Thorax",
    "Gastrointestinal", "Hematologic", "Genitourinary",
    "Gynecologic", "Head & Neck",
]


# ══════════════════════════════════════════════════════════════════
# Phase 1: Data Loading
# ══════════════════════════════════════════════════════════════════

def load_metadata():
    """Load metadata.json, index by case_id -> dataset mapping."""
    print("Loading metadata.json...")
    with open(METADATA_PATH) as f:
        data = json.load(f)
    print(f"  {len(data):,} entries loaded")

    # Build case_id -> metadata mapping
    cases = {}
    for entry in data:
        cm = entry.get("case_mapping", "")
        # Extract dataset and case_id from case_mapping
        # Format: "source_c/source C-skin-b1/case_0927" or "source_c/source C-skin-b1/case_0927/slide_..."
        parts = cm.split("/")
        if len(parts) >= 3:
            dataset = parts[1]  # e.g., "source C-skin-b1"
            case_part = parts[2]  # e.g., "case_0927"
            case_match = re.match(r"case_(\d+)", case_part)
            if case_match:
                case_id = case_match.group(1)
                # Use dataset + case_id as unique key (same case_id can appear in multiple datasets)
                key = f"{dataset}:{case_id}"
                if key not in cases:
                    cases[key] = {
                        "case_id": case_id,
                        "source_dataset": dataset,
                        "entry": entry,
                    }
    print(f"  {len(cases):,} unique case+dataset combinations")
    return cases


def load_qc_progress():
    """Load qc_progress JSONs -> {dataset: {slide_dirs, total_checked, total_bad}}."""
    print("\nLoading QC progress data...")
    qc_data = {}
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.startswith("qc_progress_") or not fname.endswith(".json"):
            continue
        path = LOGS_DIR / fname
        with open(path) as f:
            d = json.load(f)
        dataset = d["dataset"]
        completed = d.get("completed_dirs", [])
        qc_data[dataset] = {
            "slide_dirs": completed,
            "total_checked": d.get("total_files_checked", 0),
            "total_bad": d.get("total_bad_files", 0),
        }
        print(f"  {dataset}: {len(completed)} slide dirs, {d['total_files_checked']:,} checked, {d['total_bad_files']:,} bad")

    return qc_data


def load_bad_files():
    """Parse bad_files CSVs -> {case_key: bad_count}."""
    print("\nLoading bad files data...")
    case_bad = defaultdict(int)
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.startswith("bad_files_") or not fname.endswith(".csv"):
            continue
        path = LOGS_DIR / fname
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_field = row.get("case", "")
                dataset = row.get("dataset", "")
                # case_field format: "case_0041_slide_H&E_0"
                case_match = re.match(r"case_(\d+)_", case_field)
                if case_match and dataset:
                    case_id = case_match.group(1)
                    key = f"{dataset}:{case_id}"
                    case_bad[key] += 1
    print(f"  {len(case_bad):,} cases with bad patches ({sum(case_bad.values()):,} total bad patches)")
    return dict(case_bad)


def build_slide_index(qc_data):
    """Build per-case slide index from qc_progress completed_dirs.

    Returns: {case_key: {"slide_count": N, "he_slides": N, "stain_types": [...]}}
    """
    print("\nBuilding slide index...")
    index = defaultdict(lambda: {"slide_count": 0, "he_slides": 0, "stain_types": set()})

    for dataset, qc in qc_data.items():
        for slide_dir in qc["slide_dirs"]:
            # slide_dir format: "case_0000_slide_H&E_0" or "case_0531_slide_SMA(1A4)_4"
            case_match = re.match(r"case_(\d+)_slide_(.+)", slide_dir)
            if not case_match:
                continue
            case_id = case_match.group(1)
            stain_part = case_match.group(2)
            # Remove trailing _<number>
            stain_name = re.sub(r"_\d+$", "", stain_part)

            key = f"{dataset}:{case_id}"
            index[key]["slide_count"] += 1
            index[key]["stain_types"].add(stain_name)

            # Check if H&E
            if re.search(r"[hH][&eE]|[hH]\s*&?\s*[eE]", stain_name):
                index[key]["he_slides"] += 1

    # Convert sets to sorted lists
    for key in index:
        index[key]["stain_types"] = sorted(index[key]["stain_types"])

    total_cases = len(index)
    total_slides = sum(v["slide_count"] for v in index.values())
    print(f"  {total_cases:,} cases with slide data, {total_slides:,} total slides")
    return dict(index)


# ══════════════════════════════════════════════════════════════════
# Phase 2: Tissue Site Classification
# ══════════════════════════════════════════════════════════════════

def classify_mixed_case(entry):
    """Classify a source C-mixed case into a tissue site. Returns (site, method)."""
    # Tier 1: specialization field
    spec = (entry.get("specialization") or "").strip()
    if spec in SPEC_TO_SITE and SPEC_TO_SITE[spec] != "Other":
        return SPEC_TO_SITE[spec], "specialization"

    # Tier 2: diagnosis + conclusion keywords
    diag = (entry.get("diagnosis") or "").lower()
    concl = (entry.get("conclusion") or "").lower()
    combined = diag + " " + concl

    for site, keywords in DIAG_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return site, "diagnosis_keyword"

    # Tier 3: ICD-10
    icd = (entry.get("icd10") or "").strip().upper()
    # Try prefix match (first 3 chars)
    icd_prefix = icd[:3] if len(icd) >= 3 else icd
    if icd_prefix in ICD10_TO_SITE:
        return ICD10_TO_SITE[icd_prefix], "icd10"

    # Fallback
    if spec in SPEC_TO_SITE:
        return SPEC_TO_SITE[spec], "specialization_fallback"

    return "Other", "unclassified"


def classify_all_cases(metadata_cases):
    """Classify all cases into tissue sites. Returns {site: [case_objects]}."""
    print("\nClassifying cases into tissue sites...")
    site_cases = defaultdict(list)
    method_counts = Counter()

    for key, case_data in metadata_cases.items():
        dataset = case_data["source_dataset"]
        entry = case_data["entry"]

        if dataset in DATASET_TO_SITE:
            site = DATASET_TO_SITE[dataset]
            method = "case_mapping"
        elif dataset == "source C-mixed":
            site, method = classify_mixed_case(entry)
        else:
            site = "Other"
            method = "unknown"

        method_counts[method] += 1

        site_cases[site].append({
            "key": key,
            "case_id": case_data["case_id"],
            "source_dataset": dataset,
            "diagnosis": entry.get("diagnosis", ""),
            "conclusion": entry.get("conclusion", ""),
            "diff_diagnosis": entry.get("diff_diagnostic", ""),
            "grossing": entry.get("grossing", ""),
            "micro_protocol": entry.get("micro_protocol", ""),
            "additional_info": entry.get("additional_info", ""),
            "icd10": entry.get("icd10", ""),
            "age": entry.get("age"),
            "gender": entry.get("gender", ""),
            "specialization": entry.get("specialization", ""),
            "classification_method": method,
            "classification_site": site,
        })

    # Print summary
    print("\n  Classification method distribution:")
    for method, count in method_counts.most_common():
        print(f"    {method}: {count:,}")

    print("\n  Cases per tissue site:")
    for site in ALL_SITES:
        if site in site_cases:
            print(f"    {site:<25} {len(site_cases[site]):>8,}")
    other_count = len(site_cases.get("Other", []))
    if other_count:
        print(f"    {'Other (excluded)':<25} {other_count:>8,}")

    return dict(site_cases)


# ══════════════════════════════════════════════════════════════════
# Phase 3: Quality Scoring
# ══════════════════════════════════════════════════════════════════

def score_metadata_completeness(entry):
    """Score metadata completeness (0-25)."""
    score = 0
    if entry.get("diagnosis", "").strip():
        score += 6
    if entry.get("conclusion", "").strip():
        score += 6
    if entry.get("grossing", "").strip():
        score += 5
    if entry.get("micro_protocol", "").strip():
        score += 5
    if entry.get("icd10", "").strip():
        score += 3
    return score


def score_diagnosis_richness(entry):
    """Score diagnosis detail richness (0-20)."""
    score = 0
    if entry.get("diff_diagnostic", "").strip():
        score += 5
    if len(entry.get("diagnosis", "")) > 100:
        score += 5
    if len(entry.get("conclusion", "")) > 200:
        score += 5
    if entry.get("additional_info", "").strip():
        score += 5
    return score


def score_slides_and_stains(slide_info, all_patch_yields):
    """Score slides & stains quality (0-30) plus slide data availability bonus (0-25).
    - H&E stain preference: 10 pts max
    - Slide count: 10 pts max
    - Patch yield percentile within site: 10 pts max
    - BONUS: having any slide data: +15 pts (ensures cases with actual WSI processing are preferred)
    - BONUS: having H&E slides: +10 additional pts
    """
    score = 0
    he_count = slide_info.get("he_slides", 0)
    slide_count = slide_info.get("slide_count", 0)
    stain_types = slide_info.get("stain_types", [])

    # H&E preference
    if he_count > 0:
        score += 10
    elif any("he" in s.lower() or "h&e" in s.lower() for s in stain_types):
        score += 8
    elif slide_count > 0:
        score += 3

    # Slide count (capped at 10+)
    score += min(10, slide_count)

    # Bonus for having any slide data
    if slide_count > 0:
        score += 15  # substantial bonus for WSI-processed cases
    if he_count > 0:
        score += 10  # extra bonus for H&E availability

    return score


def score_bad_ratio(case_key, bad_files, slide_info):
    """Score based on bad patch ratio (0-25). Uses estimated total patches from QC data."""
    bad_count = bad_files.get(case_key, 0)

    if slide_info["slide_count"] == 0:
        return 0

    # Estimate total patches: we don't have per-case patch counts,
    # but we have per-case slide counts from qc_progress.
    # For scoring, use bad_count relative to slide_count as proxy.
    # More than 50 bad patches per slide is concerning.
    bad_per_slide = bad_count / slide_info["slide_count"] if slide_info["slide_count"] > 0 else float("inf")

    if bad_count == 0:
        return 25
    elif bad_per_slide < 1:
        return 20
    elif bad_per_slide < 5:
        return 15
    elif bad_per_slide < 10:
        return 10
    else:
        return 0


def score_all_cases(site_cases, slide_index, bad_files):
    """Compute quality scores for all cases."""
    print("\nScoring cases...")

    for site, cases in site_cases.items():
        for case in cases:
            key = case["key"]
            entry = case["entry"] if "entry" in case else None

            # Metadata scores (from the metadata dict we need to resolve)
            # For now use the stored fields
            meta_score = score_metadata_completeness(case)
            rich_score = score_diagnosis_richness(case)

            # Slide/image scores
            slide_info = slide_index.get(key, {"slide_count": 0, "he_slides": 0, "stain_types": []})
            slide_score = score_slides_and_stains(slide_info, None)
            bad_score = score_bad_ratio(key, bad_files, slide_info)

            total = meta_score + slide_score + bad_score + rich_score

            case["quality_score"] = total
            case["meta_score"] = meta_score
            case["slide_score"] = slide_score
            case["bad_score"] = bad_score
            case["rich_score"] = rich_score
            case["slide_count"] = slide_info["slide_count"]
            case["he_slides"] = slide_info["he_slides"]
            case["stain_types"] = slide_info["stain_types"]
            case["bad_patches"] = bad_files.get(key, 0)

            if total >= 70:
                case["tier"] = "A"
            elif total >= 40:
                case["tier"] = "B"
            else:
                case["tier"] = "C"

    # Print score distribution
    all_scores = [c["quality_score"] for cases in site_cases.values() for c in cases]
    print(f"  Score range: {min(all_scores)}-{max(all_scores)}, mean: {sum(all_scores)/len(all_scores):.1f}")
    tiers = Counter(c["tier"] for cases in site_cases.values() for c in cases)
    print(f"  Tiers: A={tiers.get('A',0):,}, B={tiers.get('B',0):,}, C={tiers.get('C',0):,}")


# ══════════════════════════════════════════════════════════════════
# Phase 4: Selection
# ══════════════════════════════════════════════════════════════════

def extract_diagnosis_keyword(diagnosis):
    """Extract primary diagnosis keyword for stratification."""
    if not diagnosis:
        return "unknown"
    diag = diagnosis.lower().strip().rstrip(".")

    # Try common patterns
    for pattern, label in [
        (r"melanoma", "melanoma"),
        (r"nevus", "nevus"),
        (r"basal cell carcinoma|basalioma", "basal_cell_carcinoma"),
        (r"squamous cell carcinoma", "squamous_cell_carcinoma"),
        (r"seborrheic keratosis|keratoma", "seborrheic_keratosis"),
        (r"adenocarcinoma", "adenocarcinoma"),
        (r"invasive ductal carcinoma|idc", "breast_carcinoma"),
        (r"fibroadenoma", "fibroadenoma"),
        (r"polyp", "polyp"),
        (r"lymphoma", "lymphoma"),
        (r"leukemia", "leukemia"),
        (r"carcinoma", "carcinoma"),
        (r"sarcoma", "sarcoma"),
        (r"tumor|neoplasm|lesion|mass", "tumor"),
    ]:
        if re.search(pattern, diag):
            return label
    return "other"


def select_for_site(cases, target, random_seed):
    """Select target cases from a site's pool using quality-tier stratified sampling."""
    rng = random.Random(random_seed)

    if len(cases) <= target:
        # Select all
        for c in cases:
            c["selection_tier"] = c.get("tier", "C")
        return cases, "all_cases"

    # Split by tier
    tier_a = [c for c in cases if c.get("tier") == "A"]
    tier_b = [c for c in cases if c.get("tier") == "B"]
    tier_c = [c for c in cases if c.get("tier") == "C"]

    selected = []

    # Allocate slots
    if len(tier_a) >= target:
        # All from tier A
        pool = tier_a
    elif len(tier_a) >= target * 0.7:
        # Mostly tier A + some tier B
        pool = tier_a + tier_b
    else:
        # Take all tier A + tier B, fill from tier C
        pool = tier_a + tier_b + tier_c

    # Sort by quality score descending
    pool.sort(key=lambda c: c["quality_score"], reverse=True)

    # Stratify by diagnosis keyword to ensure diversity
    # Build groups
    groups = defaultdict(list)
    for c in pool:
        kw = extract_diagnosis_keyword(c.get("diagnosis", ""))
        groups[kw].append(c)

    # Sort groups by size (rarest first to ensure they're included)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]))

    # Allocate: ensure at least 1 per group, then proportional
    remaining_slots = min(target, len(pool))
    allocated = 0

    # First pass: 1 per group
    for kw, group_cases in sorted_groups:
        if allocated >= remaining_slots:
            break
        if group_cases:
            selected.append(group_cases[0])
            allocated += 1

    # Second pass: proportional fill
    remaining = remaining_slots - allocated
    if remaining > 0:
        # Flatten remaining unselected cases, sorted by score
        remaining_cases = []
        for kw, group_cases in sorted_groups:
            for c in group_cases[1:]:  # skip first (already selected)
                remaining_cases.append(c)
        remaining_cases.sort(key=lambda c: c["quality_score"], reverse=True)

        # Select top remaining
        extra = remaining_cases[:remaining]
        selected.extend(extra)

    # Mark selection tier
    for c in selected:
        c["selection_tier"] = c.get("tier", "C")

    method = f"quality_tier_stratified_{len(selected)}_from_{len(cases)}"
    return selected, method


def run_selection(site_cases):
    """Run selection for all tissue sites."""
    print("\n" + "=" * 80)
    print("SELECTION RESULTS")
    print("=" * 80)

    selections = {}
    summary = {}

    for site in ALL_SITES:
        if site not in site_cases:
            continue
        cases = site_cases[site]
        pool_size = len(cases)

        selected, method = select_for_site(cases, TARGET_PER_SITE, RANDOM_SEED)

        # Count tiers in selection
        tier_counts = Counter(c.get("tier") for c in selected)

        selections[site] = {
            "total_in_pool": pool_size,
            "selected": len(selected),
            "selection_method": method,
            "cases": selected,
        }

        summary[site] = {
            "pool_size": pool_size,
            "selected": len(selected),
            "tier_a": tier_counts.get("A", 0),
            "tier_b": tier_counts.get("B", 0),
            "tier_c": tier_counts.get("C", 0),
        }

        flag = "" if len(selected) >= TARGET_PER_SITE else " ⚠ INCOMPLETE"
        print(f"  {site:<25} pool={pool_size:>8,}  selected={len(selected):>3}  "
              f"A={tier_counts.get('A',0):>3} B={tier_counts.get('B',0):>3} C={tier_counts.get('C',0):>3}{flag}")

    return selections, summary


# ══════════════════════════════════════════════════════════════════
# Phase 5: Output Generation
# ══════════════════════════════════════════════════════════════════

def generate_outputs(selections, summary):
    """Generate JSON and CSV output files."""
    print("\nGenerating output files...")

    # Build JSON output
    json_output = {
        "description": "source C case selection for pathologist annotation (tumor grounding + chain-of-thought)",
        "selection_date": "2026-05-22",
        "random_seed": RANDOM_SEED,
        "target_per_site": TARGET_PER_SITE,
        "selection_config": {
            "quality_thresholds": {"tier_a": 70, "tier_b": 40},
            "tissue_sites": len(selections),
        },
        "summary": summary,
        "selections": {},
    }

    for site, sel_data in selections.items():
        site_output = {
            "total_in_pool": sel_data["total_in_pool"],
            "selected": sel_data["selected"],
            "selection_method": sel_data["selection_method"],
            "cases": [],
        }
        for case in sel_data["cases"]:
            site_output["cases"].append({
                "case_id": case["case_id"],
                "source_dataset": case["source_dataset"],
                "tissue_site": site,
                "s3_path": f"s3://<YOUR_BUCKET>/source_c/{case['source_dataset']}/case_{case['case_id']}/",
                "s3_available": case["source_dataset"] in S3_DATASETS,
                "quality_score": case["quality_score"],
                "quality_tier": case["tier"],
                "meta_score": case["meta_score"],
                "slide_score": case["slide_score"],
                "bad_score": case["bad_score"],
                "rich_score": case["rich_score"],
                "slide_count": case["slide_count"],
                "he_slides": case["he_slides"],
                "stain_types": case.get("stain_types", []),
                "bad_patches": case.get("bad_patches", 0),
                "diagnosis": case.get("diagnosis", ""),
                "conclusion": case.get("conclusion", ""),
                "icd10": case.get("icd10", ""),
                "specialization": case.get("specialization", ""),
                "age": case.get("age"),
                "gender": case.get("gender", ""),
                "classification_method": case.get("classification_method", ""),
            })
        json_output["selections"][site] = site_output

    # Write JSON
    with open(SELECTION_OUTPUT, "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON: {SELECTION_OUTPUT}")

    # Write CSV
    csv_fields = [
        "tissue_site", "case_id", "source_dataset", "quality_score", "quality_tier",
        "meta_score", "slide_score", "bad_score", "rich_score",
        "slide_count", "he_slides", "stain_types", "bad_patches",
        "diagnosis", "conclusion", "icd10", "specialization",
        "age", "gender", "classification_method", "s3_path", "s3_available",
    ]

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for site, sel_data in selections.items():
            for case in sel_data["cases"]:
                row = {
                    "tissue_site": site,
                    "case_id": case["case_id"],
                    "source_dataset": case["source_dataset"],
                    "quality_score": case["quality_score"],
                    "quality_tier": case["tier"],
                    "meta_score": case["meta_score"],
                    "slide_score": case["slide_score"],
                    "bad_score": case["bad_score"],
                    "rich_score": case["rich_score"],
                    "slide_count": case["slide_count"],
                    "he_slides": case["he_slides"],
                    "stain_types": ";".join(case.get("stain_types", [])),
                    "bad_patches": case.get("bad_patches", 0),
                    "diagnosis": case.get("diagnosis", ""),
                    "conclusion": case.get("conclusion", ""),
                    "icd10": case.get("icd10", ""),
                    "specialization": case.get("specialization", ""),
                    "age": case.get("age", ""),
                    "gender": case.get("gender", ""),
                    "classification_method": case.get("classification_method", ""),
                    "s3_path": f"s3://<YOUR_BUCKET>/source_c/{case['source_dataset']}/case_{case['case_id']}/",
                    "s3_available": case["source_dataset"] in S3_DATASETS,
                }
                writer.writerow(row)
    print(f"  CSV: {CSV_OUTPUT}")

    # Print grand total
    total_selected = sum(s["selected"] for s in selections.values())
    print(f"\n  Total cases selected: {total_selected} across {len(selections)} tissue sites")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("source C Case Selection for Pathologist Annotation")
    print("=" * 80)

    # Phase 1
    metadata_cases = load_metadata()
    qc_data = load_qc_progress()
    bad_files = load_bad_files()
    slide_index = build_slide_index(qc_data)

    # Phase 2
    site_cases = classify_all_cases(metadata_cases)

    # Phase 3
    score_all_cases(site_cases, slide_index, bad_files)

    # Phase 4
    selections, summary = run_selection(site_cases)

    # Phase 5
    generate_outputs(selections, summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
