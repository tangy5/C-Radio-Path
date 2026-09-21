#!/usr/bin/env python3
"""Select a balanced subset of skin cases for processing.

This script randomly samples cases from source C-skin-b1 and source C-skin-b2
to create a balanced dataset while preserving important diagnostic categories.

Selection strategy:
- Skin-b1: Process ~40% of cases (melanoma-focused, already balanced)
- Skin-b2: Process ~10% of cases (mostly benign nevi, heavily downsampled)

Output: selected_cases.json with list of selected case IDs per dataset.
"""

import json
import random
import os
from pathlib import Path
from collections import defaultdict

# Configuration
RANDOM_SEED = 42  # For reproducibility
SELECTION_CONFIG = {
    'source C-skin-b1': {
        'target_cases': 1400,  # ~80% of 1,778 cases - MORE DIVERSITY
        'description': 'Large melanoma-focused subset with high diversity'
    },
    'source C-skin-b2': {
        'target_cases': 6000,  # ~30% of 20,621 cases - MORE DIVERSITY
        'description': 'Large balanced sample with diverse benign nevi'
    }
}

OUTPUT_FILE = 'selected_cases.json'


def load_metadata(metadata_path='../source_c/source C-metadata/metadata.json'):
    """Load source C metadata."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_cases_by_dataset(metadata, dataset_name):
    """Extract case IDs for a specific dataset from metadata."""
    cases = []
    for entry in metadata:
        case_mapping = entry.get('case_mapping', '')
        if dataset_name in case_mapping:
            # Extract case number from 'source_c/source C-skin-b1/case_0927'
            try:
                case_id = case_mapping.split('case_')[-1]
                cases.append({
                    'case_id': case_id,
                    'diagnosis': entry.get('diagnosis', ''),
                    'conclusion': entry.get('conclusion', '')
                })
            except:
                continue
    return cases


def stratified_sample(cases, target_count, random_seed=RANDOM_SEED):
    """Sample cases with stratification by diagnosis keywords.
    
    Ensures rare/important cases (melanoma) are preserved.
    """
    random.seed(random_seed)
    
    # Categorize cases
    melanoma_cases = []
    nevus_cases = []
    other_cases = []
    
    for case in cases:
        diagnosis = case['diagnosis'].lower()
        conclusion = case['conclusion'].lower()
        
        if 'melanoma' in diagnosis or 'melanoma' in conclusion:
            melanoma_cases.append(case)
        elif 'nevus' in diagnosis or 'nevus' in conclusion or 'nevi' in diagnosis:
            nevus_cases.append(case)
        else:
            other_cases.append(case)
    
    print(f"  Category breakdown:")
    print(f"    Melanoma: {len(melanoma_cases)}")
    print(f"    Nevus: {len(nevus_cases)}")
    print(f"    Other: {len(other_cases)}")
    
    selected = []
    
    # Keep all melanoma cases (they're important and relatively rare)
    selected.extend(melanoma_cases)
    print(f"  → Keeping all {len(melanoma_cases)} melanoma cases")
    
    # Calculate remaining slots
    remaining_slots = target_count - len(selected)
    
    if remaining_slots <= 0:
        # If we have more melanoma than target, sample melanoma only
        selected = random.sample(melanoma_cases, target_count)
        print(f"  → Melanoma exceeds target, sampled {target_count} melanoma only")
        return selected
    
    # Split remaining between nevus and other
    # Prioritize nevus (more common in skin pathology)
    nevus_target = int(remaining_slots * 0.8)
    other_target = remaining_slots - nevus_target
    
    # Sample nevus
    if len(nevus_cases) <= nevus_target:
        selected.extend(nevus_cases)
        print(f"  → Keeping all {len(nevus_cases)} nevus cases")
        other_target += (nevus_target - len(nevus_cases))
    else:
        selected.extend(random.sample(nevus_cases, nevus_target))
        print(f"  → Sampled {nevus_target}/{len(nevus_cases)} nevus cases")
    
    # Sample other
    if len(other_cases) <= other_target:
        selected.extend(other_cases)
        print(f"  → Keeping all {len(other_cases)} other cases")
    else:
        selected.extend(random.sample(other_cases, other_target))
        print(f"  → Sampled {other_target}/{len(other_cases)} other cases")
    
    return selected


def simple_random_sample(cases, target_count, random_seed=RANDOM_SEED):
    """Simple random sampling for smaller datasets."""
    random.seed(random_seed)
    if len(cases) <= target_count:
        print(f"  → Dataset has only {len(cases)} cases, keeping all")
        return cases
    selected = random.sample(cases, target_count)
    print(f"  → Randomly sampled {target_count}/{len(cases)} cases")
    return selected


def save_selection(selections, output_path=OUTPUT_FILE):
    """Save selected cases to JSON file."""
    output = {
        'description': 'Selected skin cases for balanced source C dataset',
        'random_seed': RANDOM_SEED,
        'selection_strategy': {
            'source C-skin-b1': '40% random sample (melanoma-focused)',
            'source C-skin-b2': 'Stratified 10% sample (preserves melanoma, downsamples nevi)'
        },
        'datasets': {}
    }
    
    for dataset_name, data in selections.items():
        output['datasets'][dataset_name] = {
            'total_cases_available': data['total'],
            'selected_cases': len(data['selected']),
            'case_ids': [c['case_id'] for c in data['selected']],
            'selection_method': data['method']
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Selection saved to: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("source C Skin Case Selection")
    print("=" * 70)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata()
    print(f"  Loaded {len(metadata)} metadata entries")
    print()
    
    selections = {}
    
    # Process each skin dataset
    for dataset_name, config in SELECTION_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"Target: {config['target_cases']} cases")
        print()
        
        # Get all cases for this dataset
        cases = get_cases_by_dataset(metadata, dataset_name)
        print(f"Total cases in metadata: {len(cases)}")
        
        if len(cases) == 0:
            print(f"  WARNING: No cases found for {dataset_name}")
            continue
        
        # Apply selection strategy
        if dataset_name == 'source C-skin-b2':
            # Use stratified sampling for b2 (large, imbalanced)
            selected = stratified_sample(cases, config['target_cases'])
            method = 'stratified'
        else:
            # Use simple random sampling for b1 (smaller, already balanced)
            selected = simple_random_sample(cases, config['target_cases'])
            method = 'random'
        
        selections[dataset_name] = {
            'total': len(cases),
            'selected': selected,
            'method': method
        }
        
        print(f"\n  Selected: {len(selected)} cases ({100*len(selected)/len(cases):.1f}%)")
    
    # Save selection
    print(f"\n{'='*70}")
    output_path = save_selection(selections)
    
    # Summary
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    total_selected = 0
    for dataset_name, data in selections.items():
        selected_count = len(data['selected'])
        total_selected += selected_count
        print(f"{dataset_name}:")
        print(f"  Selected: {selected_count} / {data['total']} ({100*selected_count/data['total']:.1f}%)")
    print(f"\nTotal skin cases selected: {total_selected}")
    print(f"  (vs 51,467 available = {100*total_selected/51467:.1f}%)")
    print(f"\nSelection file: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
