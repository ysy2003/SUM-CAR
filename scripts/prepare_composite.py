#!/usr/bin/env python3
import json, argparse
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumcar.data import finqa_rc

# Function to load GSM8K samples
def load_gsm8k(max_samples):
    ds = load_dataset('gsm8k', 'main', split='train')
    return [
        {
            'id': f'gsm8k:{i}',
            'prompt': f"Question: {ex['question']}\n\nAnswer:",
            'gold_numbers': [],
            'tests': ''
        }
        for i, ex in enumerate(ds.select(range(max_samples)))
    ]

# Function to load CodeXGLUE samples
def load_codexglue(max_samples):
    ds = load_dataset('code_x_glue_cc_code_refinement', 'small', split='train')
    return [
        {
            'id': f'codexglue:{i}',
            'prompt': f"Fix the following buggy Python code:\n\n{ex['buggy']}\n\nCorrected code:",
            'gold_numbers': [],
            'tests': f"# Test cases for the corrected code\n{ex['fixed']}"
        }
        for i, ex in enumerate(ds.select(range(max_samples)))
    ]

# Function to load FinQA samples
def load_finqa(max_samples):
    """Load FinQA samples."""
    ds = finqa_rc.load(split='train', use_cot=False)
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        samples.append({
            'id': f'finqa:{ex["uid"]}',
            'prompt': ex['prompt'],
            'gold_numbers': [],
            'tests': ''
        })
    return samples

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Output file for the composite dataset')
    ap.add_argument('--max_samples', type=int, default=100, help='Maximum samples per dataset')
    args = ap.parse_args()

    # Load datasets
    gsm8k_samples = load_gsm8k(args.max_samples)
    codexglue_samples = load_codexglue(args.max_samples)
    finqa_samples = load_finqa(args.max_samples)

    # Combine datasets
    composite_dataset = gsm8k_samples + codexglue_samples + finqa_samples

    # Save to output file
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in composite_dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f'Composite dataset saved to {args.out}')
