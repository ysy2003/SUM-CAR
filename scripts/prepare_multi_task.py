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
    actual_samples = min(max_samples, len(ds))
    return [
        {
            'id': f'gsm8k:{i}',
            'prompt': f"Question: {ex['question']}\n\nAnswer:",
            'gold_numbers': [],
            'tests': ''
        }
        for i, ex in enumerate(ds.select(range(actual_samples)))
    ]

# Function to load CodeXGLUE samples
def load_codexglue(max_samples):
    ds = load_dataset('code_x_glue_cc_code_refinement', 'small', split='train')
    actual_samples = min(max_samples, len(ds))
    return [
        {
            'id': f'codexglue:{i}',
            'prompt': f"Fix the following buggy Python code:\n\n{ex['buggy']}\n\nCorrected code:",
            'gold_numbers': [],
            'tests': f"# Test cases for the corrected code\n{ex['fixed']}"
        }
        for i, ex in enumerate(ds.select(range(actual_samples)))
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

def calculate_max_samples_with_ratio(ratio_math, ratio_code, ratio_finance):
    """Calculate maximum samples maintaining the given ratio.

    Args:
        ratio_math: Math ratio (e.g., 7)
        ratio_code: Code ratio (e.g., 2)
        ratio_finance: Finance ratio (e.g., 1)

    Returns:
        Tuple of (math_samples, code_samples, finance_samples)
    """
    # Get actual dataset sizes
    gsm8k_size = len(load_dataset('gsm8k', 'main', split='train'))
    codex_size = len(load_dataset('code_x_glue_cc_code_refinement', 'small', split='train'))
    finqa_size = len(finqa_rc.load(split='train', use_cot=False))

    print(f'Available dataset sizes:')
    print(f'  GSM8K: {gsm8k_size}')
    print(f'  CodeXGLUE: {codex_size}')
    print(f'  FinQA: {finqa_size}')

    # Calculate unit size based on each dataset being the limiting factor
    unit_from_math = gsm8k_size / ratio_math
    unit_from_code = codex_size / ratio_code
    unit_from_finance = finqa_size / ratio_finance

    # Use the smallest unit (most limiting dataset)
    unit_size = min(unit_from_math, unit_from_code, unit_from_finance)

    # Calculate final sample counts
    math_samples = int(unit_size * ratio_math)
    code_samples = int(unit_size * ratio_code)
    finance_samples = int(unit_size * ratio_finance)

    return math_samples, code_samples, finance_samples

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Output file for the composite dataset')
    ap.add_argument('--max_samples', type=int, default=None, help='Maximum samples per dataset (equal for all). If not specified, uses ratio-based maximum.')
    ap.add_argument('--ratio', type=str, default='7:2:1', help='Ratio of math:code:finance (default: 7:2:1)')
    args = ap.parse_args()

    if args.max_samples is None:
        # Use ratio-based maximum
        ratio_parts = [int(x) for x in args.ratio.split(':')]
        if len(ratio_parts) != 3:
            raise ValueError('Ratio must be in format math:code:finance (e.g., 7:2:1)')

        ratio_math, ratio_code, ratio_finance = ratio_parts
        print(f'Using ratio {ratio_math}:{ratio_code}:{ratio_finance} (math:code:finance)')

        math_count, code_count, finance_count = calculate_max_samples_with_ratio(
            ratio_math, ratio_code, ratio_finance
        )

        print(f'\nCalculated sample counts:')
        print(f'  Math (GSM8K): {math_count}')
        print(f'  Code (CodeXGLUE): {code_count}')
        print(f'  Finance (FinQA): {finance_count}')
        print(f'  Total: {math_count + code_count + finance_count}')
    else:
        # Use equal samples (old behavior)
        math_count = code_count = finance_count = args.max_samples
        print(f'Using equal samples: {args.max_samples} per dataset')

    # Load datasets with calculated counts
    print(f'\nLoading datasets...')
    gsm8k_samples = load_gsm8k(math_count)
    codexglue_samples = load_codexglue(code_count)
    finqa_samples = load_finqa(finance_count)

    print(f'Loaded: {len(gsm8k_samples)} GSM8K, {len(codexglue_samples)} CodeXGLUE, {len(finqa_samples)} FinQA')

    # Combine datasets
    composite_dataset = gsm8k_samples + codexglue_samples + finqa_samples

    # Save to output file
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in composite_dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f'\nComposite dataset saved to {args.out}')
    print(f'Total samples: {len(composite_dataset)}')
