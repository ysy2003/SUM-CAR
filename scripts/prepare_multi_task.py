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
            'prompt': f"Question: {ex['question']}\n\nThink step by step, then provide your final numeric answer in the last sentence.",
            'gold_numbers': [],
            'tests': ''
        }
        for i, ex in enumerate(ds.select(range(actual_samples)))
    ]

# Function to load MBPP samples (aligned with noLoRA/code_only)
def load_mbpp(max_samples):
    from sumcar.data import mbpp
    ds = mbpp.load(split='train')
    actual_samples = min(max_samples, len(ds))
    return [
        {
            'id': f'mbpp:{ds[i]["task_id"]}',
            'prompt': ds[i]['prompt'],
            'gold_numbers': [],
            'tests': ds[i]['target']
        }
        for i in range(actual_samples)
    ]

# Function to load FinQA samples
def load_finqa(max_samples):
    """Load FinQA samples with CoT prompts."""
    ds = finqa_rc.load(split='train', use_cot=True)  # Enable CoT
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
    from sumcar.data import mbpp

    # Get actual dataset sizes
    gsm8k_size = len(load_dataset('gsm8k', 'main', split='train'))
    mbpp_size = len(mbpp.load(split='train'))
    finqa_size = len(finqa_rc.load(split='train', use_cot=True))

    print(f'Available dataset sizes:')
    print(f'  GSM8K: {gsm8k_size}')
    print(f'  MBPP: {mbpp_size}')
    print(f'  FinQA: {finqa_size}')

    # Calculate unit size based on each dataset being the limiting factor
    unit_from_math = gsm8k_size / ratio_math
    unit_from_code = mbpp_size / ratio_code
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
    ap.add_argument('--max_samples', type=int, default=None, help='Max samples per task (default: use all)')
    ap.add_argument('--ratio', type=str, default=None, help='Ratio of math:code:finance (e.g., 7:2:1). If set, limits samples based on ratio.')
    args = ap.parse_args()

    if args.ratio is not None:
        # Use ratio-based sampling (limited by smallest dataset)
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
        print(f'  Code (MBPP): {code_count}')
        print(f'  Finance (FinQA): {finance_count}')
        print(f'  Total: {math_count + code_count + finance_count}')
    elif args.max_samples is not None:
        # Use equal samples per task
        math_count = code_count = finance_count = args.max_samples
        print(f'Using equal samples: {args.max_samples} per dataset')
    else:
        # Default: use all samples from each dataset
        math_count = code_count = finance_count = 999999
        print(f'Using all available samples from each dataset')

    # Load datasets with calculated counts
    print(f'\nLoading datasets...')
    gsm8k_samples = load_gsm8k(math_count)
    mbpp_samples = load_mbpp(code_count)
    finqa_samples = load_finqa(finance_count)

    print(f'Loaded: {len(gsm8k_samples)} GSM8K, {len(mbpp_samples)} MBPP, {len(finqa_samples)} FinQA')

    # Combine datasets
    composite_dataset = gsm8k_samples + mbpp_samples + finqa_samples

    # Save to output file
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in composite_dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f'\nComposite dataset saved to {args.out}')
    print(f'Total samples: {len(composite_dataset)}')
