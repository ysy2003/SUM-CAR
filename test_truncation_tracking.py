#!/usr/bin/env python3
"""Test truncation tracking in predictions"""
import json

# Test with existing code_only_results_cot.json to see the new fields would appear
sample_prediction = {
    'question': 'What is 2+2?',
    'prediction': 'The answer is 4',
    'gold': '4',
    'correct': True,
    'generation_status': 'finished',  # or 'truncated'
    'generated_tokens': 8
}

print("="*60)
print("Sample Prediction with Truncation Tracking")
print("="*60)
print(json.dumps(sample_prediction, indent=2))

print("\n" + "="*60)
print("Explanation")
print("="*60)
print("New fields added to each prediction:")
print("  • 'generation_status': 'finished' | 'truncated'")
print("  • 'generated_tokens': int (number of tokens generated)")
print()
print("New summary statistics:")
print("  • 'truncated': int (count of truncated predictions)")
print()
print("Detection logic:")
print("  • If generated_tokens >= max_new_tokens → 'truncated'")
print("  • Otherwise → 'finished'")
print()
print("Example output:")
print("  Truncated: 15/100 (15.0%)")
