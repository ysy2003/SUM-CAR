# FinQA Evaluation Method Update

## Problem
The original FinQA evaluation used **Exact Match (EM)**, which was too strict:
- Required perfect string match: `"127.40"` vs `"The answer is 127.4"` → WRONG
- Failed on precision differences: `"127.4"` vs `"127.40"` → WRONG  
- Ignored correct reasoning: Model calculates correctly but answer embedded in text → WRONG

**Result:** 0% accuracy on 100 samples (misleading baseline)

## Solution
Implemented **Numeric Extraction with Tolerance** (similar to GSM8K):
- Extracts last number from model output
- Compares using floating-point tolerance (±1e-6)
- Handles format differences: `127.4` == `127.40` ✓
- Handles percentage mismatches: `0.24` == `24%` ✓

**Result:** 7% accuracy on 100 samples (true baseline)

## Changes Made

### 1. Added New Metric Function
**File:** `src/sumcar/eval/metrics.py`

```python
def acc_numeric_tolerant(pred: str, gold: str, tolerance: float = 1e-6) -> float:
    """
    Numeric comparison with floating-point tolerance.
    Better for FinQA where answers may have precision differences.
    """
    # Extract last number from both strings
    # Compare with floating-point tolerance
    # Handle percentage format mismatches
```

### 2. Updated FinQA Evaluation
**File:** `baselines/eval_base_model.py`

- Import: Added `acc_numeric_tolerant`
- Function: Updated `eval_finqa()` to use numeric extraction as primary metric
- Output: Reports both metrics:
  - `accuracy` (numeric tolerant) - **primary metric**
  - `em` (exact match) - legacy comparison

### 3. Updated Output Format
Results now include:
```json
{
  "finqa": {
    "accuracy": 0.07,     // NEW: Primary metric (numeric tolerant)
    "em": 0.00,           // Legacy metric (for comparison)
    "predictions": [...]
  },
  "config": {
    "finqa_eval_method": "numeric_tolerant"
  }
}
```

## Impact

| Metric | Old (EM) | New (Numeric) | Improvement |
|--------|----------|---------------|-------------|
| FinQA Accuracy | 0% | 7% | +7 correct answers |

**Note:** 7% is still low because:
- 5% of responses collapse into repetition loops (truncation)
- 87/94 responses with valid numbers are genuinely wrong (model errors)

This is the **true baseline** for comparing SUM-CAR memory improvements.

## Backward Compatibility

- Legacy EM score still reported as `em` field
- New primary metric is `accuracy` field
- Config includes `finqa_eval_method: "numeric_tolerant"` flag

## Usage

Run evaluation as before:
```bash
python baselines/eval_base_model.py --use_cot --max_samples 100
```

Output will show both metrics:
```
✓ FinQA Accuracy (numeric): 0.0700
✓ FinQA EM (legacy):        0.0000
```
