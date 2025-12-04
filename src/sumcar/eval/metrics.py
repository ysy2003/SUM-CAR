import re, math
from typing import List, Dict


# GSM8K-style numeric compare
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(s: str):
    m = _NUM.findall(str(s).replace(',', ''))
    return m[-1] if m else None


def acc_numeric(pred: str, gold: str) -> float:
    pn, gn = last_number(pred), last_number(gold)
    return 1.0 if (pn is not None and gn is not None and pn == gn) else 0.0


def extract_answer_number(text: str) -> str:
    """
    Extract answer number from text using smart patterns.
    For best results, pass only the final answer portion (not thinking/reasoning).
    Falls back to last number if no explicit answer pattern found.
    """
    text_clean = str(text).replace(',', '')

    # Pattern 1: Final answer markers (most specific)
    # e.g., "Final answer: 60.3", "The final answer is 60.3"
    final_patterns = [
        r'final answer[:\s]+\$?(-?\d+(?:\.\d+)?)',
        r'answer:[:\s]+\$?(-?\d+(?:\.\d+)?)',
    ]

    for pattern in final_patterns:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        if matches:
            return matches[-1]

    # Pattern 2: "The answer is X" patterns
    # e.g., "The answer is 18.6", "the answer is $18.6 million"
    answer_patterns = [
        r'(?:the|my) answer is[:\s]+\$?(-?\d+(?:\.\d+)?)',
        r'(?:the )?result is[:\s]+\$?(-?\d+(?:\.\d+)?)',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        if matches:
            return matches[-1]

    # Fallback: Use last number in text (most reliable for short answers)
    return last_number(text_clean)


def acc_numeric_tolerant(pred: str, gold: str, tolerance: float = 1e-6) -> float:
    """
    Numeric comparison with floating-point tolerance and format handling.
    Better for FinQA where answers may have precision differences or format variations.

    Handles:
    - Floating point precision: 127.4 == 127.40
    - Percentage formats: 60.3% == 0.603 (tries both interpretations)
    - Embedded answers: "The answer is 18.6 million" -> 18.6
    - Dollar signs and commas: $1,234.56 -> 1234.56
    - Smart answer extraction: looks for "answer is X" patterns first
    """
    # Extract numbers using smart answer extraction
    pred_str = str(pred).replace(',', '')
    gold_str = str(gold).replace(',', '')

    pred_num_str = extract_answer_number(pred)
    gold_num_str = extract_answer_number(gold)

    if pred_num_str is None or gold_num_str is None:
        return 0.0

    try:
        pred_val = float(pred_num_str)
        gold_val = float(gold_num_str)

        # Direct comparison with tolerance
        if abs(pred_val - gold_val) <= tolerance:
            return 1.0

        # Handle percentage format mismatches
        # Case 1: pred is percentage (60.3%), gold is decimal (0.603)
        if '%' in pred and '%' not in gold:
            pred_as_decimal = pred_val / 100.0
            if abs(pred_as_decimal - gold_val) <= tolerance:
                return 1.0

        # Case 2: gold is percentage (60.3%), pred is decimal (0.603)
        if '%' in gold and '%' not in pred:
            gold_as_decimal = gold_val / 100.0
            if abs(pred_val - gold_as_decimal) <= tolerance:
                return 1.0

        # Case 3: Both have % but different scales (unlikely but handle it)
        # e.g., pred="0.603%" should match gold="60.3%"
        if '%' in pred and '%' in gold:
            # Try comparing pred*100 vs gold
            if abs(pred_val * 100.0 - gold_val) <= tolerance:
                return 1.0
            # Try comparing pred vs gold*100
            if abs(pred_val - gold_val * 100.0) <= tolerance:
                return 1.0

        return 0.0

    except (ValueError, TypeError):
        return 0.0


# EM/F1 for FinQA RC (string-level; for numbers use numeric equality where possible)


def em(pred: str, gold: str) -> float:
    return float(str(pred).strip() == str(gold).strip())


# HumanEval pass@k estimator given n samples and c correct
# Expected pass@k = 1 - C(n-c, k)/C(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k: return 0.0
    from math import comb
    return 1.0 - (comb(n-c, k) / comb(n, k)) if n>0 else 0.0


# Composite success: both NL extraction and code execution checks pass


def composite_success(nl_ok: bool, code_ok: bool) -> float:
    return 1.0 if (nl_ok and code_ok) else 0.0