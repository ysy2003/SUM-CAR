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