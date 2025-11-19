# src/sumcar/data/finqa_rc.py
from datasets import load_dataset
from typing import Dict, Iterable
import re

_SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}

_DEF_INST = "Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"

def _table_to_tsv(table_2d):
    """Convert 2D table (List[List[str]]) to TSV format"""
    if not table_2d:
        return ""
    rows = ["\t".join(map(str, row)) for row in table_2d]
    return "\n".join(rows)

def _build_context(item: Dict) -> str:
    """Build unified context from pre_text, table, and post_text"""
    pre = " ".join(item["pre_text"]) if isinstance(item["pre_text"], list) else str(item["pre_text"])
    post = " ".join(item["post_text"]) if isinstance(item["post_text"], list) else str(item["post_text"])
    table = _table_to_tsv(item.get("table", []))
    return f"[PRE]\n{pre}\n\n[TABLE]\n{table}\n\n[POST]\n{post}"

def _format_example(item: Dict) -> Dict:
    """Format example into prompt/target structure"""
    ans = item.get("answer") or item.get("final_result") or ""
    ctx = _build_context(item)
    q = item["question"]
    return {
        "context": ctx,
        "question": q,
        "answer": str(ans).strip(),
        "prompt": _DEF_INST.format(ctx=ctx, q=q),
        "target": str(ans).strip(),
        "uid": item["id"],
        # Keep metadata for debugging
        "program": item.get("program_re", ""),
        "gold_inds": item.get("gold_inds", []),
    }

def _rc_filter(ex: Dict) -> bool:
    """
    Lightweight RC subset filter:
    1) Answer is pure number or appears in context (no external knowledge needed)
    2) Keep samples with single table (FinQA is single report + single table)
    """
    ctx = (ex["context"] or "").lower()
    ans = (ex["output"] or "").lower()
    if not ans:
        return False
    # Pure number or number with decimal/percent
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%?", ans):
        return True
    # Directly matchable text answer
    return ans in ctx

def load(split: str = 'train', use_rc_filter: bool = False) -> Iterable[Dict]:
    """
    Load FinQA dataset from official HF source (ibm-research/finqa).
    
    Args:
        split: 'train', 'dev', or 'test'
        use_rc_filter: If True, filter to "non-retrieval RC subset"
    """
    hf_split = _SPLIT_MAP.get(split, split)
    # Official HF dataset (script downloads from GitHub automatically)
    raw = load_dataset("ibm-research/finqa", split=hf_split)
    
    data = [_format_example(item) for item in raw]
    
    # Optional: filter to RC subset (no retrieval needed)
    if use_rc_filter:
        data = [ex for ex in data if _rc_filter(ex)]
    
    # Filter out empty prompts
    data = [ex for ex in data if ex.get('prompt') and len(ex['prompt']) > 0]
    
    return data