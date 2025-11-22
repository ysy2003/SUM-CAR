# src/sumcar/data/finqa_rc.py
from datasets import load_dataset, Dataset
from typing import Dict, Iterable, List
import os
from pathlib import Path
import re
import json
import urllib.request

_SPLIT_MAP = {"train": "train", "dev": "dev", "test": "test"}

# 原始prompt
_DEF_INST = "Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"

# CoT prompt
_COT_INST = """Answer the question using ONLY the given context. Think step by step.

Context:
{ctx}

Question: {q}

Let me analyze this step by step:"""

def _table_to_tsv(table_2d):
    """Convert 2D table (List[List[str]]) to TSV format"""
    if not table_2d:
        return ""
    rows = ["\t".join(map(str, row)) for row in table_2d]
    return "\n".join(rows)

def _build_context(item: Dict) -> str:
    """Build unified context from pre_text, table, and post_text"""
    pre = " ".join(item.get("pre_text", [])) if isinstance(item.get("pre_text", []), list) else str(item.get("pre_text", ""))
    post = " ".join(item.get("post_text", [])) if isinstance(item.get("post_text", []), list) else str(item.get("post_text", ""))
    table = _table_to_tsv(item.get("table", []))
    return f"[PRE]\n{pre}\n\n[TABLE]\n{table}\n\n[POST]\n{post}"

def _format_example(item: Dict, use_cot: bool = False) -> Dict:
    """Format example into prompt/target structure"""
    # FinQA数据结构：question和answer在'qa'字段里
    qa = item.get("qa", {})
    ans = qa.get("answer", "") or qa.get("exe_ans", "") or ""
    q = qa.get("question", "")
    ctx = _build_context(item)
    
    # 根据use_cot选择prompt
    if use_cot:
        prompt = _COT_INST.format(ctx=ctx, q=q)
        # CoT模式下，可以添加推理步骤（如果有program的话）
        program = qa.get("program_re", "") or qa.get("program", "")
        if program:
            target = f"Based on the context, I need to: {program}\n\nThe answer is: {str(ans).strip()}"
        else:
            target = f"The answer is: {str(ans).strip()}"
    else:
        prompt = _DEF_INST.format(ctx=ctx, q=q)
        target = str(ans).strip()
    
    return {
        "context": ctx,
        "question": q,
        "answer": str(ans).strip(),
        "prompt": prompt,
        "target": target,
        "uid": item.get("id", ""),
        # Keep metadata for debugging
        "program": qa.get("program_re", "") or qa.get("program", ""),
        "gold_inds": qa.get("gold_inds", {}),
    }

def _rc_filter(ex: Dict) -> bool:
    """
    Lightweight RC subset filter:
    1) Answer is pure number or appears in context (no external knowledge needed)
    2) Keep samples with single table (FinQA is single report + single table)
    """
    ctx = (ex.get("context") or "").lower()
    ans = (ex.get("answer") or ex.get("target") or "").lower()
    if not ans:
        return False
    # Pure number or number with decimal/percent
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%?", ans):
        return True
    # Directly matchable text answer
    return ans in ctx

def _resolve_data_files(split: str):
    """Resolve data file path - try local first, then GitHub"""
    # Allow local override (set FINQA_DIR to directory containing dataset/)
    local_dir = os.getenv("FINQA_DIR")
    fname = {"train": "train.json", "dev": "dev.json", "test": "test.json"}[split]
    if local_dir:
        return str(Path(local_dir) / "dataset" / fname)
    # Otherwise fetch from GitHub raw files
    base = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
    return f"{base}/{fname}"

def load(split: str = 'train', use_rc_filter: bool = False, use_cot: bool = False) -> Dataset:
    """
    Load FinQA dataset directly from GitHub JSON files (bypasses HF script limitations).
    
    Args:
        split: 'train', 'dev', or 'test'
        use_rc_filter: If True, filter to "non-retrieval RC subset"
        use_cot: If True, use Chain-of-Thought prompting
    
    Returns:
        Dataset object compatible with HuggingFace datasets API
    """
    sp = _SPLIT_MAP.get(split, split)
    data_file = _resolve_data_files(sp)
    
    # Load JSON directly to avoid Arrow type inference issues
    try:
        if data_file.startswith("http://") or data_file.startswith("https://"):
            with urllib.request.urlopen(data_file) as response:
                raw_data = json.loads(response.read().decode())
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load FinQA data from {data_file}: {e}")
    
    # raw_data is a dict with keys like "train", "dev", "test" or direct list
    if isinstance(raw_data, dict):
        # Try common keys
        raw_items = raw_data.get(sp) or raw_data.get('data') or list(raw_data.values())[0]
    else:
        raw_items = raw_data
    
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Expected list of items, got {type(raw_items)}")
    
    # Format all examples
    data = [_format_example(item, use_cot=use_cot) for item in raw_items]
    
    # Optional: filter to RC subset (no retrieval needed)
    if use_rc_filter:
        data = [ex for ex in data if _rc_filter(ex)]
    
    # Filter out empty prompts
    data = [ex for ex in data if ex.get('prompt') and len(ex['prompt']) > 0]
    
    # Convert to Dataset object for compatibility with HF datasets API
    return Dataset.from_list(data)