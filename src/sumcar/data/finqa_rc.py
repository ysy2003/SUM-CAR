import re
from typing import Any
from datasets import load_dataset


_DEF_INST = "Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(s: str):
    return _NUM.findall(s.replace(',', ''))


def _looks_like_table_ref(x: Any) -> bool:
    if not x:
        return False
    s = str(x)
    pats = [r"\btable\b", r"\bcell\b", r"R\d+C\d+", r"\bT\[\d+,\d+\]", r"\bcol\b", r"\brow\b"]
    import regex as re
    return any(re.search(p, s, flags=re.I) for p in pats)


def load(split: str='train', join_pre_post: bool=True, require_answer_in_context: bool=True, forbid_table_refs: bool=True):
    base = None
    for name in ['cais/finqa', 'finqa/finqa']:
        try:
            base = load_dataset(name)[split]
            break
        except:
            pass
    if base is None:
        raise RuntimeError('FinQA not found on HF Hub')
    
    def _map(ex):
        pre = "\n".join(ex.get('pre_text', [])) if isinstance(ex.get('pre_text'), list) else ex.get('pre_text', '')
        post = "\n".join(ex.get('post_text', [])) if isinstance(ex.get('post_text'), list) else ex.get('post_text', '')
        ctx = (pre + "\n" + post) if join_pre_post else pre
        q = ex.get('question') or ex.get('qa', {}).get('question', '')
        a = ex.get('answer') or ex.get('qa', {}).get('answer', '')
        prog = ex.get('program') or ex.get('derivation') or ''
        uid = ex.get('uid') or ex.get('id')
        if forbid_table_refs and _looks_like_table_ref(prog):
            return None
        if require_answer_in_context:
            nums_ctx, nums_ans = set(_numbers(ctx)), set(_numbers(str(a)))
            if nums_ans and not nums_ans.issubset(nums_ctx):
                return None
        return {
            'context': ctx, 'question': q, 'answer': str(a),
            'prompt': _DEF_INST.format(ctx=ctx, q=q), 'target': str(a), 'uid': uid
        }
    
    mapped = base.map(_map, remove_columns=base.column_names)
    return mapped.filter(lambda ex: ex['prompt'] is not None and len(ex['prompt'])>0)