import json, re
from transformers import AutoTokenizer
from .metrics import composite_success
from ..utils.sandbox import safe_exec
import torch


@torch.no_grad()
def evaluate_composite(model, base_model_name: str, composite_path: str, max_new_tokens: int=256):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(composite_path, 'r', encoding='utf-8')]
    total, success = 0, 0
    for ex in rows:
        prompt = ex['prompt']
        gold_numbers = ex.get('gold_numbers', [])
        tests = ex.get('tests', '')
        enc = tok(prompt, return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=max_new_tokens, do_sample=False)
        txt = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        # Very simple heuristic split: look for code fences
        code_match = re.search(r"```python\n([\s\S]*?)```", txt)
        code = code_match.group(1) if code_match else txt
        # 1) NL number extraction check
        nl_ok = True
        for g in gold_numbers:
            if str(g) not in txt:
                nl_ok = False
                break
        # 2) Execute tests
        res = safe_exec(code + "\n\n" + tests)
        code_ok = res.ok and ('passed' in res.stdout.lower() or len(res.error)==0)
        success += composite_success(nl_ok, code_ok)
        total += 1
    return {'composite_success': success/total, 'total': total}