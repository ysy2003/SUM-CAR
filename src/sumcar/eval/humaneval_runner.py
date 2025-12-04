from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from ..utils.sandbox import safe_exec
import torch


@torch.no_grad()
def evaluate_humaneval_pass1(model, base_model_name: str, max_new_tokens: int=2048):
    # Detect device from model
    device = next(model.parameters()).device
    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('nuprl/humaneval')['test']
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    total, correct = 0, 0
    for ex in tqdm(ds, desc="HumanEval Evaluation"):
        enc = tok(ex['prompt'], return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out_ids = model.generate(enc['input_ids'], max_new_tokens=max_new_tokens, do_sample=False)
        code = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        # run provided tests
        test_code = ex.get('test', '')
        res = safe_exec(code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1
    return {'pass@1': correct/total, 'total': total}