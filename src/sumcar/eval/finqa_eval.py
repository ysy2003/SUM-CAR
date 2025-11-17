from .metrics import em
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


@torch.no_grad()
def evaluate_finqa_rc(model, base_model_name: str, split: str='validation', max_new_tokens: int=64):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    try:
        ds = load_dataset('cais/finqa')[split]
    except:
        ds = load_dataset('finqa/finqa')[split]
    # Expect the dataset has been pre-filtered similarly to data/finqa_rc.py in training pipeline
    total, correct = 0, 0
    for ex in ds:
        pre = "\n".join(ex.get('pre_text', [])) if isinstance(ex.get('pre_text'), list) else ex.get('pre_text', '')
        post = "\n".join(ex.get('post_text', [])) if isinstance(ex.get('post_text'), list) else ex.get('post_text', '')
        ctx = pre + "\n" + post
        q = ex.get('question') or ex.get('qa', {}).get('question', '')
        gold = ex.get('answer') or ex.get('qa', {}).get('answer', '')
        prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        enc = tok(prompt, return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=max_new_tokens)
        pred = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        correct += em(pred, gold)
        total += 1
    return {'em': correct/total, 'total': total}