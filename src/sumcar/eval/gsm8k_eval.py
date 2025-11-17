from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from .metrics import acc_numeric


@torch.no_grad()
def evaluate_gsm8k(model, base_model_name: str, split: str='test', max_new_tokens: int=64, batch_size: int=4):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    ds = load_dataset('gsm8k', 'main')[split]
    total, correct = 0, 0
    for ex in ds:
        prompt = f"Solve the problem and give only the final numeric answer.\n\n{ex['question']}\n\nAnswer:"
        enc = tok(prompt, return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=max_new_tokens)
        pred = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        gold = ex['answer']
        correct += acc_numeric(pred, gold)
        total += 1
    return {'accuracy': correct/total, 'total': total}