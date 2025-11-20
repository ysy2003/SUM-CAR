from .metrics import em
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


@torch.no_grad()
def evaluate_finqa_rc(model, base_model_name: str, split: str='validation', max_new_tokens: int=64):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    # Use our custom finqa_rc loader that loads from GitHub
    from sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev' if split == 'validation' else split, use_rc_filter=False)
    # Expect the dataset has been pre-filtered similarly to data/finqa_rc.py in training pipeline
    total, correct = 0, 0
    skipped = 0
    for ex in ds:
        # Use formatted fields from our custom loader
        ctx = ex.get('context', '')
        q = ex.get('question', '')
        gold = ex.get('answer', '')
        prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        enc = tok(prompt, return_tensors='pt', truncation=True, max_length=960)  # Leave room for generation
        try:
            out_ids = model.generate(enc['input_ids'], max_new_tokens=max_new_tokens)
            pred = tok.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            correct += em(pred, gold)
            total += 1
        except Exception as e:
            # Skip examples that cause errors (e.g., too long)
            skipped += 1
            continue
    if total == 0:
        return {'em': 0.0, 'total': 0, 'skipped': skipped}
    return {'em': correct/total, 'total': total, 'skipped': skipped}