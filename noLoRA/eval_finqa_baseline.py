"""
Evaluate BASE model (no memory) on FinQA for baseline comparison.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.eval.metrics import acc_numeric_tolerant


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=99999, use_cot=True):
    """Evaluate on FinQA validation set."""
    device = next(model.parameters()).device

    from src.sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev', use_rc_filter=False)

    if hasattr(ds, 'select'):
        end_idx = min(max_samples, len(ds))
        ds_list = [ds[i] for i in range(end_idx)]
    else:
        ds_list = list(ds)[:max_samples]

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""

    print(f"\n  Testing {len(ds_list)} FinQA samples{cot_str}...")

    for ex in tqdm(ds_list, desc="  FinQA", unit="sample"):
        ctx = ex.get('context', '')
        q = ex.get('question', '')
        gold = ex.get('answer', '')

        if use_cot:
            prompt = (
                f"Answer the question using ONLY the given context.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {q}\n\n"
                f"Think step by step, then provide your final numeric answer."
            )
        else:
            prompt = (
                f"Answer the question using ONLY the given context.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {q}\n"
                f"Answer:"
            )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([text], return_tensors='pt', truncation=True, max_length=2048).to(device)

        max_tokens = 512
        input_length = enc['input_ids'].shape[1]

        try:
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            gen_ids = out_ids[0, input_length:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

            is_correct = acc_numeric_tolerant(pred, gold) > 0
            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                'question': q,
                'gold': gold,
                'pred': pred,
                'correct': is_correct
            })
        except Exception as e:
            print(f"  Warning: Skipped sample due to error: {e}")
            continue

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'predictions': predictions
    }


def main(model_name='meta-llama/Meta-Llama-3-8B-Instruct',
         out='noLoRA/baseline/finqa_baseline.json',
         mode='full',
         use_cot=True,
         use_fp16=False):
    """
    Evaluate base model (no memory) on FinQA.
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 100

    print(f"=== FinQA Baseline Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Mode: {mode} ({max_samples} samples)")
    print(f"Use CoT: {use_cot}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    results = eval_finqa(model, tokenizer, max_samples=max_samples, use_cot=use_cot)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'finqa': results,
        'model': model_name,
        'config': {'mode': mode, 'use_cot': use_cot, 'use_fp16': use_fp16}
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"FinQA Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
