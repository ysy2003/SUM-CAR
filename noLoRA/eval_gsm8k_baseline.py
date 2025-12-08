"""
Evaluate BASE model (no memory) on GSM8K for baseline comparison.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant


@torch.no_grad()
def eval_gsm8k(model, tokenizer, max_samples=99999, use_cot=True):
    """Evaluate on GSM8K test set."""
    device = next(model.parameters()).device
    ds = load_dataset('gsm8k', 'main')['test']

    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""

    print(f"\n  Testing {len(ds_subset)} GSM8K samples{cot_str}...")

    for ex in tqdm(ds_subset, desc="  GSM8K", unit="sample"):
        if use_cot:
            prompt = f"Question: {ex['question']}\n\nThink step by step, then provide your final numeric answer."
        else:
            prompt = f"Question: {ex['question']}\n\nProvide your final numeric answer."

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([text], return_tensors='pt').to(device)

        max_tokens = 512
        input_length = enc['input_ids'].shape[1]
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        gen_ids = out_ids[0, input_length:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

        answer_text = ex['answer']
        gold = answer_text.split('####')[-1].strip() if '####' in answer_text else answer_text.strip()

        is_correct = acc_numeric(pred, gold) or acc_numeric_tolerant(pred, gold) > 0
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            'question': ex['question'],
            'gold': gold,
            'pred': pred,
            'correct': is_correct
        })

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'predictions': predictions
    }


def main(model_name='meta-llama/Meta-Llama-3-8B-Instruct',
         out='noLoRA/baseline/gsm8k_baseline.json',
         mode='full',
         use_cot=True,
         use_fp16=False):
    """
    Evaluate base model (no memory) on GSM8K.
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 100

    print(f"=== GSM8K Baseline Evaluation ===")
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

    results = eval_gsm8k(model, tokenizer, max_samples=max_samples, use_cot=use_cot)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'gsm8k': results,
        'model': model_name,
        'config': {'mode': mode, 'use_cot': use_cot, 'use_fp16': use_fp16}
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"GSM8K Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
