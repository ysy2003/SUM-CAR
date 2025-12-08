"""
Evaluate BASE model (no memory) on HumanEval for baseline comparison.
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

from src.sumcar.utils.sandbox import safe_exec


def extract_code(text: str) -> str:
    """Extract code from model output, handling markdown blocks."""
    if '```python' in text:
        start = text.find('```python') + len('```python')
        end = text.find('```', start)
        if end > start:
            return text[start:end].strip()
    if '```' in text:
        start = text.find('```') + 3
        end = text.find('```', start)
        if end > start:
            return text[start:end].strip()
    return text


@torch.no_grad()
def eval_humaneval(model, tokenizer, max_samples=99999):
    """Evaluate on HumanEval test set with pass@1."""
    device = next(model.parameters()).device

    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('openai/openai_humaneval')['test']

    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    total, correct = 0, 0
    predictions = []

    print(f"\n  Testing {len(ds_subset)} HumanEval samples...")

    for ex in tqdm(ds_subset, desc="  HumanEval", unit="sample"):
        prompt = ex['prompt']
        task_id = ex['task_id']
        test_code = ex.get('test', '')

        messages = [{"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"}]
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
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
            code = extract_code(raw_output)
            full_code = prompt + code

            exec_code = full_code + "\n\n" + test_code
            res = safe_exec(exec_code, timeout=5)

            passed = res.ok and len(res.error) == 0
            if passed:
                correct += 1
            total += 1

            predictions.append({
                'task_id': task_id,
                'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                'generated': code[:500] + '...' if len(code) > 500 else code,
                'passed': passed,
                'error': res.error[:200] if res.error else ''
            })

        except Exception as e:
            total += 1
            predictions.append({
                'task_id': task_id,
                'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                'generated': '',
                'passed': False,
                'error': str(e)[:200]
            })

    return {
        'correct': correct,
        'total': total,
        'pass_at_1': correct / total if total > 0 else 0,
        'predictions': predictions
    }


def main(model_name='meta-llama/Meta-Llama-3-8B-Instruct',
         out='noLoRA/baseline/humaneval_baseline.json',
         mode='full',
         use_fp16=False):
    """
    Evaluate base model (no memory) on HumanEval pass@1.
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 164

    print(f"=== HumanEval Baseline Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Mode: {mode} ({max_samples} samples)")
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

    results = eval_humaneval(model, tokenizer, max_samples=max_samples)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'humaneval': results,
        'model': model_name,
        'config': {'mode': mode, 'use_fp16': use_fp16}
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"HumanEval pass@1: {results['pass_at_1']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
