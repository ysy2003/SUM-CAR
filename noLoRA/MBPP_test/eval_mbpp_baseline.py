"""
Evaluate BASE model (no memory) on MBPP test set with pass@1.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

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
def eval_mbpp(model, tokenizer, max_samples=99999):
    """Evaluate on MBPP sanitized test set with pass@1."""
    device = next(model.parameters()).device

    ds = load_dataset('mbpp', 'sanitized')['test']

    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    total, correct = 0, 0
    predictions = []

    print(f"\n  Testing {len(ds_subset)} MBPP samples...")

    for ex in tqdm(ds_subset, desc="  MBPP", unit="sample"):
        task_id = ex['task_id']
        prompt_text = ex['prompt']
        test_list = ex['test_list']
        test_imports = ex.get('test_imports', [])

        # Create prompt for code generation
        user_prompt = f"Write a Python function that satisfies the following specification:\n\n{prompt_text}\n\nProvide only the function code."

        messages = [{"role": "user", "content": user_prompt}]
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

            # Build exec code: imports + generated code + tests
            exec_parts = []
            if test_imports:
                exec_parts.extend(test_imports)
            exec_parts.append(code)
            exec_parts.extend(test_list)
            exec_code = "\n".join(exec_parts)

            # Run tests
            res = safe_exec(exec_code, timeout=10)

            passed = res.ok and len(res.error) == 0
            if passed:
                correct += 1
            total += 1

            predictions.append({
                'task_id': task_id,
                'prompt': prompt_text[:200] + '...' if len(prompt_text) > 200 else prompt_text,
                'generated': code[:500] + '...' if len(code) > 500 else code,
                'passed': passed,
                'error': res.error[:200] if res.error else ''
            })

        except Exception as e:
            total += 1
            predictions.append({
                'task_id': task_id,
                'prompt': prompt_text[:200] + '...' if len(prompt_text) > 200 else prompt_text,
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
         out='noLoRA/MBPP_test/baseline_mbpp.json',
         mode='full',
         use_fp16=False):
    """
    Evaluate base model (no memory) on MBPP test set pass@1.
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 257

    print(f"=== MBPP Baseline Evaluation ===")
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

    results = eval_mbpp(model, tokenizer, max_samples=max_samples)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'mbpp': results,
        'model': model_name,
        'config': {'mode': mode, 'use_fp16': use_fp16}
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"MBPP pass@1: {results['pass_at_1']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
