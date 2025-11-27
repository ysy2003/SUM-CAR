"""
Evaluate Qwen3-8B with thinking mode enabled on three tasks.
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

from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant, em
from src.sumcar.utils.sandbox import safe_exec


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks or raw text."""
    import re

    # Try to find code in markdown blocks (```python ... ``` or ``` ... ```)
    # Match opening ```, optional "python", newline, content, closing ```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # If no markdown blocks, return the text as-is (might be raw code)
    return text.strip()


def parse_qwen3_output(output_ids, tokenizer):
    """Parse Qwen3 output to extract thinking and final content."""
    try:
        # Find </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
        # Successfully found </think>, split at that point
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        # No </think> token found - assume no thinking mode was used
        # All output is content
        thinking_content = ""
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return thinking_content, content


@torch.no_grad()
def eval_gsm8k(model, tokenizer, max_samples=None, enable_thinking=True):
    """Evaluate on GSM8K math problems."""
    ds = load_dataset('gsm8k', 'main')['test']
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    device = next(model.parameters()).device
    total, correct = 0, 0
    predictions = []

    for ex in tqdm(ds, desc="GSM8K", unit="problems"):
        # Format as chat message
        prompt = f"Question: {ex['question']}\n\nProvide your final numeric answer in the last sentence."
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        input_length = model_inputs.input_ids.shape[1]
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=False
        )

        # Extract only generated tokens
        if len(generated_ids[0]) > input_length:
            output_ids = generated_ids[0][input_length:].tolist()
        else:
            output_ids = generated_ids[0].tolist()
        thinking, content = parse_qwen3_output(output_ids, tokenizer)

        gold = ex['answer']
        is_correct = acc_numeric(content, gold)
        correct += is_correct
        total += 1

        predictions.append({
            'question': ex['question'],
            'thinking': thinking,
            'prediction': content,
            'gold': gold,
            'correct': bool(is_correct)
        })

    return {'accuracy': correct/total, 'total': total, 'predictions': predictions}


@torch.no_grad()
def eval_humaneval(model, tokenizer, max_samples=None, enable_thinking=True):
    """Evaluate on HumanEval code generation.
    """
    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('nuprl/humaneval')['test']

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    device = next(model.parameters()).device
    total, correct = 0, 0
    predictions = []

    for ex in tqdm(ds, desc="HumanEval", unit="problems"):
        # Format as chat message
        prompt = ex['prompt']
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        input_length = model_inputs.input_ids.shape[1]
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False
        )

        # Extract only generated tokens
        if len(generated_ids[0]) > input_length:
            output_ids = generated_ids[0][input_length:].tolist()
        else:
            output_ids = generated_ids[0].tolist()
        thinking, content = parse_qwen3_output(output_ids, tokenizer)

        # Extract code from markdown blocks if present
        code = extract_code_from_markdown(content)

        # Combine with prompt for execution
        executed_code = ex['prompt'] + code

        test_code = ex.get('test', '')
        res = safe_exec(executed_code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1

        predictions.append({
            'prompt': ex['prompt'],
            'thinking': thinking,
            'raw_content': content,  # Content after thinking extraction
            'extracted_code': code,  # Code extracted from markdown
            'executed_code': executed_code,  # Complete executable code
            'passed': bool(ok),
            'error': res.error if not ok else None
        })

    return {'pass@1': correct/total, 'total': total, 'correct': correct, 'predictions': predictions}


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=None, enable_thinking=True):
    """Evaluate on FinQA."""
    from src.sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev', use_rc_filter=False)

    if max_samples:
        if hasattr(ds, 'select'):
            ds = ds.select(range(min(max_samples, len(ds))))
        else:
            ds = ds[:max_samples]

    device = next(model.parameters()).device
    total, correct = 0, 0
    skipped = 0
    predictions = []

    for ex in tqdm(ds, desc="FinQA", unit="problems"):
        ctx = ex.get('context', '')
        q = ex.get('question', '')
        gold = ex.get('answer', '')

        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\n\nProvide your final numeric answer in the last sentence."
        messages = [{"role": "user", "content": prompt}]

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )

            model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=4096).to(device)

            input_length = model_inputs.input_ids.shape[1]
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,  # Same as GSM8K - both need reasoning with thinking mode
                do_sample=False
            )

            # Extract only generated tokens
            if len(generated_ids[0]) > input_length:
                output_ids = generated_ids[0][input_length:].tolist()
            else:
                output_ids = generated_ids[0].tolist()
            thinking, content = parse_qwen3_output(output_ids, tokenizer)

            # Use tolerant metric for financial data
            is_correct = acc_numeric_tolerant(content, gold)
            correct += is_correct
            total += 1

            predictions.append({
                'question': q,
                'context': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                'thinking': thinking,
                'prediction': content,
                'gold': gold,
                'correct': bool(is_correct)
            })
        except Exception as e:
            skipped += 1
            print(f"Skipped sample due to error: {str(e)}")

    return {
        'accuracy': correct/total if total > 0 else 0.0,
        'total': total,
        'correct': correct,
        'skipped': skipped,
        'predictions': predictions
    }


def main(out='baselines/qwen3_8b_thinking.json', max_samples=None):
    """
    Evaluate Qwen3-8B with thinking mode on three tasks.

    Args:
        out: Output JSON file path
        max_samples: Maximum samples per task (None = use all)
    """
    import warnings
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)

    print("="*60)
    print("Qwen3-8B Baseline Evaluation (Thinking Mode)")
    print("="*60)
    print()

    # Load model
    print("Loading Qwen3-8B...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    print()

    results = {}

    # GSM8K
    print("="*60)
    print("Task 1/3: GSM8K (Math)")
    print("="*60)
    results['gsm8k'] = eval_gsm8k(model, tokenizer, max_samples)
    print(f"  Accuracy: {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['total']} samples)")
    print()

    # HumanEval
    print("="*60)
    print("Task 2/3: HumanEval (Code)")
    print("="*60)
    results['humaneval'] = eval_humaneval(model, tokenizer, max_samples)
    print(f"  Pass@1: {results['humaneval']['pass@1']:.4f} ({results['humaneval']['total']} samples)")
    print()

    # FinQA
    print("="*60)
    print("Task 3/3: FinQA (Finance)")
    print("="*60)
    results['finqa'] = eval_finqa(model, tokenizer, max_samples)
    print(f"  Accuracy: {results['finqa']['accuracy']:.4f} ({results['finqa']['total']} samples)")
    print()

    # Save results
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"GSM8K:     {results['gsm8k']['accuracy']:.4f}")
    print(f"HumanEval: {results['humaneval']['pass@1']:.4f}")
    print(f"FinQA:     {results['finqa']['accuracy']:.4f}")
    print()
    print(f"Results saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
