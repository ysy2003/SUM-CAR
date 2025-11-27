"""
Evaluate base language model (without memory) on three tasks.
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


@torch.no_grad()
def eval_gsm8k(model, tokenizer, max_samples=None, use_cot=False):
    """Evaluate on GSM8K math problems."""
    ds = load_dataset('gsm8k', 'main')['test']
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    device = next(model.parameters()).device
    total, correct = 0, 0
    predictions = []
    prompt_type = "CoT" if use_cot else "normal"
    print(f"  Using {prompt_type} prompting")
    for ex in tqdm(ds, desc="GSM8K", unit="problems"):
        if use_cot:
            prompt = f"Question: {ex['question']}\n\nThink step by step, then provide your final numeric answer in the last sentence."
        else:
            prompt = f"Question: {ex['question']}\n\nProvide your final numeric answer in the last sentence."

        # Use chat template for proper formatting
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tokenizer([text], return_tensors='pt').to(device)

        # High limit - let model finish naturally with EOS token
        max_tokens = 4096  # Generous limit for all math reasoning
        input_length = enc['input_ids'].shape[1]
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  # Stop at EOS
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode only generated tokens (skip input)
        if len(out_ids[0]) > input_length:
            pred = tokenizer.decode(out_ids[0][input_length:], skip_special_tokens=True)
        else:
            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gold = ex['answer']
        is_correct = acc_numeric(pred, gold)
        correct += is_correct
        total += 1

        # Check if generation was truncated
        generated_length = len(out_ids[0]) - len(enc['input_ids'][0])
        was_truncated = generated_length >= max_tokens

        predictions.append({
            'question': ex['question'],
            'prediction': pred,
            'gold': gold,
            'correct': bool(is_correct),
            'generated_tokens': int(generated_length),
            'was_truncated': bool(was_truncated)
        })
    
    return {'accuracy': correct/total, 'total': total, 'predictions': predictions}


@torch.no_grad()
def eval_humaneval(model, tokenizer, max_samples=None):
    """Evaluate on HumanEval code generation."""
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
        prompt = ex['prompt']

        # Use chat template for proper formatting
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tokenizer([text], return_tensors='pt').to(device)

        # High limit for code generation - let model finish naturally
        input_length = enc['input_ids'].shape[1]
        out_ids = model.generate(
            **enc,
            max_new_tokens=2048,  # Generous limit for complex functions
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode only generated tokens (skip input)
        if len(out_ids[0]) > input_length:
            raw_output = tokenizer.decode(out_ids[0][input_length:], skip_special_tokens=True)
        else:
            raw_output = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Extract code from markdown blocks if present
        code = extract_code_from_markdown(raw_output)

        # Combine prompt (function signature) with generated code (function body)
        executed_code = ex['prompt'] + code

        # Run tests
        test_code = ex.get('test', '')
        res = safe_exec(executed_code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1

        predictions.append({
            'prompt': ex['prompt'],
            'raw_output': raw_output,  # Store raw model output for debugging
            'extracted_code': code,  # Code extracted from markdown
            'executed_code': executed_code,  # Complete executable code
            'passed': bool(ok),
            'error': res.error if not ok else None
        })
    
    return {'pass@1': correct/total, 'total': total, 'predictions': predictions}


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=None, use_cot=False):
    """
    Evaluate on FinQA financial QA.
    Uses acc_numeric_tolerant (handles financial formats, percentages, precision).
    """
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
    prompt_type = "CoT" if use_cot else "normal"
    print(f"  Using {prompt_type} prompting")
    print(f"  Evaluation: acc_numeric_tolerant (handles percentages, precision)")

    for ex in tqdm(ds, desc="FinQA", unit="questions"):
        ctx = ex['context'] if 'context' in ex else ex.get('context', '')
        q = ex['question'] if 'question' in ex else ex.get('question', '')
        gold = ex['answer'] if 'answer' in ex else ex.get('answer', '')
        if use_cot:
            prompt = f"Context:\n{ctx}\n\nQuestion: {q}\n\nThink step by step, then provide your final numeric answer in the last sentence."
        else:
            prompt = f"Context:\n{ctx}\n\nQuestion: {q}\n\nProvide your final numeric answer in the last sentence."

        # Use chat template for proper formatting
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Truncate input if needed to fit context
        enc = tokenizer([text], return_tensors='pt', truncation=True, max_length=960).to(device)

        try:
            # High limit - let model finish naturally with EOS token
            # Match GSM8K limits for fair comparison between reasoning tasks
            max_tokens = 4096  # Same as GSM8K
            input_length = enc['input_ids'].shape[1]
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode only generated tokens (skip input)
            if len(out_ids[0]) > input_length:
                pred = tokenizer.decode(out_ids[0][input_length:], skip_special_tokens=True)
            else:
                pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

            # Use tolerant metric for financial data
            is_correct = acc_numeric_tolerant(pred, gold)
            correct += is_correct
            total += 1

            predictions.append({
                'question': q,
                'context': ctx[:200] + '...' if len(ctx) > 200 else ctx,  # Truncate context for readability
                'prediction': pred,
                'gold': gold,
                'correct': bool(is_correct)
            })
        except Exception as e:
            skipped += 1
            continue

    return {
        'accuracy': correct/total if total > 0 else 0.0,
        'total': total,
        'skipped': skipped,
        'predictions': predictions
    }


def main(base_model='meta-llama/Meta-Llama-3-8B-Instruct',
         out='baselines/base_model_results.json',
         max_samples=None,
         use_cot=False):
    """
    Evaluate base language model on three tasks.

    Args:
        base_model: Model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
        out: Output JSON file path (base name, will append _gsm8k.json, _humaneval.json, _finqa.json)
        max_samples: Maximum samples per task (None = use all)
        use_cot: Use Chain-of-Thought prompting (default: False)
    """
    # Silence transformers warnings
    import warnings
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)

    # Load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    print()

    # Prepare output file path
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    # Evaluate on three tasks
    results = {}

    print("Evaluating GSM8K (Math)...")
    results['gsm8k'] = eval_gsm8k(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"  ✓ GSM8K Accuracy: {results['gsm8k']['accuracy']:.4f}")
    print()

    print("Evaluating HumanEval (Code)...")
    results['humaneval'] = eval_humaneval(model, tokenizer, max_samples)
    print(f"  ✓ HumanEval Pass@1: {results['humaneval']['pass@1']:.4f}")
    print()

    print("Evaluating FinQA (Finance)...")
    results['finqa'] = eval_finqa(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"  ✓ FinQA Accuracy: {results['finqa']['accuracy']:.4f}")
    print()

    # Save combined results
    results['config'] = {'use_cot': use_cot, 'gsm8k_eval_method': 'acc_numeric', 'finqa_eval_method': 'acc_numeric_tolerant'}
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out}")
    print()
    print("Summary:")
    print(f"  GSM8K Accuracy:    {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['total']} samples) [acc_numeric]")
    print(f"  HumanEval Pass@1:  {results['humaneval']['pass@1']:.4f} ({results['humaneval']['total']} samples)")
    print(f"  FinQA Accuracy:    {results['finqa']['accuracy']:.4f} ({results['finqa']['total']} samples) [acc_numeric_tolerant]")
    print()
    print("Note: GSM8K uses acc_numeric (string match), FinQA uses acc_numeric_tolerant (float + format handling)")


if __name__ == '__main__':
    fire.Fire(main)
