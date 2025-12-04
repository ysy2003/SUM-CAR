"""
Quick evaluation of merged model on small subset.
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

from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
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


def load_merged_model(base_model, merged_dir, k_top=8, alpha=1.0, use_fp16=True):
    """Load merged memory-augmented model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(os.path.join(merged_dir, 'memory.pt'), map_location=device, weights_only=False)

    torch_dtype = torch.float16 if use_fp16 else torch.float32
    d_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype
    ).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=state['keys'].shape[0], k_top=k_top, alpha=alpha)

    with torch.no_grad():
        mem.keys.data[:] = state['keys']
        mem.vals.data[:] = state['vals']

    # Convert memory to FP16 if requested
    if use_fp16:
        mem = mem.half()

    model = MemoryAugmentedCausalLM(base_model, mem, use_fp16=use_fp16)
    model = model.to(device)
    precision = "FP16" if use_fp16 else "FP32"
    print(f"  Loaded with {precision} precision")
    return model


@torch.no_grad()
def eval_gsm8k(model, tokenizer, max_samples=20, use_cot=False):
    """Quick eval on GSM8K."""
    device = next(model.parameters()).device
    ds = load_dataset('gsm8k', 'main')['test']
    ds = ds.select(range(min(max_samples, len(ds))))

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""
    print(f"\n  Testing {len(ds)} GSM8K samples{cot_str}...")
    for i, ex in enumerate(tqdm(ds, desc="  GSM8K", unit="sample")):
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

        # High limit for all math reasoning
        max_tokens = 4096
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

        gold = ex['answer']
        is_correct = acc_numeric(pred, gold)
        correct += is_correct
        total += 1

        predictions.append({
            'question': ex['question'],
            'prediction': pred,
            'gold': gold,
            'correct': bool(is_correct)
        })

    return {
        'accuracy': correct/total,
        'total': total,
        'correct': correct,
        'predictions': predictions
    }


@torch.no_grad()
def eval_humaneval(model, tokenizer, max_samples=20):
    """Quick eval on HumanEval."""
    device = next(model.parameters()).device
    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('nuprl/humaneval')['test']

    ds = ds.select(range(min(max_samples, len(ds))))

    total, correct = 0, 0
    predictions = []
    print(f"\n  Testing {len(ds)} HumanEval samples...")
    for i, ex in enumerate(tqdm(ds, desc="  HumanEval", unit="sample")):
        prompt = ex['prompt']

        # Use chat template for proper formatting
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tokenizer([text], return_tensors='pt').to(device)

        # High limit for code generation
        max_tokens = 2048
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
            raw_output = tokenizer.decode(out_ids[0][input_length:], skip_special_tokens=True)
        else:
            raw_output = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Extract code from markdown blocks if present
        code = extract_code_from_markdown(raw_output)

        # Combine prompt (function signature) with generated code (function body)
        executed_code = ex['prompt'] + code

        test_code = ex.get('test', '')
        res = safe_exec(executed_code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1

        predictions.append({
            'prompt': ex['prompt'],
            'raw_output': raw_output,
            'extracted_code': code,
            'executed_code': executed_code,
            'passed': bool(ok),
            'error': res.error if not ok else None
        })

    return {
        'pass@1': correct/total,
        'total': total,
        'correct': correct,
        'predictions': predictions
    }


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=20, use_cot=False):
    """
    Quick eval on FinQA.
    Uses acc_numeric_tolerant (handles financial formats, percentages, precision).
    """
    device = next(model.parameters()).device
    from src.sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev', use_rc_filter=False)
    # Select samples (handle Dataset object)
    if hasattr(ds, 'select'):
        ds = ds.select(range(min(max_samples, len(ds))))
    else:
        ds = ds[:min(max_samples, len(ds))]

    total, correct = 0, 0
    skipped = 0
    predictions = []
    prompt_type = "CoT" if use_cot else "normal"
    print(f"\n  Testing {len(ds)} FinQA samples ({prompt_type})...")
    print(f"  Evaluation: acc_numeric_tolerant (handles percentages, precision)")
    for i, ex in enumerate(tqdm(ds, desc="  FinQA", unit="sample")):
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
        enc = tokenizer([text], return_tensors='pt', truncation=True, max_length=960).to(device)

        try:
            # High limit for financial reasoning - match GSM8K
            max_tokens = 4096
            input_length = enc['input_ids'].shape[1]
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            # Handle different output formats
            output_length = len(out_ids[0])
            if output_length > input_length:
                # Normal case: output includes input + generated
                pred = tokenizer.decode(out_ids[0][input_length:], skip_special_tokens=True)
            else:
                # MemoryAugmentedCausalLM returns only generated tokens
                pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

            # Use tolerant metric for financial data
            is_correct = acc_numeric_tolerant(pred, gold)
            correct += is_correct
            total += 1

            predictions.append({
                'question': q,
                'context': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                'prediction': pred,
                'gold': gold,
                'correct': bool(is_correct)
            })
        except Exception as e:
            print(f"\n  [ERROR] Exception on sample {i}: {str(e)}")
            skipped += 1

    return {
        'accuracy': correct/total if total > 0 else 0.0,
        'total': total,
        'correct': correct,
        'skipped': skipped,
        'predictions': predictions
    }


def main(base_model='gpt2',
         merged_dir='out/merged',
         out='scripts/merged_model_results_quick.json',
         k_top=8,
         alpha=1.0,
         max_samples=20,
         use_cot=False,
         use_fp16=True):
    """
    Quick evaluation of merged model.

    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt
        out: Output JSON path
        k_top: Top-k for memory retrieval (should match training)
        alpha: Alpha parameter (should match training)
        max_samples: Samples per task (default: 20)
        use_cot: Use Chain-of-Thought prompting (default: False)
        use_fp16: Use FP16 precision (default: True)
    """
    print(f"=== Quick Evaluation: Merged Model ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Max samples per task: {max_samples}")
    print()
    
    # Load model
    print("Loading merged model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha, use_fp16=use_fp16)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("✓ Model loaded")
    
    # Evaluate
    results = {}
    
    print("\n" + "="*50)
    print("GSM8K (Math)")
    print("="*50)
    results['gsm8k'] = eval_gsm8k(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"\n  Result: {results['gsm8k']['correct']}/{results['gsm8k']['total']} correct")
    print(f"  Accuracy: {results['gsm8k']['accuracy']:.4f}")

    print("\n" + "="*50)
    print("HumanEval (Code)")
    print("="*50)
    results['humaneval'] = eval_humaneval(model, tokenizer, max_samples)
    print(f"\n  Result: {results['humaneval']['correct']}/{results['humaneval']['total']} passed")
    print(f"  Pass@1: {results['humaneval']['pass@1']:.4f}")

    print("\n" + "="*50)
    print("FinQA (Finance)")
    print("="*50)
    results['finqa'] = eval_finqa(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"\n  Result: {results['finqa']['correct']}/{results['finqa']['total']} correct")
    print(f"  Accuracy: {results['finqa']['accuracy']:.4f}")
    if results['finqa']['skipped'] > 0:
        print(f"  Skipped: {results['finqa']['skipped']}")

    # Save
    results['config'] = {'use_cot': use_cot, 'k_top': k_top, 'alpha': alpha, 'gsm8k_eval_method': 'acc_numeric', 'finqa_eval_method': 'acc_numeric_tolerant'}
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"GSM8K:     {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['correct']}/{results['gsm8k']['total']}) [acc_numeric]")
    print(f"HumanEval: {results['humaneval']['pass@1']:.4f} ({results['humaneval']['correct']}/{results['humaneval']['total']})")
    print(f"FinQA:     {results['finqa']['accuracy']:.4f} ({results['finqa']['correct']}/{results['finqa']['total']}) [acc_numeric_tolerant]")
    print()
    print("Note: GSM8K uses acc_numeric (string match), FinQA uses acc_numeric_tolerant (float + format handling)")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
