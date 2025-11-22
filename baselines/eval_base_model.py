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

from src.sumcar.eval.metrics import acc_numeric, em
from src.sumcar.utils.sandbox import safe_exec


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
            prompt = f"Let's solve this step by step.\n\nQuestion: {ex['question']}\n\nLet me think through this carefully:"
        else:
            prompt = f"Solve the problem and give only the final numeric answer.\n\n{ex['question']}\n\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt').to(device)
        # High limit - let model finish naturally with EOS token
        max_tokens = 2048 if use_cot else 512
        out_ids = model.generate(
            enc['input_ids'],
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  # Stop at EOS
            pad_token_id=tokenizer.pad_token_id
        )
        pred = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
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
        enc = tokenizer(ex['prompt'], return_tensors='pt').to(device)
        # High limit for code generation - let model finish naturally
        out_ids = model.generate(
            enc['input_ids'],
            max_new_tokens=1024,  # Generous limit for complex functions
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        code = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)

        # Combine prompt (function signature) with generated code (function body)
        full_code = ex['prompt'] + code

        # Run tests
        test_code = ex.get('test', '')
        res = safe_exec(full_code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1
        
        predictions.append({
            'prompt': ex['prompt'],
            'generated_code': code,
            'full_code': full_code,  # Include complete executable code
            'passed': bool(ok),
            'error': res.error if not ok else None
        })
    
    return {'pass@1': correct/total, 'total': total, 'predictions': predictions}


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=None, use_cot=False):
    """Evaluate on FinQA financial QA."""
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
    for ex in tqdm(ds, desc="FinQA", unit="questions"):
        ctx = ex['context'] if 'context' in ex else ex.get('context', '')
        q = ex['question'] if 'question' in ex else ex.get('question', '')
        gold = ex['answer'] if 'answer' in ex else ex.get('answer', '')
        if use_cot:
            prompt = f"Answer the question using ONLY the given context. Think step by step.\n\nContext:\n{ctx}\n\nQuestion: {q}\nLet me think through this carefully:\n"
        else:
            prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=960).to(device)

        try:
            # High limit - let model finish naturally with EOS token
            max_tokens = 2048 if use_cot else 512
            out_ids = model.generate(
                enc['input_ids'],
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            pred = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            is_correct = em(pred, gold)
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
    
    return {'em': correct/total if total > 0 else 0.0, 'total': total, 'skipped': skipped, 'predictions': predictions}


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

    # Prepare output file paths
    out_base = out.replace('.json', '')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    # Evaluate on three tasks and save after each
    results = {}

    print("Evaluating GSM8K (Math)...")
    results['gsm8k'] = eval_gsm8k(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"  ✓ GSM8K Accuracy: {results['gsm8k']['accuracy']:.4f}")
    # Save GSM8K results immediately
    gsm8k_out = f"{out_base}_gsm8k.json"
    with open(gsm8k_out, 'w') as f:
        json.dump({'gsm8k': results['gsm8k'], 'config': {'use_cot': use_cot}}, f, indent=2)
    print(f"  ✓ Saved to: {gsm8k_out}")
    print()

    print("Evaluating HumanEval (Code)...")
    results['humaneval'] = eval_humaneval(model, tokenizer, max_samples)
    print(f"  ✓ HumanEval Pass@1: {results['humaneval']['pass@1']:.4f}")
    # Save HumanEval results immediately
    humaneval_out = f"{out_base}_humaneval.json"
    with open(humaneval_out, 'w') as f:
        json.dump({'humaneval': results['humaneval'], 'config': {'use_cot': use_cot}}, f, indent=2)
    print(f"  ✓ Saved to: {humaneval_out}")
    print()

    print("Evaluating FinQA (Finance)...")
    results['finqa'] = eval_finqa(model, tokenizer, max_samples, use_cot=use_cot)
    print(f"  ✓ FinQA EM: {results['finqa']['em']:.4f}")
    # Save FinQA results immediately
    finqa_out = f"{out_base}_finqa.json"
    with open(finqa_out, 'w') as f:
        json.dump({'finqa': results['finqa'], 'config': {'use_cot': use_cot}}, f, indent=2)
    print(f"  ✓ Saved to: {finqa_out}")
    print()

    # Also save combined results
    results['config'] = {'use_cot': use_cot}
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Combined results saved to: {out}")
    print()
    print("Summary:")
    print(f"  GSM8K Accuracy:    {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['total']} samples)")
    print(f"  HumanEval Pass@1:  {results['humaneval']['pass@1']:.4f} ({results['humaneval']['total']} samples)")
    print(f"  FinQA EM:          {results['finqa']['em']:.4f} ({results['finqa']['total']} samples)")


if __name__ == '__main__':
    fire.Fire(main)
