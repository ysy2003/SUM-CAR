"""
Evaluate GPT-2 base model (without memory) on three tasks.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
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
    
    total, correct = 0, 0
    predictions = []
    prompt_type = "CoT" if use_cot else "normal"
    print(f"  Using {prompt_type} prompting")
    for ex in ds:
        if use_cot:
            prompt = f"Let's solve this step by step.\n\nQuestion: {ex['question']}\n\nLet me think through this carefully:"
        else:
            prompt = f"Solve the problem and give only the final numeric answer.\n\n{ex['question']}\n\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt')
        max_tokens = 128 if use_cot else 64
        out_ids = model.generate(enc['input_ids'], max_new_tokens=max_tokens, do_sample=False)
        pred = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
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
        
        if total <= 3:
            print(f"    Example {total}: {'✓' if is_correct else '✗'}")
            print(f"      Q: {ex['question'][:60]}...")
            print(f"      Pred: {pred[:100]}")
            print(f"      Gold: {gold[:60]}")
        if total % 100 == 0:
            print(f"  GSM8K: {total}/{len(ds)} processed, acc={correct/total:.4f}")
    
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
    
    total, correct = 0, 0
    predictions = []
    for ex in ds:
        enc = tokenizer(ex['prompt'], return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=256, do_sample=False)
        code = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Run tests
        test_code = ex.get('test', '')
        res = safe_exec(code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1
        
        predictions.append({
            'prompt': ex['prompt'],
            'generated_code': code,
            'passed': bool(ok),
            'error': res.error if not ok else None
        })
        
        if total <= 3:
            print(f"    Example {total}: {'✓' if ok else '✗'}")
            print(f"      Generated: {code[:100]}...")
        if total % 20 == 0:
            print(f"  HumanEval: {total}/{len(ds)} processed, pass@1={correct/total:.4f}")
    
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
    
    total, correct = 0, 0
    skipped = 0
    predictions = []
    prompt_type = "CoT" if use_cot else "normal"
    print(f"  Using {prompt_type} prompting")
    for ex in ds:
        ctx = ex['context'] if 'context' in ex else ex.get('context', '')
        q = ex['question'] if 'question' in ex else ex.get('question', '')
        gold = ex['answer'] if 'answer' in ex else ex.get('answer', '')
        if use_cot:
            prompt = f"Answer the question using ONLY the given context. Think step by step.\n\nContext:\n{ctx}\n\nQuestion: {q}\nLet me think through this carefully:\n"
        else:
            prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=960)
        
        try:
            max_tokens = 128 if use_cot else 64
            out_ids = model.generate(enc['input_ids'], max_new_tokens=max_tokens, do_sample=False)
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
            
            if total <= 3:
                print(f"    Example {total}: {'✓' if is_correct else '✗'}")
                print(f"      Q: {q[:60]}...")
                print(f"      Pred: {pred[:100]}")
                print(f"      Gold: {gold}")
        except Exception as e:
            skipped += 1
            continue
        
        if total % 100 == 0:
            print(f"  FinQA: {total} processed, em={correct/total:.4f}")
    
    return {'em': correct/total if total > 0 else 0.0, 'total': total, 'skipped': skipped, 'predictions': predictions}


def main(base_model='gpt2', 
         out='baselines/base_model_results.json',
         max_samples=None,
         use_cot=False):
    """
    Evaluate base GPT-2 model on three tasks.
    
    Args:
        base_model: Model name (default: gpt2)
        out: Output JSON file path
        max_samples: Maximum samples per task (None = use all)
        use_cot: Use Chain-of-Thought prompting (default: False)
    """
    print(f"=== Evaluating Base Model: {base_model} ===")
    if max_samples:
        print(f"Using max {max_samples} samples per task")
    print()
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print()
    
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
    print(f"  ✓ FinQA EM: {results['finqa']['em']:.4f}")
    print()
    
    # Save results
    results['config'] = {'use_cot': use_cot}
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {out}")
    print()
    print("Summary:")
    print(f"  GSM8K Accuracy:    {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['total']} samples)")
    print(f"  HumanEval Pass@1:  {results['humaneval']['pass@1']:.4f} ({results['humaneval']['total']} samples)")
    print(f"  FinQA EM:          {results['finqa']['em']:.4f} ({results['finqa']['total']} samples)")


if __name__ == '__main__':
    fire.Fire(main)
