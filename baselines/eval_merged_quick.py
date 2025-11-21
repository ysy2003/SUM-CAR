"""
Quick evaluation of merged model on small subset.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
from src.sumcar.eval.metrics import acc_numeric, em
from src.sumcar.utils.sandbox import safe_exec


def load_merged_model(base_model, merged_dir, k_top=4, alpha=1.0):
    """Load merged memory-augmented model."""
    state = torch.load(os.path.join(merged_dir, 'memory.pt'), map_location='cpu', weights_only=False)
    
    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=state['keys'].shape[0], k_top=k_top, alpha=alpha)
    
    with torch.no_grad():
        mem.keys.data[:] = state['keys']
        mem.vals.data[:] = state['vals']
    
    return MemoryAugmentedCausalLM(base_model, mem)


@torch.no_grad()
def eval_gsm8k(model, tokenizer, max_samples=20):
    """Quick eval on GSM8K."""
    ds = load_dataset('gsm8k', 'main')['test']
    ds = ds.select(range(min(max_samples, len(ds))))
    
    total, correct = 0, 0
    print(f"\n  Testing {len(ds)} GSM8K samples...")
    for i, ex in enumerate(ds):
        prompt = f"Solve the problem and give only the final numeric answer.\n\n{ex['question']}\n\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=64, do_sample=False)
        pred = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        gold = ex['answer']
        is_correct = acc_numeric(pred, gold)
        correct += is_correct
        total += 1
        if i < 3:  # Show first 3 examples
            print(f"    Example {i+1}: {'✓' if is_correct else '✗'}")
            print(f"      Q: {ex['question'][:60]}...")
            print(f"      Pred: {pred[:50]}")
            print(f"      Gold: {gold[:50]}")
    
    return {'accuracy': correct/total, 'total': total, 'correct': correct}


@torch.no_grad()
def eval_humaneval(model, tokenizer, max_samples=20):
    """Quick eval on HumanEval."""
    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('nuprl/humaneval')['test']
    
    ds = ds.select(range(min(max_samples, len(ds))))
    
    total, correct = 0, 0
    print(f"\n  Testing {len(ds)} HumanEval samples...")
    for i, ex in enumerate(ds):
        enc = tokenizer(ex['prompt'], return_tensors='pt')
        out_ids = model.generate(enc['input_ids'], max_new_tokens=256, do_sample=False)
        code = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        test_code = ex.get('test', '')
        res = safe_exec(code + "\n\n" + test_code)
        ok = (res.ok and 'passed' in res.stdout.lower()) or (res.ok and len(res.error)==0)
        correct += 1 if ok else 0
        total += 1
        if i < 3:
            print(f"    Example {i+1}: {'✓' if ok else '✗'}")
    
    return {'pass@1': correct/total, 'total': total, 'correct': correct}


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=20):
    """Quick eval on FinQA."""
    from src.sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev', use_rc_filter=False)
    ds = ds[:min(max_samples, len(ds))]
    
    total, correct = 0, 0
    skipped = 0
    print(f"\n  Testing {len(ds)} FinQA samples...")
    for i, ex in enumerate(ds):
        ctx = ex.get('context', '')
        q = ex.get('question', '')
        gold = ex.get('answer', '')
        prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=960)
        
        try:
            out_ids = model.generate(enc['input_ids'], max_new_tokens=64, do_sample=False)
            pred = tokenizer.decode(out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            is_correct = em(pred, gold)
            correct += is_correct
            total += 1
            if i < 3:
                print(f"    Example {i+1}: {'✓' if is_correct else '✗'}")
                print(f"      Pred: {pred[:50]}")
                print(f"      Gold: {gold[:50]}")
        except Exception as e:
            skipped += 1
    
    return {'em': correct/total if total > 0 else 0.0, 'total': total, 'correct': correct, 'skipped': skipped}


def main(base_model='gpt2',
         merged_dir='out/merged',
         out='baselines/merged_model_results_quick.json',
         k_top=4,
         alpha=1.0,
         max_samples=20):
    """
    Quick evaluation of merged model.
    
    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt
        out: Output JSON path
        k_top: Top-k for memory retrieval (should match training)
        alpha: Alpha parameter (should match training)
        max_samples: Samples per task (default: 20)
    """
    print(f"=== Quick Evaluation: Merged Model ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Max samples per task: {max_samples}")
    print()
    
    # Load model
    print("Loading merged model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha)
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
    results['gsm8k'] = eval_gsm8k(model, tokenizer, max_samples)
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
    results['finqa'] = eval_finqa(model, tokenizer, max_samples)
    print(f"\n  Result: {results['finqa']['correct']}/{results['finqa']['total']} correct")
    print(f"  EM: {results['finqa']['em']:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"GSM8K:     {results['gsm8k']['accuracy']:.4f} ({results['gsm8k']['correct']}/{results['gsm8k']['total']})")
    print(f"HumanEval: {results['humaneval']['pass@1']:.4f} ({results['humaneval']['correct']}/{results['humaneval']['total']})")
    print(f"FinQA:     {results['finqa']['em']:.4f} ({results['finqa']['correct']}/{results['finqa']['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
