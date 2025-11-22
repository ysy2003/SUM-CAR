"""Test individual task patches before merge to check training effectiveness"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sumcar.models.base_model import MemoryAugmentedCausalLM
from sumcar.memory.kv_memory import KVMemoryLayer
from sumcar.data.gsm8k import load as load_gsm8k
from sumcar.data.humaneval import load as load_humaneval
from sumcar.data.finqa_rc import load as load_finqa
from sumcar.eval.metrics import last_number
import json
import os

def test_gsm8k_patch(patch_path, max_samples=3):
    """Test GSM8K task patch"""
    print("\n" + "="*60)
    print("Testing GSM8K Patch (BEFORE merge)")
    print("="*60)
    
    # Load patch
    with open(patch_path, 'r') as f:
        patch = json.load(f)
    
    print(f"Patch contains {len(patch['keys'])} slots")
    
    # Create memory
    keys = torch.tensor(patch['keys'])
    vals = torch.tensor(patch['vals'])
    d_model = keys.shape[1]
    num_slots = keys.shape[0]
    
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=4, alpha=1.0)
    with torch.no_grad():
        mem.keys.data[:] = keys
        mem.vals.data[:] = vals
    
    # Create model
    model = MemoryAugmentedCausalLM("gpt2", mem)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check memory stats
    print(f"Keys norm: {mem.keys.norm().item():.4f}")
    print(f"Vals norm: {mem.vals.norm().item():.4f}")
    print(f"Non-zero vals: {(mem.vals.abs() > 1e-6).sum().item()}/{mem.vals.numel()}")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load data
    ds = load_gsm8k(split='test')
    ds = ds.select(range(min(max_samples, len(ds))))
    
    correct = 0
    for i, ex in enumerate(ds):
        question = ex['raw_question']
        gold = ex['raw_answer'].split('####')[-1].strip()
        
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        pred_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = last_number(pred_text)
        gold_num = last_number(gold)
        
        is_correct = (pred is not None and gold_num is not None and pred == gold_num)
        correct += int(is_correct)
        
        print(f"\nExample {i+1}:")
        print(f"Q: {question[:80]}...")
        print(f"Gold: {gold} (num: {gold_num})")
        print(f"Pred text: {pred_text[:80]}")
        print(f"Pred num: {pred}")
        print(f"Correct: {is_correct}")
    
    acc = correct / len(ds)
    print(f"\n{'='*60}")
    print(f"GSM8K Single Patch Accuracy: {acc:.2%} ({correct}/{len(ds)})")
    print(f"{'='*60}")
    return acc

def test_humaneval_patch(patch_path, max_samples=3):
    """Test HumanEval/CodeXGLUE Patch"""
    print("\n" + "="*60)
    print("Testing HumanEval/CodeXGLUE Patch (BEFORE merge)")
    print("="*60)
    
    # Load patch
    with open(patch_path, 'r') as f:
        patch = json.load(f)
    
    print(f"Patch contains {len(patch['keys'])} slots")
    
    # Create memory
    keys = torch.tensor(patch['keys'])
    vals = torch.tensor(patch['vals'])
    d_model = keys.shape[1]
    num_slots = keys.shape[0]
    
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=4, alpha=1.0)
    with torch.no_grad():
        mem.keys.data[:] = keys
        mem.vals.data[:] = vals
    
    # Create model
    model = MemoryAugmentedCausalLM("gpt2", mem)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Keys norm: {mem.keys.norm().item():.4f}")
    print(f"Vals norm: {mem.vals.norm().item():.4f}")
    print(f"Non-zero vals: {(mem.vals.abs() > 1e-6).sum().item()}/{mem.vals.numel()}")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load data
    ds = load_humaneval(split='test')
    ds = ds.select(range(min(max_samples, len(ds))))
    
    for i, ex in enumerate(ds):
        task_id = ex.get('task_id', f'HumanEval/{i}')
        prompt = ex['prompt']
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\nExample {i+1} ({task_id}):")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Generated: {completion[:150]}")
    
    print(f"\n{'='*60}")
    print(f"Note: Full evaluation requires running test cases")
    print(f"{'='*60}")
    return None

def test_finqa_patch(patch_path, max_samples=3):
    """Test FinQA task patch"""
    print("\n" + "="*60)
    print("Testing FinQA Patch (BEFORE merge)")
    print("="*60)
    
    # Load patch
    with open(patch_path, 'r') as f:
        patch = json.load(f)
    
    print(f"Patch contains {len(patch['keys'])} slots")
    
    # Create memory
    keys = torch.tensor(patch['keys'])
    vals = torch.tensor(patch['vals'])
    d_model = keys.shape[1]
    num_slots = keys.shape[0]
    
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=4, alpha=1.0)
    with torch.no_grad():
        mem.keys.data[:] = keys
        mem.vals.data[:] = vals
    
    # Create model
    model = MemoryAugmentedCausalLM("gpt2", mem)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Keys norm: {mem.keys.norm().item():.4f}")
    print(f"Vals norm: {mem.vals.norm().item():.4f}")
    print(f"Non-zero vals: {(mem.vals.abs() > 1e-6).sum().item()}/{mem.vals.numel()}")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load data
    ds = load_finqa(split='dev')
    ds = ds.select(range(min(max_samples, len(ds))))
    
    correct = 0
    for i, ex in enumerate(ds):
        question = ex['qa']['question']
        gold = str(ex['qa']['answer'])
        
        # Build prompt with context
        pre = ' '.join(ex.get('pre_text', []))
        table = str(ex.get('table', ''))
        post = ' '.join(ex.get('post_text', []))
        context = f"{pre}\n{table}\n{post}".strip()[:1000]
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        pred_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = last_number(pred_text)
        gold_num = last_number(gold)
        
        is_correct = (pred is not None and gold_num is not None and pred == gold_num)
        correct += int(is_correct)
        
        print(f"\nExample {i+1}:")
        print(f"Q: {question[:80]}...")
        print(f"Gold: {gold} (num: {gold_num})")
        print(f"Pred text: {pred_text[:80]}")
        print(f"Pred num: {pred}")
        print(f"Correct: {is_correct}")
    
    acc = correct / len(ds)
    print(f"\n{'='*60}")
    print(f"FinQA Single Patch Accuracy: {acc:.2%} ({correct}/{len(ds)})")
    print(f"{'='*60}")
    return acc

if __name__ == "__main__":
    # Test each patch individually BEFORE merge
    test_gsm8k_patch("out/math/patch_gsm8k.json", max_samples=3)
    test_humaneval_patch("out/code/patch_codexglue.json", max_samples=3)
    test_finqa_patch("out/finqa/patch_finqa.json", max_samples=3)
