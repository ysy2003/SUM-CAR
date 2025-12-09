"""
Evaluate memory-augmented model on MBPP test set with pass@1.
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

from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
from src.sumcar.utils.sandbox import safe_exec


def load_merged_model(base_model, merged_dir, k_top=64, alpha=1.0, use_fp16=False, memory_position='middle'):
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

    if use_fp16:
        mem = mem.half()

    model = MemoryAugmentedCausalLM(base_model, mem, use_fp16=use_fp16, memory_position=memory_position)
    model = model.to(device)
    precision = "FP16" if use_fp16 else "FP32"
    print(f"  Loaded with {precision} precision, memory_position={memory_position}")
    return model


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


def main(base_model='meta-llama/Meta-Llama-3-8B-Instruct',
         merged_dir='noLoRA/code_only/merged',
         out='noLoRA/MBPP_test/code_only_mbpp.json',
         k_top=64,
         alpha=1.0,
         use_fp16=False,
         mode='full',
         memory_position='middle'):
    """
    Evaluate memory-augmented model on MBPP test set pass@1.

    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt
        out: Output JSON path
        k_top: Top-k for memory retrieval
        alpha: Alpha parameter
        use_fp16: Use FP16 precision
        mode: Number of samples or 'full' (257 total)
        memory_position: 'embedding' or 'middle' (after layer 16)
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 257

    print(f"=== MBPP pass@1 Evaluation ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Mode: {mode} ({max_samples} samples)")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Memory position: {memory_position}")
    print()

    # Load model
    print("Loading merged model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha, use_fp16=use_fp16, memory_position=memory_position)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    print("\n" + "="*50)
    print("MBPP Evaluation")
    print("="*50)

    results = eval_mbpp(model, tokenizer, max_samples=max_samples)

    # Save results
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'mbpp': results,
        'config': {
            'merged_dir': merged_dir,
            'k_top': k_top,
            'alpha': alpha,
            'use_fp16': use_fp16,
            'max_samples': max_samples,
            'memory_position': memory_position
        }
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"MBPP pass@1: {results['pass_at_1']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
