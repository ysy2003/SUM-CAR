"""
Cross-task evaluation: Test specified model on HumanEval (code) test set.
Default: math-trained memory model.
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


def load_merged_model(base_model, merged_dir, k_top=8, alpha=1.0, use_fp16=True, memory_position='embedding'):
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


def main(base_model='meta-llama/Meta-Llama-3-8B-Instruct',
         merged_dir='noLoRA/math_only/acc_72%/merged',
         out='noLoRA/math_only/eval/humaneval_cross_results.json',
         k_top=64,
         alpha=1.0,
         use_fp16=False,
         mode='100',
         memory_position='middle'):
    """
    Cross-task evaluation: Test model on HumanEval (code) test set.

    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt (default: math_only)
        out: Output JSON path
        k_top: Top-k for memory retrieval
        alpha: Alpha parameter
        use_fp16: Use FP16 precision
        mode: Number of samples or 'full' (164 total)
        memory_position: 'embedding' or 'middle' (after layer 16)
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 100

    # Determine source task from merged_dir path
    if 'math_only' in merged_dir:
        source_task = 'gsm8k (math)'
    elif 'code_only' in merged_dir:
        source_task = 'mbpp (code)'
    else:
        source_task = 'unknown'

    print(f"=== Cross-Task Evaluation: {source_task} -> HumanEval ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Mode: {mode} ({max_samples} samples)")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Memory position: {memory_position}")
    print()

    print("Loading memory-augmented model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha, use_fp16=use_fp16, memory_position=memory_position)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    print("\n" + "="*50)
    print("HumanEval Evaluation")
    print("="*50)

    results = eval_humaneval(model, tokenizer, max_samples=max_samples)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'humaneval': results,
        'source_task': source_task,
        'config': {
            'merged_dir': merged_dir,
            'k_top': k_top,
            'alpha': alpha,
            'use_fp16': use_fp16,
            'memory_position': memory_position
        }
    }
    with open(out, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Source: {source_task}")
    print(f"Target: HumanEval (code)")
    print(f"HumanEval pass@1: {results['pass_at_1']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
