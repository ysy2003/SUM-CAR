"""
Cross-task evaluation: Test code-trained memory (MBPP) on FinQA task.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
from src.sumcar.eval.metrics import acc_numeric_tolerant


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


@torch.no_grad()
def eval_finqa(model, tokenizer, max_samples=99999, use_cot=True):
    """Evaluate on FinQA validation set."""
    device = next(model.parameters()).device

    from src.sumcar.data.finqa_rc import load as load_finqa
    ds = load_finqa(split='dev', use_rc_filter=False)

    if hasattr(ds, 'select'):
        end_idx = min(max_samples, len(ds))
        ds_list = [ds[i] for i in range(end_idx)]
    else:
        ds_list = list(ds)[:max_samples]

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""

    print(f"\n  Testing {len(ds_list)} FinQA samples{cot_str}...")

    for ex in tqdm(ds_list, desc="  FinQA", unit="sample"):
        ctx = ex.get('context', '')
        q = ex.get('question', '')
        gold = ex.get('answer', '')

        if use_cot:
            prompt = (
                f"Answer the question using ONLY the given context.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {q}\n\n"
                f"Think step by step, then provide your final numeric answer."
            )
        else:
            prompt = (
                f"Answer the question using ONLY the given context.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {q}\n"
                f"Answer:"
            )

        messages = [{"role": "user", "content": prompt}]
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
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

            is_correct = acc_numeric_tolerant(pred, gold) > 0
            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                'question': q,
                'gold': gold,
                'pred': pred,
                'correct': is_correct
            })
        except Exception as e:
            print(f"  Warning: Skipped sample due to error: {e}")
            continue

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'predictions': predictions
    }


def main(base_model='meta-llama/Meta-Llama-3-8B-Instruct',
         merged_dir='noLoRA/code_only/merged',
         out='noLoRA/code_only/eval/finqa_cross_results.json',
         k_top=64,
         alpha=1.0,
         use_cot=True,
         use_fp16=False,
         mode='100',
         memory_position='middle'):
    """
    Cross-task evaluation: Test code-trained memory on FinQA.
    """
    if str(mode) == 'full':
        max_samples = 99999
    else:
        try:
            max_samples = int(mode)
        except ValueError:
            max_samples = 100

    print(f"=== Cross-Task Evaluation: Code Memory -> FinQA ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Mode: {mode} ({max_samples} samples)")
    print(f"Use CoT: {use_cot}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Memory position: {memory_position}")
    print()

    print("Loading code-trained memory model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha, use_fp16=use_fp16, memory_position=memory_position)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    print("\n" + "="*50)
    print("FinQA Evaluation (using code-trained memory)")
    print("="*50)

    results = eval_finqa(model, tokenizer, max_samples=max_samples, use_cot=use_cot)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    output_data = {
        'finqa': results,
        'source_task': 'mbpp',
        'config': {
            'merged_dir': merged_dir,
            'use_cot': use_cot,
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
    print(f"Source: Code-trained memory (MBPP)")
    print(f"Target: FinQA")
    print(f"FinQA Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
