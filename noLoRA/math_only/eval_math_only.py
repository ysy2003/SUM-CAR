"""
Evaluate memory-augmented model on GSM8K (math) only.
Supports checkpoint at 100 samples and resume from checkpoint.
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
from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant


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
def eval_gsm8k(model, tokenizer, max_samples=99999, use_cot=True, skip_samples=0, checkpoint_at=0, checkpoint_path=None):
    """Evaluate on GSM8K test set with checkpoint support."""
    device = next(model.parameters()).device
    ds = load_dataset('gsm8k', 'main')['test']

    # Determine range to evaluate
    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(skip_samples, end_idx))

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""

    if skip_samples > 0:
        print(f"\n  Resuming from sample {skip_samples + 1}, evaluating {len(ds_subset)} samples{cot_str}...")
    else:
        print(f"\n  Testing {len(ds_subset)} GSM8K samples{cot_str}...")

    for i, ex in enumerate(tqdm(ds_subset, desc="  GSM8K", unit="sample", total=len(ds_subset))):
        actual_idx = skip_samples + i  # Track actual index in full dataset

        if use_cot:
            prompt = f"Question: {ex['question']}\n\nThink step by step, then provide your final numeric answer in the last sentence."
        else:
            prompt = f"Question: {ex['question']}\n\nProvide your final numeric answer in the last sentence."

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tokenizer([text], return_tensors='pt').to(device)

        max_tokens = 4096
        input_length = enc['input_ids'].shape[1]
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode only generated part
        gen_ids = out_ids[0, input_length:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract gold answer
        answer_text = ex['answer']
        gold = answer_text.split('####')[-1].strip() if '####' in answer_text else answer_text.strip()

        # Check correctness
        is_correct = acc_numeric(pred, gold) or acc_numeric_tolerant(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            'question': ex['question'],
            'gold': gold,
            'pred': pred,
            'correct': is_correct
        })

        # Save checkpoint at specified sample count (only if starting from 0)
        if checkpoint_at > 0 and skip_samples == 0 and (actual_idx + 1) == checkpoint_at and checkpoint_path:
            checkpoint_data = {
                'correct': correct,
                'total': total,
                'accuracy': correct / total if total > 0 else 0,
                'predictions': predictions.copy()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump({'gsm8k': checkpoint_data, 'checkpoint_at': checkpoint_at}, f, indent=2)
            print(f"\n  + Checkpoint saved at {checkpoint_at} samples: {checkpoint_path}")

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'predictions': predictions
    }


def main(base_model='meta-llama/Meta-Llama-3-8B-Instruct',
         merged_dir='noLoRA/math_only/merged',
         out='noLoRA/math_only/eval/math_results_cot.json',
         k_top=64,
         alpha=1.0,
         use_cot=True,
         use_fp16=False,
         mode='full',
         memory_position='middle'):
    """
    Evaluate memory-augmented model on GSM8K only.

    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt
        out: Output JSON path
        k_top: Top-k for memory retrieval
        alpha: Alpha parameter
        use_cot: Use Chain-of-Thought prompting
        use_fp16: Use FP16 precision
        mode: '100' for first 100 only, 'full' for full dataset (with checkpoint resume)
        memory_position: 'embedding' or 'middle' (after layer 16)
    """
    print(f"=== GSM8K (Math) Evaluation ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Mode: {mode}")
    print(f"Use CoT: {use_cot}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Memory position: {memory_position}")
    print()

    # Determine checkpoint path
    checkpoint_path = out.replace('.json', '_checkpoint_100.json')

    # Load model
    print("Loading merged model...")
    model = load_merged_model(base_model, merged_dir, k_top=k_top, alpha=alpha, use_fp16=use_fp16, memory_position=memory_position)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    print("\n" + "="*50)
    print("GSM8K (Math)")
    print("="*50)

    if str(mode) == '100':
        # Run first 100 only
        print("Running first 100 samples only...")
        results = eval_gsm8k(model, tokenizer, max_samples=100, use_cot=use_cot)

        # Save as checkpoint
        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
        output_data = {
            'gsm8k': results,
            'checkpoint_at': 100,
            'config': {'use_cot': use_cot, 'k_top': k_top, 'alpha': alpha, 'use_fp16': use_fp16}
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n+ Results saved to: {checkpoint_path}")
        final_results = results

    else:  # mode == 'full'
        # Check for existing checkpoint
        if os.path.exists(checkpoint_path):
            print(f"+ Found checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            ckpt_results = checkpoint_data['gsm8k']
            print(f"  Checkpoint has {ckpt_results['total']} samples, accuracy: {ckpt_results['accuracy']:.4f}")
            print(f"  Resuming from sample 101...")

            # Run from 101 onwards
            remaining_results = eval_gsm8k(model, tokenizer, max_samples=99999, use_cot=use_cot, skip_samples=100)

            # Merge results
            final_results = {
                'correct': ckpt_results['correct'] + remaining_results['correct'],
                'total': ckpt_results['total'] + remaining_results['total'],
                'predictions': ckpt_results['predictions'] + remaining_results['predictions']
            }
            final_results['accuracy'] = final_results['correct'] / final_results['total']

        else:
            print("No checkpoint found, running full evaluation...")
            print("Will save checkpoint at 100 samples...")

            # Run full with checkpoint saving at 100
            final_results = eval_gsm8k(model, tokenizer, max_samples=99999, use_cot=use_cot,
                                       checkpoint_at=100, checkpoint_path=checkpoint_path)

        # Save full results
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        output_data = {
            'gsm8k': final_results,
            'config': {'use_cot': use_cot, 'k_top': k_top, 'alpha': alpha, 'use_fp16': use_fp16}
        }
        with open(out, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n+ Results saved to: {out}")

    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"GSM8K Accuracy: {final_results['accuracy']:.4f} ({final_results['correct']}/{final_results['total']})")


if __name__ == '__main__':
    fire.Fire(main)
