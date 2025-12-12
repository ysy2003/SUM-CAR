"""
Evaluate memory-augmented model on FinQA (finance) only.
Supports checkpoint at 100 samples and resume from checkpoint.
"""
import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add project root so `src` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant
from src.sumcar.data import finqa_rc


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
def eval_finqa(model,
               tokenizer,
               split: str = "dev",
               max_samples: int = 99999,
               use_cot: bool = True,
               skip_samples: int = 0,
               checkpoint_at: int = 0,
               checkpoint_path: str | None = None):
    """Evaluate on FinQA split with checkpoint support."""
    device = next(model.parameters()).device

    # 使用你写的 loader
    ds = finqa_rc.load(split=split, use_rc_filter=False, use_cot=use_cot)

    # Determine range to evaluate
    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(skip_samples, end_idx))

    total, correct = 0, 0
    predictions = []
    cot_str = " with CoT" if use_cot else ""

    if skip_samples > 0:
        print(f"\n  Resuming from sample {skip_samples + 1}, evaluating {len(ds_subset)} samples{cot_str}...")
    else:
        print(f"\n  Testing {len(ds_subset)} FinQA {split} samples{cot_str}...")

    for i, ex in enumerate(tqdm(ds_subset, desc=f"  FinQA-{split}", unit="sample", total=len(ds_subset))):
        actual_idx = skip_samples + i  # index in full dataset

        # prompt 已经在 finqa_rc 里构造好
        prompt = ex.get("prompt")
        if not prompt:  # fallback，理论上不会走到
            ctx = ex.get("context", "")
            q = ex.get("question", "")
            prompt = f"Context:\n{ctx}\n\nQuestion: {q}\n\nProvide your final numeric answer in the last sentence."

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

        # Gold answer：finqa_rc 里是纯数字字符串
        gold = str(ex.get("answer", "")).strip()

        # Check correctness (numeric EM / tolerant)
        is_correct = acc_numeric(pred, gold) or acc_numeric_tolerant(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "uid": ex.get("uid", ""),
            "question": ex.get("question", ""),
            "context": ex.get("context", ""),
            "gold": gold,
            "pred": pred,
            "correct": is_correct
        })

        # Save checkpoint at specified sample count (only if starting from 0)
        if checkpoint_at > 0 and skip_samples == 0 and (actual_idx + 1) == checkpoint_at and checkpoint_path:
            checkpoint_data = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0,
                "predictions": predictions.copy()
            }
            with open(checkpoint_path, "w") as f:
                json.dump({"finqa": checkpoint_data, "checkpoint_at": checkpoint_at, "split": split}, f, indent=2)
            print(f"\n  + Checkpoint saved at {checkpoint_at} samples: {checkpoint_path}")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
        "predictions": predictions
    }


def main(base_model: str,
         merged_dir: str,
         out: str,
         k_top: int = 64,
         alpha: float = 1.0,
         use_cot: bool = True,
         use_fp16: bool = False,
         mode: str = "full",   # '100' or 'full'
         memory_position: str = "middle",
         split: str = "dev"):
    """
    Evaluate memory-augmented model on FinQA only.

    Args:
        base_model: Base model name
        merged_dir: Directory with merged memory.pt
        out: Output JSON path
        k_top: Top-k for memory retrieval
        alpha: Alpha parameter
        use_cot: Use Chain-of-Thought prompting
        use_fp16: Use FP16 precision
        mode: '100' for first 100 only, 'full' for full dataset (with checkpoint resume)
        memory_position: 'embedding' or 'middle' (after some layer)
        split: 'dev' or 'test'
    """
    print(f"=== FinQA Evaluation ===")
    print(f"Base model: {base_model}")
    print(f"Merged dir: {merged_dir}")
    print(f"k_top: {k_top}, alpha: {alpha}")
    print(f"Mode: {mode}")
    print(f"Use CoT: {use_cot}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Memory position: {memory_position}")
    print(f"Split: {split}")
    print()

    # Determine checkpoint path
    checkpoint_path = out.replace(".json", f"_checkpoint_100_{split}.json")

    # Load model
    print("Loading merged model...")
    model = load_merged_model(base_model, merged_dir,
                              k_top=k_top, alpha=alpha,
                              use_fp16=use_fp16,
                              memory_position=memory_position)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    print("\n" + "=" * 50)
    print(f"FinQA ({split})")
    print("=" * 50)

    # Run evaluation
    if str(mode) == "100":
        # Run first 100 only
        print("Running first 100 samples only...")
        final_results = eval_finqa(
            model,
            tokenizer,
            split=split,
            max_samples=100,
            use_cot=use_cot,
            skip_samples=0,
            checkpoint_at=0,
            checkpoint_path=None
        )
    else:  # mode == 'full'
        # Check for existing checkpoint
        if os.path.exists(checkpoint_path):
            print(f"+ Found checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            ckpt_results = checkpoint_data["finqa"]
            print(f"  Checkpoint has {ckpt_results['total']} samples, accuracy: {ckpt_results['accuracy']:.4f}")
            print(f"  Resuming from sample 101...")

            # Run from 101 onwards
            remaining_results = eval_finqa(
                model,
                tokenizer,
                split=split,
                max_samples=99999,
                use_cot=use_cot,
                skip_samples=100,
                checkpoint_at=0,
                checkpoint_path=None
            )

            # Merge results
            final_results = {
                "correct": ckpt_results["correct"] + remaining_results["correct"],
                "total": ckpt_results["total"] + remaining_results["total"],
                "predictions": ckpt_results["predictions"] + remaining_results["predictions"]
            }
            final_results["accuracy"] = final_results["correct"] / final_results["total"]
        else:
            print("No checkpoint found, running full evaluation (with checkpoint at 100)...")
            final_results = eval_finqa(
                model,
                tokenizer,
                split=split,
                max_samples=99999,
                use_cot=use_cot,
                skip_samples=0,
                checkpoint_at=100,
                checkpoint_path=checkpoint_path
            )

    # Save results
    if out:
        output_data = {
            "finqa": final_results,
            "split": split,
            "mode": str(mode),
            "k_top": k_top,
            "alpha": alpha,
            "use_cot": use_cot,
            "use_fp16": use_fp16,
            "memory_position": memory_position,
            "base_model": base_model,
            "merged_dir": merged_dir
        }
        with open(out, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n+ Results saved to: {out}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"FinQA ({split}) Accuracy: {final_results['accuracy']:.4f} "
          f"({final_results['correct']}/{final_results['total']})")


if __name__ == "__main__":
    fire.Fire(main)
