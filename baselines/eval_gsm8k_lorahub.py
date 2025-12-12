import os
import sys
import json
import re
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant


def extract_last_number(text: str) -> str:
    """
    从模型输出中提取最后一个数字（整数或小数）。
    如果找不到数字，就返回原始字符串（去掉首尾空格）。
    例子：
      "4000 lbs" -> "4000"
      "Answer is 125.5 dollars." -> "125.5"
      "no number" -> "no number"
    """
    # 去掉逗号对数字的干扰：1,200 -> 1 200
    cleaned = text.replace(",", " ")
    # 匹配整数 / 小数：-12, 3.14 之类
    pattern = r"-?\d+(?:\.\d+)?"
    matches = re.findall(pattern, cleaned)
    if not matches:
        return text.strip()
    return matches[-1]


@torch.no_grad()
def eval_gsm8k_t5(
    model,
    tokenizer,
    split: str = "test",
    max_samples: int = 99999,
    max_new_tokens: int = 256,
):
    device = next(model.parameters()).device

    # 统一处理 split 名，gsm8k 只有 "train" 和 "test"
    ds_all = load_dataset("gsm8k", "main")
    if split in ("dev", "val", "validation"):
        split_key = "test"
    else:
        split_key = split

    if split_key not in ds_all:
        raise ValueError(f"gsm8k only has splits: {list(ds_all.keys())}, got '{split}'")

    ds = ds_all[split_key]

    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    print(f"\nTesting {len(ds_subset)} GSM8K {split_key} samples...")

    total, correct = 0, 0
    predictions = []

    for ex in tqdm(ds_subset, desc=f"GSM8K-{split_key}", unit="sample"):
        q = ex["question"]
        answer_text = ex["answer"]
        gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()

        prompt = (
            "You are a math reasoning assistant.\n"
            "Read the following question, reason step by step, and provide the final numeric answer.\n"
            "Output only the final number (no units, no words).\n\n"
            f"Question: {q}\n\n"
            "Answer:"
        )

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        raw_pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        pred = extract_last_number(raw_pred)

        is_correct = acc_numeric(pred, gold) or acc_numeric_tolerant(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        predictions.append(
            {
                "question": q,
                "gold": gold,
                "raw_pred": raw_pred,
                "pred": pred,      # 清洗后用于打分的数字
                "correct": is_correct,
            }
        )

    acc = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "predictions": predictions,
    }


def main(
    model_dir: str,
    out: str,
    split: str = "test",
    max_samples: int = 99999,
    max_new_tokens: int = 1024,
):
    print("=== GSM8K LoRAHub–Flan-T5 Evaluation ===")
    print(f"Model dir: {model_dir}")
    print(f"Requested split: {split}")
    print(f"Max samples: {max_samples}")
    print(f"max_new_tokens: {max_new_tokens}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading fused LoRAHub model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("+ Model loaded")

    results = eval_gsm8k_t5(
        model,
        tokenizer,
        split=split,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
    )

    if out:
        output = {
            "gsm8k": results,
            "split": split,
            "model_dir": model_dir,
            "max_new_tokens": max_new_tokens,
        }
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n+ Results saved to: {out}")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(
        f"GSM8K ({split}) Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )


if __name__ == "__main__":
    fire.Fire(main)
