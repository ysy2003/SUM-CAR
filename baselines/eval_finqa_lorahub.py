"""
Evaluate LoRAHub fused Flan-T5 model on FinQA RC subset.

用法示例：
PYTHONPATH=. python noLoRA/eval_finqa_lorahub.py \
  --model_dir lorahub_flan_t5_finqa \
  --out analysis/lorahub_flan_t5_finqa_dev.json \
  --split dev \
  --max_samples 200 \
  --use_cot True
"""
import os
import sys
import json
import re
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 让 src 可 import（假设本脚本在项目根的 noLoRA/ 目录下）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant
from src.sumcar.data import finqa_rc


def extract_last_number_or_percent(text: str) -> str:
    """
    从输出中提取最后一个数字（可带小数和 %）。
    若找不到数字，就返回原始字符串（去掉首尾空格）。
    例：
      "The answer is 127.40." -> "127.40"
      "93.5% is the return." -> "93.5%"
      "about 20 percent" -> "20"
    """
    cleaned = text.replace(",", " ")
    # 先找带 % 的数字
    percent_pattern = r"-?\d+(?:\.\d+)?%"
    m_all = re.findall(percent_pattern, cleaned)
    if m_all:
        return m_all[-1]

    # 再找纯数字
    num_pattern = r"-?\d+(?:\.\d+)?"
    m_all = re.findall(num_pattern, cleaned)
    if m_all:
        return m_all[-1]

    return text.strip()


@torch.no_grad()
def eval_finqa_t5(
    model,
    tokenizer,
    split: str = "dev",
    max_samples: int = 99999,
    use_cot: bool = True,
    max_new_tokens: int = 256,
):
    """用 Seq2Seq (Flan-T5) 在 FinQA 上评测。"""
    device = next(model.parameters()).device

    ds = finqa_rc.load(split=split, use_rc_filter=False, use_cot=use_cot)
    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    print(
        f"\nTesting {len(ds_subset)} FinQA {split} samples"
        f"{' with CoT' if use_cot else ''}..."
    )

    total, correct = 0, 0
    predictions = []

    for ex in tqdm(ds_subset, desc=f"FinQA-{split}", unit="sample"):
        ctx = ex.get("context", "")
        q = ex.get("question", "")

        # 指令式 prompt，强制只输出一个数
        prompt = (
            "You are a financial question answering assistant.\n"
            "Read the following context and answer the question with a single numeric value.\n"
            "If the answer is a percentage, include the % sign.\n"
            "Do not provide any explanation.\n\n"
            f"Context:\n{ctx}\n\n"
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
        pred = extract_last_number_or_percent(raw_pred)

        gold = str(ex.get("answer", "")).strip()
        is_correct = acc_numeric(pred, gold) or acc_numeric_tolerant(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        predictions.append(
            {
                "uid": ex.get("uid", ""),
                "question": ex.get("question", ""),
                "context": ex.get("context", ""),
                "gold": gold,
                "raw_pred": raw_pred,
                "pred": pred,
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
    split: str = "dev",
    max_samples: int = 99999,
    use_cot: bool = True,
    max_new_tokens: int = 256,
):
    """
    Args:
        model_dir: LoRAHub fused T5 模型目录（save_pretrained 的目录，或 HF 模型名）
        out: 评测结果 JSON 输出路径
        split: 'dev' or 'test'
        max_samples: 最多评多少条
        use_cot: 只是传给 finqa_rc.load，影响上下文构造
        max_new_tokens: 生成最大 token 数
    """
    print("=== FinQA LoRAHub–Flan-T5 Evaluation ===")
    print(f"Model dir: {model_dir}")
    print(f"Split: {split}")
    print(f"Max samples: {max_samples}")
    print(f"Use CoT: {use_cot}")
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

    results = eval_finqa_t5(
        model,
        tokenizer,
        split=split,
        max_samples=max_samples,
        use_cot=use_cot,
        max_new_tokens=max_new_tokens,
    )

    if out:
        output = {
            "finqa": results,
            "split": split,
            "model_dir": model_dir,
            "use_cot": use_cot,
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
        f"FinQA ({split}) Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )


if __name__ == "__main__":
    fire.Fire(main)
