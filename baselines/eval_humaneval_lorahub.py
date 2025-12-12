"""
Evaluate LoRAHub fused Flan-T5 model on HumanEval with pass@1.

用法示例：
PYTHONPATH=. python noLoRA/eval_humaneval_lorahub.py \
  --model_dir lorahub_flan_t5_mbpp \
  --out analysis/lorahub_flan_t5_humaneval.json \
  --max_samples 50
"""
import os
import sys
import json
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# 假设本脚本在项目根的 noLoRA/ 目录下
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sumcar.utils.sandbox import safe_exec


def extract_code(text: str) -> str:
    """从输出中抽 Python 代码，兼容 markdown code fence。"""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


@torch.no_grad()
def eval_humaneval_t5(
    model,
    tokenizer,
    split: str = "test",
    max_samples: int = 99999,
    max_new_tokens: int = 256,
):
    """
    用 Seq2Seq (Flan-T5) 在 HumanEval 上评测 pass@1。
    数据：HuggingFace openai_humaneval。
    """
    device = next(model.parameters()).device

    ds = load_dataset("openai_humaneval")[split]
    end_idx = min(max_samples, len(ds))
    ds_subset = ds.select(range(end_idx))

    print(f"\nTesting {len(ds_subset)} HumanEval {split} samples (pass@1)...")

    total, correct = 0, 0
    predictions = []

    for ex in tqdm(ds_subset, desc=f"HumanEval-{split}", unit="sample"):
        task_id = ex["task_id"]
        prompt = ex["prompt"]       # 含函数签名、docstring、pass 等
        test_code = ex["test"]      # 官方测试代码字符串

        # 给 T5 的指令：补全函数实现
        t5_prompt = (
            "You are a Python coding assistant.\n"
            "Complete the following Python function implementation.\n"
            "Return only Python code, no explanations.\n\n"
            f"{prompt}\n"
        )

        enc = tokenizer(t5_prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        raw_output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        code_suffix = extract_code(raw_output)

        # HumanEval 里的 prompt 通常是函数头 + docstring + "pass"
        # 这里直接把 prompt + 生成的代码当作完整实现
        # 如果 prompt 里有 "pass"，你也可以先把 "pass" 去掉再拼接，这里简单处理：
        if "pass" in prompt:
            prefix = prompt.replace("pass", "")
        else:
            prefix = prompt
        full_code = prefix + code_suffix

        # 运行测试：candidate code + 官方 test
        exec_code = full_code + "\n\n" + test_code
        res = safe_exec(exec_code, timeout=10)

        passed = res.ok and (not res.error)

        if passed:
            correct += 1
        total += 1

        predictions.append(
            {
                "task_id": task_id,
                "prompt_head": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "generated": code_suffix[:500] + "..." if len(code_suffix) > 500 else code_suffix,
                "passed": passed,
                "error": (res.error or "")[:200],
            }
        )

    pass_at_1 = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "pass_at_1": pass_at_1,
        "predictions": predictions,
    }


def main(
    model_dir: str,
    out: str,
    split: str = "test",
    max_samples: int = 99999,
    max_new_tokens: int = 256,
):
    """
    Args:
        model_dir: LoRAHub fused T5 模型目录（一般是 MBPP 训练出来的那个）
        out: 评测结果 JSON 输出路径
        split: HumanEval 只有 'test'，保持默认即可
        max_samples: 最多评多少题
        max_new_tokens: 每题生成最大 token 数
    """
    print("=== HumanEval LoRAHub–Flan-T5 Evaluation ===")
    print(f"Model dir: {model_dir}")
    print(f"Split: {split}")
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

    results = eval_humaneval_t5(
        model,
        tokenizer,
        split=split,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
    )

    if out:
        output = {
            "humaneval": results,
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
        f"HumanEval pass@1: {results['pass_at_1']:.4f} "
        f"({results['correct']}/{results['total']})"
    )


if __name__ == "__main__":
    fire.Fire(main)
