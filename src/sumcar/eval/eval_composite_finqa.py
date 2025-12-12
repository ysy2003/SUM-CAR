import json
import re
import argparse
import os
import sys
from pathlib import Path

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.sumcar.utils.sandbox import safe_exec

# Regex to find the python code block
CODE_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def parse_answer_from_text(generation: str) -> float | None:
    """
    Parses the final numerical answer from the generation text.
    It assumes the last number in the text is the answer.
    """
    # Find all numbers (including those with commas) in the generation
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", generation)
    if not numbers:
        return None
    try:
        # Take the last number found and convert to float
        return float(numbers[-1].replace(",", ""))
    except (ValueError, TypeError):
        return None

def extract_code(generation: str) -> str | None:
    """Extracts the python code block."""
    match = CODE_RE.search(generation)
    return match.group(1) if match else None

def compare_floats(a: float | None, b: float | None, tol=1e-2) -> bool:
    """Compare two floats with a tolerance."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

def check_one(generation: str, gold_answer: float):
    """Performs the two-part check on a single generation."""
    
    # 1. Answer check from text
    predicted_answer = parse_answer_from_text(generation)
    answer_correct = compare_floats(predicted_answer, gold_answer)

    # 2. Code execution check
    code_result = None
    code_correct = False
    code_block = extract_code(generation)

    if code_block and "compute_answer" in code_block:
        # Append print statement to get the result from stdout
        exec_code = code_block + "\nprint(compute_answer())"
        
        result = safe_exec(exec_code)
        
        if result.ok and result.stdout:
            try:
                # The result is printed to stdout, parse it
                code_result = float(result.stdout.strip())
                code_correct = compare_floats(code_result, gold_answer)
            except (ValueError, TypeError):
                # The stdout was not a valid float
                pass

    joint_success = answer_correct and code_correct
    
    return {
        "answer_correct": answer_correct,
        "code_correct": code_correct,
        "joint_success": joint_success,
        "predicted_answer": predicted_answer,
        "code_result": code_result,
        "gold_answer": gold_answer,
    }

def evaluate(generations_file: str, ground_truth_file: str, results_file: str):
    """
    Evaluates model generations against a ground truth file using a two-part check.
    """
    # Load ground truth
    ground_truth_lines = open(ground_truth_file).readlines()
    ground_truth = {json.loads(line)['id']: json.loads(line) for line in ground_truth_lines}
    print(f"Loaded {len(ground_truth)} ground truth records from {ground_truth_file}")
    ground_truth_ids = set(ground_truth.keys())

    # Process generations
    total = 0
    answer_correct_count = 0
    code_correct_count = 0
    joint_success_count = 0
    
    details = []

    skipped_missing_id_or_gen = 0
    skipped_id_not_in_gt = 0
    skipped_missing_gold = 0
    skipped_bad_gold = 0
    
    generation_lines = open(generations_file, 'r').readlines()
    print(f"Processing {len(generation_lines)} generated records from {generations_file}")

    from tqdm import tqdm
    with open(results_file, 'w') as f_res:
        for i, line in enumerate(tqdm(generation_lines, desc="Evaluating")):
            try:
                gen_record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i+1} in {generations_file} due to JSON decode error.")
                continue

            gen_id = gen_record.get("id")
            generation = gen_record.get("generation")

            if not gen_id or not generation:
                skipped_missing_id_or_gen += 1
                continue
            
            if gen_id not in ground_truth_ids:
                skipped_id_not_in_gt += 1
                # print(f"Skipping id {gen_id} not found in ground truth.") # This might be too verbose
                continue

            gt_record = ground_truth[gen_id]
            gold_answer_str = gt_record.get("gold")
            if gold_answer_str is None:
                skipped_missing_gold += 1
                continue
            
            try:
                gold_answer = float(str(gold_answer_str).replace(",", ""))
            except (ValueError, TypeError):
                skipped_bad_gold += 1
                continue
            
            total += 1
            result = check_one(generation, gold_answer)
            result['id'] = gen_id
            
            details.append(result)
            f_res.write(json.dumps(result) + '\n')

            if result["answer_correct"]:
                answer_correct_count += 1
            if result["code_correct"]:
                code_correct_count += 1
            if result["joint_success"]:
                joint_success_count += 1
    
    print("\nEvaluation complete.")
    print("--- Filter stats ---")
    print(f"Generation records processed: {len(generation_lines)}")
    print(f"Skipped (missing id or generation): {skipped_missing_id_or_gen}")
    print(f"Skipped (id not in ground truth): {skipped_id_not_in_gt}")
    print(f"Skipped (missing gold answer): {skipped_missing_gold}")
    print(f"Skipped (invalid gold answer): {skipped_bad_gold}")
    print("--------------------")

    # Print metrics
    print(f"Total evaluated: {total}")
    if total > 0:
        metrics = {
            "total": total,
            "answer_accuracy": answer_correct_count / total,
            "code_accuracy": code_correct_count / total,
            "joint_success_rate": joint_success_count / total,
        }
        print(f"Answer Accuracy: {metrics['answer_accuracy']:.4f}")
        print(f"Code Accuracy: {metrics['code_accuracy']:.4f}")
        print(f"Joint Success Rate: {metrics['joint_success_rate']:.4f}")
        
        # Save aggregate metrics
        metrics_file = Path(results_file).with_suffix('.metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("generations_file", help="Path to the generations file (.jsonl)")
    parser.add_argument("--ground_truth_file", default="out/finqa/finqa_composite_dev.jsonl", help="Path to the ground truth file (.jsonl)")
    parser.add_argument("--results_file", default="out/finqa/eval_results.jsonl", help="Path to save detailed evaluation results")
    args = parser.parse_args()
    
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)

    evaluate(args.generations_file, args.ground_truth_file, args.results_file)
