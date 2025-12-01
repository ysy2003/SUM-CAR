import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.build_composite_finqa import build_composite_finqa_dev
from src.sumcar.eval.eval_composite_finqa import evaluate as run_eval

def evaluate_composite_finqa(model, tokenizer, device, ground_truth_file, generations_file):
    """
    Orchestrates the composite FinQA evaluation.
    """
    # 1. Load or generate dataset
    if not Path(ground_truth_file).exists():
        print(f"{ground_truth_file} not found. Generating it now...")
        build_composite_finqa_dev()

    with open(ground_truth_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Ensure compatibility with 'gold' field
    for sample in dataset:
        if 'gold' in sample and 'gold_answer' not in sample:
            sample['gold_answer'] = sample['gold']

    # Filter out invalid gold answers
    valid_dataset = []
    for sample in dataset:
        gold_answer = sample.get('gold_answer', None)
        if gold_answer is not None:
            try:
                # Attempt to parse gold_answer as a number or valid string
                float(gold_answer.strip('%')) if isinstance(gold_answer, str) else gold_answer
                valid_dataset.append(sample)
            except ValueError:
                print(f"Skipping invalid gold answer: {gold_answer} in sample ID {sample['id']}")
        else:
            print(f"Missing gold answer in sample ID {sample['id']}")

    dataset = valid_dataset
    dataset_by_id = {sample['id']: sample for sample in dataset}

    # 2. Generate responses from model, with resume capability
    existing_generations = {}
    if Path(generations_file).exists():
        with open(generations_file, 'r') as f:
            for line in f:
                try:
                    gen = json.loads(line)
                    if 'id' in gen:
                        existing_generations[gen['id']] = gen
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line in {generations_file}: {line.strip()}")
        print(f"Loaded {len(existing_generations)} existing generations from {generations_file}.")

    samples_to_generate_ids = [sample_id for sample_id in dataset_by_id if sample_id not in existing_generations]
    
    if not samples_to_generate_ids:
        print("All generations already exist. Skipping generation.")
    else:
        print(f"Generating responses for {len(samples_to_generate_ids)} new samples...")
        Path(generations_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(generations_file, 'a') as f_out:
            for sample_id in tqdm(samples_to_generate_ids, desc="Processing samples"):
                sample = dataset_by_id[sample_id]
                prompt = sample['prompt']

                # Define generation parameters
                max_new_tokens = 512
                # Some models like gpt2 have a small context window, so we need to be careful
                model_max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 1024
            
                # Truncate prompt to leave space for new tokens
                max_prompt_len = model_max_length - max_new_tokens
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_prompt_len
                ).to(device)
                input_len = inputs.input_ids.shape[1]

                # Generate output
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

                # This is for decoder-only models.
                new_tokens = outputs[0, input_len:]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

                result = {
                    "id": sample['id'],
                    "prompt": prompt,
                    "generation": completion,
                }
                f_out.write(json.dumps(result) + '\n')

        print(f"\nGenerations saved to {generations_file}")

    # 3. Run evaluation script
    print("Running evaluation...")
    
    results_file = Path(generations_file).parent / 'eval_results.metrics.json'
    run_eval(generations_file, ground_truth_file, str(results_file))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gpt2", help="Model to evaluate")
    parser.add_argument("--ground_truth_file", default="out/finqa/finqa_composite_dev.jsonl")
    parser.add_argument("--generations_file", default="out/finqa/generations.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    evaluate_composite_finqa(
        model, 
        tokenizer, 
        device,
        args.ground_truth_file, 
        args.generations_file
    )