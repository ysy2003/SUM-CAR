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

def evaluate_composite_finqa(model, tokenizer, device, ground_truth_file, generations_file, log_skipped_records=False, max_samples=None):
    """
    Orchestrates the composite FinQA evaluation.
    """
    # 1. Load or generate dataset
    if not Path(ground_truth_file).exists():
        print(f"{ground_truth_file} not found. Generating it now...")
        generated_path = build_composite_finqa_dev()
        print(f"Generated dataset at: {generated_path}")

    with open(ground_truth_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Ensure compatibility with 'gold' field
    for sample in dataset:
        if 'gold' in sample and 'gold_answer' not in sample:
            sample['gold_answer'] = sample['gold']

    # Enhanced cleaning and validation for gold answers with debug output
    def clean_gold_answer(gold_answer):
        if isinstance(gold_answer, str):
            # Convert 'yes' and 'no' to numeric values
            if gold_answer.lower() == 'yes':
                return 1.0
            elif gold_answer.lower() == 'no':
                return 0.0

            # Handle percentage values
            if gold_answer.endswith('%'):
                try:
                    return float(gold_answer.strip('%')) / 100
                except ValueError:
                    return None

            # Handle semicolon-separated values
            if ';' in gold_answer:
                gold_answer = gold_answer.split(';')[0].strip()  # Take the first value

            # Remove other unwanted characters
            gold_answer = gold_answer.replace('$', '').replace('million', '000000').replace('thousand', '000')

            # Handle multiple values (e.g., "$ 386797190 or $ 386.8 million")
            if 'or' in gold_answer:
                gold_answer = gold_answer.split('or')[0].strip()

            # Extract numeric part
            gold_answer = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', gold_answer))

        try:
            # Convert to float if possible
            return float(gold_answer)
        except ValueError:
            return None

    valid_dataset = []
    invalid_records = []  # Collect invalid records for debugging
    for sample in dataset:
        gold_answer = sample.get('gold_answer', None)
        if gold_answer is not None:
            cleaned_gold = clean_gold_answer(gold_answer)
            if cleaned_gold is not None:
                sample['gold_answer'] = cleaned_gold  # Update with cleaned value
                valid_dataset.append(sample)
            else:
                invalid_records.append({"id": sample['id'], "gold_answer": gold_answer})
        else:
            invalid_records.append({"id": sample['id'], "gold_answer": "MISSING"})

    # Print all invalid records for debugging if log_skipped_records is True
    if log_skipped_records:
        print("Invalid gold answers:")
        for record in invalid_records:
            print(record)

    dataset = valid_dataset

    # Limit the dataset if max_samples is specified
    print(f"Original dataset size: {len(dataset)}")
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"Dataset size after applying max_samples: {len(dataset)}")

    # Update dataset_by_id to reflect the limited dataset
    dataset_by_id = {sample['id']: sample for sample in dataset}

    # Initialize existing_generations before using it
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

    # Ensure samples_to_generate_ids is based on the limited dataset
    samples_to_generate_ids = [sample_id for sample_id in dataset_by_id if sample_id not in existing_generations]

    # Debugging: Print the number of samples to generate
    print(f"Samples to generate after applying max_samples: {len(samples_to_generate_ids)}")

    # 2. Generate responses from model, with resume capability
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
                # Use reasonable context length (Llama-3 has 8192, but model_max_length can be huge)
                model_max_length = min(
                    getattr(tokenizer, 'model_max_length', 8192),
                    8192  # Cap at 8192 for Llama-3
                )

                # Truncate prompt to leave space for new tokens
                max_prompt_len = model_max_length - max_new_tokens

                # Use chat template for instruct models
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt

                inputs = tokenizer(
                    formatted_prompt,
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
    parser.add_argument("--ground_truth_file", default="noLoRA/composite_eval/finqa_composite_dev_cleaned.jsonl")
    parser.add_argument("--generations_file", default="noLoRA/composite_eval/generations.jsonl")
    parser.add_argument("--log_skipped_records", action="store_true", help="Log details of skipped records")
    parser.add_argument("--mode", default="full", help="'full' or number of samples (e.g., '100')")
    args = parser.parse_args()

    # Parse mode
    if str(args.mode) == 'full':
        max_samples = None
    else:
        try:
            max_samples = int(args.mode)
        except ValueError:
            max_samples = None

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
        args.generations_file,
        log_skipped_records=args.log_skipped_records,
        max_samples=max_samples
    )