"""
Compare base model vs memory-augmented model outputs on GSM8K examples.
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.sumcar.memory.kv_memory import KVMemoryLayer
from src.sumcar.models.base_model import MemoryAugmentedCausalLM
from src.sumcar.eval.metrics import acc_numeric, acc_numeric_tolerant


def load_finetuned_model(base_model, merged_dir, k_top=8, alpha=1.0, memory_position='middle'):
    """Load memory-augmented model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(os.path.join(merged_dir, 'memory.pt'), map_location=device, weights_only=False)

    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=state['keys'].shape[0], k_top=k_top, alpha=alpha)

    with torch.no_grad():
        mem.keys.data[:] = state['keys']
        mem.vals.data[:] = state['vals']

    model = MemoryAugmentedCausalLM(base_model, mem, use_fp16=False, memory_position=memory_position)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tokenizer, question, use_cot=True):
    """Generate answer for a question."""
    device = next(model.parameters()).device

    if use_cot:
        prompt = f"Question: {question}\n\nThink step by step, then provide your final numeric answer in the last sentence."
    else:
        prompt = f"Question: {question}\n\nProvide your final numeric answer in the last sentence."

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([text], return_tensors='pt').to(device)

    input_length = enc['input_ids'].shape[1]
    out_ids = model.generate(
        **enc,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    gen_ids = out_ids[0, input_length:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main(num_examples=3, merged_dir='noLoRA/merged'):
    base_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K test set
    ds = load_dataset('gsm8k', 'main')['test']

    # Load base model
    print("Loading base model...")
    base_lm = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    base_lm.eval()
    print("Base model loaded.\n")

    # Check if finetuned model exists
    finetuned_model = None
    if os.path.exists(os.path.join(merged_dir, 'memory.pt')):
        print("Loading finetuned (memory-augmented) model...")
        finetuned_model = load_finetuned_model(base_model, merged_dir)
        print("Finetuned model loaded.\n")
    else:
        print(f"No finetuned model found at {merged_dir}/memory.pt\n")

    # Compare outputs
    print("=" * 80)
    print("COMPARISON: Base Model vs Finetuned Model")
    print("=" * 80)

    for i in range(min(num_examples, len(ds))):
        ex = ds[i]
        question = ex['question']
        gold = ex['answer'].split('####')[-1].strip() if '####' in ex['answer'] else ex['answer'].strip()

        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"\nQUESTION:\n{question}")
        print(f"\nGOLD ANSWER: {gold}")

        # Base model
        print(f"\n{'-'*40}")
        print("BASE MODEL OUTPUT:")
        print(f"{'-'*40}")
        base_output = generate_answer(base_lm, tokenizer, question)
        print(base_output[:1000] + "..." if len(base_output) > 1000 else base_output)
        base_correct = acc_numeric(base_output, gold) or acc_numeric_tolerant(base_output, gold)
        print(f"\n[Base model correct: {bool(base_correct)}]")

        # Finetuned model
        if finetuned_model:
            print(f"\n{'-'*40}")
            print("FINETUNED MODEL OUTPUT:")
            print(f"{'-'*40}")
            ft_output = generate_answer(finetuned_model, tokenizer, question)
            print(ft_output[:1000] + "..." if len(ft_output) > 1000 else ft_output)
            ft_correct = acc_numeric(ft_output, gold) or acc_numeric_tolerant(ft_output, gold)
            print(f"\n[Finetuned model correct: {bool(ft_correct)}]")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
