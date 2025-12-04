import torch
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Custom Dataset for Multi-Task Fine-Tuning
class CompositeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Fine-tuning function
def fine_tune(data_file, base_model='meta-llama/Meta-Llama-3-8B-Instruct', output_dir='finetuned_model', epochs=3, batch_size=1, lr=5e-5, gradient_accumulation_steps=8, use_fp16=True):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}, Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Mixed precision (fp16): {use_fp16}")

    # Load dataset
    dataset = CompositeDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model and tokenizer with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map='auto'  # Automatically handle device placement
    )

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    print(f"Model loaded to device")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()

    # Best checkpoint tracking
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            prompts = batch['prompt']
            tests = batch['tests']

            # Combine prompt and test for Causal LM fine-tuning
            full_texts = [p + t for p, t in zip(prompts, tests)]

            # Tokenize the combined texts (with max_length to avoid warning)
            inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

            # Move inputs to device (model may be on multiple devices with device_map='auto')
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Create labels by cloning input_ids
            labels = inputs['input_ids'].clone()

            # Mask the prompt part of the labels
            prompts_tokenized = tokenizer(prompts, padding=False, truncation=False)
            for i in range(len(prompts)):
                prompt_len = len(prompts_tokenized['input_ids'][i])
                labels[i, :prompt_len] = -100

            # Also mask padding tokens in labels
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps
            current_loss = loss.item() * gradient_accumulation_steps

            # Track best checkpoint
            if current_loss < best_loss:
                best_loss = current_loss
                # Save best model state (state_dict is more memory efficient than full model)
                best_model_state = {
                    'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'step': step,
                    'epoch': epoch + 1,
                    'loss': current_loss
                }

            progress_bar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{epoch_loss/(step+1):.4f}', 'best_loss': f'{best_loss:.4f}'})

    # Restore best checkpoint
    if best_model_state is not None:
        print(f"\n{'='*60}")
        print(f"Restoring best checkpoint:")
        print(f"  Step: {best_model_state['step']}")
        print(f"  Epoch: {best_model_state['epoch']}/{epochs}")
        print(f"  Best loss: {best_model_state['loss']:.4f}")
        print(f"{'='*60}\n")

        # Load best model state back to device
        model_device = next(model.parameters()).device
        state_dict_on_device = {k: v.to(model_device) for k, v in best_model_state['model_state_dict'].items()}
        model.load_state_dict(state_dict_on_device)
    else:
        print("\nWarning: No best checkpoint found, using final state")

    # Save the model (now it's the best checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, help='Path to the composite dataset')
    parser.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B-Instruct', help='Base model name')
    parser.add_argument('--output_dir', default='finetuned_model', help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (use small value with gradient accumulation)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--use_fp16', action='store_true', default=True, help='Use mixed precision (fp16) training')
    parser.add_argument('--no_fp16', action='store_false', dest='use_fp16', help='Disable mixed precision training')
    args = parser.parse_args()

    fine_tune(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        use_fp16=args.use_fp16
    )