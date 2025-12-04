import torch
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

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
def fine_tune(data_file, base_model='gpt2', output_dir='finetuned_model', epochs=3, batch_size=8, lr=5e-5):
    # Load dataset
    dataset = CompositeDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            prompts = batch['prompt']
            tests = batch['tests']
            
            # Combine prompt and test for Causal LM fine-tuning
            full_texts = [p + t for p, t in zip(prompts, tests)]
            
            # Tokenize the combined texts
            inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True)
            
            # Create labels by cloning input_ids
            labels = inputs.input_ids.clone()
            
            # Mask the prompt part of the labels
            prompts_tokenized = tokenizer(prompts, padding=False, truncation=False)
            for i in range(len(prompts)):
                prompt_len = len(prompts_tokenized['input_ids'][i])
                labels[i, :prompt_len] = -100
            
            # Also mask padding tokens in labels
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, help='Path to the composite dataset')
    parser.add_argument('--base_model', default='gpt2', help='Base model name')
    parser.add_argument('--output_dir', default='finetuned_model', help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    fine_tune(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )