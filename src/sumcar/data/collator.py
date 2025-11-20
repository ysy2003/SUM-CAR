from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class CLMCollator:
    tokenizer: any
    max_length: int = 512

    def __call__(self, batch: List[Dict]):
        prompts = [ex['prompt'] for ex in batch]
        targets = [ex['target'] for ex in batch]
        
        # Tokenize targets first to reserve space for them
        target_encodings = [
            self.tokenizer(t, add_special_tokens=False)['input_ids']
            for t in targets
        ]
        
        # Calculate max prompt length (reserve space for target + separator + padding)
        max_target_len = max(len(t) for t in target_encodings)
        max_prompt_len = self.max_length - max_target_len - 2  # -2 for safety margin
        
        # Truncate prompts if needed
        prompt_encodings = []
        for p in prompts:
            p_enc = self.tokenizer(p, add_special_tokens=True, truncation=True, max_length=max_prompt_len)
            prompt_encodings.append(p_enc['input_ids'])
        
        # Build full sequences manually
        input_ids_list = []
        labels_list = []
        
        for p_ids, t_ids in zip(prompt_encodings, target_encodings):
            # Combine prompt + target
            full_ids = p_ids + t_ids
            
            # Create labels (mask prompt, keep target)
            labels = [-100] * len(p_ids) + t_ids
            
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        
        # Pad sequences
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, labs in zip(input_ids_list, labels_list):
            # Truncate if necessary
            ids = ids[:max_len]
            labs = labs[:max_len]
            
            # Calculate padding
            pad_len = max_len - len(ids)
            
            # Pad
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }