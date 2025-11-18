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
        # Build input: prompt + target; mask loss on prompt tokens
        inputs = [p + " " + t for p, t in zip(prompts, targets)]
        enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Prepare labels (target tokens only)
        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(targets, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        labels = enc['input_ids'].clone()
        # mask prompt part: set labels of prompt tokens to -100
        for i in range(len(batch)):
            p = self.tokenizer(batch[i]['prompt'], truncation=True, max_length=self.max_length, return_tensors='pt')
            plen = p['input_ids'].shape[1]
            labels[i, :min(plen, labels.shape[1])] = -100
        enc['labels'] = labels
        return enc