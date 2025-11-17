from typing import Optional
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM


class MemoryAugmentedCausalLM(nn.Module):

    def __init__(self, base_model_name: str, kv_memory: nn.Module):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.mem = kv_memory
        self.embed = self.lm.get_input_embeddings()


    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        assert input_ids is not None, "input_ids required"
        inputs_embeds = self.embed(input_ids)
        # memory augments token representations BEFORE LM forward
        aug = self.mem(inputs_embeds) # shape like inputs_embeds
        inputs_embeds = inputs_embeds + aug
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kw)


    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **gen_kw):
        inputs_embeds = self.embed(input_ids)
        aug = self.mem(inputs_embeds)
        inputs_embeds = inputs_embeds + aug
        return self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kw)
