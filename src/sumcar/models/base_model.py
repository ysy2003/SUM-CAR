from typing import Optional
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM


class MemoryAugmentedCausalLM(nn.Module):

    def __init__(self, base_model_name: str, kv_memory: nn.Module, use_fp16: bool = False):
        """
        Args:
            base_model_name: HuggingFace model name
            kv_memory: KV memory layer
            use_fp16: Whether to use FP16 precision (default: False for FP32)
        """
        super().__init__()
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype
        )
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

        # Generate with inputs_embeds
        # For Llama models, this INCLUDES input tokens in output
        generated_ids = self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kw)

        # Always return as-is (input already included by Llama)
        return generated_ids
