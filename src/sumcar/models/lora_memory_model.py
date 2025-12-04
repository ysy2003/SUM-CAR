"""
LoRA + Memory Augmented Model
Applies LoRA adapters to the base model, then adds memory augmentation.
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


class LoRAMemoryAugmentedCausalLM(nn.Module):
    """
    Model with LoRA adapters applied to base LM + Memory augmentation.

    Architecture:
        Input → Embeddings → Memory Augmentation → LoRA-adapted LM → Output
    """

    def __init__(self, base_model_name: str, kv_memory: nn.Module, lora_config: dict = None, use_fp16: bool = False):
        """
        Args:
            base_model_name: HuggingFace model name
            kv_memory: KV memory layer
            lora_config: LoRA configuration dict with keys:
                - r: rank (default: 8)
                - lora_alpha: scaling factor (default: 16)
                - target_modules: modules to adapt (default: ["q_proj", "v_proj"])
                - lora_dropout: dropout rate (default: 0.05)
            use_fp16: Whether to use FP16 precision (default: False for FP32)
        """
        super().__init__()

        # Load base model with specified precision
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype
        )

        # Apply LoRA to base model
        if lora_config is None:
            lora_config = {}

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias="none"
        )

        self.lm = get_peft_model(self.lm, peft_config)
        self.lm.print_trainable_parameters()  # Show what's trainable

        # Memory layer
        self.mem = kv_memory

        # Embeddings
        self.embed = self.lm.get_base_model().get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        """Forward pass: embeddings → memory augmentation → LoRA LM"""
        assert input_ids is not None, "input_ids required"

        # Get embeddings
        inputs_embeds = self.embed(input_ids)

        # Memory augmentation (before LoRA LM)
        aug = self.mem(inputs_embeds)
        inputs_embeds = inputs_embeds + aug

        # Forward through LoRA-adapted LM
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kw)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **gen_kw):
        """Generate with memory augmentation + LoRA"""
        inputs_embeds = self.embed(input_ids)
        aug = self.mem(inputs_embeds)
        inputs_embeds = inputs_embeds + aug

        # Generate with LoRA-adapted model
        generated_ids = self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kw)

        # Check if input_ids are already in generated_ids
        if generated_ids.shape[1] >= input_ids.shape[1]:
            if torch.equal(generated_ids[0, :min(3, input_ids.shape[1])], input_ids[0, :min(3, input_ids.shape[1])]):
                return generated_ids

        # Concatenate input_ids with generated_ids
        return torch.cat([input_ids, generated_ids], dim=1)
