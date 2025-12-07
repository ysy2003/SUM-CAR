from typing import Optional
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM


class MemoryAugmentedCausalLM(nn.Module):

    def __init__(self, base_model_name: str, kv_memory: nn.Module, use_fp16: bool = False, memory_position: str = 'embedding'):
        """
        Args:
            base_model_name: HuggingFace model name
            kv_memory: KV memory layer
            use_fp16: Whether to use FP16 precision (default: False for FP32)
            memory_position: Where to inject memory - 'embedding' (after embed) or 'middle' (after layer 16)
        """
        super().__init__()
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype
        )
        self.mem = kv_memory
        self.embed = self.lm.get_input_embeddings()
        self.memory_position = memory_position

        # For middle injection, use a hook-based approach
        if memory_position == 'middle':
            self.num_layers = len(self.lm.model.layers)
            self.middle_layer = self.num_layers // 2  # Layer 16 for 32-layer model
            self._register_middle_hook()

    def _register_middle_hook(self):
        """Register a forward hook to inject memory after the middle layer."""
        self._hook_handle = None

        def hook_fn(module, input, output):
            # Handle different output formats from different transformers versions
            # output can be: tuple (hidden_states, ...) or just tensor hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                is_tuple = True
            else:
                hidden_states = output
                is_tuple = False

            # Ensure 3D tensor [batch, seq_len, hidden_dim]
            was_2d = hidden_states.dim() == 2
            if was_2d:
                hidden_states = hidden_states.unsqueeze(0)

            # Apply memory augmentation
            aug = self.mem(hidden_states)
            new_hidden_states = hidden_states + aug

            # Restore original shape if needed
            if was_2d:
                new_hidden_states = new_hidden_states.squeeze(0)

            # Return in the same format as input
            if is_tuple:
                return (new_hidden_states,) + output[1:]
            else:
                return new_hidden_states

        # Register hook on the middle layer
        middle_layer = self.lm.model.layers[self.middle_layer - 1]  # -1 because we want after layer 16
        self._hook_handle = middle_layer.register_forward_hook(hook_fn)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        assert input_ids is not None, "input_ids required"

        if self.memory_position == 'embedding':
            # Original: inject after embedding
            inputs_embeds = self.embed(input_ids)
            aug = self.mem(inputs_embeds)
            inputs_embeds = inputs_embeds + aug
            return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kw)

        else:  # middle - hook handles memory injection
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **gen_kw):
        if self.memory_position == 'embedding':
            # Original generation
            inputs_embeds = self.embed(input_ids)
            aug = self.mem(inputs_embeds)
            inputs_embeds = inputs_embeds + aug
            generated_ids = self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kw)
            return generated_ids

        else:  # middle - hook handles memory injection during generate
            generated_ids = self.lm.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kw)
            return generated_ids
