import json
import fire
import torch
from transformers import AutoModelForCausalLM
from ..memory.kv_memory import KVMemory
from ..models.base_model import MemoryAugmentedCausalLM
from ..eval.composite_eval import evaluate_composite

def _load_merged(base_model: str, merged_dir: str, k_top: int=32, alpha: float=1.0, device: str=None, use_fp16: bool=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(merged_dir + '/memory.pt', map_location='cpu')
    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemory(num_slots=state['keys'].shape[0], d_model=d_model, k_top=k_top, alpha=alpha)
    with torch.no_grad():
        mem.keys[:] = state['keys']
        mem.vals[:] = state['vals']

    # Convert memory to FP16 if needed
    if use_fp16:
        mem = mem.half()

    model = MemoryAugmentedCausalLM(base_model, mem, use_fp16=use_fp16)
    model = model.to(device)
    precision = "FP16" if use_fp16 else "FP32"
    print(f"Model loaded with {precision} precision and moved to {device}")
    return model

def main(base_model: str='gpt2', merged: str=None, composite: str=None, out: str='out/eval_composite.json', use_fp16: bool=True):
    assert merged and composite, "Require --merged and --composite"
    model = _load_merged(base_model, merged, use_fp16=use_fp16)
    res = evaluate_composite(model, base_model, composite)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print('composite eval saved to', out)

if __name__ == '__main__':
    fire.Fire(main)
