import os, json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..memory.kv_memory import KVMemoryLayer
from ..models.base_model import MemoryAugmentedCausalLM
from ..eval.gsm8k_eval import evaluate_gsm8k
from ..eval.finqa_eval import evaluate_finqa_rc
from ..eval.humaneval_runner import evaluate_humaneval_pass1

def _build_model_with_memory(base_model: str, mem_state: dict, k_top: int=32, alpha: float=1.0):
    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=mem_state['keys'].shape[0], k_top=k_top, alpha=alpha)
    with torch.no_grad():
        mem.keys.data[:] = mem_state['keys']
        mem.vals.data[:] = mem_state['vals']
    return MemoryAugmentedCausalLM(base_model, mem)

def _load_merged(base_model: str, merged_dir: str):
    state = torch.load(os.path.join(merged_dir, 'memory.pt'), map_location='cpu')
    return _build_model_with_memory(base_model, state)

def _load_from_patch(base_model: str, patch_json: str, num_slots: int=65536, k_top: int=32, alpha: float=1.0):
    # initialize empty memory then apply patch rows to their slot ids
    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=k_top, alpha=alpha)
    patch = json.load(open(patch_json, 'r', encoding='utf-8'))
    mem.apply_patch(patch)
    return MemoryAugmentedCausalLM(base_model, mem)

def main(base_model: str='gpt2',
         merged: str=None,
         patch: str=None,
         out: str='out/eval_single.json',
         max_new_tokens: int=128):
    """Evaluate either a merged model (preferred) or a single patch model.

    One of --merged or --patch must be provided.
    """
    assert (merged is not None) ^ (patch is not None), "Provide exactly one of --merged or --patch"

    if merged:
        model = _load_merged(base_model, merged)
    else:
        model = _load_from_patch(base_model, patch)

    # Run the three single-task suites
    res = {
        'gsm8k': evaluate_gsm8k(model, base_model),
        'humaneval': evaluate_humaneval_pass1(model, base_model),
        'finqa': evaluate_finqa_rc(model, base_model)
    }
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print('single-task eval saved to', out)

if __name__ == '__main__':
    fire.Fire(main)
