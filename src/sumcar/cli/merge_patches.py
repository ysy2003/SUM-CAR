import os, json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..memory.kv_memory import KVMemoryLayer
from ..memory.merge import sumcar_merge

def main(base_model: str = 'gpt2',
         patches: list = None,
         out: str = 'out/merged',
         num_slots: int = 65536,
         k_top: int = 32,
         alpha: float = 1.0,
         use_tfidf_scoring: bool = True,
         use_capacity_budgeting: bool = True):
    """Merge multiple skill patches with conflict-aware remapping.

    Args:
      base_model: HF id of the base LM.
      patches: list of JSON patch files (order doesn't matter).
      out: output directory for merged memory + remap map.
      num_slots: initial number of memory slots.
      k_top: top-k slots retrieved per token.
      alpha: scaling for memory contribution.
      use_tfidf_scoring: use TF-IDF driven scoring for conflict resolution.
      use_capacity_budgeting: allocate capacity quota per task.
    """
    assert patches and len(patches) > 0, "Provide --patches a list of patch_*.json"
    os.makedirs(out, exist_ok=True)

    # infer d_model from base model embeddings
    d_model = AutoModelForCausalLM.from_pretrained(base_model).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=k_top, alpha=alpha)

    plist = [json.load(open(p, 'r', encoding='utf-8')) for p in patches]
    # fill missing task tag if any
    for i, p in enumerate(plist):
        if 'task' not in p:
            p['task'] = f't{i}'

    res = sumcar_merge(mem, plist, use_tfidf_scoring=use_tfidf_scoring, 
                      use_capacity_budgeting=use_capacity_budgeting)

    # save merged memory state tensors
    torch.save({'keys': mem.keys.detach().cpu(), 'vals': mem.vals.detach().cpu()}, os.path.join(out, 'memory.pt'))
    
    # Convert tuple keys to strings for JSON serialization
    remap_serializable = {f"{task}:{sid}": new_sid for (task, sid), new_sid in res['remap'].items()}
    res_serializable = {'remap': remap_serializable, 'final_num_slots': res['final_num_slots']}
    
    with open(os.path.join(out, 'remap.json'), 'w') as f:
        json.dump(res_serializable, f, indent=2)

    # also save a compact patch_meta for metrics (slot ids per task)
    patch_meta = {'total_slots': mem.num_slots}
    for p in plist:
        patch_meta[p['task']] = {'slot_ids': p['slot_ids'], 'n_slots': len(p['slot_ids'])}
    with open(os.path.join(out, 'patch_meta.json'), 'w') as f:
        json.dump(patch_meta, f, indent=2)

    print('Merged. Final slots:', res['final_num_slots'])

if __name__ == '__main__':
    fire.Fire(main)
