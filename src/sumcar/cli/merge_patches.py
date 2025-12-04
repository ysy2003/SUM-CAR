import os, json
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..memory.kv_memory import KVMemoryLayer
from ..memory.merge import sumcar_merge

def main(base_model: str = 'gpt2',
         patches: str = None,  # Changed to comma-separated string
         out: str = 'out/merged',
         num_slots: int = 65536,
         k_top: int = 32,
         alpha: float = 1.0,
         use_tfidf_scoring: bool = True,
         use_capacity_budgeting: bool = True,
         verbose: bool = False,
         use_fp16: bool = True):
    """Merge multiple skill patches with conflict-aware remapping.

    Args:
      base_model: HF id of the base LM.
      patches: comma-separated list of JSON patch files (order doesn't matter).
      out: output directory for merged memory + remap map.
      num_slots: initial number of memory slots.
      k_top: top-k slots retrieved per token.
      alpha: scaling for memory contribution.
      use_tfidf_scoring: use TF-IDF driven scoring for conflict resolution.
      use_capacity_budgeting: allocate capacity quota per task.
      verbose: print detailed merge statistics.
      use_fp16: whether to use FP16 precision (default: True).
    """
    # Parse patches parameter
    if isinstance(patches, str):
        patch_list = [p.strip() for p in patches.split(',') if p.strip()]
    else:
        patch_list = patches if patches else []
    
    assert patch_list and len(patch_list) > 0, "Provide --patches as comma-separated list of patch_*.json"
    os.makedirs(out, exist_ok=True)

    # infer d_model from base model embeddings
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    d_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype
    ).get_input_embeddings().weight.shape[1]
    mem = KVMemoryLayer(d_model=d_model, num_slots=num_slots, k_top=k_top, alpha=alpha)

    # Convert memory to FP16 if requested
    if use_fp16:
        mem = mem.half()
        print(f"Using FP16 precision for memory merging")

    plist = [json.load(open(p, 'r', encoding='utf-8')) for p in patch_list]
    # fill missing task tag if any
    for i, p in enumerate(plist):
        if 'task' not in p:
            p['task'] = f't{i}'

    res = sumcar_merge(mem, plist, use_tfidf_scoring=use_tfidf_scoring, 
                      use_capacity_budgeting=use_capacity_budgeting,
                      verbose=verbose)

    # save merged memory state tensors
    torch.save({'keys': mem.keys.detach().cpu(), 'vals': mem.vals.detach().cpu()}, os.path.join(out, 'memory.pt'))
    
    # Convert tuple keys to strings for JSON serialization
    remap_serializable = {f"{task}:{sid}": new_sid for (task, sid), new_sid in res['remap'].items()}
    res_serializable = {
        'remap': remap_serializable, 
        'final_num_slots': res['final_num_slots'],
        'conflict_stats': res.get('conflict_stats', {})
    }
    
    with open(os.path.join(out, 'remap.json'), 'w') as f:
        json.dump(res_serializable, f, indent=2)

    # also save a compact patch_meta for metrics (slot ids per task)
    patch_meta = {'total_slots': mem.num_slots}
    for p in plist:
        patch_meta[p['task']] = {'slot_ids': p['slot_ids'], 'n_slots': len(p['slot_ids'])}
    with open(os.path.join(out, 'patch_meta.json'), 'w') as f:
        json.dump(patch_meta, f, indent=2)

    print('Merged. Final slots:', res['final_num_slots'])

    # Display conflict statistics
    if 'conflict_stats' in res:
        stats = res['conflict_stats']
        print(f"\nConflict Resolution:")
        print(f"  Total conflicts: {stats.get('total_conflicts', 0)}")
        print(f"  Resolved by TF-IDF: {stats.get('conflicts_resolved_by_tfidf', 0)}")
        print(f"  High-spec winners: {stats.get('high_specificity_winners', 0)}")
        print(f"  Hub slots avoided: {stats.get('hub_slots_avoided', 0)}")

if __name__ == '__main__':
    fire.Fire(main)
