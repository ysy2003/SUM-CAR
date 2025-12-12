#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import fire
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM

from ..memory.kv_memory import KVMemoryLayer
from ..memory.merge import sumcar_merge


def main(
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    patches: str = None,  # comma-separated list
    out: str = "out/merged",
    num_slots: int = 65536,
    k_top: int = 32,
    alpha: float = 1.0,
    use_tfidf_scoring: bool = True,
    use_capacity_budgeting: bool = True,
    verbose: bool = False,
    use_fp16: bool = True,
    max_slots_per_task: int | None = None,
):
    """
    Merge multiple skill patches with conflict-aware remapping.

    输出:
      - memory.pt: merged KV memory tensors (keys, vals)
      - remap.json: (task, local_slot) -> global_slot 
      - patch_meta.json
      - slot_owners.json
    """
    if isinstance(patches, str):
        patch_list = [p.strip() for p in patches.split(",") if p.strip()]
    else:
        patch_list = patches if patches else []

    assert patch_list, "Provide --patches as comma-separated list of patch_*.json"
    os.makedirs(out, exist_ok=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    d_model = base.get_input_embeddings().weight.shape[1]
    del base

    mem = KVMemoryLayer(
        d_model=d_model,
        num_slots=num_slots,
        k_top=k_top,
        alpha=alpha,
    )

    if use_fp16:
        mem = mem.half()
        print("Using FP16 precision for memory merging")

    plist = []
    for p_path in patch_list:
        with open(p_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        plist.append(pdata)

    for i, p in enumerate(plist):
        patch_name = patch_list[i]

        if "task" not in p:
            p["task"] = f"t{i}"
        task_name = str(p["task"]).lower()

        if (
            "specificity" in p
            and isinstance(p["specificity"], list)
            and len(p["specificity"]) > 0
        ):
            spec = torch.tensor(p["specificity"], dtype=torch.float32)
            s_min, s_max = spec.min(), spec.max()
            if (s_max - s_min) > 1e-6:
                spec = (spec - s_min) / (s_max - s_min)
            else:
                spec = torch.full_like(spec, 0.5)
            p["specificity"] = spec.tolist()

        if "usage" in p and isinstance(p["usage"], list) and len(p["usage"]) > 0:
            usage = torch.tensor(p["usage"], dtype=torch.float32)
            u_min, u_max = usage.min(), usage.max()
            if (u_max - u_min) > 1e-6:
                usage = (usage - u_min) / (u_max - u_min)
            else:
                usage = torch.full_like(usage, 0.5)
            p["usage"] = usage.tolist()

        if "slot_ids" not in p:
            if "keys" in p and isinstance(p["keys"], list):
                n_slots_local = len(p["keys"])
                p["slot_ids"] = list(range(n_slots_local))
                if verbose:
                    print(
                        f"[merge] Patch {patch_name}: inferred slot_ids = "
                        f"0..{n_slots_local - 1}"
                    )
            else:
                raise ValueError(
                    f"Patch {patch_name} has no 'slot_ids' and no 'keys' to infer from."
                )

        if max_slots_per_task is not None:
            scores_tensor = None

            if "specificity" in p and isinstance(p["specificity"], list):
                scores_tensor = torch.tensor(p["specificity"], dtype=torch.float32)
            elif "usage" in p and isinstance(p["usage"], list):
                scores_tensor = torch.tensor(p["usage"], dtype=torch.float32)
            elif "vals" in p and isinstance(p["vals"], list):
                vals_tensor = torch.tensor(p["vals"], dtype=torch.float32)
                if vals_tensor.ndim == 2:
                    scores_tensor = torch.norm(vals_tensor, dim=1)
            elif "keys" in p and isinstance(p["keys"], list):
                keys_tensor = torch.tensor(p["keys"], dtype=torch.float32)
                if keys_tensor.ndim == 2:
                    scores_tensor = torch.norm(keys_tensor, dim=1)

            if scores_tensor is None:
                if verbose:
                    print(
                        f"[merge] Patch {patch_name}: cannot compute scores, "
                        f"skip selection."
                    )
            else:
                n_slots = scores_tensor.shape[0]
                k_keep = min(max_slots_per_task, n_slots)
                topk_scores, topk_idx = torch.topk(
                    scores_tensor, k=k_keep, largest=True
                )
                idx_list = topk_idx.tolist()

                p["slot_ids"] = [p["slot_ids"][j] for j in idx_list]

                if "keys" in p and isinstance(p["keys"], list):
                    p["keys"] = [p["keys"][j] for j in idx_list]

                if "vals" in p and isinstance(p["vals"], list):
                    p["vals"] = [p["vals"][j] for j in idx_list]

                if "specificity" in p and isinstance(p["specificity"], list):
                    p["specificity"] = [p["specificity"][j] for j in idx_list]

                if "usage" in p and isinstance(p["usage"], list):
                    p["usage"] = [p["usage"][j] for j in idx_list]

                if verbose:
                    print(
                        f"[merge] Patch {p['task']} ({patch_name}): "
                        f"selected {k_keep}/{n_slots} most active slots"
                    )

    res = sumcar_merge(
        mem,
        plist,
        use_tfidf_scoring=use_tfidf_scoring,
        use_capacity_budgeting=use_capacity_budgeting,
        verbose=verbose,
    )

    slot_owners: dict[int, list[str]] = defaultdict(list)
    per_task_global: dict[str, list[int]] = defaultdict(list)

    for (task, local_sid), new_sid in res["remap"].items():
        new_sid = int(new_sid)
        slot_owners[new_sid].append(str(task))
        per_task_global[str(task)].append(new_sid)

    for s in slot_owners:
        slot_owners[s] = sorted(set(slot_owners[s]))
    for t in per_task_global:
        per_task_global[t] = sorted(set(per_task_global[t]))

    with torch.no_grad():
        key_norms = mem.keys.norm(dim=1, keepdim=True)  # [num_slots, 1]
        key_norms.clamp_(min=1e-6)
        mem.keys.data /= key_norms

    torch.save(
        {"keys": mem.keys.detach().cpu(), "vals": mem.vals.detach().cpu()},
        os.path.join(out, "memory.pt"),
    )

    remap_serializable = {
        f"{task}:{sid}": int(new_sid)
        for (task, sid), new_sid in res["remap"].items()
    }
    res_serializable = {
        "remap": remap_serializable,
        "final_num_slots": int(res["final_num_slots"]),
        "conflict_stats": res.get("conflict_stats", {}),
    }
    with open(os.path.join(out, "remap.json"), "w", encoding="utf-8") as f:
        json.dump(res_serializable, f, indent=2)

    patch_meta = {
        "total_slots": int(mem.num_slots),
        "k_top": int(k_top),
        "alpha": float(alpha),
    }
    for p in plist:
        t = str(p["task"])
        local_ids = p["slot_ids"]
        global_ids = per_task_global.get(t, [])
        patch_meta[t] = {
            "slot_ids": local_ids,               
            "n_slots": len(local_ids),
            "global_slot_ids": global_ids,       
            "n_global_slots": len(global_ids),
        }

    with open(os.path.join(out, "patch_meta.json"), "w", encoding="utf-8") as f:
        json.dump(patch_meta, f, indent=2)

    num_used_slots = len(slot_owners)
    num_conflict_slots = sum(
        1 for owners in slot_owners.values() if len(owners) >= 2
    )
    util_static = float(num_used_slots) / float(mem.num_slots)
    conflict_static = float(num_conflict_slots) / float(max(1, num_used_slots))

    slot_owners_meta = {
        "total_slots": int(mem.num_slots),
        "num_used_slots": int(num_used_slots),
        "num_conflict_slots": int(num_conflict_slots),
        "util_static": util_static,
        "conflict_static": conflict_static,
        "owners": {str(s): owners for s, owners in slot_owners.items()},
    }
    with open(os.path.join(out, "slot_owners.json"), "w", encoding="utf-8") as f:
        json.dump(slot_owners_meta, f, indent=2)

    print("Merged. Final slots:", res["final_num_slots"])

    if "conflict_stats" in res:
        stats = res["conflict_stats"]
        print("\nConflict Resolution:")
        print(f"  Total conflicts: {stats.get('total_conflicts', 0)}")
        print(f"  Resolved by TF-IDF: {stats.get('conflicts_resolved_by_tfidf', 0)}")
        print(f"  High-specificity winners: {stats.get('high_specificity_winners', 0)}")
        print(f"  Hub slots avoided: {stats.get('hub_slots_avoided', 0)}")

    print("\nStatic routing structure:")
    print(
        f"  Used slots: {num_used_slots}/{mem.num_slots} "
        f"(util_static={util_static:.3f})"
    )
    print(
        f"  Conflict slots: {num_conflict_slots}/{max(1, num_used_slots)} "
        f"(conflict_static={conflict_static:.3f})"
    )


if __name__ == "__main__":
    fire.Fire(main)
