import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM

from ..memory.kv_memory import KVMemoryLayer
from ..memory.merge import sumcar_merge


def main(
    base_model: str = "gpt2",
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
        max_slots_per_task: if set, per-task active memory selection (keep top-K slots).
    """
    # 1) 解析 patch 列表
    if isinstance(patches, str):
        patch_list = [p.strip() for p in patches.split(",") if p.strip()]
    else:
        patch_list = patches if patches else []

    assert patch_list and len(patch_list) > 0, (
        "Provide --patches as comma-separated list of patch_*.json"
    )
    os.makedirs(out, exist_ok=True)

    # 2) 从 base model 推 d_model，初始化 KVMemory
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    d_model = (
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
        .get_input_embeddings()
        .weight.shape[1]
    )

    mem = KVMemoryLayer(
        d_model=d_model,
        num_slots=num_slots,
        k_top=k_top,
        alpha=alpha,
    )

    if use_fp16:
        mem = mem.half()
        print("Using FP16 precision for memory merging")

    # 3) 读入所有 patch，并做 slot_ids + 归一化 + active selection 预处理
    plist = [json.load(open(p, "r", encoding="utf-8")) for p in patch_list]

    for i, p in enumerate(plist):
        patch_name = patch_list[i]

        # 3.1 确保有 task 字段
        if "task" not in p:
            p["task"] = f"t{i}"

        task_name = str(p["task"]).lower()

        # 3.2 对 specificity / usage 做 per-patch 归一化（防止不同任务尺度差异互相压制）
        if (
            "specificity" in p
            and isinstance(p["specificity"], list)
            and len(p["specificity"]) > 0
        ):
            spec = torch.tensor(p["specificity"], dtype=torch.float32)
            s_min, s_max = spec.min(), spec.max()
            if (s_max - s_min) > 1e-6:
                spec = (spec - s_min) / (s_max - s_min)  # 缩放到 [0,1]
            else:
                spec = torch.full_like(spec, 0.5)  # 所有值相同就给常数
            p["specificity"] = spec.tolist()

        if "usage" in p and isinstance(p["usage"], list) and len(p["usage"]) > 0:
            usage = torch.tensor(p["usage"], dtype=torch.float32)
            u_min, u_max = usage.min(), usage.max()
            if (u_max - u_min) > 1e-6:
                usage = (usage - u_min) / (u_max - u_min)
            else:
                usage = torch.full_like(usage, 0.5)
            p["usage"] = usage.tolist()

        # 如果以后想对某个 task 进一步加权，可以在这里对 spec/usage 乘一个系数
        # 例如：
        # if 'gsm8k' in task_name or 'math' in task_name:
        #     if 'specificity' in p:
        #         spec = torch.tensor(p['specificity'])
        #         spec = spec * 1.1
        #         p['specificity'] = spec.tolist()

        # 3.3 如果没有 slot_ids，就根据 keys 长度自动补上
        if "slot_ids" not in p:
            if "keys" in p and isinstance(p["keys"], list):
                n_slots_local = len(p["keys"])
                p["slot_ids"] = list(range(n_slots_local))
                if verbose:
                    print(
                        f"[merge] Patch {patch_name}: inferred slot_ids = 0..{n_slots_local - 1}"
                    )
            else:
                raise ValueError(
                    f"Patch {patch_name} has no 'slot_ids' and no 'keys' to infer from. "
                    f"Please check its schema."
                )

        # 3.4 可选：按 max_slots_per_task 做 active memory selection
        if max_slots_per_task is not None:
            scores_tensor = None

            # 优先使用归一化后的 specificity / usage
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
                        f"[merge] Patch {patch_name}: cannot compute scores, skip selection."
                    )
            else:
                n_slots = scores_tensor.shape[0]
                k_keep = min(max_slots_per_task, n_slots)

                # top-k 选择
                topk_scores, topk_idx = torch.topk(
                    scores_tensor, k=k_keep, largest=True
                )
                idx_list = topk_idx.tolist()

                # 按选择结果 slice 各个字段
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

    # 4) 调用 SUM-CAR merge
    res = sumcar_merge(
        mem,
        plist,
        use_tfidf_scoring=use_tfidf_scoring,
        use_capacity_budgeting=use_capacity_budgeting,
        verbose=verbose,
    )

    # 5) 对合并后的 keys 做 L2 归一化，减弱不同任务 key 范数差异带来的抢占效应
    with torch.no_grad():
        key_norms = mem.keys.norm(dim=1, keepdim=True)  # [num_slots, 1]
        key_norms.clamp_(min=1e-6)
        mem.keys.data /= key_norms

    # 6) 保存 merged memory state tensors
    torch.save(
        {"keys": mem.keys.detach().cpu(), "vals": mem.vals.detach().cpu()},
        os.path.join(out, "memory.pt"),
    )

    # 7) 保存 remap.json
    remap_serializable = {
        f"{task}:{sid}": new_sid for (task, sid), new_sid in res["remap"].items()
    }
    res_serializable = {
        "remap": remap_serializable,
        "final_num_slots": res["final_num_slots"],
        "conflict_stats": res.get("conflict_stats", {}),
    }

    with open(os.path.join(out, "remap.json"), "w") as f:
        json.dump(res_serializable, f, indent=2)

    # 8) 保存 patch_meta，方便做统计 / 可视化
    patch_meta = {"total_slots": mem.num_slots}
    for p in plist:
        patch_meta[p["task"]] = {
            "slot_ids": p["slot_ids"],
            "n_slots": len(p["slot_ids"]),
        }
    with open(os.path.join(out, "patch_meta.json"), "w") as f:
        json.dump(patch_meta, f, indent=2)

    print("Merged. Final slots:", res["final_num_slots"])

    # 9) 打印冲突统计
    if "conflict_stats" in res:
        stats = res["conflict_stats"]
        print("\nConflict Resolution:")
        print(f"  Total conflicts: {stats.get('total_conflicts', 0)}")
        print(
            f"  Resolved by TF-IDF: {stats.get('conflicts_resolved_by_tfidf', 0)}"
        )
        print(
            f"  High-specificity winners: {stats.get('high_specificity_winners', 0)}"
        )
        print(f"  Hub slots avoided: {stats.get('hub_slots_avoided', 0)}")


if __name__ == "__main__":
    fire.Fire(main)
