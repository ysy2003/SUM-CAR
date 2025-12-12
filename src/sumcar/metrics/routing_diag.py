
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Routing diagnostics for merged SUM-CAR memories.

Reads:
  - remap.json (from merge step)
  - patch_meta.json (from merge step)

Outputs:
  - Global utilization / conflict stats
  - Per-task home-hit / cross-hit / conflict-share
"""

import os
import json
from collections import defaultdict
from typing import Optional
import fire


def main(
    root: str,
    remap: Optional[str] = None,
    patch_meta: Optional[str] = None,
    json_out: Optional[str] = None,
):
    """
    Args:
        root: 目录，比如 "noLoRA/merged_math_finqa_new"
        remap: 可选，自定义 remap.json 路径；默认 root/remap.json
        patch_meta: 可选，自定义 patch_meta.json 路径；默认 root/patch_meta.json
        json_out: 如果给定，就把所有指标也 dump 成一个 JSON 文件
    """

    # ---------- 1) 解析路径并加载 ----------
    if remap is None:
        remap = os.path.join(root, "remap.json")
    if patch_meta is None:
        patch_meta = os.path.join(root, "patch_meta.json")

    with open(remap, "r", encoding="utf-8") as f:
        remap_raw = json.load(f)
    remap_map = remap_raw["remap"]
    final_num_slots = remap_raw.get("final_num_slots")

    with open(patch_meta, "r", encoding="utf-8") as f:
        patch_meta_data = json.load(f)

    total_slots_capacity = int(
        patch_meta_data.get("total_slots", final_num_slots or 0)
    )

    # ---------- 2) 建 forward / reverse 映射 ----------
    # forward: (task, local_slot_id) -> global_slot_id
    # owners_by_global: global_slot_id -> [(task, local_slot_id), ...]
    forward = {}
    owners_by_global = defaultdict(list)

    for k, g in remap_map.items():
        # k 形如 "finqa:123"
        task, sid_str = k.split(":", 1)
        local_sid = int(sid_str)
        global_sid = int(g)

        forward[(task, local_sid)] = global_sid
        owners_by_global[global_sid].append((task, local_sid))

    used_slots = len(owners_by_global)
    conflict_slots = 0
    for _, owners in owners_by_global.items():
        owner_tasks = {t for t, _ in owners}
        if len(owner_tasks) > 1:
            conflict_slots += 1

    utilization = used_slots / total_slots_capacity if total_slots_capacity else 0.0
    conflict_rate_used = conflict_slots / used_slots if used_slots else 0.0
    conflict_rate_total = (
        conflict_slots / total_slots_capacity if total_slots_capacity else 0.0
    )

    # ---------- 3) 按 task 统计 home / cross / conflict ----------
    tasks = [t for t in patch_meta_data.keys() if t != "total_slots"]
    per_task_stats = {}

    for t in sorted(tasks):
        meta = patch_meta_data[t]
        slot_ids = meta["slot_ids"]
        n_slots = len(slot_ids)

        home = 0           # 这个 task 的 slot 映射到只属于自己的 global slot
        cross = 0          # 这个 task 的 slot 跟别的 task 共享 global slot
        conflict_slots_t = 0  # 这个 task 参与 conflict 的 slot 数量

        for local_sid in slot_ids:
            key = (t, local_sid)
            if key not in forward:
                # 例如被剪枝掉的 slot
                continue

            g = forward[key]
            owners = owners_by_global[g]
            owner_tasks = {tt for tt, _ in owners}

            if len(owner_tasks) == 1 and t in owner_tasks:
                home += 1
            else:
                cross += 1

            if len(owner_tasks) > 1:
                conflict_slots_t += 1

        denom = n_slots if n_slots else 1
        per_task_stats[t] = {
            "n_slots": n_slots,
            "home_slots": home,
            "cross_slots": cross,
            "home_hit_rate": home / denom,
            "cross_hit_rate": cross / denom,
            "conflict_slots": conflict_slots_t,
            "conflict_share": conflict_slots_t / denom,
        }

    # ---------- 4) 打印结果 ----------
    print("=== Global routing stats ===")
    print(f"Merge root           : {root}")
    print(f"Capacity slots       : {total_slots_capacity}")
    print(f"Final num slots      : {final_num_slots}")
    print(f"Used slots           : {used_slots}")
    print(f"Utilization          : {utilization:.4f}")
    print(f"Conflict slots       : {conflict_slots}")
    print(f"Conflict rate (used) : {conflict_rate_used:.4f}")
    print(f"Conflict rate (total): {conflict_rate_total:.4f}")
    print()

    print("=== Per-task slot routing ===")
    header = (
        f"{'task':12s} {'n_slots':7s} {'home':7s} {'cross':7s} "
        f"{'home_rate':10s} {'cross_rate':11s} {'conflict_share':14s}"
    )
    print(header)
    print("-" * len(header))

    for t, s in per_task_stats.items():
        print(
            f"{t:12s} "
            f"{s['n_slots']:7d} "
            f"{s['home_slots']:7d} "
            f"{s['cross_slots']:7d} "
            f"{s['home_hit_rate'] * 100:9.2f}% "
            f"{s['cross_hit_rate'] * 100:10.2f}% "
            f"{s['conflict_share'] * 100:13.2f}%"
        )

    # ---------- 5) 可选：保存 JSON ----------
    metrics = {
        "root": root,
        "total_slots_capacity": total_slots_capacity,
        "final_num_slots": final_num_slots,
        "used_slots": used_slots,
        "utilization": utilization,
        "conflict_slots": conflict_slots,
        "conflict_rate_used": conflict_rate_used,
        "conflict_rate_total": conflict_rate_total,
        "per_task": per_task_stats,
        "raw_conflict_stats": remap_raw.get("conflict_stats", {}),
    }

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print()
        print(f"Saved JSON metrics to {json_out}")


if __name__ == "__main__":
    fire.Fire(main)
