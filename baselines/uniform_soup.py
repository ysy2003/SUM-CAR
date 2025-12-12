import json
import os
from typing import List, Dict, Any

import fire
import numpy as np


def load_patch(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_slot_dict(
    patch: Dict[str, Any],
    slot_ids_key: str = "slot_ids",
    keys_key: str = "keys",
    values_key: str = "values",
) -> Dict[int, tuple]:
    """
    把一个 patch 变成: {slot_id: (key_vec, value_vec)} 的字典
    """
    ids = patch[slot_ids_key]
    keys = patch[keys_key]
    values = patch.get(values_key, None)

    has_values = values is not None
    slot_dict: Dict[int, tuple] = {}

    if has_values:
        assert len(ids) == len(keys) == len(values)
    else:
        assert len(ids) == len(keys)

    for i, sid in enumerate(ids):
        k_vec = np.asarray(keys[i], dtype=np.float32)
        if has_values:
            v_vec = np.asarray(values[i], dtype=np.float32)
        else:
            v_vec = None
        slot_dict[int(sid)] = (k_vec, v_vec)

    return slot_dict


def uniform_soup_patches_sparse(
    patches: str,
    out_patch: str = "patch_uniform_soup.json",
    slot_ids_key: str = "slot_ids",
    keys_key: str = "keys",
    values_key: str = "values",
    gate_key: str = "gate",
) -> None:
    """
    对多个 SUM-CAR memory patch 做 uniform soup
    （在 slot_ids + keys + values 上做稀疏均值）。

    Args:
        patches: 逗号分隔的 patch 路径，例如：
                 "noLoRA/patch_finqa.json,noLoRA_math/patch_gsm8k.json,noLoRA/patch_mbpp.json"
        out_patch: 输出的 merged patch 路径
    """
    patch_paths: List[str] = [p.strip() for p in patches.split(",") if p.strip()]
    assert len(patch_paths) >= 2, "uniform soup 至少需要两个 patch"

    print("[Soup] patches:")
    for p in patch_paths:
        print("  -", p)

    # 1) 读入所有 patch
    patch_objs = [load_patch(p) for p in patch_paths]
    num_patches = len(patch_objs)

    # 2) 基于第一个 patch 推断 key/value 维度
    first = patch_objs[0]
    has_values = values_key in first

    d_key = len(first[keys_key][0])
    d_val = len(first[values_key][0]) if has_values else None
    print(f"[Soup] d_key={d_key}, d_val={d_val}")

    # 3) 每个 patch 做成 slot 字典
    slot_dicts: List[Dict[int, tuple]] = []
    for i, p in enumerate(patch_objs):
        print(f"[Soup] 构建 slot dict: 第 {i+1}/{num_patches} 个 patch")
        sd = build_slot_dict(p, slot_ids_key=slot_ids_key, keys_key=keys_key, values_key=values_key)
        slot_dicts.append(sd)

    # 4) union 所有 slot_id
    all_slot_ids = set()
    for sd in slot_dicts:
        all_slot_ids.update(sd.keys())
    all_slot_ids = sorted(all_slot_ids)
    print(f"[Soup] union slot_ids 数量: {len(all_slot_ids)}")

    # 5) 对每个 slot_id 做均值
    merged_slot_ids: List[int] = []
    merged_keys: List[list] = []
    merged_values: List[list] = [] if has_values else None

    for sid in all_slot_ids:
        sum_k = np.zeros(d_key, dtype=np.float32)
        sum_v = np.zeros(d_val, dtype=np.float32) if has_values else None

        for sd in slot_dicts:
            if sid in sd:
                k_vec, v_vec = sd[sid]
                sum_k += k_vec
                if has_values and v_vec is not None:
                    sum_v += v_vec
            # 如果该 patch 没有这个 slot，就相当于加 0

        avg_k = (sum_k / num_patches).tolist()
        merged_slot_ids.append(int(sid))
        merged_keys.append(avg_k)

        if has_values:
            avg_v = (sum_v / num_patches).tolist()
            merged_values.append(avg_v)

    # 6) gate 均值（如果有）
    avg_gate: Dict[str, Any] = {}
    if gate_key in patch_objs[0]:
        gate_keys = patch_objs[0][gate_key].keys()
        for gk in gate_keys:
            vals = []
            for p in patch_objs:
                v = np.asarray(p[gate_key][gk], dtype=np.float32)
                vals.append(v)
            vals = np.stack(vals, axis=0)
            avg_gate[gk] = vals.mean(axis=0).tolist()

    # 7) 以第一个 patch 为模板，构造新的 patch
    soup_patch = dict(patch_objs[0])
    soup_patch["task"] = "uniform_soup"
    soup_patch[slot_ids_key] = merged_slot_ids
    soup_patch[keys_key] = merged_keys
    if has_values:
        soup_patch[values_key] = merged_values
    if avg_gate:
        soup_patch[gate_key] = avg_gate

    # 8) 更新 meta（可选）
    if "train_stats" in soup_patch:
        soup_patch["train_stats"]["dataset"] = "uniform_soup_multi_task"
        soup_patch["train_stats"]["num_examples"] = sum(
            p.get("train_stats", {}).get("num_examples", 0) for p in patch_objs
        )

    os.makedirs(os.path.dirname(out_patch) or ".", exist_ok=True)
    with open(out_patch, "w") as f:
        json.dump(soup_patch, f, indent=2)

    print(f"[Soup] uniform soup patch saved to: {out_patch}")


def main(
    patches: str,
    out_patch: str = "patch_uniform_soup.json",
):
    """
    命令行用法：

    python uniform_soup_patch_sparse.py \\
      --patches noLoRA/patch_finqa.json,noLoRA_math/patch_gsm8k.json,noLoRA/patch_mbpp.json \\
      --out_patch noLoRA/patch_uniform_soup.json
    """
    uniform_soup_patches_sparse(
        patches=patches,
        out_patch=out_patch,
    )


if __name__ == "__main__":
    fire.Fire(main)
