# sumcar/experiments/slot_utils.py
import json
from pathlib import Path
from typing import List

def load_task_global_slots(
    task: str,
    patch_meta_path: str,
    remap_path: str,
) -> List[int]:
    patch_meta = json.loads(Path(patch_meta_path).read_text())
    remap_obj = json.loads(Path(remap_path).read_text())
    remap = remap_obj["remap"]

    local_ids = patch_meta[task]["slot_ids"]   # 0..4095
    global_ids = []
    for s in local_ids:
        key = f"{task}:{s}"
        if key not in remap:

            continue
        global_ids.append(remap[key])

    global_ids = sorted(set(global_ids))
    return global_ids
