# noLoRA/toggle_math_slots_offline.py

import os
import sys
import json
import subprocess
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent      # .../SUM-CAR/noLoRA
ROOT_DIR = THIS_DIR.parent                      # .../SUM-CAR
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


MERGED_DIR = ROOT_DIR / "noLoRA" / "merged_selection_3task"
PATCH_META_PATH = MERGED_DIR / "patch_meta.json"
REMAP_PATH = MERGED_DIR / "remap.json"

MEMORY_PATH = MERGED_DIR / "memory.pt"


def load_task_global_slots(task: str,
                           patch_meta_path: Path,
                           remap_path: Path):
    patch_meta = json.loads(patch_meta_path.read_text())
    remap_obj = json.loads(remap_path.read_text())
    remap = remap_obj["remap"]

    if task not in patch_meta:
        raise KeyError(f"Task {task!r} not found in patch_meta.json")

    local_ids = patch_meta[task]["slot_ids"]  # e.g. [0..4095]
    global_ids = []
    for s in local_ids:
        key = f"{task}:{s}"
        if key not in remap:
            raise KeyError(f"Missing remap for {key}")
        global_ids.append(int(remap[key]))

    global_ids = sorted(set(global_ids))
    return global_ids


def zero_math_slots_in_file(memory_path: Path, math_slots):
    state = torch.load(memory_path, map_location="cpu")
    keys = state["keys"]
    vals = state["vals"]

    idx = torch.as_tensor(math_slots, dtype=torch.long)
    backup = {
        "keys": keys[idx].clone(),
        "vals": vals[idx].clone(),
    }

    keys[idx].zero_()
    vals[idx].zero_()

    state["keys"] = keys
    state["vals"] = vals

    torch.save(state, memory_path)
    return backup


def restore_math_slots_in_file(memory_path: Path, math_slots, backup):
    state = torch.load(memory_path, map_location="cpu")
    keys = state["keys"]
    vals = state["vals"]

    idx = torch.as_tensor(math_slots, dtype=torch.long)

    if backup["keys"].shape[0] != idx.shape[0]:
        raise ValueError("Backup size and math_slots length do not match")

    keys[idx] = backup["keys"]
    vals[idx] = backup["vals"]

    state["keys"] = keys
    state["vals"] = vals
    torch.save(state, memory_path)

def run_all_evals(tag: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    print(f"\n========== [{tag}] ==========")
    cmds = [
        [
            "python", "/content/drive/MyDrive/SUM-CAR/noLoRA/eval_math_only.py",
            "--merged_dir", str(MERGED_DIR),
            "--mode", "100",     
        ],
        [
            "python", "/content/drive/MyDrive/SUM-CAR/noLoRA/eval_finqa_only.py",
            "--merged_dir", str(MERGED_DIR),
            "--mode", "100",
        ],
        [
            "python", "/content/drive/MyDrive/SUM-CAR/noLoRA/code_only/eval_humaneval.py",
            "--merged_dir", str(MERGED_DIR),
            "--mode", "100",
        ],
    ]


    for cmd in cmds:
        print("\n>>> Running:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True, cwd=str(ROOT_DIR))

def main():
    print("Root dir:", ROOT_DIR)
    print("Merged dir:", MERGED_DIR)
    print("Memory path:", MEMORY_PATH)

    math_slots = load_task_global_slots(
        task="gsm8k",
        patch_meta_path=PATCH_META_PATH,
        remap_path=REMAP_PATH,
    )
    print("Math(gsm8k) global slot count:", len(math_slots))
    print("\n==> Zeroing math slots in memory.pt ...")
    backup = zero_math_slots_in_file(MEMORY_PATH, math_slots)
    run_all_evals("math_zero")
    print("\n==> Restoring math slots in memory.pt ...")
    restore_math_slots_in_file(MEMORY_PATH, math_slots, backup)
    run_all_evals("math_restored")


if __name__ == "__main__":
    main()
