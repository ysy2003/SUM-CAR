import json
import pandas as pd
import os
import glob

def get_score(data, task):
    """Extracts score from result data for a given task."""
    if task in data:
        task_data = data[task]
        if "accuracy" in task_data: return task_data["accuracy"]
        if "pass@1" in task_data: return task_data["pass@1"]
        if "em" in task_data: return task_data["em"]
    return 0.0

def normalize_task_name(task):
    """Converts codexglue variations to humaneval."""
    if not isinstance(task, str):
        return ""
    task_lower = task.lower()
    if 'codexglue' in task_lower:
        return "humaneval"
    return task_lower

def create_scores_csv(base_results_path, merged_results_path, output_path):
    """
    Generates scores.csv from base and merged model results.
    Schema: model,task,score_single,score_merged
    """
    with open(base_results_path, 'r') as f:
        base_data = json.load(f)
    with open(merged_results_path, 'r') as f:
        merged_data = json.load(f)

    tasks = set(base_data.keys()) | set(merged_data.keys())
    tasks.discard("config")

    records = []
    for task in tasks:
        records.append({
            "model": "SUM-CAR",
            "task": task,
            "score_single": get_score(base_data, task),
            "score_merged": get_score(merged_data, task),
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")

def create_memory_hits_csv(patches_dir, output_path):
    """
    Generates memory_hits.csv from patch metadata files.
    Schema: task,batch_id,slot_id,weight
    """
    print("--- Creating memory_hits.csv ---")
    patch_files = glob.glob(os.path.join(patches_dir, "patch_*.meta.json"))
    if not patch_files:
        print(f"Warning: No patch files found in {patches_dir}. {output_path} will be empty.")
    
    all_hits = []
    for patch_file in patch_files:
        print(f"Processing {patch_file} for memory hits...")
        with open(patch_file, 'r') as f:
            meta = json.load(f)
        
        task = meta.get("task")
        if not task:
            print(f"Warning: Skipping {patch_file}, 'task' key not found.")
            continue
            
        task_norm = normalize_task_name(task)
        slot_ids = meta.get("slot_ids", [])
        weights = meta.get("access_counts", [])

        if not slot_ids or not weights:
            print(f"Warning: Skipping {patch_file}, 'slot_ids' or 'access_counts' are missing or empty.")
            continue
        
        for slot_id, weight in zip(slot_ids, weights):
            all_hits.append({
                "task": task_norm,
                "batch_id": 0, # Assuming one batch for simplicity
                "slot_id": slot_id,
                "weight": weight,
            })

    if not all_hits:
        print(f"Warning: No data was extracted for memory_hits.csv.")

    df = pd.DataFrame(all_hits)
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path} with {len(df)} rows.")

def create_remap_events_csv(remap_path, patches_dir, output_path):
    """
    Generates remap_events.csv from a merge log file.
    """
    print("--- Creating remap_events.csv ---")
    if not os.path.exists(remap_path):
        print(f"Warning: Remap log not found at {remap_path}. Creating empty {output_path}")
        pd.DataFrame(columns=["task_from", "slot_old", "task_to", "slot_new", "weight"]).to_csv(output_path, index=False)
        return

    # 1. Build a weight map from all individual patch files
    weights_map = {}
    patch_files = glob.glob(os.path.join(patches_dir, "patch_*.meta.json"))
    if not patch_files:
        print(f"Warning: No patch files found in {patches_dir} to build weight map.")
        
    for patch_file in patch_files:
        with open(patch_file, 'r') as f:
            meta = json.load(f)
        
        task = meta.get("task")
        if not task: continue
        
        task_norm = normalize_task_name(task)
        slot_ids = meta.get("slot_ids", [])
        access_counts = meta.get("access_counts", [])
        for slot, count in zip(slot_ids, access_counts):
            weights_map[(task_norm, slot)] = count
            
    if not weights_map:
        print("Warning: Weight map for remapping is empty. All flows will have weight=0.")

    # 2. Parse remap.json and use the weight map
    with open(remap_path, 'r') as f:
        remap_data = json.load(f).get("remap", {})
        
    events = []
    for from_key, slot_new in remap_data.items():
        try:
            task_from, slot_old_str = from_key.split(':')
            slot_old = int(slot_old_str)
            task_norm = normalize_task_name(task_from)
            
            weight = weights_map.get((task_norm, slot_old), 0)
            
            events.append({
                "task_from": task_norm,
                "slot_old": slot_old,
                "task_to": task_norm, # In this format, task_to is the same
                "slot_new": slot_new,
                "weight": weight,
            })
        except ValueError:
            print(f"Warning: Skipping malformed remap key '{from_key}'.")
            continue

    df = pd.DataFrame(events)
    # Filter out flows that are not remapped and have no weight
    df_filtered = df[(df['slot_old'] != df['slot_new']) | (df['weight'] > 0)]
    
    df_filtered.to_csv(output_path, index=False)
    print(f"Successfully created {output_path} with {len(df_filtered)} rows.")


if __name__ == "__main__":
    # Define paths
    LOGS_DIR = "logs"
    BASELINES_DIR = "baselines"
    PATCHES_DIR = "patches"
    MERGED_DIR = "out/merged"

    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- Create scores.csv ---
    base_results = os.path.join(BASELINES_DIR, "base_model_results_quick.json")
    merged_results = os.path.join(BASELINES_DIR, "llama3_8b_results_quick_cot.json")
    scores_csv_path = os.path.join(LOGS_DIR, "scores.csv")
    create_scores_csv(base_results, merged_results, scores_csv_path)

    # --- Create memory_hits.csv ---
    memory_hits_csv_path = os.path.join(LOGS_DIR, "memory_hits.csv")
    create_memory_hits_csv(PATCHES_DIR, memory_hits_csv_path)

    # --- Create remap_events.csv ---
    remap_log = os.path.join(MERGED_DIR, "remap.json")
    remap_events_csv_path = os.path.join(LOGS_DIR, "remap_events.csv")
    create_remap_events_csv(remap_log, PATCHES_DIR, remap_events_csv_path)