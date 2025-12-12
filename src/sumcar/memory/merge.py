"""
SUM-CAR Merge Algorithm
冲突感知的记忆槽合并策略
"""
from typing import List, Dict, Tuple
import math


def sumcar_merge(memory, patches: List[Dict], use_tfidf_scoring: bool = True, 
                 use_capacity_budgeting: bool = True, verbose: bool = False) -> Dict:
    num_tasks = len(patches)

    conflict_stats = {
        'total_conflicts': 0,
        'conflicts_resolved_by_tfidf': 0,
        'high_specificity_winners': 0,
        'hub_slots_avoided': 0,
    }

    if use_capacity_budgeting and num_tasks > 0:
        capacity_budget = _allocate_capacity(patches, memory.num_slots)
    else:
        capacity_budget = {patches[i].get('task', f't{i}'): memory.num_slots 
                          for i in range(num_tasks)}
    
    used = set()
    free_ptr = memory.num_slots
    remap = {}  # (task, old_sid) -> new_sid

    for p in patches:
        for sid in p['slot_ids']:
            used.add(sid)

    sid_map = {}
    for i, p in enumerate(patches):
        specificities = p.get('specificity', [1.0] * len(p['slot_ids']))
        idf_df_counts = p.get('idf_df_counts', {}) 
        
        for j, sid in enumerate(p['slot_ids']):
            acc = p['access_counts'][j]
            spec = specificities[j] if j < len(specificities) else 1.0
            df = idf_df_counts.get(sid, 0) if isinstance(idf_df_counts, dict) else 0
            sid_map.setdefault(sid, []).append((i, acc, spec, df))
    

    for sid, lst in sid_map.items():
        if len(lst) == 1:
            # no conflict
            i, acc, spec, df = lst[0]
            task = patches[i].get('task', f't{i}')
            if _within_budget(task, capacity_budget, remap):
                _apply_one(memory, sid, patches[i], sid)
                remap[(task, sid)] = sid
            else:
                if spec > 0.5:  # keep high-specificity slots even if over budget
                    if free_ptr == memory.num_slots:
                        memory.expand_slots(1024)
                    _apply_one(memory, sid, patches[i], free_ptr)
                    remap[(task, sid)] = free_ptr
                    free_ptr += 1
        else:
            # have conflict
            conflict_stats['total_conflicts'] += 1
            
            if use_tfidf_scoring:
                winner_idx = _select_winner_tfidf(lst, patches, num_tasks, verbose=verbose)
                conflict_stats['conflicts_resolved_by_tfidf'] += 1
                
                win_spec = lst[winner_idx][2]
                if win_spec > 0.7:
                    conflict_stats['high_specificity_winners'] += 1
                
                win_df = lst[winner_idx][3]
                if win_df == num_tasks:
                    pass
                else:
                    has_hub = any(df == num_tasks for _, _, _, df in lst)
                    if has_hub:
                        conflict_stats['hub_slots_avoided'] += 1
            else:
                winner_idx = max(range(len(lst)), key=lambda x: lst[x][1])
            
            win_i, _, _, _ = lst[winner_idx]
            win_task = patches[win_i].get('task', f't{win_i}')

            _apply_one(memory, sid, patches[win_i], sid)
            remap[(win_task, sid)] = sid
            
            for k, (j, acc, spec, df) in enumerate(lst):
                if k == winner_idx:
                    continue
                    
                task = patches[j].get('task', f't{j}')
                if spec > 0.5 or _within_budget(task, capacity_budget, remap):
                    if free_ptr == memory.num_slots:
                        memory.expand_slots(1024)
                    
                    _apply_one(memory, sid, patches[j], free_ptr)
                    remap[(task, sid)] = free_ptr
                    free_ptr += 1

    if verbose:
        print(f"\n=== Merge Statistics ===")
        print(f"Total conflicts: {conflict_stats['total_conflicts']}")
        print(f"Resolved by TF-IDF: {conflict_stats['conflicts_resolved_by_tfidf']}")
        print(f"High-specificity winners (>0.7): {conflict_stats['high_specificity_winners']}")
        print(f"Hub slots avoided: {conflict_stats['hub_slots_avoided']}")
    
    return {
        'remap': remap, 
        'final_num_slots': memory.num_slots,
        'conflict_stats': conflict_stats
    }


def _allocate_capacity(patches: List[Dict], total_slots: int) -> Dict[str, int]:
    task_access = {}
    task_names = []
    
    for i, p in enumerate(patches):
        task = p.get('task', f't{i}')
        task_names.append(task)
        task_access[task] = sum(p.get('access_counts', []))

    math_tasks = [t for t in task_names if 'gsm8k' in t.lower() or 'math' in t.lower()]
    
    budget = {}
    reserved_for_math = int(total_slots * 0.4)  
    
    if math_tasks:
        math_quota = reserved_for_math // len(math_tasks)
        for task in math_tasks:
            budget[task] = math_quota

        remaining_slots = total_slots - reserved_for_math
        non_math_tasks = [t for t in task_names if t not in math_tasks]
        
        if non_math_tasks:
            total_non_math_access = sum(task_access.get(t, 1) for t in non_math_tasks)
            for task in non_math_tasks:
                ratio = task_access.get(task, 1) / (total_non_math_access or 1)
                budget[task] = int(remaining_slots * ratio)
    else:
        total_access = sum(task_access.values()) or 1
        for task in task_names:
            ratio = task_access.get(task, 1) / total_access
            budget[task] = int(total_slots * ratio)
    
    return budget


def _within_budget(task: str, budget: Dict[str, int], remap: Dict) -> bool:
    if task not in budget:
        return True 

    used = sum(1 for (t, _), _ in remap.items() if t == task)
    return used < budget[task]


def _select_winner_tfidf(candidates: List[Tuple], patches: List[Dict], num_tasks: int, verbose: bool = False) -> int:
    beta, gamma = 1.2, 0.2     
    hub_penalty = 0.8           
    
    scores = []
    max_acc = max(acc for _, acc, _, _ in candidates) or 1.0
    
    for i, (patch_idx, acc, spec, df) in enumerate(candidates):
        score = (spec ** beta) * ((acc / max_acc) ** gamma)
        if df == num_tasks:
            score *= hub_penalty
        
        scores.append(score)

    winner_idx = max(range(len(scores)), key=lambda i: scores[i])

    if verbose and len(candidates) > 1:
        print(f'  Conflict: {len(candidates)} tasks compete for same slot')
        for i, (patch_idx, acc, spec, df) in enumerate(candidates):
            task_name = patches[patch_idx].get('task', f't{patch_idx}')
            marker = '→ WINNER' if i == winner_idx else ''
            print(f'    {task_name}: spec={spec:.3f}, acc={acc}, df={df}, score={scores[i]:.4f} {marker}')
    
    return winner_idx


def _apply_one(memory, src_sid: int, patch: Dict, dst_sid: int):

    idx = patch['slot_ids'].index(src_sid)
    k = patch['keys'][idx]
    v = patch['vals'][idx]

    patch_one = {
        'slot_ids': [dst_sid],
        'keys': [k],
        'vals': [v],
        'access_counts': [patch['access_counts'][idx]]
    }
    
    memory.apply_patch(patch_one)
