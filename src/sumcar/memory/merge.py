"""
SUM-CAR Merge Algorithm
冲突感知的记忆槽合并策略
"""
from typing import List, Dict, Tuple
import math


def sumcar_merge(memory, patches: List[Dict], use_tfidf_scoring: bool = True, 
                 use_capacity_budgeting: bool = True) -> Dict:
    """
    SUM-CAR 合并算法：冲突感知的重映射
    
    策略：
    - 使用 TF-IDF 驱动的打分选择胜者（高特异性 + 中高频）
    - 配额合并：按任务重要性分配槽位，给弱势任务保底
    - 根据需要扩展记忆表
    
    参数:
        memory: KVMemoryLayer 实例
        patches: patch 字典列表，每个包含 'slot_ids', 'keys', 'vals', 'access_counts'
                以及可选的 'task', 'specificity', 'idf_df_counts' 元信息
        use_tfidf_scoring: 是否使用 TF-IDF 打分（默认 True）
        use_capacity_budgeting: 是否使用配额分配（默认 True）
    
    返回:
        包含 'remap' 和 'final_num_slots' 的字典
    """
    num_tasks = len(patches)
    
    # 配额分配（如果启用）
    if use_capacity_budgeting and num_tasks > 0:
        capacity_budget = _allocate_capacity(patches, memory.num_slots)
    else:
        capacity_budget = {patches[i].get('task', f't{i}'): memory.num_slots 
                          for i in range(num_tasks)}
    
    used = set()
    free_ptr = memory.num_slots
    remap = {}  # (task, old_sid) -> new_sid
    
    # 收集所有使用的槽位
    for p in patches:
        for sid in p['slot_ids']:
            used.add(sid)
    
    # 构建槽位映射: sid -> [(patch_idx, acc_count, specificity, df), ...]
    sid_map = {}
    for i, p in enumerate(patches):
        specificities = p.get('specificity', [1.0] * len(p['slot_ids']))
        idf_df = p.get('idf_df_counts', [0] * len(p['slot_ids']))
        
        for j, sid in enumerate(p['slot_ids']):
            acc = p['access_counts'][j]
            spec = specificities[j] if j < len(specificities) else 1.0
            df = idf_df[j] if j < len(idf_df) else 0
            sid_map.setdefault(sid, []).append((i, acc, spec, df))
    
    # 处理每个槽位
    for sid, lst in sid_map.items():
        if len(lst) == 1:
            # 无冲突：直接写入（检查配额）
            i, acc, spec, df = lst[0]
            task = patches[i].get('task', f't{i}')
            if _within_budget(task, capacity_budget, remap):
                _apply_one(memory, sid, patches[i], sid)
                remap[(task, sid)] = sid
            else:
                # 超出配额：跳过或重映射到扩展区域
                if spec > 0.5:  # 高特异性的仍然保留
                    if free_ptr == memory.num_slots:
                        memory.expand_slots(1024)
                    _apply_one(memory, sid, patches[i], free_ptr)
                    remap[(task, sid)] = free_ptr
                    free_ptr += 1
        else:
            # 有冲突：使用 TF-IDF 打分选择胜者
            if use_tfidf_scoring:
                winner_idx = _select_winner_tfidf(lst, patches, num_tasks)
            else:
                # 回退到原始策略：按访问计数
                winner_idx = max(range(len(lst)), key=lambda x: lst[x][1])
            
            win_i, _, _, _ = lst[winner_idx]
            win_task = patches[win_i].get('task', f't{win_i}')
            
            # 胜者保留原位置
            _apply_one(memory, sid, patches[win_i], sid)
            remap[(win_task, sid)] = sid
            
            # 失败者重映射到新槽位（如果在配额内或高特异性）
            for k, (j, acc, spec, df) in enumerate(lst):
                if k == winner_idx:
                    continue
                    
                task = patches[j].get('task', f't{j}')
                # 高特异性或在配额内的失败者才重映射
                if spec > 0.5 or _within_budget(task, capacity_budget, remap):
                    if free_ptr == memory.num_slots:
                        memory.expand_slots(1024)
                    
                    _apply_one(memory, sid, patches[j], free_ptr)
                    remap[(task, sid)] = free_ptr
                    free_ptr += 1
    
    return {'remap': remap, 'final_num_slots': memory.num_slots}


def _allocate_capacity(patches: List[Dict], total_slots: int) -> Dict[str, int]:
    """
    按任务访问总量和重要性分配配额，给弱势任务保底
    
    策略：
    - 计算每个任务的访问总量
    - 数学任务保底 40%
    - 剩余按访问量比例分配
    """
    task_access = {}
    task_names = []
    
    for i, p in enumerate(patches):
        task = p.get('task', f't{i}')
        task_names.append(task)
        task_access[task] = sum(p.get('access_counts', []))
    
    # 识别数学任务（通常命名包含 'gsm8k' 或 'math'）
    math_tasks = [t for t in task_names if 'gsm8k' in t.lower() or 'math' in t.lower()]
    
    budget = {}
    reserved_for_math = int(total_slots * 0.4)  # 40% 保底
    
    if math_tasks:
        # 数学任务平分保底配额
        math_quota = reserved_for_math // len(math_tasks)
        for task in math_tasks:
            budget[task] = math_quota
        
        # 剩余槽位按访问量比例分配给非数学任务
        remaining_slots = total_slots - reserved_for_math
        non_math_tasks = [t for t in task_names if t not in math_tasks]
        
        if non_math_tasks:
            total_non_math_access = sum(task_access.get(t, 1) for t in non_math_tasks)
            for task in non_math_tasks:
                ratio = task_access.get(task, 1) / (total_non_math_access or 1)
                budget[task] = int(remaining_slots * ratio)
    else:
        # 没有数学任务：按访问量比例分配
        total_access = sum(task_access.values()) or 1
        for task in task_names:
            ratio = task_access.get(task, 1) / total_access
            budget[task] = int(total_slots * ratio)
    
    return budget


def _within_budget(task: str, budget: Dict[str, int], remap: Dict) -> bool:
    """检查任务是否在配额内"""
    if task not in budget:
        return True  # 未设置配额，允许
    
    # 统计该任务已使用的槽位数
    used = sum(1 for (t, _), _ in remap.items() if t == task)
    return used < budget[task]


def _select_winner_tfidf(candidates: List[Tuple], patches: List[Dict], num_tasks: int) -> int:
    """
    使用 TF-IDF 打分选择胜者
    
    参数:
        candidates: [(patch_idx, acc_count, specificity, df), ...]
        patches: patch 列表
        num_tasks: 任务总数
    
    返回:
        胜者在 candidates 中的索引
    """
    beta, gamma = 1.2, 0.2  # beta 偏向特异性，gamma 偏向频率
    
    scores = []
    max_acc = max(acc for _, acc, _, _ in candidates)
    
    for i, (patch_idx, acc, spec, df) in enumerate(candidates):
        # 基础分数：特异性 ^ beta * (访问频率 ^ gamma)
        score = (spec ** beta) * ((acc / (max_acc or 1.0)) ** gamma)
        
        # Hub 惩罚：跨任务通用槽（df == T）降低优先级
        if df == num_tasks:
            score *= 0.8
        
        scores.append(score)
    
    # 返回得分最高的索引
    return max(range(len(scores)), key=lambda i: scores[i])


def _apply_one(memory, src_sid: int, patch: Dict, dst_sid: int):
    """
    应用单个槽位的更新
    
    参数:
        memory: KVMemoryLayer 实例
        src_sid: 源槽位 ID（在 patch 中的 ID）
        patch: patch 字典
        dst_sid: 目标槽位 ID（在 memory 中的 ID）
    """
    # 找到 src_sid 在 patch 中的索引
    idx = patch['slot_ids'].index(src_sid)
    k = patch['keys'][idx]
    v = patch['vals'][idx]
    
    # 构造单槽位 patch
    patch_one = {
        'slot_ids': [dst_sid],
        'keys': [k],
        'vals': [v],
        'access_counts': [patch['access_counts'][idx]]
    }
    
    memory.apply_patch(patch_one)
