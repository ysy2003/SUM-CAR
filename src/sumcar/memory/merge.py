"""
SUM-CAR Merge Algorithm
冲突感知的记忆槽合并策略
"""
from typing import List, Dict, Tuple


def sumcar_merge(memory, patches: List[Dict]) -> Dict:
    """
    SUM-CAR 合并算法：冲突感知的重映射
    
    策略：
    - 保留访问计数更高的版本在原始索引
    - 将其他版本重映射到新的空闲槽位
    - 根据需要扩展记忆表
    
    参数:
        memory: KVMemoryLayer 实例
        patches: patch 字典列表，每个包含 'slot_ids', 'keys', 'vals', 'access_counts'
                以及可选的 'task', 'specificity' 元信息
    
    返回:
        包含 'remap' 和 'final_num_slots' 的字典
    """
    used = set()
    free_ptr = memory.num_slots
    remap = {}  # (task, old_sid) -> new_sid
    
    # 收集所有使用的槽位
    for p in patches:
        for sid in p['slot_ids']:
            used.add(sid)
    
    # 构建槽位映射: sid -> [(patch_idx, acc_count), ...]
    sid_map = {}
    for i, p in enumerate(patches):
        for sid, acc in zip(p['slot_ids'], p['access_counts']):
            sid_map.setdefault(sid, []).append((i, acc))
    
    # 处理每个槽位
    for sid, lst in sid_map.items():
        if len(lst) == 1:
            # 无冲突：直接写入
            i, _ = lst[0]
            _apply_one(memory, sid, patches[i], sid)
            remap[(patches[i].get('task', 't' + str(i)), sid)] = sid
        else:
            # 有冲突：选择访问计数最高的为胜者
            lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
            win_i, _ = lst_sorted[0]
            
            # 胜者保留原位置
            _apply_one(memory, sid, patches[win_i], sid)
            remap[(patches[win_i].get('task', 't' + str(win_i)), sid)] = sid
            
            # 失败者重映射到新槽位
            for j, _ in lst_sorted[1:]:
                if free_ptr == memory.num_slots:
                    # 需要扩展记忆表
                    memory.expand_slots(1024)  # 以块为单位增长
                
                _apply_one(memory, sid, patches[j], free_ptr)
                remap[(patches[j].get('task', 't' + str(j)), sid)] = free_ptr
                free_ptr += 1
    
    return {'remap': remap, 'final_num_slots': memory.num_slots}


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
