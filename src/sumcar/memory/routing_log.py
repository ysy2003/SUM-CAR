from typing import Dict


def specificity_from_counts(task_counts: Dict[int,int], global_counts: Dict[int,int]):
    # Simple tf-idf like score: freq_task / (1 + freq_global)
    return {sid: task_counts.get(sid,0) / (1.0 + global_counts.get(sid,0)) for sid in task_counts}