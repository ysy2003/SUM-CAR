import json
import fire

_DEF_METRIC_KEYS = {
    'gsm8k': 'accuracy',
    'humaneval': 'pass@1',
    'finqa': 'em'
}

def _pick_metric(d: dict):
    # choose the first float/int value
    for k, v in d.items():
        if isinstance(v, (int, float)):
            return k, float(v)
    raise ValueError('No numeric metric found in dict')

def main(per_task: str,
         merged: str,
         composite: str,
         patch_meta: str,
         out: str = 'out/metrics.json',
         metric_map: str = None,
         gmax: float = 0.01,
         eps: float = 0.02,
         Z: float = 0.2):
    """Compute Retention, Joint Accuracy, Slot Growth per Task, Reversibility (schema).

    Args:
      per_task: JSON — standalone scores per task (from evaluating each patch separately).
      merged: JSON — merged single-task scores.
      composite: JSON — composite success results.
      patch_meta: JSON — includes total_slots and each task's slot_ids.
      metric_map: JSON string mapping task->metric key (optional, else auto-detect or _DEF_METRIC_KEYS).
    """
    S_patch = json.load(open(per_task))
    S_merge = json.load(open(merged))
    S_comp = json.load(open(composite))
    Pmeta = json.load(open(patch_meta))

    mm = _DEF_METRIC_KEYS.copy()
    if metric_map:
        mm.update(json.loads(metric_map))

    # 1) Retention per task
    retention = {}
    for task in ['gsm8k', 'humaneval', 'finqa']:
        key = mm.get(task)
        if key is None:
            key, _ = _pick_metric(S_merge[task])
        r = (S_merge[task][key]) / max(float(S_patch[task][key]), 1e-8)
        retention[task] = r
    retention_avg = sum(retention.values()) / len(retention)

    # 2) Joint accuracy (macro avg across tasks + composite)
    joint_parts = [
        float(S_merge['gsm8k'][mm['gsm8k']]),
        float(S_merge['humaneval'][mm['humaneval']]),
        float(S_merge['finqa'][mm['finqa']]),
        float(S_comp.get('composite_success', 0.0))
    ]
    joint_acc = sum(joint_parts) / len(joint_parts)

    # 3) Slot growth per task
    total_slots = int(Pmeta.get('total_slots', 65536))
    slot_growth = {}
    for task in ['gsm8k', 'humaneval', 'finqa']:
        tag = 'math' if task == 'gsm8k' else ('code' if task == 'humaneval' else 'finqa')
        if tag in Pmeta:
            slot_growth[task] = len(Pmeta[tag]['slot_ids']) / max(total_slots, 1)

    # 4) Reversibility — schema only (fill if ablation results are produced elsewhere)
    reversibility = {t: {'Delta_self': None, 'Delta_cross_max': None} for t in ['gsm8k', 'humaneval', 'finqa']}

    outj = {
        'retention': retention,
        'retention_avg': retention_avg,
        'joint_acc': joint_acc,
        'slot_growth': slot_growth,
        'reversibility': reversibility
    }
    with open(out, 'w') as f:
        json.dump(outj, f, indent=2)
    print('metrics saved to', out)

if __name__ == '__main__':
    fire.Fire(main)
