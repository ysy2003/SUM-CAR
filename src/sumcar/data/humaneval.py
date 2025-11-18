from datasets import load_dataset


def load():
    try:
        ds = load_dataset('openai_humaneval')['test']
    except:
        ds = load_dataset('nuprl/humaneval')['test']

    def _map(ex):
        return {
'prompt': ex['prompt'],
'target': ex.get('canonical_solution', ''),
'task_id': ex['task_id'],
'tests': ex.get('test', '')
}
    return ds.map(_map)