from datasets import load_dataset


def load(split: str='train'):
    try:
        base = load_dataset('mbpp', 'sanitized')[split]
    except:
        base = load_dataset('google-research-datasets/mbpp', 'sanitized')[split]

    def _map(ex):
        spec = ex.get('prompt', ex.get('text', ''))
        prompt = (
f"Write a correct Python function that satisfies the specification.\n\n"
f"Specification:\n{spec}\n\n# Your code:\n"
)
        return {
            'prompt': prompt,
            'target': ex.get('code', ''),
            'task_id': ex.get('task_id', -1),
            'test_list': ex.get('test_list', []),
            'test_setup_code': ex.get('test_setup_code', '')
        }

    return base.map(_map, remove_columns=base.column_names)