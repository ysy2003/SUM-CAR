from datasets import load_dataset


def load(split: str='train'):
    try:
        base = load_dataset('mbpp', 'sanitized')[split]
    except:
        base = load_dataset('google-research-datasets/mbpp', 'sanitized')[split]

    def _map(ex):
        prompt = (
"Write a correct Python function that satisfies the specification.\n\n"
f"Specification:\n{ex['text']}\n\n# Your code:\n"
)
        return {
            'prompt': prompt,
            'target': ex.get('code', ''),
            'task_id': ex.get('task_id', -1),
            'test_list': ex.get('test_list', []),
            'test_setup_code': ex.get('test_setup_code', '')
        }

    return base.map(_map, remove_columns=base.column_names)