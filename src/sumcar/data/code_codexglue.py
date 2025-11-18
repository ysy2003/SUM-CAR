from datasets import load_dataset


_DEF_INST = (
"Fix the following buggy Python code. Return ONLY the corrected code.\n\n"
"# Buggy code:\n{buggy}\n\n# Corrected code:\n"
)


def load(split: str = 'train'):
    ds = load_dataset('code_x_glue_cc_code_refinement', 'python')[split]

    def _map(ex):
        return {
            'prompt': _DEF_INST.format(buggy=ex['buggy']),
            'target': ex['fixed'],
            'buggy': ex['buggy'],
            'fixed': ex['fixed']
        }

    return ds.map(_map, remove_columns=ds.column_names)