from datasets import load_dataset


_DEF_INST = (
"Fix the following buggy Python code. Return ONLY the corrected code.\n\n"
"# Buggy code:\n{buggy}\n\n# Corrected code:\n"
)


def load(split: str = 'train'):
    # 尝试不同的配置名称（数据集可能已更新）
    try:
        # 首先尝试 'python'
        ds = load_dataset('code_x_glue_cc_code_refinement', 'python')[split]
    except ValueError:
        try:
            # 尝试 'small'（新版本）
            ds = load_dataset('code_x_glue_cc_code_refinement', 'small')[split]
        except:
            # 最后尝试 'medium'
            ds = load_dataset('code_x_glue_cc_code_refinement', 'medium')[split]

    def _map(ex):
        return {
            'prompt': _DEF_INST.format(buggy=ex['buggy']),
            'target': ex['fixed'],
            'buggy': ex['buggy'],
            'fixed': ex['fixed']
        }

    return ds.map(_map, remove_columns=ds.column_names)