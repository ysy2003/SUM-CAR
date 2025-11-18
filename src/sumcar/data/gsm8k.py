import re
from datasets import load_dataset


_DEF_PROMPT = "Solve the problem and give only the final numeric answer.\n\n{q}\n\nAnswer:"


def _last_number(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else s.strip()


def load(split: str = 'train'):
    ds = load_dataset('gsm8k', 'main')[split]

    def _map(ex):
        return {
            'prompt': _DEF_PROMPT.format(q=ex['question']),
            'target': _last_number(ex['answer']),
            'raw_question': ex['question'],
            'raw_answer': ex['answer']
        }

    return ds.map(_map, remove_columns=ds.column_names)