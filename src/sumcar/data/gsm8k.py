import re
from datasets import load_dataset


_DEF_PROMPT = "Question: {q}\n\nProvide your final numeric answer in the last sentence."
_COT_PROMPT = "Question: {q}\n\nThink step by step, then provide your final numeric answer in the last sentence."


def _last_number(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else s.strip()


def load(split: str = 'train', use_cot: bool = False):

    ds = load_dataset('gsm8k', 'main')[split]

    def _map(ex):
        if '####' in ex['answer']:
            reasoning, final_answer = ex['answer'].split('####')
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = ""
            final_answer = _last_number(ex['answer'])
        
        if use_cot:
            prompt = _COT_PROMPT.format(q=ex['question'])
            target = f"{reasoning}\n\nThe answer is: {final_answer}"
        else:
            prompt = _DEF_PROMPT.format(q=ex['question'])
            target = final_answer
        
        return {
            'prompt': prompt,
            'target': target,
            'raw_question': ex['question'],
            'raw_answer': ex['answer']
        }

    return ds.map(_map, remove_columns=ds.column_names)