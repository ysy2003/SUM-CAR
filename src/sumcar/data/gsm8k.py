import re
from datasets import load_dataset


# 原始prompt（直接给答案）
_DEF_PROMPT = "Solve the problem and give only the final numeric answer.\n\n{q}\n\nAnswer:"

# CoT prompt（让模型生成推理过程）
_COT_PROMPT = """Let's solve this step by step.

Question: {q}

Let me think through this carefully:"""


def _last_number(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else s.strip()


def load(split: str = 'train', use_cot: bool = False):
    """
    加载GSM8K数据集
    
    参数:
        split: 数据集划分 ('train' 或 'test')
        use_cot: 是否使用Chain-of-Thought prompting
    """
    ds = load_dataset('gsm8k', 'main')[split]

    def _map(ex):
        # 提取推理过程和最终答案
        if '####' in ex['answer']:
            reasoning, final_answer = ex['answer'].split('####')
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = ""
            final_answer = _last_number(ex['answer'])
        
        # 根据use_cot选择prompt和target
        if use_cot:
            prompt = _COT_PROMPT.format(q=ex['question'])
            # CoT模式下，target包含推理过程和答案
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