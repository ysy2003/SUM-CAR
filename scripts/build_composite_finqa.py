import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sumcar.data import finqa_rc

# Define the new composite prompt template
COMPOSITE_FINQA_PROMPT = """Context:
{context}

Question: {question}

Based on the context, first lay out the step-by-step reasoning to answer the question. Then, write a Python code snippet containing a function `compute_answer()` that programmatically calculates the final answer. The function should not take any arguments and should retrieve any necessary numbers from the context provided above.

Answer:
"""

def build_composite_finqa_dev():
    """
    Generates a composite FinQA dev set with specialized prompts for evaluating
    reasoning and code generation abilities.
    """
    output_dir = Path('out/finqa')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'finqa_composite_dev.jsonl'
    
    # Load the FinQA dev set
    # use_cot=True gives us the pre-parsed program which is good for reference
    dev_dataset = finqa_rc.load(split='dev', use_cot=True) 
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in dev_dataset:
            answer_str = ex.get('answer')
            # Skip examples with no answer or non-numeric answers
            if not answer_str:
                continue
            try:
                float(str(answer_str).replace(",", ""))
            except (ValueError, TypeError):
                continue

            prompt = COMPOSITE_FINQA_PROMPT.format(
                context=ex['context'],
                question=ex['question']
            )
            
            record = {
                "id": ex['uid'],
                "prompt": prompt,
                "gold": ex['answer'],
                "program": ex['program'] # for reference
            }
            f.write(json.dumps(record) + '\n')
            count += 1
            
    print(f"Successfully generated {count} samples to {output_file}")

if __name__ == '__main__':
    build_composite_finqa_dev()
