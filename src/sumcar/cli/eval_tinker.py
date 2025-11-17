"""
使用 Tinker LoRA 模型进行评估
对应原来的 eval_single.py，但使用 Tinker API
"""
import os
import json
import fire
from dotenv import load_dotenv

import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer

from ..utils.io import ensure_dir, dump_json, load_json
from ..utils.logger import Logger
from ..data import gsm8k as gsm
from ..data import code_codexglue as cglue
from ..data import finqa_rc as finqa

load_dotenv()

EVAL_LOADERS = {
    'gsm8k': lambda: gsm.load('test'),
    'codexglue_refine': lambda: cglue.load('test'),
    'finqa_rc': lambda: finqa.load('validation'),
}


def evaluate_with_tinker(
    sampling_client,
    renderer,
    tokenizer,
    dataset,
    task_name: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    logger=None,
):
    """
    使用 Tinker sampling client 评估数据集
    
    Args:
        sampling_client: Tinker sampling client
        renderer: Tinker renderer
        tokenizer: Tokenizer
        dataset: 评估数据集
        task_name: 任务名称
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        logger: Logger
    
    Returns:
        dict: 评估结果
    """
    logger = logger or Logger(f'[eval-{task_name}]')
    logger.log(f"Evaluating {task_name} with {len(dataset)} examples...")
    
    predictions = []
    references = []
    
    for i, example in enumerate(dataset):
        # 构造 messages
        messages = [{"role": "user", "content": example['prompt']}]
        
        # 构建输入
        model_input = renderer.build_generation_prompt(messages)
        
        # 采样
        resp_future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
            ),
        )
        
        resp = resp_future.result()
        tokens = resp.sequences[0].tokens
        pred_text = tokenizer.decode(tokens)
        
        # 提取预测（移除 prompt 部分）
        # 注意：需要根据具体任务调整后处理逻辑
        pred = pred_text.strip()
        ref = example['target'].strip()
        
        predictions.append(pred)
        references.append(ref)
        
        if (i + 1) % 50 == 0:
            logger.log(f"Processed {i + 1}/{len(dataset)} examples")
    
    # 计算准确率（简单的完全匹配）
    correct = sum(1 for p, r in zip(predictions, references) if p.lower().strip() == r.lower().strip())
    accuracy = correct / len(references) if references else 0.0
    
    logger.log(f"{task_name} Accuracy: {accuracy:.4f} ({correct}/{len(references)})")
    
    return {
        'task': task_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(references),
        'predictions': predictions[:10],  # 保存前 10 个预测样例
        'references': references[:10],
    }


def main(
    patch_path: str,
    out: str = 'out/eval_tinker.json',
    max_new_tokens: int = 128,
    temperature: float = 0.0,
):
    """
    评估 Tinker LoRA patch
    
    Args:
        patch_path: tinker_patch_*.json 文件路径
        out: 输出结果路径
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
    """
    logger = Logger('[eval-tinker]')
    
    # 加载 patch 信息
    patch_info = load_json(patch_path)
    task = patch_info['task']
    base_model = patch_info['base_model']
    model_path = patch_info['model_path']
    
    logger.log(f"Evaluating task: {task}")
    logger.log(f"Base model: {base_model}")
    logger.log(f"Model path: {model_path}")
    
    # 创建 service client 和 sampling client
    service_client = tinker.ServiceClient()
    
    # 从保存的模型路径创建 sampling client
    # 注意：这里假设 model_path 可以直接用来创建 sampling client
    # 实际使用时可能需要根据 Tinker API 调整
    sampling_client = service_client.create_sampling_client(
        base_model=model_path  # 使用微调后的模型路径
    )
    
    # 准备 tokenizer 和 renderer
    tokenizer = get_tokenizer(base_model)
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # 根据任务加载评估数据
    dataset_key = patch_info['dataset']
    if dataset_key not in EVAL_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    dataset = EVAL_LOADERS[dataset_key]()
    
    # 限制评估样本数（用于快速测试）
    max_eval_examples = 100  # 可以通过参数配置
    if len(dataset) > max_eval_examples:
        dataset = dataset.select(range(max_eval_examples))
        logger.log(f"Limited evaluation to {max_eval_examples} examples")
    
    # 评估
    result = evaluate_with_tinker(
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        dataset=dataset,
        task_name=task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logger=logger,
    )
    
    # 保存结果
    ensure_dir(os.path.dirname(out) or '.')
    dump_json(result, out)
    logger.log(f"Results saved to: {out}")


if __name__ == '__main__':
    fire.Fire(main)
