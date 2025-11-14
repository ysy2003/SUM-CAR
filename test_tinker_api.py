"""
Tinker API 使用示例
"""
import tinker
import os
from dotenv import load_dotenv
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# 从 .env 文件加载环境变量
load_dotenv()

# 创建服务客户端
service_client = tinker.ServiceClient()

# 方式1: 创建采样客户端(用于推理/生成文本)
base_model = "meta-llama/Llama-3.2-1B"
sampling_client = service_client.create_sampling_client(
    base_model=base_model
)

# 创建 tokenizer 和 renderer
tokenizer = get_tokenizer(base_model)
renderer = renderers.get_renderer("llama3", tokenizer)

# 准备对话消息
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

# 使用 renderer 构建 ModelInput
model_input = renderer.build_generation_prompt(messages)

# 使用采样客户端生成文本
response = sampling_client.sample(
    prompt=model_input,
    num_samples=1,
    sampling_params=tinker.SamplingParams(
        max_tokens=100,
        temperature=0.7,
    )
)

result = response.result()
print("生成的回复 (原始):", result)

# 解码 tokens 为文本
generated_tokens = result.sequences[0].tokens
generated_text = tokenizer.decode(generated_tokens)
print("\n生成的文本:", generated_text)

# 方式2: 创建训练客户端(用于微调模型)
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32,
)

# 进行训练步骤
# training_client.forward_backward(...)
# training_client.optim_step(...)

# 保存模型并获取新的采样客户端
# sampling_client = training_client.save_weights_and_get_sampling_client(
#     name="my_finetuned_model"
# )

# 下载模型权重
# rest_client = service_client.create_rest_client()
# future = rest_client.download_checkpoint_archive_from_tinker_path(
#     sampling_client.model_path
# )
# with open("model-checkpoint.tar.gz", "wb") as f:
#     f.write(future.result())
