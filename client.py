import yaml
from openai import OpenAI

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
task_cfg = cfg["task"]

client = OpenAI(
    base_url=model_cfg["api_base"],
    api_key=model_cfg["api_key"],
)

resp = client.chat.completions.create(
    model=model_cfg["name"],
    messages=[
        {"role": "user", "content": task_cfg["prompt"]}
    ],
)

print(resp.choices[0].message.content)
