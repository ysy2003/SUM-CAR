# 环境设置说明

## Python 环境要求

- Python 3.11 或更高版本

## 安装步骤

1. 创建虚拟环境:
```bash
/opt/homebrew/bin/python3.11 -m venv nlp
```

2. 激活虚拟环境:
```bash
source nlp/bin/activate
```

3. 安装依赖:
```bash
pip install --upgrade pip
pip install tinker
pip install -e tinker-cookbook/
```

4. 设置 API Key:
创建 `.env` 文件并添加:
```
TINKER_API_KEY=your-api-key-here
```

## 运行示例

```bash
python test_tinker_api.py
```

## 注意事项

- `nlp/` 虚拟环境文件夹不会被提交到 Git（已在 .gitignore 中排除）
- 每次克隆仓库后都需要重新创建虚拟环境
