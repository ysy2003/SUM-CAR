import json, os
from typing import Any, Dict


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def dump_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)