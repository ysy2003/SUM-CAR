# Reads a JSONL prepared by scripts/prepare_composite.py
import json
from datasets import Dataset


def load(path: str):
    rows = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]
    return Dataset.from_list(rows)