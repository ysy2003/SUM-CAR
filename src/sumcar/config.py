from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemCfg:
    num_slots: int = 65536
    k_top: int = 32
    alpha: float = 1.0


@dataclass
class TrainCfg:
    epochs: int = 1
    lr: float = 5e-4
    batch_size: int = 4
    max_length: int = 512
    probe_steps: int = 1000
    top_t: int = 2048
    save_dir: str = "out/task"
    seed: int = 42
    dataset: str = "gsm8k"


@dataclass
class EvalCfg:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 4


@dataclass
class GlobalCfg:
    base_model: str = "gpt2"
    mem: MemCfg = field(default_factory=MemCfg)