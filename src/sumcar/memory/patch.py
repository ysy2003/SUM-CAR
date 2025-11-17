from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SkillPatch:
    task: str
    slot_ids: List[int]
    keys: List[List[float]]
    vals: List[List[float]]
    access_counts: List[int]
    meta: Dict