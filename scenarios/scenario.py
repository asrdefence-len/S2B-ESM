from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    emitters: List[Dict]
    noise_std: float
