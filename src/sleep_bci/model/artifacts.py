from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import joblib


@dataclass
class ModelBundle:
    pipeline: Any
    label_map: Dict[int, str]
    fs: float = 100.0


def save_bundle(path: str, bundle: ModelBundle) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle:
    return joblib.load(path)
