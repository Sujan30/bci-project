from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import joblib


@dataclass
class ModelBundle:
    pipeline: Any
    label_map: Dict[int, str]
    fs: float = 100.0
    model_type: str = "lda"
    training_config: Optional[Dict[str, Any]] = field(default=None)
    dataset_hash: Optional[str] = field(default=None)


def save_bundle(path: str, bundle: ModelBundle) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle:
    return joblib.load(path)
