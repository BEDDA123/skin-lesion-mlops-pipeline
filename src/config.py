from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path | None = None) -> dict:
    p = path or ROOT / "params.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_raw_csv(params: dict) -> Path:
    env_path = os.environ.get("SKIN_CSV_PATH")
    if env_path:
        return Path(env_path)
    return ROOT / params["data"]["raw_csv"]


def is_fast_train() -> bool:
    return os.environ.get("FAST_TRAIN", "").lower() in ("1", "true", "yes")
