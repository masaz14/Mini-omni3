# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Minimal utils subset for offline PASKAL inference (load_checkpoint + tokenizer JSON fixups)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def fix_and_load_json(s: str):
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    pattern = r'(?<=[}\]0-9truefalsenull"])\s*(\n\s*)"'
    s = re.sub(pattern, r',\1"', s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after fixing: {e}") from e


def get_default_supported_precision(training: bool) -> str:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed" if training else "bf16-true"
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def load_checkpoint(
    fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True
) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)
