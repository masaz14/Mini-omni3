# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Small helper extracted from litgpt.scripts.convert_hf_checkpoint (checkpoint compat only)."""

from __future__ import annotations

import torch

from litgpt.config import Config


def qkv_reassemble(param: torch.Tensor, config: Config) -> torch.Tensor:
    """Reassemble from a normal to an interleaved placement in a QKV matrix."""
    q_per_kv = config.n_head // config.n_query_groups
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(
            chunk,
            [config.head_size * q_per_kv, config.head_size, config.head_size],
        )
        qs.append(split[0])
        ks.append(split[1])
        vs.append(split[2])
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)
    return torch.cat((q, k, v))
