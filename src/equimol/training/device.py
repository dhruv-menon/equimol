from __future__ import annotations

from typing import Any

import torch


def to_device(batch: Any, device: torch.device | str) -> Any:
    """Move every tensor in `batch` to `device`, keeping the structure.

    Handles, recursively:
        torch.Tensor                   -> tensor.to(device)
        dict                           -> same keys, values moved
        list / tuple                   -> same container type, items moved
        objects with a .to() method    -> obj.to(device)   (PyG Data/Batch)
        anything else (str, int, None) -> returned unchanged
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(item, device) for item in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch