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

    Why it exists: the MD17/QM9 loaders yield PyG Batches, the protein loader
    yields dicts of tensors, and TensorDataset yields lists. One helper keeps
    that knowledge out of the Trainer.

    Watch the order of the checks: dict/list/tuple have no .to(), but some
    objects you might meet do both (a tensor subclass, a PyG Batch). Decide
    which test runs first so each type takes the branch you intend.
    """
    # TODO: implement. Tests: tests/test_training.py::test_to_device_*
    raise NotImplementedError
