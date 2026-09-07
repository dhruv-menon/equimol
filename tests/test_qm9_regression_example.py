import importlib.util
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from equimol.data import build_radial_basis
from equimol.models import EGNNRegressor


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "qm9_regression.py"
_SPEC = importlib.util.spec_from_file_location("qm9_regression_example", _SCRIPT_PATH)
qm9_regression = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(qm9_regression)


def _fake_qm9_loader() -> DataLoader:
    data = [
        Data(
            z=torch.tensor([6, 1, 1]),
            pos=torch.randn(3, 3),
            y=torch.arange(19, dtype=torch.float32).reshape(1, 19),
        ),
        Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.randn(3, 3),
            y=(1 + torch.arange(19, dtype=torch.float32)).reshape(1, 19),
        ),
    ]
    return DataLoader(data, batch_size=2, shuffle=False)


def test_qm9_training_functions_run_on_fake_batch():
    device = torch.device("cpu")
    loader = _fake_qm9_loader()
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0).to(device)
    model = EGNNRegressor(
        node_feat_dim=10,
        num_layers=1,
        hidden_dim=16,
        edge_attr_dim=4,
        message_dim=16,
        pooling="mean",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    target_mean = torch.tensor(0.5)
    target_std = torch.tensor(1.0)

    loss = qm9_regression.train_epoch(
        model=model,
        optimizer=optimizer,
        loader=loader,
        radial_basis=radial_basis,
        device=device,
        target_name="mu",
        target_mean=target_mean,
        target_std=target_std,
        num_atom_types=10,
    )
    mae = qm9_regression.evaluate(
        model=model,
        loader=loader,
        radial_basis=radial_basis,
        device=device,
        target_name="mu",
        target_mean=target_mean,
        target_std=target_std,
        num_atom_types=10,
    )

    assert isinstance(loss, float)
    assert isinstance(mae, float)
    assert loss >= 0.0
    assert mae >= 0.0
