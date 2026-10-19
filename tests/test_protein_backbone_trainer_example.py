import importlib.util
from pathlib import Path

import torch

from equimol.data import ProteinBackboneBatchConfig
from equimol.diffusion import linear_beta_schedule
from equimol.layers import GaussianRadialBasis
from equimol.models import VectorEGNNCoordinateDenoiser

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "trainers" / "protein_backbone_denoiser.py"
_SPEC = importlib.util.spec_from_file_location("protein_backbone_denoiser_example", _SCRIPT_PATH)
trainer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(trainer)


def _fake_protein(num_residues: int = 6) -> dict[str, torch.Tensor]:
    return {
        "coordinates": 3.0 * torch.randn(num_residues, 4, 3),
        "atom_mask": torch.ones(num_residues, 4, dtype=torch.bool),
        "residue_types": torch.zeros(num_residues, dtype=torch.long),
        "residue_index": torch.arange(num_residues),
    }


def test_train_and_eval_build_edges_from_noisy_coordinates(monkeypatch):
    # Edges/RBFs built from clean x_0 leak the target structure to the denoiser.
    torch.manual_seed(0)
    protein = _fake_protein()
    clean_x = protein["coordinates"].reshape(-1, 3)
    graph_inputs = []
    build_graph = trainer.build_protein_backbone_graph

    def recording_build_graph(x, batch, **kwargs):
        graph_inputs.append(x.detach().clone())
        return build_graph(x, batch, **kwargs)

    monkeypatch.setattr(trainer, "build_protein_backbone_graph", recording_build_graph)

    config = ProteinBackboneBatchConfig(graph="backbone_radius", radius=8.0)
    radial_basis = GaussianRadialBasis(num_basis=4, cutoff=8.0)
    model = VectorEGNNCoordinateDenoiser(node_dim=5, num_layers=1, hidden_dim=16, edge_attr_dim=6, vector_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = linear_beta_schedule(10)
    loader = [protein]

    trainer.train_epoch(model, optimizer, loader, schedule, radial_basis, torch.device("cpu"), protein_config=config)
    trainer.evaluate(model, loader, schedule, radial_basis, torch.device("cpu"), protein_config=config)

    assert len(graph_inputs) == 2
    for x in graph_inputs:
        assert not torch.allclose(x, clean_x)
        assert not torch.allclose(x, clean_x - clean_x.mean(dim=0))
