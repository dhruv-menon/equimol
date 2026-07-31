import torch

from equimol.utils.geometry import random_rotation, rotate, squared_distances, translate
from equimol.graph import fully_connected_edges


def test_squared_distances_are_rigid_motion_invariant():
    torch.manual_seed(0)
    x = torch.randn(6, 3)
    edge_index = fully_connected_edges(x.size(0))
    rotation = random_rotation(3)
    shift = torch.tensor([2.0, -1.0, 0.5])

    before = squared_distances(x, edge_index)
    after = squared_distances(translate(rotate(x, rotation), shift), edge_index)

    assert torch.allclose(before, after, atol=1e-5)
