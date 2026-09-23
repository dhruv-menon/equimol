import torch

from equimol.visualization import save_backbone_trajectory_gif


def test_save_backbone_trajectory_gif_writes_file(tmp_path):
    trajectory = torch.randn(3, 4, 4, 3)
    path = save_backbone_trajectory_gif(
        trajectory,
        tmp_path / "trajectory.gif",
        fps=4,
        dpi=40,
    )

    assert path.is_file()
    assert path.stat().st_size > 0
