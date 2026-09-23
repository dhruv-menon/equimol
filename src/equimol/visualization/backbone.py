from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import torch
from matplotlib import animation
import matplotlib.pyplot as plt


def _as_backbone_ca(trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if trajectory.ndim == 4:
        if trajectory.shape[2:] != (4, 3):
            raise ValueError(f"Expected trajectory shape [T, R, 4, 3], got {tuple(trajectory.shape)}")
        return trajectory[:, :, 1, :], trajectory
    if trajectory.ndim == 3:
        if trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory shape [T, N, 3], got {tuple(trajectory.shape)}")
        return trajectory, None
    raise ValueError(f"Expected trajectory shape [T, N, 3] or [T, R, 4, 3], got {tuple(trajectory.shape)}")


def _axis_limits(coords: torch.Tensor, pad: float = 2.0) -> tuple[float, float]:
    low = float(coords.min().item())
    high = float(coords.max().item())
    center = 0.5 * (low + high)
    radius = 0.5 * max(high - low, 1.0) + pad
    return center - radius, center + radius


def save_backbone_trajectory_gif(
    trajectory: torch.Tensor,
    path: str | Path,
    *,
    fps: int = 16,
    stride: int = 1,
    dpi: int = 140,
    title: str | None = None,
) -> Path:
    """Render a coordinate denoising trajectory as a polished backbone GIF.

    Shapes:
        - trajectory: [T, N, 3] or [T, R, 4, 3]
        - output: GIF written to path
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = torch.as_tensor(trajectory).detach().cpu().float()[::stride]
    ca, backbone = _as_backbone_ca(trajectory)
    xlim = _axis_limits(ca[..., 0])
    ylim = _axis_limits(ca[..., 1])
    zlim = _axis_limits(ca[..., 2])

    fig = plt.figure(figsize=(7.2, 5.0), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def draw(frame_idx: int) -> None:
        ax.clear()
        frame = ca[frame_idx]
        ax.set_facecolor("#05070d")
        ax.plot(
            frame[:, 0],
            frame[:, 1],
            frame[:, 2],
            color="#72f6ff",
            linewidth=2.4,
            alpha=0.95,
        )
        ax.scatter(
            frame[:, 0],
            frame[:, 1],
            frame[:, 2],
            s=12,
            color="#ffb35c",
            alpha=0.9,
            edgecolors="none",
        )

        if backbone is not None:
            atoms = backbone[frame_idx].reshape(-1, 3)
            ax.scatter(
                atoms[:, 0],
                atoms[:, 1],
                atoms[:, 2],
                s=5,
                color="#f7f7ff",
                alpha=0.25,
                edgecolors="none",
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=22, azim=35 + 0.5 * frame_idx)
        ax.set_axis_off()
        ax.grid(False)
        ax.set_box_aspect((1, 1, 1))
        if title is not None:
            ax.text2D(
                0.04,
                0.93,
                title,
                transform=ax.transAxes,
                color="#f7f7ff",
                fontsize=16,
                fontweight="bold",
            )

    anim = animation.FuncAnimation(fig, draw, frames=ca.shape[0], interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
