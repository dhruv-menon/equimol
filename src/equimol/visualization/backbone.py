from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import torch.nn.functional as F


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


def _smooth_trace(points: torch.Tensor, passes: int = 2) -> torch.Tensor:
    if points.shape[0] < 3:
        return points
    smoothed = points
    for _ in range(passes):
        padded = F.pad(smoothed.T.unsqueeze(0), (1, 1), mode="replicate").squeeze(0).T
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return smoothed


def _resample_trace(points: torch.Tensor, samples_per_segment: int = 5) -> torch.Tensor:
    if points.shape[0] < 2:
        return points
    parts = []
    weights = torch.linspace(0.0, 1.0, samples_per_segment, dtype=points.dtype)
    for start, end in zip(points[:-1], points[1:]):
        parts.append((1.0 - weights[:, None]) * start + weights[:, None] * end)
    parts.append(points[-1:].clone())
    return torch.cat(parts, dim=0)


def _ribbon_edges(points: torch.Tensor, width: float = 0.65) -> tuple[torch.Tensor, torch.Tensor]:
    tangents = torch.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents = tangents / tangents.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    up = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype)
    normals = torch.linalg.cross(tangents, up.expand_as(tangents), dim=-1)
    bad = normals.norm(dim=-1) < 1e-3
    if bad.any():
        alt = torch.tensor([0.0, 1.0, 0.0], dtype=points.dtype)
        normals[bad] = torch.linalg.cross(tangents[bad], alt.expand_as(tangents[bad]), dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return points - width * normals, points + width * normals


def _draw_cartoon_ribbon(ax, frame: torch.Tensor) -> None:
    frame = _resample_trace(_smooth_trace(frame), samples_per_segment=5)
    left, right = _ribbon_edges(frame)
    surface = torch.stack([left, right], dim=0).numpy()

    ax.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color="#f2be7e",
        shade=False,
        linewidth=0,
        antialiased=True,
        alpha=1.0,
    )
    for edge in (left, right):
        ax.plot(
            edge[:, 0].numpy(),
            edge[:, 1].numpy(),
            edge[:, 2].numpy(),
            color="#090909",
            linewidth=0.85,
            alpha=0.95,
        )
    ax.plot(
        frame[:, 0].numpy(),
        frame[:, 1].numpy(),
        frame[:, 2].numpy(),
        color="#d99d5b",
        linewidth=1.15,
        alpha=0.7,
    )


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
    xlim = _axis_limits(ca[..., 0], pad=4.0)
    ylim = _axis_limits(ca[..., 1], pad=4.0)
    zlim = _axis_limits(ca[..., 2], pad=4.0)

    fig = plt.figure(figsize=(7.6, 5.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def draw(frame_idx: int) -> None:
        ax.clear()
        frame = ca[frame_idx]
        ax.set_facecolor("white")
        _draw_cartoon_ribbon(ax, frame)

        if backbone is not None:
            atoms = backbone[frame_idx]
            atom_points = atoms.reshape(-1, 3)
            ax.scatter(
                atom_points[:, 0],
                atom_points[:, 1],
                atom_points[:, 2],
                s=3,
                color="#303030",
                alpha=0.18,
                edgecolors="none",
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=18, azim=35 + 0.35 * frame_idx)
        ax.set_axis_off()
        ax.grid(False)
        ax.set_box_aspect((1, 1, 1))
        if title is not None:
            ax.text2D(
                0.04,
                0.93,
                title,
                transform=ax.transAxes,
                color="#111111",
                fontsize=16,
                fontweight="bold",
            )

    anim = animation.FuncAnimation(fig, draw, frames=ca.shape[0], interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
