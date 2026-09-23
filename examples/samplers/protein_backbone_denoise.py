"""Denoise a real protein backbone with a trained coordinate denoiser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from equimol.data import ProteinBackboneBatchConfig, prepare_protein_backbone_batch
from equimol.diffusion import (
    cosine_beta_schedule,
    ddim_sample_coordinates_step,
    linear_beta_schedule,
    p_sample_coordinates_step,
    q_sample_coordinates,
)
from equimol.io import read_backbone_pdb, write_backbone_pdb
from equimol.layers import GaussianRadialBasis
from equimol.visualization import save_backbone_trajectory_gif

from examples.trainers.protein_backbone_denoiser import build_model


def load_schedule(args, device: torch.device):
    if args.schedule == "linear":
        return linear_beta_schedule(
            args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=device,
        )
    if args.schedule == "cosine":
        return cosine_beta_schedule(args.timesteps, device=device)
    raise ValueError(f"Unsupported schedule: {args.schedule}")


@torch.no_grad()
def denoise_from_timestep(
    model,
    x_t: torch.Tensor,
    h: torch.Tensor,
    edge_index: torch.Tensor,
    schedule,
    noise_timestep: int,
    *,
    batch: torch.Tensor,
    edge_attr: torch.Tensor | None,
    sampler: str,
    eta: float,
    return_trajectory: bool,
) -> torch.Tensor:
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler: {sampler}")

    trajectory = [x_t.clone()] if return_trajectory else None
    for timestep in reversed(range(noise_timestep + 1)):
        t = torch.tensor(timestep, device=x_t.device, dtype=torch.long)
        if sampler == "ddpm":
            x_t = p_sample_coordinates_step(
                model,
                h,
                x_t,
                t,
                schedule,
                edge_index,
                batch=batch,
                edge_attr=edge_attr,
            )
        else:
            x_t = ddim_sample_coordinates_step(
                model,
                h,
                x_t,
                t,
                schedule,
                edge_index,
                batch=batch,
                edge_attr=edge_attr,
                eta=eta,
            )
        if trajectory is not None:
            trajectory.append(x_t.clone())

    if trajectory is not None:
        return torch.stack(trajectory, dim=0)
    return x_t


def write_pdb_snapshots(
    trajectory: torch.Tensor,
    frames_dir: str | Path,
    *,
    num_residues: int,
    atom_mask: torch.Tensor,
    residue_types: torch.Tensor,
    residue_index: torch.Tensor,
    stride: int,
) -> list[Path]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    pdb_dir = Path(frames_dir) / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    frames = trajectory[::stride].reshape(-1, num_residues, 4, 3).detach().cpu()
    for frame_idx, coordinates in enumerate(frames):
        path = pdb_dir / f"frame_{frame_idx:04d}.pdb"
        write_backbone_pdb(
            path,
            coordinates,
            atom_mask=atom_mask,
            residue_types=residue_types,
            residue_index=residue_index,
        )
        paths.append(path)
    return paths


def write_pymol_render_script(
    pdb_paths: list[Path],
    frames_dir: str | Path,
    *,
    script_path: str | Path | None,
    width: int,
    height: int,
    cartoon_fraction: float,
) -> Path:
    if not pdb_paths:
        raise ValueError("pdb_paths must contain at least one frame")
    if not 0.0 <= cartoon_fraction <= 1.0:
        raise ValueError(f"cartoon_fraction must be in [0, 1], got {cartoon_fraction}")

    frames_dir = Path(frames_dir)
    png_dir = frames_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(script_path) if script_path is not None else frames_dir / "render.pml"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir_abs = png_dir.resolve()

    lines = [
        "reinitialize\n",
        "bg_color white\n",
        "set antialias, 2\n",
        "set ray_trace_mode, 1\n",
        "set ray_trace_color, black\n",
        "set cartoon_fancy_helices, on\n",
        "set cartoon_smooth_loops, on\n",
        "set cartoon_flat_sheets, off\n",
        "set cartoon_loop_radius, 0.18\n",
        "set cartoon_tube_radius, 0.24\n",
        "set cartoon_highlight_color, grey70\n",
        "set ribbon_width, 5\n",
        f"viewport {width}, {height}\n",
        f'python\nimport os\nos.makedirs(r"{png_dir_abs.as_posix()}", exist_ok=True)\npython end\n',
    ]

    cartoon_start = int(round((1.0 - cartoon_fraction) * len(pdb_paths)))

    for frame_idx, pdb_path in enumerate(pdb_paths):
        png_path = (png_dir_abs / f"frame_{frame_idx:04d}.png").with_suffix("")
        pdb_path = pdb_path.resolve()
        lines.extend(
            [
                f"load {pdb_path.as_posix()}, frame\n",
                "dss frame\n",
                "hide everything, frame\n",
            ]
        )
        if frame_idx >= cartoon_start:
            lines.extend(
                [
                    "show cartoon, frame\n",
                    "color wheat, frame\n",
                ]
            )
        else:
            lines.extend(
                [
                    "show ribbon, frame\n",
                    "color wheat, frame\n",
                ]
            )
        if frame_idx == 0:
            lines.extend(["orient frame\n", "zoom frame, 5\n"])
        lines.extend(
            [
                f"png {png_path.as_posix()}, width={width}, height={height}, ray=1\n",
                "delete frame\n",
            ]
        )

    script_path.write_text("".join(lines), encoding="utf-8")
    return script_path


def write_gif_compiler(
    frames_dir: str | Path,
    *,
    gif_path: str | Path | None,
    duration_ms: int,
) -> Path:
    frames_dir = Path(frames_dir)
    gif_path = Path(gif_path) if gif_path is not None else frames_dir / "denoise_pymol.gif"
    compiler_path = frames_dir / "compile_gif.py"
    compiler_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from PIL import Image",
                "",
                f'png_dir = Path(r"{(frames_dir / "png").as_posix()}")',
                f'gif_path = Path(r"{gif_path.as_posix()}")',
                "frames = [Image.open(path).convert('P') for path in sorted(png_dir.glob('frame_*.png'))]",
                "if not frames:",
                "    raise SystemExit(f'No PNG frames found in {png_dir}')",
                "gif_path.parent.mkdir(parents=True, exist_ok=True)",
                "frames[0].save(",
                "    gif_path,",
                "    save_all=True,",
                "    append_images=frames[1:],",
                f"    duration={duration_ms},",
                "    loop=0,",
                ")",
                "print(f'wrote {gif_path}')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return compiler_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-timestep", type=int, default=700)
    parser.add_argument("--gif-stride", type=int, default=5)
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--pymol-script", type=str, default=None)
    parser.add_argument("--pymol-gif", type=str, default=None)
    parser.add_argument("--pymol-width", type=int, default=1400)
    parser.add_argument("--pymol-height", type=int, default=1000)
    parser.add_argument("--pymol-duration-ms", type=int, default=80)
    parser.add_argument("--pymol-cartoon-fraction", type=float, default=0.25)

    parser.add_argument("--graph", choices=["backbone", "radius", "backbone_radius"], default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default=None)
    parser.add_argument("--beta-start", type=float, default=None)
    parser.add_argument("--beta-end", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})

    merged = dict(train_args)
    merged["graph"] = args.graph if args.graph is not None else train_args.get("graph", "backbone_radius")
    merged["radius"] = args.radius if args.radius is not None else train_args.get("radius", 8.0)
    merged["timesteps"] = args.timesteps if args.timesteps is not None else train_args.get("timesteps", 1000)
    merged["schedule"] = args.schedule if args.schedule is not None else train_args.get("schedule", "linear")
    merged["beta_start"] = args.beta_start if args.beta_start is not None else train_args.get("beta_start", 1e-4)
    merged["beta_end"] = args.beta_end if args.beta_end is not None else train_args.get("beta_end", 2e-2)
    model_args = SimpleNamespace(**merged)

    schedule = load_schedule(model_args, device)
    if args.noise_timestep < 0 or args.noise_timestep >= schedule.betas.numel():
        raise ValueError(
            f"noise_timestep must be in [0, {schedule.betas.numel() - 1}], got {args.noise_timestep}"
        )

    data = read_backbone_pdb(args.input)
    data = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }

    protein_config = ProteinBackboneBatchConfig(
        graph=model_args.graph,
        radius=model_args.radius,
        use_residue_types=bool(getattr(model_args, "use_residue_types", False)),
        use_residue_index=not bool(getattr(model_args, "no_residue_index", False)),
        use_edge_type=not bool(getattr(model_args, "no_edge_type", False)),
        use_radial_basis=int(getattr(model_args, "num_radial", 0)) > 0,
    )
    radial_basis = (
        GaussianRadialBasis(
            num_basis=int(getattr(model_args, "num_radial", 0)),
            cutoff=float(getattr(model_args, "radial_cutoff", model_args.radius)),
        ).to(device)
        if protein_config.use_radial_basis
        else None
    )

    geometric_batch = prepare_protein_backbone_batch(
        data,
        config=protein_config,
        radial_basis=radial_basis,
    )
    node_dim = geometric_batch.h.shape[-1]
    edge_attr_dim = geometric_batch.edge_attr.shape[-1] if geometric_batch.edge_attr is not None else 0

    model = build_model(model_args, node_dim=node_dim, edge_attr_dim=edge_attr_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    t = torch.tensor(args.noise_timestep, device=device, dtype=torch.long)
    x_t, _ = q_sample_coordinates(
        geometric_batch.x,
        t,
        schedule,
        batch=geometric_batch.batch,
        center=True,
    )

    trajectory = denoise_from_timestep(
        model,
        x_t,
        geometric_batch.h,
        geometric_batch.edge_index,
        schedule,
        args.noise_timestep,
        batch=geometric_batch.batch,
        edge_attr=geometric_batch.edge_attr,
        sampler=args.sampler,
        eta=args.eta,
        return_trajectory=args.gif is not None or args.frames_dir is not None,
    )

    final_x = trajectory[-1] if trajectory.ndim == 3 else trajectory
    num_residues = data["coordinates"].shape[0]
    final_coordinates = final_x.reshape(num_residues, 4, 3).detach().cpu()
    atom_mask = data["atom_mask"].detach().cpu()
    residue_types = data["residue_types"].detach().cpu()
    residue_index = data["residue_index"].detach().cpu()

    write_backbone_pdb(
        args.output,
        final_coordinates,
        atom_mask=atom_mask,
        residue_types=residue_types,
        residue_index=residue_index,
    )

    if args.gif is not None:
        gif_trajectory = trajectory.reshape(-1, num_residues, 4, 3)
        save_backbone_trajectory_gif(
            gif_trajectory,
            args.gif,
            stride=args.gif_stride,
            title="equimol backbone denoising",
        )

    if args.frames_dir is not None:
        pdb_paths = write_pdb_snapshots(
            trajectory,
            args.frames_dir,
            num_residues=num_residues,
            atom_mask=atom_mask,
            residue_types=residue_types,
            residue_index=residue_index,
            stride=args.gif_stride,
        )
        pymol_script = write_pymol_render_script(
            pdb_paths,
            args.frames_dir,
            script_path=args.pymol_script,
            width=args.pymol_width,
            height=args.pymol_height,
            cartoon_fraction=args.pymol_cartoon_fraction,
        )
        compiler = write_gif_compiler(
            args.frames_dir,
            gif_path=args.pymol_gif,
            duration_ms=args.pymol_duration_ms,
        )
        print(f"wrote {Path(args.frames_dir) / 'pdb'}")
        print(f"wrote {pymol_script}")
        print(f"wrote {compiler}")

    print(f"wrote {args.output}")
    if args.gif is not None:
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
