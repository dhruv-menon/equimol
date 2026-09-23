"""Sample de novo protein backbones from a trained coordinate denoiser."""

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
    linear_beta_schedule,
    sample_coordinates_loop,
)
from equimol.io import write_backbone_pdb
from equimol.layers import GaussianRadialBasis
from equimol.visualization import save_backbone_trajectory_gif

from examples.trainers.protein_backbone_denoiser import build_model


def build_template_backbone(num_residues: int, device: torch.device) -> dict[str, torch.Tensor]:
    if num_residues <= 0:
        raise ValueError(f"num_residues must be positive, got {num_residues}")

    residue_i = torch.arange(num_residues, device=device, dtype=torch.float32)
    coordinates = torch.zeros((num_residues, 4, 3), device=device)

    # Simple extended backbone template. It defines topology/features only; the
    # diffusion sampler still starts from Gaussian coordinate noise.
    coordinates[:, 0, 0] = 3.8 * residue_i - 1.2  # N
    coordinates[:, 1, 0] = 3.8 * residue_i        # CA
    coordinates[:, 2, 0] = 3.8 * residue_i + 1.5  # C
    coordinates[:, 3, 0] = 3.8 * residue_i + 2.2  # O
    coordinates[:, 0, 1] = 0.5
    coordinates[:, 2, 1] = -0.5
    coordinates[:, 3, 1] = -1.2

    return {
        "coordinates": coordinates,
        "atom_mask": torch.ones((num_residues, 4), dtype=torch.bool, device=device),
        "residue_types": torch.full((num_residues,), 7, dtype=torch.long, device=device),
        "residue_index": torch.arange(1, num_residues + 1, dtype=torch.long, device=device),
    }


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-residues", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--graph", choices=["backbone", "radius", "backbone_radius"], default="backbone")
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
    merged["graph"] = args.graph
    merged["radius"] = args.radius if args.radius is not None else train_args.get("radius", 8.0)
    merged["timesteps"] = args.timesteps if args.timesteps is not None else train_args.get("timesteps", 1000)
    merged["schedule"] = args.schedule if args.schedule is not None else train_args.get("schedule", "linear")
    merged["beta_start"] = args.beta_start if args.beta_start is not None else train_args.get("beta_start", 1e-4)
    merged["beta_end"] = args.beta_end if args.beta_end is not None else train_args.get("beta_end", 2e-2)
    model_args = SimpleNamespace(**merged)

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

    template = build_template_backbone(args.num_residues, device)
    batch = prepare_protein_backbone_batch(
        template,
        config=protein_config,
        radial_basis=radial_basis,
    )
    node_dim = batch.h.shape[-1]
    edge_attr_dim = batch.edge_attr.shape[-1] if batch.edge_attr is not None else 0

    model = build_model(model_args, node_dim=node_dim, edge_attr_dim=edge_attr_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = load_schedule(model_args, device)
    trajectory = sample_coordinates_loop(
        model,
        batch.h,
        batch.edge_index,
        schedule,
        batch=batch.batch,
        edge_attr=batch.edge_attr,
        sampler=args.sampler,
        eta=args.eta,
        return_trajectory=args.gif is not None,
    )

    if args.gif is not None:
        final_x = trajectory[-1]
    else:
        final_x = trajectory

    coordinates = final_x.reshape(args.num_residues, 4, 3).detach().cpu()
    output = Path(args.output)
    write_backbone_pdb(
        output,
        coordinates,
        atom_mask=torch.ones((args.num_residues, 4), dtype=torch.bool),
        residue_index=torch.arange(1, args.num_residues + 1, dtype=torch.long),
    )

    if args.gif is not None:
        gif_trajectory = trajectory.reshape(-1, args.num_residues, 4, 3)
        save_backbone_trajectory_gif(
            gif_trajectory,
            args.gif,
            title="equimol backbone diffusion",
        )

    print(f"wrote {output}")
    if args.gif is not None:
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
