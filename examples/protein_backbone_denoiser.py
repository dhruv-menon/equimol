"""Train a protein backbone denoiser on flatten backbone coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from equimol.data import ProteinBackboneBatchConfig, prepare_protein_backbone_batch
from equimol.diffusion import (
    coordinate_noise_mse,
    DiffusionSchedule,
    cosine_beta_schedule,
    linear_beta_schedule,
    q_sample_coordinates,
)
from equimol.io import read_backbone_pdb
from equimol.layers import GaussianRadialBasis
from equimol.models import EGNNCoordinateDenoiser, VectorEGNNCoordinateDenoiser


class BackbonePDBDataset(Dataset):
    def __init__(self, pdb_dir: str | Path, limit: int | None = None) -> None:
        pdb_dir = Path(pdb_dir)
        self.paths = sorted(
            path for path in pdb_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No PDB-like files found in {pdb_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return read_backbone_pdb(self.paths[idx])

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_item_collate(items):
    if len(items) != 1:
        raise ValueError("Protein backbone trainer currently supports batch_size=1")
    return items[0]

def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        loader: DataLoader,
        schedule: DiffusionSchedule,
        radial_basis: GaussianRadialBasis | None,
        device: torch.device,
        protein_config: ProteinBackboneBatchConfig | None = None,
        grad_clip: float | None = 1.0,
        fixed_overfit_noise: bool = False,
        ) -> float:
     
    epoch_loss = 0.
    batch_count = 0
    protein_config = protein_config or ProteinBackboneBatchConfig()
    if radial_basis is not None:
        radial_basis = radial_basis.to(device)

    model.train()
    progress = tqdm(loader, desc='train', leave=False)
    for data in progress:
        if isinstance(data, dict):
            data = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in data.items()
            }
        elif hasattr(data, "to"):
            data = data.to(device)
        optimizer.zero_grad(set_to_none = True)

        geometric_batch = prepare_protein_backbone_batch(
            data = data,
            config = protein_config,
            radial_basis = radial_basis
        )

        # ----- Forward pass -----
        h = geometric_batch.h
        x = geometric_batch.x
        edge_index = geometric_batch.edge_index
        edge_attr = geometric_batch.edge_attr
        graph_batch = geometric_batch.batch
        mask = geometric_batch.mask

        # ----- Sample timesteps and noisy coordinates -----
        num_graphs = int(graph_batch.max().item()) + 1
        if fixed_overfit_noise:
            t = torch.full(
                (num_graphs,),
                schedule.betas.numel() // 2,
                device=device,
                dtype=torch.long,
            )
            noise = torch.sin(
                torch.arange(x.numel(), device=device, dtype=x.dtype)
            ).view_as(x)
        else:
            t = torch.randint(
                low = 0,
                high = schedule.betas.numel(),
                size = (num_graphs,),
                device = device,
                dtype = torch.long,
            )
            noise = None

        x_t, eps = q_sample_coordinates(
            x0 = x, 
            t = t,
            schedule = schedule,
            batch = graph_batch,
            noise = noise,
            center = True
        )

        eps_hat = model(
            h = h,
            x_t = x_t,
            t = t,
            edge_index = edge_index,
            batch = graph_batch,
            edge_attr = edge_attr
        )

        # ----- Calculate loss -----
        loss = coordinate_noise_mse(
            eps_hat = eps_hat,
            eps = eps, 
            node_mask = mask)
        loss.backward()

        # ----- backward pass -----
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += float(loss.item())
        batch_count += 1
        progress.set_postfix(loss=epoch_loss / batch_count)

    return epoch_loss / max(batch_count, 1)

@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        schedule: DiffusionSchedule,
        radial_basis: GaussianRadialBasis | None,
        device: torch.device,
        protein_config: ProteinBackboneBatchConfig | None = None,
        fixed_overfit_noise: bool = False,
        ) -> float:
    protein_config = protein_config or ProteinBackboneBatchConfig()
    if radial_basis is not None:
        radial_basis = radial_basis.to(device)

    model.eval()
    total_loss = 0.0
    batch_count = 0

    progress = tqdm(loader, desc='eval', leave=False)
    for data in progress:
        if isinstance(data, dict):
            data = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in data.items()
            }
        elif hasattr(data, "to"):
            data = data.to(device)

        geometric_batch = prepare_protein_backbone_batch(
            data=data,
            config=protein_config,
            radial_basis=radial_basis,
        )

        h = geometric_batch.h
        x = geometric_batch.x
        edge_index = geometric_batch.edge_index
        edge_attr = geometric_batch.edge_attr
        graph_batch = geometric_batch.batch
        mask = geometric_batch.mask

        num_graphs = int(graph_batch.max().item()) + 1
        if fixed_overfit_noise:
            t = torch.full(
                (num_graphs,),
                schedule.betas.numel() // 2,
                device=device,
                dtype=torch.long,
            )
            noise = torch.sin(
                torch.arange(x.numel(), device=device, dtype=x.dtype)
            ).view_as(x)
        else:
            t = torch.randint(
                low=0,
                high=schedule.betas.numel(),
                size=(num_graphs,),
                device=device,
                dtype=torch.long,
            )
            noise = None

        x_t, eps = q_sample_coordinates(
            x0=x,
            t=t,
            schedule=schedule,
            batch=graph_batch,
            noise=noise,
            center=True,
        )

        eps_hat = model(
            h=h,
            x_t=x_t,
            t=t,
            edge_index=edge_index,
            batch=graph_batch,
            edge_attr=edge_attr,
        )

        loss = coordinate_noise_mse(
            eps_hat=eps_hat,
            eps=eps,
            node_mask=mask,
        )

        total_loss += float(loss.item())
        batch_count += 1
        progress.set_postfix(loss=total_loss / batch_count)

    return total_loss / max(batch_count, 1)


def split_indices(
    num_items: int,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_items <= 1:
        raise ValueError("Need at least two PDB files for train/val split")
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator)
    val_count = max(1, int(num_items * val_frac))
    train_count = num_items - val_count
    if train_count <= 0:
        raise ValueError("train split is empty")
    return perm[:train_count], perm[train_count:]


def build_schedule(args, device: torch.device) -> DiffusionSchedule:
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


def build_model(args, node_dim: int, edge_attr_dim: int) -> nn.Module:
    if args.model == "egnn":
        return EGNNCoordinateDenoiser(
            node_dim=node_dim,
            num_layers=args.layers,
            hidden_dim=args.hidden_dim,
            edge_attr_dim=edge_attr_dim,
            message_dim=args.message_dim,
            attention=args.attention,
            attention_dim=args.attention_dim,
            time_embedding_dim=args.time_embedding_dim,
            dropout=args.dropout,
            coord_step_size=args.coord_step_size,
        )
    if args.model == "vector-egnn":
        return VectorEGNNCoordinateDenoiser(
            node_dim=node_dim,
            num_layers=args.layers,
            hidden_dim=args.hidden_dim,
            edge_attr_dim=edge_attr_dim,
            message_dim=args.message_dim,
            vector_dim=args.vector_dim,
            attention=args.attention,
            attention_dim=args.attention_dim,
            vector_gate=args.vector_gate,
            time_embedding_dim=args.time_embedding_dim,
            dropout=args.dropout,
            coord_step_size=args.coord_step_size,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model", choices=["egnn", "vector-egnn"], default="vector-egnn")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-dim", type=int, default=128)
    parser.add_argument("--vector-dim", type=int, default=64)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--vector-gate", action="store_true")
    parser.add_argument("--time-embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--coord-step-size", type=float, default=0.1)

    parser.add_argument("--graph", choices=["backbone", "radius", "backbone_radius"], default="backbone_radius")
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--use-residue-types", action="store_true")
    parser.add_argument("--no-residue-index", action="store_true")
    parser.add_argument("--no-edge-type", action="store_true")
    parser.add_argument("--num-radial", type=int, default=32)
    parser.add_argument("--radial-cutoff", type=float, default=8.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/protein_backbone_denoiser.pt")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--fixed-overfit-noise", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for now; variable-length PDB batching is not wired yet")
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = BackbonePDBDataset(args.pdb_dir, limit=args.limit)
    train_idx, val_idx = split_indices(len(dataset), val_frac=args.val_frac, seed=args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=1,
        shuffle=True,
        collate_fn=_one_item_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=1,
        shuffle=False,
        collate_fn=_one_item_collate,
    )

    protein_config = ProteinBackboneBatchConfig(
        graph=args.graph,
        radius=args.radius,
        use_residue_types=args.use_residue_types,
        use_residue_index=not args.no_residue_index,
        use_edge_type=not args.no_edge_type,
        use_radial_basis=args.num_radial > 0,
    )
    radial_basis = (
        GaussianRadialBasis(num_basis=args.num_radial, cutoff=args.radial_cutoff).to(device)
        if args.num_radial > 0
        else None
    )
    schedule = build_schedule(args, device)

    sample = dataset[int(train_idx[0])]
    sample = {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}
    sample_batch = prepare_protein_backbone_batch(sample, config=protein_config, radial_basis=radial_basis)
    node_dim = sample_batch.h.shape[-1]
    edge_attr_dim = sample_batch.edge_attr.shape[-1] if sample_batch.edge_attr is not None else 0

    model = build_model(args, node_dim=node_dim, edge_attr_dim=edge_attr_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")

    print(
        f"device={device} model={args.model} train={len(train_idx)} val={len(val_idx)} "
        f"node_dim={node_dim} edge_attr_dim={edge_attr_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            schedule,
            radial_basis,
            device,
            protein_config=protein_config,
            grad_clip=args.grad_clip,
            fixed_overfit_noise=args.fixed_overfit_noise,
        )
        row = {"epoch": epoch, "train_loss": train_loss}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss = evaluate(
                model,
                val_loader,
                schedule,
                radial_basis,
                device,
                protein_config=protein_config,
                fixed_overfit_noise=args.fixed_overfit_noise,
            )
            row["val_loss"] = val_loss
            print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "history": history + [row],
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
        else:
            print(f"epoch={epoch:03d} train_loss={train_loss:.5f}")

        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "args": vars(args),
            },
            checkpoint_path.with_suffix(".last.pt"),
        )

    print(f"best_val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    main()
