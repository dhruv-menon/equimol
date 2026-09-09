"""Train an EGNN regressor on a single QM9 target."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from equimol.data import (
    build_radial_basis,
    compute_target_stats,
    get_qm9_target_index,
    load_qm9,
    prepare_qm9_batch,
    split_qm9,
)

from equimol.models import EGNNRegressor
from equimol.layers import GaussianRadialBasis


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shrink_indices(
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        train_idx[:train_size],
        val_idx[:val_size],
        test_idx[:test_size],
    )

def train_epoch(
        model: EGNNRegressor,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        radial_basis: GaussianRadialBasis,
        device: torch.device,
        target_name: str | int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        graph_type: str = "fully_connected",
        num_atom_types: int = 100,
        radius: float | None = None,
        k: int | None = None,
        ) -> float:

    epoch_loss = 0.
    batch_count = 0

    model.train()
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        geometric_batch = prepare_qm9_batch(
            data = batch,
            target=target_name,
            radial_basis = radial_basis,
            graph = graph_type,
            num_atom_types = num_atom_types,
            radius = radius,
            k = k,
            )

        # ----- Forward pass ------
        h = geometric_batch.h
        x = geometric_batch.x
        edge_index = geometric_batch.edge_index
        graph_batch = geometric_batch.batch
        edge_attr = geometric_batch.edge_attr
        y = geometric_batch.y
        pred = model(h, x, edge_index, graph_batch, edge_attr)
        # ------ MSE Loss ------
        target_norm = (y - target_mean.to(device)) / target_std.to(device)
        loss = nn.MSELoss()(pred, target_norm)
        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss + loss.detach().item()
        batch_count = batch_count + 1
        progress.set_postfix(loss=epoch_loss / batch_count)

    if batch_count == 0:
        raise ValueError("train loader is empty")
    return epoch_loss / batch_count

@torch.no_grad()
def evaluate(
    model: EGNNRegressor,
    loader: DataLoader,
    radial_basis: GaussianRadialBasis,
    device: torch.device,
    target_name: str | int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    graph_type: str = "fully_connected",
    num_atom_types: int = 100,
    radius: float | None = None,
    k: int | None = None,
) -> float:
    model.eval()
    total_abs_error = 0.0
    total_graphs = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        geometric_batch = prepare_qm9_batch(
            data=batch,
            target=target_name,
            radial_basis=radial_basis,
            graph=graph_type,
            num_atom_types=num_atom_types,
            radius=radius,
            k=k,
        )

        pred_norm = model(
            geometric_batch.h,
            geometric_batch.x,
            geometric_batch.edge_index,
            geometric_batch.batch,
            geometric_batch.edge_attr,
        )
        y = geometric_batch.y
        pred = pred_norm * target_std.to(device) + target_mean.to(device)
        total_abs_error += (pred - y).abs().sum().item()
        total_graphs += y.numel()

    if total_graphs == 0:
        raise ValueError("evaluation loader is empty")
    return total_abs_error / total_graphs
	    

def trainer(
        model: EGNNRegressor,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        radial_basis: GaussianRadialBasis,
        device: torch.device,
        target_name: str | int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        epochs: int,
        checkpoint_path: str,
        metadata: dict | None = None,
        graph_type: str = "fully_connected",
        num_atom_types: int = 100,
        radius: float | None = None,
        k: int | None = None,
        eval_every: int = 20,
        ):

    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    history = []

    for epoch_idx in range(1, epochs + 1):
        loss = train_epoch(
            model = model, 
            optimizer = optimizer,
            loader = train_loader,
            radial_basis = radial_basis,
            device = device, 
            target_name = target_name,
            target_mean = target_mean,
            target_std = target_std, 
            graph_type = graph_type, 
            num_atom_types = num_atom_types,
            radius = radius,
            k = k)

        if epoch_idx % eval_every == 0 or epoch_idx == epochs:
            val_mae = evaluate(
                model=model,
                loader=val_loader,
                radial_basis=radial_basis,
                device=device,
                target_name=target_name,
                target_mean=target_mean,
                target_std=target_std,
                graph_type=graph_type,
                num_atom_types=num_atom_types,
                radius=radius,
                k=k,
            )
            print(f"epoch={epoch_idx:03d} train_loss={loss:.6g} val_mae={val_mae:.6g}")
            history.append(
                {
                    "epoch": epoch_idx,
                    "train_loss": loss,
                    "val_mae": val_mae,
                }
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch_idx,
                        "best_val_mae": best_val_mae,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "history": history,
                        "test_mae": None,
                        "metadata": metadata or {},
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_mae = evaluate(
        model=model,
        loader=test_loader,
        radial_basis=radial_basis,
        device=device,
        target_name=target_name,
        target_mean=target_mean,
        target_std=target_std,
        graph_type=graph_type,
        num_atom_types=num_atom_types,
        radius=radius,
        k=k,
    )
    checkpoint["history"] = history
    checkpoint["test_mae"] = test_mae
    checkpoint["best_val_mae"] = best_val_mae
    torch.save(checkpoint, checkpoint_path)
    print(f"best_val_mae={best_val_mae:.6g} test_mae={test_mae:.6g}")
    return best_val_mae, test_mae


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trainer for QM9 regression using equimol")
    ap.add_argument("--datapath", type=str, required=True, help="Path to QM9 dataset root")
    ap.add_argument("--target", type=str, default="mu", help="Regression target from QM9")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic splitting")
    ap.add_argument("--train-frac", type=float, default=0.9, help="Training set fraction in (0, 1)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="Validation set fraction in [0, 1)")
    ap.add_argument("--fast-dev-run", action="store_true", help="Use small deterministic subsets")
    ap.add_argument("--fast-train-size", type=int, default=1000)
    ap.add_argument("--fast-val-size", type=int, default=200)
    ap.add_argument("--fast-test-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-atom-types", type=int, default=100)
    ap.add_argument("--lr", type = float, default = 3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--graph", choices=["fully_connected", "radius", "knn"], default="fully_connected")
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--edge-featurizer-num-basis", type=int, default=32)
    ap.add_argument("--edge-featurizer-cutoff", type=float, default=10.0)
    ap.add_argument("--edge-featurizer-gamma", type=float, default=None)
    ap.add_argument("--edge-featurizer-eps", type=float, default=1e-8)
    ap.add_argument("--egnn-num-layers", type=int, default=4)
    ap.add_argument("--egnn-hidden-dim", type=int, default=128)
    ap.add_argument("--egnn-message-dim", type=int, default=128)
    ap.add_argument("--egnn-dropout", type=float, default=0.0)
    ap.add_argument("--egnn-coord-step-size", type=float, default=0.1)
    ap.add_argument("--egnn-pooling", choices=["sum", "mean"], default="mean")
    ap.add_argument("--egnn-eps", type=float, default=1e-8)
    args = ap.parse_args(argv)
    set_seed(args.seed)

    # ----- Resolve device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load QM9 & generate splits -----
    datapath = Path(args.datapath)
    dataset = load_qm9(str(datapath))
    target_idx = get_qm9_target_index(target=args.target)
    train_idx, val_idx, test_idx = split_qm9(
        num_items=len(dataset),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    if args.fast_dev_run:
        train_idx, val_idx, test_idx = shrink_indices(
            train_idx,
            val_idx,
            test_idx,
            train_size=args.fast_train_size,
            val_size=args.fast_val_size,
            test_size=args.fast_test_size,
        )

    # ----- Target stats -----
    target_mean, target_std = compute_target_stats(
        dataset=dataset,
        indices=train_idx,
        target=args.target,
    )

    print(
        f"device={device} target={args.target} target_idx={target_idx} "
        f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()} "
        f"mean={target_mean.item():.6g} std={target_std.item():.6g}"
    )

    # ----- DataLoaders -----
    train_loader = DataLoader(
        dataset[train_idx.tolist()],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset[val_idx.tolist()],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset[test_idx.tolist()],
        batch_size=args.batch_size,
        shuffle=False,
    )

    edge_featurizer = build_radial_basis(
        num_basis=args.edge_featurizer_num_basis,
        cutoff=args.edge_featurizer_cutoff,
        gamma=args.edge_featurizer_gamma,
        eps=args.edge_featurizer_eps,
    ).to(device)

    model = EGNNRegressor(
        node_feat_dim=args.num_atom_types,
        num_layers=args.egnn_num_layers,
        hidden_dim=args.egnn_hidden_dim,
        edge_attr_dim=args.edge_featurizer_num_basis,
        message_dim=args.egnn_message_dim,
        dropout=args.egnn_dropout,
        coord_step_size=args.egnn_coord_step_size,
        pooling=args.egnn_pooling,
        eps=args.egnn_eps,
    ).to(device)

    optimizer = optim.AdamW(
        params = model.parameters(),
        lr = args.lr,
        weight_decay=args.weight_decay)

    checkpoint_path = args.checkpoint_path or f"checkpoints/qm9/{args.target}.pt"
    metadata = vars(args).copy()
    metadata["checkpoint_path"] = checkpoint_path
    metadata["target_idx"] = target_idx

    trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        radial_basis=edge_featurizer,
        device=device,
        target_name=args.target,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        graph_type=args.graph,
        num_atom_types=args.num_atom_types,
        radius=args.radius,
        k=args.k,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
