"""Train an EGNN energy/force regressor on MD17 or revised MD17 (rMD17).

rMD17 follows the NequIP/MACE protocol: for each official split, 950 train and
50 validation frames from the official train indices, the official test
indices as the test set, and test MAEs reported in meV and meV/A.

    python examples/trainers/md17_regression.py --dataset rmd17 --datapath data/rmd17 --splits 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from equimol.data import (
    KCAL_MOL_TO_MEV,
    build_radial_basis,
    compute_energy_stats,
    load_md17,
    load_rmd17,
    prepare_md17_batch,
    read_rmd17_split,
    split_md17,
    split_rmd17,
)
from equimol.models import (
    AttentiveEGNNRegressor,
    EGNNRegressor,
    IrrepEGNNRegressor,
    VectorEGNNRegressor,
)
from equimol.tasks import EnergyForceTask
from equimol.training import Trainer, TrainerConfig


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


def load_rmd17_dataset(datapath: str | Path, molecule: str) -> tuple[list, Path]:
    """Load an rMD17 molecule from the extracted rMD17 directory.

    Layout (as in the official rmd17.tar.bz2):
        <datapath>/npz_data/rmd17_<molecule>.npz
        <datapath>/splits/index_{train,test}_0k.csv

    Returns the frames and the splits directory. Load once and reuse the
    frames for every split; only the indices change between splits.
    """
    datapath = Path(datapath)
    npz_path = datapath / "npz_data" / f"rmd17_{molecule}.npz"
    splits_dir = datapath / "splits"
    if not npz_path.is_file():
        raise FileNotFoundError(f"rMD17 file not found: {npz_path}")
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"rMD17 splits directory not found: {splits_dir}")
    return load_rmd17(npz_path), splits_dir


def rmd17_split_indices(
    splits_dir: str | Path,
    split: int,
    num_val: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official split -> (train, val, test): val is carved from the official train set."""
    official_train_idx, test_idx = read_rmd17_split(splits_dir, split=split)
    train_idx, val_idx = split_rmd17(official_train_idx, num_val=num_val, seed=seed)
    return train_idx, val_idx, test_idx


def subset_dataset(dataset, indices: torch.Tensor):
    """Select frames by index from a PyG dataset (md17) or a list of frames (rmd17)."""
    if isinstance(dataset, list):
        return [dataset[int(i)] for i in indices]
    return dataset[indices.tolist()]

def main(argv=None):
    ap = argparse.ArgumentParser(description="Trainer for MD17 energy regression using equimol")
    ap.add_argument("--dataset", choices=["md17", "rmd17"], default="md17", help="Original MD17 or revised MD17")
    ap.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="MD17: PyG dataset root. rMD17: extracted rmd17 directory with npz_data/ and splits/",
    )
    ap.add_argument("--molecule", type=str, default="aspirin", help="Molecule name")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic splitting")
    ap.add_argument("--splits", type=int, nargs="+", default=[1], help="rMD17: official split ids to run (1-5)")
    ap.add_argument("--num-val", type=int, default=50, help="rMD17: validation frames held out of the official train set")
    ap.add_argument("--train-frac", type=float, default=0.9, help="Training set fraction in (0, 1)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="Validation set fraction in [0, 1)")
    ap.add_argument("--fast-dev-run", action="store_true", help="Use small deterministic subsets")
    ap.add_argument("--fast-train-size", type=int, default=1000)
    ap.add_argument("--fast-val-size", type=int, default=200)
    ap.add_argument("--fast-test-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--num-atom-types", type=int, default=100)
    ap.add_argument("--lr", type = float, default = 3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-energy", type=float, default=1.0)
    ap.add_argument("--lambda-force", type=float, default=1.0)
    ap.add_argument(
        "--energy-loss",
        choices=["per-atom", "total"],
        default="per-atom",
        help="Energy MSE on per-atom errors (NequIP/MACE convention) or on total energies",
    )
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--resume-from-checkpoint", type=str, default=None)
    ap.add_argument("--graph", choices=["fully_connected", "radius", "knn"], default="radius")
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--edge-featurizer-num-basis", type=int, default=32)
    ap.add_argument("--edge-featurizer-cutoff", type=float, default=10.0)
    ap.add_argument("--edge-featurizer-gamma", type=float, default=None)
    ap.add_argument("--edge-featurizer-eps", type=float, default=1e-8)
    ap.add_argument("--use-bond-features", action="store_true")
    ap.add_argument("--use-angle-features", action="store_true")
    ap.add_argument("--use-torsion-features", action="store_true")
    ap.add_argument("--model", choices=["egnn", "attentive-egnn", "vector-egnn", "irrep-egnn"], default="egnn")
    ap.add_argument("--egnn-num-layers", type=int, default=4)
    ap.add_argument("--egnn-hidden-dim", type=int, default=128)
    ap.add_argument("--egnn-message-dim", type=int, default=128)
    ap.add_argument("--egnn-vector-dim", type=int, default=64)
    ap.add_argument("--egnn-vector-gate", action="store_true")
    ap.add_argument("--egnn-attention-dim", type=int, default=128)
    ap.add_argument("--irreps-hidden", type=str, default="64x0e + 32x1o + 16x2e")
    ap.add_argument("--irreps-edge", type=str, default="0e + 1o + 2e")
    ap.add_argument("--irrep-radial-hidden-dim", type=int, default=128)
    ap.add_argument("--egnn-dropout", type=float, default=0.0)
    ap.add_argument("--egnn-coord-step-size", type=float, default=0.1)
    ap.add_argument("--egnn-pooling", choices=["sum", "mean"], default="mean")
    ap.add_argument("--egnn-eps", type=float, default=1e-8)
    ap.add_argument("--select-metric", choices=["energy", "force"], default="force",
                    help="Validation MAE that picks the best checkpoint and drives the LR scheduler")
    ap.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    ap.add_argument("--lr-factor", type=float, default=0.5, help="LR multiplier on a plateau")
    ap.add_argument("--lr-patience", type=int, default=10, help="Validations without improvement before cutting the LR")
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--ema-decay", type=float, default=None, help="EMA decay of the weights in (0, 1); off by default")
    args = ap.parse_args(argv)

    if args.dataset == "rmd17" and args.resume_from_checkpoint is not None and len(args.splits) > 1:
        raise ValueError("--resume-from-checkpoint resumes one run; pass a single --splits value")

    # ----- Resolve device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath = Path(args.datapath)
    checkpoint_name = args.molecule.replace(" ", "_").replace("/", "_")

    if args.dataset == "md17":
        # ----- Load MD17 & generate a random split -----
        dataset = load_md17(str(datapath), name=args.molecule)
        train_idx, val_idx, test_idx = split_md17(
            num_items=len(dataset),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        checkpoint_path = args.checkpoint_path or f"checkpoints/md17/{checkpoint_name}_energy.pt"
        run_split(args, device, dataset, train_idx, val_idx, test_idx, checkpoint_path)
        return None

    # ----- Load rMD17 once; each official split only changes the indices -----
    dataset, splits_dir = load_rmd17_dataset(datapath, args.molecule)
    base_checkpoint = Path(args.checkpoint_path or f"checkpoints/rmd17/{checkpoint_name}_energy.pt")

    per_split = []
    for split in args.splits:
        train_idx, val_idx, test_idx = rmd17_split_indices(splits_dir, split, args.num_val, args.seed)
        checkpoint_path = base_checkpoint.with_name(f"{base_checkpoint.stem}_split{split}{base_checkpoint.suffix}")
        print(f"rmd17 split={split}", flush=True)
        _, test_energy_mae, test_force_mae = run_split(
            args, device, dataset, train_idx, val_idx, test_idx, str(checkpoint_path)
        )
        per_split.append(
            {
                "split": split,
                "energy_mae_mev": float(KCAL_MOL_TO_MEV(test_energy_mae)),
                "force_mae_mev": float(KCAL_MOL_TO_MEV(test_force_mae)),
            }
        )

    results = {
        "per_split": per_split,
        "energy_mae_mev": mean_and_std([row["energy_mae_mev"] for row in per_split]),
        "force_mae_mev": mean_and_std([row["force_mae_mev"] for row in per_split]),
    }
    energy_mae_mean, energy_mae_std = results["energy_mae_mev"]
    force_mae_mean, force_mae_std = results["force_mae_mev"]
    print(
        f"rmd17 {args.molecule} splits={args.splits} "
        f"test_energy_mae={energy_mae_mean:.4g}±{energy_mae_std:.2g} meV "
        f"test_force_mae={force_mae_mean:.4g}±{force_mae_std:.2g} meV/A"
    )
    return results


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Mean and population standard deviation (0.0 for a single value)."""
    values = torch.tensor(values, dtype=torch.float64)
    return float(values.mean()), float(values.std(unbiased=False))


def run_split(
    args: argparse.Namespace,
    device: torch.device,
    dataset,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    checkpoint_path: str,
) -> tuple[float, float, float]:
    """Build loaders, model and optimizer for one split, train, and return
    (best_val_mae, test_energy_mae, test_force_mae) in kcal/mol units."""
    set_seed(args.seed)

    if args.fast_dev_run:
        train_idx, val_idx, test_idx = shrink_indices(
            train_idx,
            val_idx,
            test_idx,
            train_size=args.fast_train_size,
            val_size=args.fast_val_size,
            test_size=args.fast_test_size,
        )

    # ----- Energy stats -----
    energy_mean, energy_std = compute_energy_stats(
        dataset=dataset,
        indices=train_idx,
    )

    print(
        f"device={device} dataset={args.dataset} molecule={args.molecule} "
        f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()} "
        f"mean={energy_mean.item():.6g} std={energy_std.item():.6g}",
        flush=True,
    )

    # ----- DataLoaders -----
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        subset_dataset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        subset_dataset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        subset_dataset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    edge_featurizer = build_radial_basis(
        num_basis=args.edge_featurizer_num_basis,
        cutoff=args.edge_featurizer_cutoff,
        gamma=args.edge_featurizer_gamma,
        eps=args.edge_featurizer_eps,
    ).to(device)
    edge_attr_dim = args.edge_featurizer_num_basis
    if args.use_bond_features:
        edge_attr_dim += 2
    if args.use_angle_features:
        edge_attr_dim += 1
    if args.use_torsion_features:
        edge_attr_dim += 2

    if args.model == "egnn":
        model = EGNNRegressor(
            node_feat_dim=args.num_atom_types,
            num_layers=args.egnn_num_layers,
            hidden_dim=args.egnn_hidden_dim,
            edge_attr_dim=edge_attr_dim,
            message_dim=args.egnn_message_dim,
            dropout=args.egnn_dropout,
            update_coords=False,
            coord_step_size=args.egnn_coord_step_size,
            pooling=args.egnn_pooling,
            eps=args.egnn_eps,
        ).to(device)
    elif args.model == "attentive-egnn":
        model = AttentiveEGNNRegressor(
            node_feat_dim=args.num_atom_types,
            num_layers=args.egnn_num_layers,
            hidden_dim=args.egnn_hidden_dim,
            edge_attr_dim=edge_attr_dim,
            message_dim=args.egnn_message_dim,
            attention_dim=args.egnn_attention_dim,
            dropout=args.egnn_dropout,
            coord_step_size=args.egnn_coord_step_size,
            pooling=args.egnn_pooling,
            eps=args.egnn_eps,
        ).to(device)
    elif args.model == "vector-egnn":
        model = VectorEGNNRegressor(
            node_feat_dim=args.num_atom_types,
            num_layers=args.egnn_num_layers,
            hidden_dim=args.egnn_hidden_dim,
            edge_attr_dim=edge_attr_dim,
            message_dim=args.egnn_message_dim,
            vector_dim=args.egnn_vector_dim,
            attention=True,
            attention_dim=args.egnn_attention_dim,
            vector_gate=args.egnn_vector_gate,
            dropout=args.egnn_dropout,
            update_coords=False,
            coord_step_size=args.egnn_coord_step_size,
            pooling=args.egnn_pooling,
            eps=args.egnn_eps,
        ).to(device)
    else:
        model = IrrepEGNNRegressor(
            node_feat_dim=args.num_atom_types,
            num_layers=args.egnn_num_layers,
            edge_attr_dim=edge_attr_dim,
            irreps_hidden=args.irreps_hidden,
            irreps_edge=args.irreps_edge,
            radial_hidden_dim=args.irrep_radial_hidden_dim,
            attention=True,
            dropout=args.egnn_dropout,
            pooling=args.egnn_pooling,
            eps=args.egnn_eps,
        ).to(device)

    optimizer = optim.AdamW(
        params = model.parameters(),
        lr = args.lr,
        weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    metadata = vars(args).copy()
    metadata["checkpoint_path"] = checkpoint_path

    # The task turns batches into graphs with these settings; energies,
    # forces, loss and metrics live in EnergyForceTask, the loop in Trainer.
    featurize = partial(
        prepare_md17_batch,
        radial_basis=edge_featurizer,
        graph=args.graph,
        num_atom_types=args.num_atom_types,
        radius=args.radius,
        k=args.k,
        use_bond_features=args.use_bond_features,
        use_angle_features=args.use_angle_features,
        use_torsion_features=args.use_torsion_features,
    )
    task = EnergyForceTask(
        featurize,
        energy_mean,
        energy_std,
        lambda_energy=args.lambda_energy,
        lambda_force=args.lambda_force,
        per_atom_energy=args.energy_loss == "per-atom",
    )
    config = TrainerConfig(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        eval_every=args.eval_every,
        select_metric=f"{args.select_metric}_mae",
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        ema_decay=args.ema_decay,
    )
    metadata["energy_mean"] = float(energy_mean)
    metadata["energy_std"] = float(energy_std)

    last_time = time.perf_counter()

    def print_progress(row: dict[str, float]) -> None:
        nonlocal last_time
        now = time.perf_counter()
        print(
            f"epoch={row['epoch']} train_loss={row['train_loss']:.6g} "
            f"val_energy_mae={row['energy_mae']:.6g} val_force_mae={row['force_mae']:.6g} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} time={now - last_time:.1f}s",
            flush=True,
        )
        last_time = now

    trainer = Trainer(
        model,
        task,
        optimizer,
        config,
        scheduler=scheduler,
        device=device,
        metadata=metadata,
        on_validation=print_progress,
    )
    print(
        f"Starting the training process: model={args.model} epochs={args.epochs} "
        f"steps_per_epoch={len(train_loader)} parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    last_time = time.perf_counter()
    result = trainer.fit(train_loader, val_loader, resume_from=args.resume_from_checkpoint)
    test_metrics = trainer.test(test_loader)
    print(
        f"best_val_{config.select_metric}={result.best_metric:.6g} best_epoch={result.best_epoch} "
        f"test_energy_mae={test_metrics['energy_mae']:.6g} test_force_mae={test_metrics['force_mae']:.6g} "
        f"wall_time={result.wall_time / 3600:.2f}h",
        flush=True,
    )
    return result.best_metric, test_metrics["energy_mae"], test_metrics["force_mae"]


if __name__ == "__main__":
    main()
