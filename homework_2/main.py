"""
main.py  –  Entry point for Assignment 2 experiments.

CLI examples
------------
# Single run (one seed)
python main.py --dataset MUTAG --model gin --pool sum --hidden 64 --layers 4 --seed 0

# Multi-seed (reports mean ± std)
python main.py --dataset PROTEINS --model gcn --pool mean --hidden 64 --layers 4 --seeds 0 1 2

# OGB dataset
python main.py --dataset ogbg-molhiv --model gin --pool sum --hidden 64 --layers 4 --seeds 0 1 2

# Full experiment grid (all models × pools × 3 seeds)
python main.py --run_all --dataset MUTAG

# train.py also works as an alias
python train.py --dataset MUTAG --model gin --pool sum --seed 0
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data import DataManager
from models import build_model
from utils.utilities import (
    EarlyStopping,
    Trainer,
    set_seed,
    get_device,
    save_curves,
    save_json,
    print_table,
    save_csv,
)

# Experiment grid (do not modify)
GRID = [
    ("linear",   "mean"),
    ("mlp",      "mean"),
    ("deepsets", "sum"),
    ("deepsets", "mean"),
    ("gin",      "mean"),
    ("gin",      "sum"),
    ("gcn",      "mean"),
    ("gcn",      "sum"),
    ("sage",     "mean"),
    ("sage",     "sum"),
    ("gat",      "mean"),
    ("gat",      "sum"),
]


class ExperimentRunner:
    """Orchestrates single-run and multi-seed training experiments."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)

        if args.run_all:
            self._run_all()
            return

        seeds = list(args.seeds)
        print(f"\n{'═'*60}")
        print(f"  Dataset : {args.dataset}")
        print(f"  Model   : {args.model.upper()} | Pool: {args.pool}")
        print(f"  Hidden  : {args.hidden}  |  Layers: {args.layers}  |  Dropout: {args.dropout}")
        print(f"  LR      : {args.lr}  |  BS: {args.batch_size}")
        print(f"  Seeds   : {seeds}")
        print(f"{'═'*60}\n")

        if len(seeds) == 1:
            rec = self._run_once(seeds[0])
            print(f"\n  Final test {rec['metric']}: {rec['test_score']:.4f}")
        else:
            rec = self._run_multi_seed(seeds)
            print_table([rec])
            csv_path = str(Path(args.results_dir) / "summary.csv")
            save_csv([rec], csv_path)

    def _run_once(self, seed: int) -> dict:
        """Train and evaluate one (model, pool, seed) configuration."""
        args = self.args
        set_seed(seed)
        device = get_device()

        dm = DataManager(args.dataset, args.batch_size, args.num_workers, seed)
        _, train_loader, val_loader, test_loader, meta = dm.get_loaders()

        model = build_model(
            model_name      =args.model,
            in_channels     =meta["node_feat_dim"],
            hidden_channels =args.hidden,
            out_channels    =meta["num_tasks"],
            num_layers      =args.layers,
            dropout         =args.dropout,
            pool            =args.pool if args.model not in ("mlp", "linear") else "mean",
            use_atom_encoder=meta["use_atom_encoder"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.verbose:
            print(f"  Model: {args.model.upper()} | Pool: {args.pool} | "
                  f"Params: {n_params:,} | Seed: {seed} | Device: {device}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        trainer = Trainer(model, optimizer, scheduler, device, meta)
        stopper = EarlyStopping(patience=args.patience)

        history = {"train_loss": [], "val_loss": [], "val_score": []}
        best_val_score = -1.0
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            tr_loss          = trainer.train_epoch(train_loader)
            val_loss, val_sc = trainer.eval_epoch(val_loader)

            scheduler.step(val_loss)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_score"].append(val_sc)

            if val_sc > best_val_score:
                best_val_score = val_sc

            if args.verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(f"    Epoch {epoch:4d} | tr_loss {tr_loss:.4f} | "
                      f"val_loss {val_loss:.4f} | val_{meta['metric']} {val_sc:.4f} "
                      f"| {elapsed:.1f}s")

            if stopper.step(val_loss, model):
                if args.verbose:
                    print(f"    Early stop at epoch {epoch}.")
                break

        stopper.restore_best(model)
        _, test_score = trainer.eval_epoch(test_loader)

        if args.verbose:
            print(f"  → Test {meta['metric']}: {test_score:.4f}  "
                  f"(best_val {best_val_score:.4f})\n")

        run_name = (f"{args.dataset}_{args.model}_{args.pool}"
                    f"_h{args.hidden}_l{args.layers}_seed{seed}")
        save_dir = Path(args.results_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_curves(history, meta["metric"], save_dir / "curves.png", run_name)
        save_json(history, save_dir / "history.json")

        return {
            "dataset"   : args.dataset,
            "model"     : args.model,
            "pool"      : args.pool,
            "hidden"    : args.hidden,
            "layers"    : args.layers,
            "dropout"   : args.dropout,
            "seed"      : seed,
            "best_val"  : round(best_val_score, 5),
            "test_score": round(test_score, 5),
            "metric"    : meta["metric"],
            "n_params"  : n_params,
            "epochs_ran": len(history["train_loss"]),
        }

    def _run_multi_seed(self, seeds: list[int]) -> dict:
        """Run _run_once for each seed and aggregate results."""
        records = [self._run_once(seed) for seed in seeds]

        val_scores  = [r["best_val"]   for r in records]
        test_scores = [r["test_score"] for r in records]

        summary = deepcopy(records[0])
        summary["seeds"]      = seeds
        summary["val_mean"]   = round(float(np.mean(val_scores)),  5)
        summary["val_std"]    = round(float(np.std(val_scores)),   5)
        summary["test_mean"]  = round(float(np.mean(test_scores)), 5)
        summary["test_std"]   = round(float(np.std(test_scores)),  5)
        summary["test_score"] = summary["test_mean"]
        summary["best_val"]   = summary["val_mean"]

        print(f"\n  ══ {self.args.model.upper()} | pool={self.args.pool} ══")
        print(f"  Val  {summary['metric']}: {summary['val_mean']:.4f} ± {summary['val_std']:.4f}")
        print(f"  Test {summary['metric']}: {summary['test_mean']:.4f} ± {summary['test_std']:.4f}\n")
        return summary

    def _run_all(self) -> list[dict]:
        """Run the full experiment grid."""
        args = self.args
        all_records = []
        seeds = list(args.seeds)
        orig_model, orig_pool = args.model, args.pool

        for model_name, pool in GRID:
            print(f"\n{'═'*60}")
            print(f"  Running: {model_name.upper()} | pool={pool} | seeds={seeds}")
            print(f"{'═'*60}")
            args.model, args.pool = model_name, pool
            rec = self._run_multi_seed(seeds)
            all_records.append(rec)

        args.model, args.pool = orig_model, orig_pool
        print_table(all_records)
        csv_path = str(Path(args.results_dir) / "summary.csv")
        save_csv(all_records, csv_path)
        return all_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GNN for graph classification (Assignment 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",      type=str,   default="ogbg-molhiv",
                   help="Dataset: ogbg-molhiv | PROTEINS | IMDB-MULTI | REDDIT-BINARY")
    p.add_argument("--model",        type=str,   default="gin",
                   choices=["linear", "mlp", "deepsets", "gin", "gcn", "sage", "gat"])
    p.add_argument("--pool",         type=str,   default="mean",
                   choices=["mean", "sum", "attention", "set2set"])
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--layers",       type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--seeds",        type=int,   nargs="+", default=[0, 1, 2])
    p.add_argument("--results_dir",  type=str,   default="results")
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--verbose",      action="store_true", default=True)
    p.add_argument("--run_all",      action="store_true", default=False,
                   help="Run full experiment grid (all models × pools × seeds)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    ExperimentRunner(args).run()
