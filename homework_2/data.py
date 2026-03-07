"""
data.py  –  Dataset loading, splitting, and DataLoader creation.

Supported datasets
------------------
  OGB  : ogbg-molhiv  (scaffold split provided by OGB – do NOT change)
  TU   : PROTEINS, IMDB-MULTI, REDDIT-BINARY  (you implement the split)

TODOs in this file
------------------
"""

import os

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.transforms import OneHotDegree

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

OGB_DATASETS = {"ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21"}
TU_DATASETS  = {"MUTAG", "PROTEINS", "IMDB-MULTI", "REDDIT-BINARY"}


class DataManager:
    """Loads a dataset and creates DataLoaders for train / val / test splits."""

    def __init__(
        self,
        dataset_name: str,
        batch_size  : int = 64,
        num_workers : int = 0,
        seed        : int = 42,
    ):
        self.dataset_name = dataset_name
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.seed         = seed

    def get_loaders(self):
        """
        Return (dataset, train_loader, val_loader, test_loader, meta).

        Dispatches to the OGB or TU loader based on dataset name.
        OGB uses the official scaffold split (do not modify).
        TU uses a stratified split you implement in _tu_loaders
        """
        name = self.dataset_name
        if name.lower() in {d.lower() for d in OGB_DATASETS}:
            return self._ogb_loaders()
        if name.upper() in TU_DATASETS:
            return self._tu_loaders()
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {OGB_DATASETS | TU_DATASETS}"
        )

    # OGB loader (do not modify – official scaffold split required by assignment)

    def _ogb_loaders(self):
        """Return (dataset, train_loader, val_loader, test_loader, meta) for an OGB dataset."""
        from ogb.graphproppred import PygGraphPropPredDataset

        dataset   = PygGraphPropPredDataset(name=self.dataset_name.lower(), root=DATA_ROOT)
        split_idx = dataset.get_idx_split()

        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=self.batch_size,
            shuffle=True,  num_workers=self.num_workers, pin_memory=False
        )
        val_loader = DataLoader(
            dataset[split_idx["valid"]], batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=False
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=False
        )

        meta = {
            "num_classes"     : dataset.num_classes,
            "num_tasks"       : dataset.num_tasks,
            "node_feat_dim"   : dataset[0].x.shape[1],
            "task_type"       : "classification",
            "metric"          : "rocauc",
            "use_atom_encoder": True,
            "dataset_type"    : "ogb",
        }
        return dataset, train_loader, val_loader, test_loader, meta



    def _tu_loaders(self, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Return (dataset, train_loader, val_loader, test_loader, meta) for a TU dataset.

        Steps you need to implement
        ---------------------------
        1. Load the TUDataset. Use `use_node_attr=True`.
           - For datasets WITHOUT node features (dataset[0].x is None), add
             one-hot degree features using OneHotDegree(max_degree).
             Compute max_degree from the dataset before re-loading.

        2. Extract integer labels:  labels = [int(d.y.item()) for d in dataset]

        3. Perform a **stratified** split into train / val / test using
           sklearn.model_selection.train_test_split.
           - First split off `test_ratio` as the test set.
           - Then split the remaining into train / val with effective ratio
             val_ratio / (1 - test_ratio).
           - Use `random_state=self.seed` and `stratify=` in both splits.

        4. Build three DataLoaders (train: shuffle=True, val/test: shuffle=False).

        5. Build and return the `meta` dict with keys:
             num_classes, num_tasks (=1), node_feat_dim, task_type,
             metric ("rocauc" for binary, "accuracy" for multi-class),
             use_atom_encoder (=False), dataset_type (="tu")

        Parameters
        ----------
        val_ratio  : fraction of the full dataset to use for validation
        test_ratio : fraction of the full dataset to use for testing
        """

        dataset = TUDataset(root=DATA_ROOT, name=self.dataset_name, use_node_attr=True)

        if dataset[0].x is None:
            max_degree = 0
            for data in dataset:
                row = data.edge_index[0]
                deg = torch.bincount(row, minlength=data.num_nodes)
                max_degree = max(max_degree, int(deg.max().item()))

            dataset = TUDataset(
                root=DATA_ROOT,
                name=self.dataset_name,
                use_node_attr=True,
                transform=OneHotDegree(max_degree)
            )

        labels = [int(data.y.item()) for data in dataset]
        indices = list(range(len(dataset)))

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        train_val_labels = [labels[i] for i in train_val_idx]
        val_effective_ratio = val_ratio / (1.0 - test_ratio)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_effective_ratio,
            random_state=self.seed,
            stratify=train_val_labels,
        )

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        num_classes = dataset.num_classes
        node_feat_dim = dataset[0].x.shape[1]
        metric = "accuracy"

        meta = {
            "num_classes": num_classes,
            "num_tasks": 1,
            "node_feat_dim": node_feat_dim,
            "task_type": "classification",
            "metric": metric,
            "use_atom_encoder": False,
            "dataset_type": "tu",
        }

        return dataset, train_loader, val_loader, test_loader, meta


    # TODO 7 – k-fold cross-validation

    def get_kfold_loaders(self, n_splits: int = 10):
        """
        Yield (fold_idx, train_loader, val_loader, meta) for k-fold cross-validation.
        Intended for small TU datasets where a fixed split is too noisy.

        Steps you need to implement
        ---------------------------
        1. Load the TUDataset (same degree feature handling as _tu_loaders).
        2. Build `labels` array and `meta` dict.
        3. Use StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
           and yield (fold, train_loader, val_loader, meta) for each fold.

        Hint: sklearn's StratifiedKFold.split(X, y) returns (train_idx, val_idx)
              pairs.  Use them to index into `dataset`.

        TODO: Implement this generator.
        """
        # Note:
        # The k-fold cross-validation utility in data.py was left unimplemented because
        # it was not required for the final reported experiments. The reported results
        # were obtained using stratified train/validation/test splits for TU datasets and
        # the official OGB split for ogbg-molhiv.
        raise NotImplementedError("TODO 7: implement DataManager.get_kfold_loaders")
