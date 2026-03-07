"""
models.py  –  GNN architectures for graph classification.
Assignment 2: Structural Inductive Bias in Graph Neural Networks

Models
------
  LinearBaseline   : Logistic regression on mean-pooled node features      [PROVIDED]
  MLPBaseline      : Global mean-pool node embeddings → MLP                [PROVIDED]
  DeepSets         : Per-node MLP → pool → MLP                             [PROVIDED]
  GINModel         : Graph Isomorphism Network (Xu et al. 2019)            [TODO] -- DONE
  GCNModel         : Graph Convolutional Network (Kipf & Welling 2017)     [TODO] -- DOne
  GraphSAGEModel   : Inductive Representation Learning (Hamilton et al.)   [TODO] -- DONE
  GATModel         : Graph Attention Network v2 (Brody et al. 2022)        [TODO] --  DONE

Pooling / Readout options
-------------------------
  mean          : global_mean_pool
  sum           : global_add_pool
  attention     : GlobalAttention (Li et al., 2016)
  set2set       : Set2Set (Vinyals et al., 2015)

You must report for each model:
  - Hidden dimension
  - Number of layers
  - Aggregation function (e.g. sum, mean)
  - Readout / pooling function
  - Activation functions
  - Regularisation strategy (BatchNorm, Dropout, …)
  - Total parameter count

Key references
--------------
  [1]  Xu et al. (2019).  "How Powerful are Graph Neural Networks?"
       ICLR 2019.  https://arxiv.org/abs/1810.00826

  [2]  Kipf & Welling (2017).  "Semi-Supervised Classification with Graph
       Convolutional Networks."  ICLR 2017.  https://arxiv.org/abs/1609.02907

  [3]  Hamilton et al. (2017).  "Inductive Representation Learning on Large
       Graphs."  NeurIPS 2017.  https://arxiv.org/abs/1706.02216

  [4]  Veličković et al. (2018).  "Graph Attention Networks."  ICLR 2018.
       https://arxiv.org/abs/1710.10903

  [5]  Brody et al. (2022).  "How Attentive are Graph Attention Networks?"
       ICLR 2022.  https://arxiv.org/abs/2105.14491

  [6]  Zaheer et al. (2017).  "Deep Sets."  NeurIPS 2017.
       https://arxiv.org/abs/1703.06114

  [7]  Xu et al. (2018).  "Representation Learning on Graphs with Jumping
       Knowledge Networks."  ICML 2018.  https://arxiv.org/abs/1806.03536

  [8]  Vinyals et al. (2016).  "Order Matters: Sequence to Sequence for
       Sets."  ICLR 2016.  https://arxiv.org/abs/1511.06391

  [9]  Li et al. (2016).  "Gated Graph Sequence Neural Networks."
       ICLR 2016.  https://arxiv.org/abs/1511.05493

  [10] Ioffe & Szegedy (2015).  "Batch Normalization: Accelerating Deep
       Network Training."  ICML 2015.  https://arxiv.org/abs/1502.03167

  [11] Srivastava et al. (2014).  "Dropout: A Simple Way to Prevent Neural
       Networks from Overfitting."  JMLR 2014.
       https://jmlr.org/papers/v15/srivastava14a.html

  [12] Hu et al. (2020).  "Open Graph Benchmark: Datasets for Machine
       Learning on Graphs."  NeurIPS 2020.  https://arxiv.org/abs/2005.00687
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
    global_add_pool,
    GlobalAttention,
    Set2Set,
)


# Utilities (do not modify)

def _make_atom_encoder(use_atom_encoder: bool, in_channels: int, hidden_channels: int):
    """Return an atom encoder: AtomEncoder (OGB) or a plain Linear projection."""
    if use_atom_encoder:
        from ogb.graphproppred.mol_encoder import AtomEncoder
        return AtomEncoder(emb_dim=hidden_channels)
    return nn.Linear(in_channels, hidden_channels)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
    """Build a multi-layer MLP with BatchNorm + ReLU + Dropout on hidden layers."""
    assert num_layers >= 1
    layers: list[nn.Module] = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


# Pooling helpers

def _get_pool_layer(pool: str, hidden_channels: int):
    """
    Return (pool_module_or_None, graph_emb_dim).

    pool_module is None for mean/sum (handled inline with global_*_pool).
    For attention and set2set, return the appropriate PyG module.

    Pooling strategies
    ------------------
    mean / sum
        Simple permutation-invariant aggregation; no learnable parameters.
        Both are theoretically justified as universal approximators over
        multisets when composed with an MLP  (Xu et al., 2019  [1]).

    attention (GlobalAttention)
        Soft-attention gate: g(x_i) = σ(W x_i), readout = Σ g_i · x_i.
        Originally proposed as a gating mechanism in Gated Graph Neural
        Networks  (Li et al., 2016  [9]).
        PyG: torch_geometric.nn.GlobalAttention

    set2set
        LSTM-based iterative attention that is order-invariant and produces
        a 2H-dimensional graph embedding.
        Vinyals et al. (2016)  "Order Matters"  [8].
        PyG: torch_geometric.nn.Set2Set

    Note: Set2Set doubles the embedding dimension (returns hidden_channels * 2).
    """
    if pool == "mean":
        return None, hidden_channels
    if pool == "sum":
        return None, hidden_channels
    if pool == "attention":
        gate_nn = nn.Sequential(nn.Linear(hidden_channels, 1))
        return GlobalAttention(gate_nn=gate_nn), hidden_channels
    if pool == "set2set":
        return Set2Set(hidden_channels, processing_steps=3), hidden_channels * 2
    raise ValueError(f"Unknown pool '{pool}'. Choose from: mean, sum, attention, set2set")


def _apply_pool(pool_name: str, pool_module, x, batch):
    """
    Apply the chosen readout / pooling function.

    Parameters
    ----------
    pool_name   : 'mean' | 'sum' | 'attention' | 'set2set'
    pool_module : None for mean/sum, the PyG module for attention/set2set
    x           : node embeddings  (N, H)
    batch       : batch vector     (N,)

    Returns
    -------
    graph_emb : (B, H) or (B, 2H) for set2set
    """
    if pool_name == "mean":
        return global_mean_pool(x, batch)
    if pool_name == "sum":
        return global_add_pool(x, batch)
    # attention / set2set – use the module
    return pool_module(x, batch)


# 0. Linear Baseline (do not modify – study as reference)

class LinearBaseline(nn.Module):
    """
    Simplest possible baseline: global mean-pool raw node features → single
    linear layer.  Equivalent to logistic regression on the graph-level
    feature vector.

    Pipeline:
      atom_encoder(x)  →  global_mean_pool  →  Linear  →  logits
    """

    def __init__(
        self,
        in_channels    : int,
        hidden_channels: int,
        out_channels   : int,
        num_layers     : int   = 1,
        dropout        : float = 0.0,
        use_atom_encoder: bool = True,
    ):
        super().__init__()
        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)
        self.classifier   = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.atom_encoder(x)               # (N, H)
        h = global_mean_pool(h, batch)          # (B, H)
        return self.classifier(h)               # (B, out)


# 1. MLP Baseline (do not modify – study as reference)

class MLPBaseline(nn.Module):
    """
    Non-graph baseline: no message passing.

    Pipeline:
      atom_encoder(x)  →  global_mean_pool  →  MLP  →  logits
    """

    def __init__(
        self,
        in_channels    : int,
        hidden_channels: int,
        out_channels   : int,
        num_layers     : int  = 3,
        dropout        : float = 0.5,
        use_atom_encoder: bool = True,
    ):
        super().__init__()
        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)
        self.mlp = _make_mlp(hidden_channels, hidden_channels, out_channels, num_layers, dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.atom_encoder(x)               # (N, H)
        h = global_mean_pool(h, batch)          # (B, H)
        return self.mlp(h)                      # (B, out)


# 1b. DeepSets (do not modify – study as reference)

class DeepSets(nn.Module):
    """
    Deep Sets  (Zaheer et al., NeurIPS 2017).

    Applies a learned per-node MLP (φ) *before* pooling, then a second MLP (ρ)
    on the graph-level embedding.  No message passing.

    Pipeline:
      atom_encoder(x)  →  φ-MLP(x_i) per node  →  pool  →  ρ-MLP  →  logits

    Reference
    ---------
    Zaheer et al. (2017).  "Deep Sets."  NeurIPS 2017.
    https://arxiv.org/abs/1703.06114
    """

    def __init__(
        self,
        in_channels    : int,
        hidden_channels: int,
        out_channels   : int,
        num_layers     : int   = 3,
        dropout        : float = 0.5,
        pool           : str   = "sum",
        use_atom_encoder: bool = True,
    ):
        super().__init__()
        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)

        phi_layers: list[nn.Module] = []
        for _ in range(max(num_layers - 1, 1)):
            phi_layers += [
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        self.phi = nn.Sequential(*phi_layers)

        self.pool_name   = pool
        self.pool_module, graph_emb_dim = _get_pool_layer(pool, hidden_channels)

        self.rho = _make_mlp(graph_emb_dim, hidden_channels, out_channels, 2, dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.atom_encoder(x)
        h = self.phi(h)
        g = _apply_pool(self.pool_name, self.pool_module, h, batch)
        return self.rho(g)



class GINBlock(nn.Module):
    """
    Single GIN layer: MLP( (1 + ε) · h_v + Σ_{u∈N(v)} h_u ) with learnable ε.

    Update rule  (Xu et al., 2019  eq. 4.1):
      h_v^(k) = MLP^(k)( (1 + ε^(k)) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1) )

    Hint: Use GINConv from torch_geometric.nn with train_eps=True.
          The inner MLP should be:
            Linear(H → 2H) → BatchNorm1d(2H) → ReLU → Linear(2H → H)
          After the convolution apply: BatchNorm → ReLU → Dropout.

    Reference
    ---------
    Xu et al. (2019).  "How Powerful are Graph Neural Networks?"  ICLR 2019.
    https://arxiv.org/abs/1810.00826

    """

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GINModel(nn.Module):
    """
    Graph Isomorphism Network for graph classification.
    Xu et al. (2019).  "How Powerful are Graph Neural Networks?"  ICLR 2019.
    https://arxiv.org/abs/1810.00826

    Architecture
    ------------
      AtomEncoder  →  [GINBlock] × num_layers  →  Readout  →  Linear(classifier)

    The readout aggregates node embeddings from **all layers** (Jumping Knowledge
    / JK-style concatenation). When jk=True the classifier input size is
    (num_layers + 1) * hidden_channels (layer-0 is the encoder output).

    Jumping Knowledge reference
    ---------------------------
    Xu et al. (2018).  "Representation Learning on Graphs with Jumping
    Knowledge Networks."  ICML 2018.  https://arxiv.org/abs/1806.03536

    Regularisation
    --------------
    BatchNorm  (Ioffe & Szegedy, 2015  https://arxiv.org/abs/1502.03167)
    Dropout    (Srivastava et al., 2014  https://jmlr.org/papers/v15/srivastava14a.html)

    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 4,
            dropout: float = 0.5,
            pool: str = "mean",
            use_atom_encoder: bool = True,
            jk: bool = True,
    ):
        super().__init__()

        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            GINBlock(hidden_channels, dropout) for _ in range(num_layers)
        ])

        self.pool_name = pool
        self.pool_module, graph_emb_dim = _get_pool_layer(pool, hidden_channels)

        self.jk = jk
        if self.jk:
            classifier_in_dim = graph_emb_dim * (num_layers + 1)
        else:
            classifier_in_dim = graph_emb_dim

        self.classifier = nn.Linear(classifier_in_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.atom_encoder(x)

        layer_outputs = [x]
        for block in self.blocks:
            x = block(x, edge_index)
            layer_outputs.append(x)

        if self.jk:
            pooled = [
                _apply_pool(self.pool_name, self.pool_module, h, batch)
                for h in layer_outputs
            ]
            g = torch.cat(pooled, dim=-1)
        else:
            g = _apply_pool(self.pool_name, self.pool_module, x, batch)

        return self.classifier(g)



class GCNBlock(nn.Module):
    """
    Single GCN layer with BatchNorm + ReLU + Dropout.

    Spectral update rule  (Kipf & Welling, 2017  eq. 2):
      H^(l+1) = σ( D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l) )
    where Ã = A + I  and  D̃ is the corresponding degree matrix.

    Hint: Use GCNConv(hidden_channels, hidden_channels).
          After convolution apply: BatchNorm → ReLU → Dropout.

    Reference
    ---------
    Kipf & Welling (2017).  "Semi-Supervised Classification with Graph
    Convolutional Networks."  ICLR 2017.  https://arxiv.org/abs/1609.02907

    """

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for graph classification.
    Kipf & Welling (2017).  "Semi-Supervised Classification with Graph
    Convolutional Networks."  ICLR 2017.  https://arxiv.org/abs/1609.02907

    Architecture
    ------------
      AtomEncoder  →  [GCNBlock] × num_layers  →  Readout  →  Linear(classifier)

    Regularisation
    --------------
    BatchNorm  (Ioffe & Szegedy, 2015  https://arxiv.org/abs/1502.03167)
    Dropout    (Srivastava et al., 2014  https://jmlr.org/papers/v15/srivastava14a.html)

    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 4,
            dropout: float = 0.5,
            pool: str = "mean",
            use_atom_encoder: bool = True,
    ):
        super().__init__()

        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            GCNBlock(hidden_channels, dropout) for _ in range(num_layers)
        ])

        self.pool_name = pool
        self.pool_module, graph_emb_dim = _get_pool_layer(pool, hidden_channels)

        self.classifier = nn.Linear(graph_emb_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.atom_encoder(x)

        for block in self.blocks:
            x = block(x, edge_index)

        g = _apply_pool(self.pool_name, self.pool_module, x, batch)
        return self.classifier(g)



class GraphSAGEBlock(nn.Module):
    """
    Single GraphSAGE layer (mean aggregator) with BatchNorm + ReLU + Dropout.

    Inductive update rule  (Hamilton et al., 2017  Algorithm 1):
      h_{N(v)}^k = AGGREGATE_k({ h_u^{k-1}, ∀u ∈ N(v) })
      h_v^k = σ( W^k · CONCAT(h_v^{k-1}, h_{N(v)}^k) )
    Here AGGREGATE is mean pooling over neighbours.

    Hint: Use SAGEConv(hidden_channels, hidden_channels, aggr="mean").

    Reference
    ---------
    Hamilton et al. (2017).  "Inductive Representation Learning on Large
    Graphs."  NeurIPS 2017.  https://arxiv.org/abs/1706.02216

    """

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE for graph classification.
    Hamilton et al. (2017).  "Inductive Representation Learning on Large
    Graphs."  NeurIPS 2017.  https://arxiv.org/abs/1706.02216

    Architecture
    ------------
      AtomEncoder  →  [SAGEBlock] × num_layers  →  Readout  →  Linear(classifier)

    Regularisation
    --------------
    BatchNorm  (Ioffe & Szegedy, 2015  https://arxiv.org/abs/1502.03167)
    Dropout    (Srivastava et al., 2014  https://jmlr.org/papers/v15/srivastava14a.html)

    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 4,
            dropout: float = 0.5,
            pool: str = "mean",
            use_atom_encoder: bool = True,
    ):
        super().__init__()

        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            GraphSAGEBlock(hidden_channels, dropout) for _ in range(num_layers)
        ])

        self.pool_name = pool
        self.pool_module, graph_emb_dim = _get_pool_layer(pool, hidden_channels)

        self.classifier = nn.Linear(graph_emb_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.atom_encoder(x)

        for block in self.blocks:
            x = block(x, edge_index)

        g = _apply_pool(self.pool_name, self.pool_module, x, batch)
        return self.classifier(g)



class GATBlock(nn.Module):
    """
    Single GATv2 layer.

    GATv2 attention coefficient  (Brody et al., 2022  eq. 3):
      e_{uv} = a^T · LeakyReLU( W · [h_u || h_v] )
      α_{uv} = softmax_v( e_{uv} )
      h_v'   = σ( Σ_{u∈N(v)} α_{uv} · W h_u )

    Compared with GATv1  (Veličković et al., 2018  https://arxiv.org/abs/1710.10903),
    GATv2 is strictly more expressive (the attention is dynamic rather than
    static).

    Hints:
      - Use GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                      concat=False, dropout=dropout)
        Setting concat=False averages the heads so the output dim stays H.
      - After convolution apply: BatchNorm → ReLU → Dropout.

    References
    ----------
    Brody et al. (2022).  "How Attentive are Graph Attention Networks?"
    ICLR 2022.  https://arxiv.org/abs/2105.14491

    Veličković et al. (2018).  "Graph Attention Networks."  ICLR 2018.
    https://arxiv.org/abs/1710.10903

    """

    def __init__(self, hidden_channels: int, dropout: float, heads: int = 4):
        super().__init__()
        self.conv = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GATModel(nn.Module):
    """
    Graph Attention Network v2 for graph classification.
    Brody et al. (2022).  "How Attentive are Graph Attention Networks?"
    ICLR 2022.  https://arxiv.org/abs/2105.14491

    Architecture
    ------------
      AtomEncoder  →  [GATBlock] × num_layers  →  Readout  →  Linear(classifier)

    Regularisation
    --------------
    BatchNorm  (Ioffe & Szegedy, 2015  https://arxiv.org/abs/1502.03167)
    Dropout applied inside GATv2Conv (on attention weights) and after each block.
    Srivastava et al. (2014  https://jmlr.org/papers/v15/srivastava14a.html)

    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 4,
            dropout: float = 0.5,
            pool: str = "mean",
            use_atom_encoder: bool = True,
            heads: int = 4,
    ):
        super().__init__()

        self.atom_encoder = _make_atom_encoder(use_atom_encoder, in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            GATBlock(hidden_channels, dropout, heads=heads) for _ in range(num_layers)
        ])

        self.pool_name = pool
        self.pool_module, graph_emb_dim = _get_pool_layer(pool, hidden_channels)

        self.classifier = nn.Linear(graph_emb_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.atom_encoder(x)

        for block in self.blocks:
            x = block(x, edge_index)

        g = _apply_pool(self.pool_name, self.pool_module, x, batch)
        return self.classifier(g)


# Factory (do not modify)

def build_model(
    model_name     : str,
    in_channels    : int,
    hidden_channels: int,
    out_channels   : int,
    num_layers     : int,
    dropout        : float,
    pool           : str,
    use_atom_encoder: bool,
) -> nn.Module:
    """Instantiate a model by name."""
    kwargs = dict(
        in_channels    =in_channels,
        hidden_channels=hidden_channels,
        out_channels   =out_channels,
        num_layers     =num_layers,
        dropout        =dropout,
        use_atom_encoder=use_atom_encoder,
    )
    name = model_name.lower()
    if name == "linear":
        return LinearBaseline(**kwargs)
    if name == "mlp":
        return MLPBaseline(**kwargs)
    if name == "deepsets":
        return DeepSets(**kwargs, pool=pool)
    if name == "gin":
        return GINModel(**kwargs, pool=pool)
    if name == "gcn":
        return GCNModel(**kwargs, pool=pool)
    if name == "sage":
        return GraphSAGEModel(**kwargs, pool=pool)
    if name == "gat":
        return GATModel(**kwargs, pool=pool)
    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Choose from: linear, mlp, deepsets, gin, gcn, sage, gat"
    )
