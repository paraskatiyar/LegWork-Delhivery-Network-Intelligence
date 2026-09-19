"""Node representations: node2vec, GraphSAGE, and a spectral control.

Both learned embedders are implemented here directly against NumPy and
PyTorch rather than pulled from `node2vec`/`torch-geometric`. That is a
deliberate choice, not a workaround: it keeps the dependency surface to
scikit-learn + torch, it makes the second-order walk bias and the aggregation
step readable and testable, and it lets both embedders be fitted strictly on
the training graph -- which is the property that actually decides whether the
measured "graph advantage" means anything.

Three representations are produced so the ablation can separate *having a
graph* from *which embedding you chose*:

  * `SVDEmbedder`     - truncated SVD of the weighted adjacency. Cheap,
                        deterministic, no training. The control: if the learned
                        embedders cannot beat this, the extra machinery is not
                        earning its keep.
  * `Node2VecEmbedder`- second-order biased random walks + skip-gram with
                        negative sampling. Transductive: it can only embed
                        facilities present in the training graph.
  * `GraphSAGEEmbedder` - two-layer mean-aggregator, trained unsupervised on a
                        graph-context loss. Inductive over *features*, so a
                        facility that is new to the graph but has known
                        neighbours can still be embedded at serve time.

Cold-start honesty: a facility with neither history nor neighbours cannot be
embedded by any of the three. Those rows get the training-set mean embedding
and are flagged via `*_is_cold_start`, so the model can learn to widen its
estimate rather than pretending it knows something.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD

from .config import Config


class BaseEmbedder(ABC):
    name: str = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.nodes: list[str] = []
        self.index: dict[str, int] = {}
        self.matrix: np.ndarray | None = None

    @abstractmethod
    def fit(self, graph: nx.DiGraph) -> "BaseEmbedder": ...

    @property
    def dim(self) -> int:
        return self.cfg.embed.dim

    def _finalise(self, matrix: np.ndarray) -> None:
        # L2-normalise so downstream distance/difference features are scale-free
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.clip(norms, 1e-9, None)

    def lookup(self, keys: pd.Series) -> np.ndarray:
        """Embed a column of facility ids; unknown ids get the training mean."""
        assert self.matrix is not None, "Embedder not fitted"
        fallback = self.matrix.mean(axis=0)
        out = np.tile(fallback, (len(keys), 1))
        positions = keys.map(self.index)
        known = positions.notna().to_numpy()
        if known.any():
            out[known] = self.matrix[positions[known].astype(int).to_numpy()]
        return out

    def frame(self, legs: pd.DataFrame) -> pd.DataFrame:
        """Source/destination embeddings plus corridor interaction terms."""
        src = self.lookup(legs["source_center"])
        dst = self.lookup(legs["destination_center"])
        data: dict[str, np.ndarray] = {}
        for i in range(self.dim):
            data[f"{self.name}_src_{i:02d}"] = src[:, i]
            data[f"{self.name}_dst_{i:02d}"] = dst[:, i]
            # The directed difference encodes "which way along the corridor",
            # which a plain concatenation makes the model rediscover.
            data[f"{self.name}_diff_{i:02d}"] = src[:, i] - dst[:, i]
        data[f"{self.name}_cosine"] = np.sum(src * dst, axis=1)
        return pd.DataFrame(data, index=legs.index)


class SVDEmbedder(BaseEmbedder):
    name = "svd"

    def fit(self, graph: nx.DiGraph) -> "SVDEmbedder":
        self.nodes = sorted(graph.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        adjacency = nx.to_numpy_array(graph, nodelist=self.nodes, weight="leg_count")
        # Symmetrise then log-damp: raw leg counts span four orders of magnitude
        # and would otherwise let a handful of trunk lanes dominate every axis.
        adjacency = np.log1p(adjacency + adjacency.T)
        components = max(1, min(self.dim, len(self.nodes) - 1))
        svd = TruncatedSVD(n_components=components, random_state=self.cfg.seed)
        matrix = svd.fit_transform(adjacency)
        if components < self.dim:
            matrix = np.hstack([matrix, np.zeros((len(self.nodes), self.dim - components))])
        self._finalise(matrix)
        return self


class Node2VecEmbedder(BaseEmbedder):
    """Second-order biased walks + skip-gram with negative sampling."""

    name = "n2v"

    def _build_adjacency(self, graph: nx.DiGraph):
        undirected = graph.to_undirected()
        neighbours: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for node in self.nodes:
            nbrs = list(undirected.neighbors(node))
            neighbours.append(np.array([self.index[n] for n in nbrs], dtype=np.int64))
            weights.append(np.array(
                [max(undirected[node][n].get("leg_count", 1.0), 1e-6) for n in nbrs],
                dtype=np.float64,
            ))
        return neighbours, weights

    def _walks(self, neighbours, weights, rng: np.random.Generator) -> np.ndarray:
        p, q = self.cfg.embed.p, self.cfg.embed.q
        length = self.cfg.embed.walk_length
        neighbour_sets = [set(n.tolist()) for n in neighbours]
        walks = []
        starts = [i for i in range(len(self.nodes)) if len(neighbours[i]) > 0]
        for _ in range(self.cfg.embed.walks_per_node):
            rng.shuffle(starts)
            for start in starts:
                walk = [start]
                while len(walk) < length:
                    current = walk[-1]
                    nbrs = neighbours[current]
                    if len(nbrs) == 0:
                        break
                    if len(walk) == 1:
                        probs = weights[current]
                    else:
                        previous = walk[-2]
                        # node2vec's search bias: 1/p to step back, 1 to stay in
                        # the previous node's neighbourhood, 1/q to move outward.
                        bias = np.where(
                            nbrs == previous, 1.0 / p,
                            np.where(
                                [n in neighbour_sets[previous] for n in nbrs.tolist()],
                                1.0, 1.0 / q,
                            ),
                        )
                        probs = weights[current] * bias
                    total = probs.sum()
                    if total <= 0:
                        break
                    walk.append(int(rng.choice(nbrs, p=probs / total)))
                if len(walk) > 1:
                    walks.append(walk + [-1] * (length - len(walk)))
        return np.array(walks, dtype=np.int64)

    @staticmethod
    def _pairs(walks: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        centres, contexts = [], []
        for offset in range(1, window + 1):
            left, right = walks[:, :-offset], walks[:, offset:]
            valid = (left >= 0) & (right >= 0)
            centres.append(left[valid]); contexts.append(right[valid])
            centres.append(right[valid]); contexts.append(left[valid])
        return np.concatenate(centres), np.concatenate(contexts)

    def fit(self, graph: nx.DiGraph) -> "Node2VecEmbedder":
        self.nodes = sorted(graph.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        neighbours, weights = self._build_adjacency(graph)
        walks = self._walks(neighbours, weights, rng)
        if len(walks) == 0:
            self._finalise(np.zeros((len(self.nodes), self.dim)))
            return self

        centres, contexts = self._pairs(walks, self.cfg.embed.window)
        n_nodes = len(self.nodes)

        # Negative sampling distribution: unigram frequency ^ 0.75, the standard
        # skip-gram smoothing that stops hub facilities monopolising negatives.
        counts = np.bincount(walks[walks >= 0].ravel(), minlength=n_nodes).astype(np.float64)
        noise = torch.tensor(counts ** 0.75 / max((counts ** 0.75).sum(), 1e-9),
                             dtype=torch.float)

        centre_emb = nn.Embedding(n_nodes, self.dim)
        context_emb = nn.Embedding(n_nodes, self.dim)
        nn.init.uniform_(centre_emb.weight, -0.5 / self.dim, 0.5 / self.dim)
        nn.init.zeros_(context_emb.weight)
        optimiser = torch.optim.Adam(
            list(centre_emb.parameters()) + list(context_emb.parameters()),
            lr=self.cfg.embed.lr,
        )

        centres_t = torch.tensor(centres, dtype=torch.long)
        contexts_t = torch.tensor(contexts, dtype=torch.long)
        batch_size = 8192
        n_pairs = len(centres_t)
        for _ in range(self.cfg.embed.epochs):
            order = torch.randperm(n_pairs)
            for start in range(0, n_pairs, batch_size):
                idx = order[start:start + batch_size]
                c, o = centres_t[idx], contexts_t[idx]
                negatives = torch.multinomial(
                    noise, len(idx) * self.cfg.embed.negative, replacement=True
                ).view(len(idx), self.cfg.embed.negative)

                v = centre_emb(c)                                   # (B, D)
                pos = torch.sum(v * context_emb(o), dim=1)          # (B,)
                neg = torch.bmm(context_emb(negatives), v.unsqueeze(2)).squeeze(2)
                loss = -(
                    nn.functional.logsigmoid(pos).mean()
                    + nn.functional.logsigmoid(-neg).sum(1).mean()
                )
                optimiser.zero_grad(); loss.backward(); optimiser.step()

        self._finalise(centre_emb.weight.detach().numpy())
        return self


class _SAGELayer(nn.Module):
    """Mean-aggregator GraphSAGE layer: h' = W . [h_self || mean(h_neighbours)]."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbour_mean = torch.sparse.mm(adjacency, features)
        return self.linear(torch.cat([features, neighbour_mean], dim=1))


class _SAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = _SAGELayer(in_dim, hidden)
        self.layer2 = _SAGELayer(hidden, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layer1(x, adjacency))
        return self.layer2(h, adjacency)


class GraphSAGEEmbedder(BaseEmbedder):
    """Unsupervised GraphSAGE over structural + operational node features."""

    name = "sage"

    #: Node inputs. Deliberately structural and operational rather than
    #: identity-based -- an identity one-hot would make the model transductive
    #: again and defeat the point of choosing SAGE.
    FEATURE_FN = [
        ("log_in_degree", lambda g, n: np.log1p(g.in_degree(n))),
        ("log_out_degree", lambda g, n: np.log1p(g.out_degree(n))),
        ("log_in_volume", lambda g, n: np.log1p(g.in_degree(n, weight="leg_count"))),
        ("log_out_volume", lambda g, n: np.log1p(g.out_degree(n, weight="leg_count"))),
    ]

    def _node_features(self, graph: nx.DiGraph) -> np.ndarray:
        undirected = graph.to_undirected()
        clustering = nx.clustering(undirected)
        rows = []
        for node in self.nodes:
            base = [fn(graph, node) for _, fn in self.FEATURE_FN]
            out_edges = [d for _, _, d in graph.out_edges(node, data=True)]
            in_edges = [d for _, _, d in graph.in_edges(node, data=True)]
            base += [
                float(np.median([e["median_delay_ratio"] for e in out_edges])) if out_edges else 1.0,
                float(np.median([e["median_delay_ratio"] for e in in_edges])) if in_edges else 1.0,
                float(np.mean([e["late_rate"] for e in out_edges])) if out_edges else 0.0,
                float(np.mean([e["late_rate"] for e in in_edges])) if in_edges else 0.0,
                float(clustering.get(node, 0.0)),
            ]
            rows.append(base)
        features = np.array(rows, dtype=np.float64)
        mean, std = features.mean(axis=0), features.std(axis=0)
        return (features - mean) / np.clip(std, 1e-9, None)

    def _adjacency(self, graph: nx.DiGraph) -> torch.Tensor:
        undirected = graph.to_undirected()
        rows, cols, vals = [], [], []
        for node in self.nodes:
            i = self.index[node]
            nbrs = [self.index[n] for n in undirected.neighbors(node)]
            if not nbrs:
                continue
            weight = 1.0 / len(nbrs)  # mean aggregation
            for j in nbrs:
                rows.append(i); cols.append(j); vals.append(weight)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        return torch.sparse_coo_tensor(
            indices, torch.tensor(vals, dtype=torch.float),
            (len(self.nodes), len(self.nodes)),
        ).coalesce()

    def fit(self, graph: nx.DiGraph) -> "GraphSAGEEmbedder":
        self.nodes = sorted(graph.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        x = torch.tensor(self._node_features(graph), dtype=torch.float)
        adjacency = self._adjacency(graph)
        model = _SAGE(x.shape[1], self.cfg.embed.sage_hidden, self.dim)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.cfg.embed.sage_lr)

        edges = np.array(
            [[self.index[u], self.index[v]] for u, v in graph.edges()], dtype=np.int64
        )
        if len(edges) == 0:
            self._finalise(np.zeros((len(self.nodes), self.dim)))
            return self
        src = torch.tensor(edges[:, 0]); dst = torch.tensor(edges[:, 1])

        # Unsupervised graph-context objective: connected facilities should sit
        # close in the embedding, random pairs should not.
        for _ in range(self.cfg.embed.sage_epochs):
            model.train()
            z = model(x, adjacency)
            negatives = torch.tensor(
                rng.integers(0, len(self.nodes), size=len(edges)), dtype=torch.long
            )
            pos_score = (z[src] * z[dst]).sum(dim=1)
            neg_score = (z[src] * z[negatives]).sum(dim=1)
            loss = -(
                nn.functional.logsigmoid(pos_score).mean()
                + nn.functional.logsigmoid(-neg_score).mean()
            )
            optimiser.zero_grad(); loss.backward(); optimiser.step()

        model.eval()
        with torch.no_grad():
            self._finalise(model(x, adjacency).numpy())
        return self


EMBEDDERS = {
    "svd": SVDEmbedder,
    "node2vec": Node2VecEmbedder,
    "graphsage": GraphSAGEEmbedder,
}


def build_embedder(kind: str, cfg: Config) -> BaseEmbedder:
    if kind not in EMBEDDERS:
        raise KeyError(f"Unknown embedder {kind!r}; choose from {sorted(EMBEDDERS)}")
    return EMBEDDERS[kind](cfg)
