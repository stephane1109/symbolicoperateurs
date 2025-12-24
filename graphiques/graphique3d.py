"""Visualisation simple des similarités cosinus sous forme de réseau.

Ce module fournit un pipeline sans JavaScript basé sur ``python-igraph`` et
Matplotlib pour représenter les modèles comme des nœuds reliés par des arêtes
pondérées par leur similarité cosinus. Le rendu est statique, ce qui le rend
compatible avec les environnements où les graphiques Plotly interactifs ne sont
pas pris en charge (par exemple Streamlit Cloud).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import igraph as ig
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CosineGraphConfig:
    """Paramètres du graphe de similarité cosinus.

    Attributes:
        min_similarity: Seuil minimal pour conserver une arête.
        layout: Nom du layout igraph (ex: ``"fruchterman_reingold"``).
        vertex_size: Taille des nœuds (en points matplotlib).
        edge_label_round: Nombre de décimales affichées sur les arêtes.
    """

    min_similarity: float = 0.0
    layout: str = "fruchterman_reingold"
    vertex_size: int = 30
    edge_label_round: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.min_similarity <= 1:
            raise ValueError("min_similarity doit être compris entre 0 et 1")


def compute_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Calcule et renvoie la matrice de similarité cosinus."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings doit être une matrice 2D")
    return cosine_similarity(embeddings)


def build_igraph_cosine_graph(
    similarities: np.ndarray,
    labels: Sequence[str],
    min_similarity: float,
    edge_label_round: int,
) -> ig.Graph:
    """Construit un graphe igraph pondéré à partir des similarités cosinus."""

    if similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities doit être carrée")
    if len(labels) != similarities.shape[0]:
        raise ValueError("labels doit avoir la même longueur que similarities")

    graph = ig.Graph()
    graph.add_vertices(len(labels))
    graph.vs["label"] = labels

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(similarities[i, j])
            if weight >= min_similarity:
                edges.append((i, j))
                weights.append(weight)

    graph.add_edges(edges)
    graph.es["weight"] = weights
    decimals = max(0, int(edge_label_round))
    graph.es["label"] = [f"{w:.{decimals}f}" for w in weights]
    return graph


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        return []
    min_w, max_w = min(weights), max(weights)
    if np.isclose(min_w, max_w):
        return [0.5 for _ in weights]
    norm = [(w - min_w) / (max_w - min_w) for w in weights]
    return [float(v) for v in norm]


def create_cosine_network_figure(
    embeddings: np.ndarray,
    labels: Sequence[str],
    config: CosineGraphConfig | None = None,
) -> plt.Figure:
    """Construit une figure Matplotlib représentant le réseau des similarités.

    Args:
        embeddings: Matrice d'embeddings (n, d).
        labels: Noms des modèles.
        config: Paramètres optionnels du graphe.
    """

    cfg = config or CosineGraphConfig()
    similarities = compute_cosine_similarities(np.asarray(embeddings))
    graph = build_igraph_cosine_graph(
        similarities, labels, cfg.min_similarity, cfg.edge_label_round
    )

    layout = graph.layout(cfg.layout)
    normalized_weights = _normalize_weights(graph.es["weight"])
    cmap = cm.get_cmap("viridis")
    edge_colors = [cmap(w) for w in normalized_weights] if normalized_weights else "gray"
    edge_widths = [2 + 6 * w for w in normalized_weights] if normalized_weights else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_size=cfg.vertex_size,
        vertex_color="#5dade2",
        vertex_frame_color="#1b4f72",
        vertex_label=labels,
        vertex_label_size=12,
        vertex_label_color="black",
        edge_width=edge_widths,
        edge_color=edge_colors,
        edge_label=graph.es["label"],
        edge_label_size=10,
        edge_curved=0,
    )

    ax.set_title("Réseau des similarités cosinus", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def create_cosine_network_from_similarity(
    similarity_matrix: np.ndarray,
    labels: Sequence[str],
    config: CosineGraphConfig | None = None,
) -> plt.Figure:
    """Crée une figure Matplotlib à partir d'une matrice de similarités cosinus.

    Args:
        similarity_matrix: Matrice carrée des similarités cosinus.
        labels: Noms des modèles associés à chaque ligne/colonne.
        config: Paramètres optionnels du graphe.
    """

    cfg = config or CosineGraphConfig()
    similarities = np.asarray(similarity_matrix)
    graph = build_igraph_cosine_graph(
        similarities, labels, cfg.min_similarity, cfg.edge_label_round
    )

    layout = graph.layout(cfg.layout)
    normalized_weights = _normalize_weights(graph.es["weight"])
    cmap = cm.get_cmap("viridis")
    edge_colors = [cmap(w) for w in normalized_weights] if normalized_weights else "gray"
    edge_widths = [2 + 6 * w for w in normalized_weights] if normalized_weights else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_size=cfg.vertex_size,
        vertex_color="#5dade2",
        vertex_frame_color="#1b4f72",
        vertex_label=labels,
        vertex_label_size=12,
        vertex_label_color="black",
        edge_width=edge_widths,
        edge_color=edge_colors,
        edge_label=graph.es["label"],
        edge_label_size=10,
        edge_curved=0,
    )

    ax.set_title("Réseau des similarités cosinus", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    n_models = 6
    embedding_dim = 10
    demo_embeddings = rng.normal(size=(n_models, embedding_dim))
    demo_labels = [f"modèle_{i}" for i in range(n_models)]

    figure = create_cosine_network_figure(
        demo_embeddings,
        demo_labels,
        config=CosineGraphConfig(min_similarity=0.2, layout="kamada_kawai"),
    )
    figure.savefig("demo_cosine_network.png", dpi=150)
