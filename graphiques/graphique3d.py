"""Visualisation 3D des distances cosinus sous forme de réseau de sphères.

Ce module fournit un pipeline complet pour :
- calculer une matrice de similarité cosinus à partir d'embeddings ;
- construire un graphe pondéré filtré par seuil ;
- générer un layout 3D (force-directed) ;
- créer une figure Plotly interactive avec sphères et liaisons codant la similarité.

Fonctionnement minimal :
>>> import numpy as np
>>> from graphiques.graphique3d import create_cosine_network_figure
>>> embeddings = np.random.rand(6, 8)
>>> labels = [f"modèle_{i}" for i in range(len(embeddings))]
>>> fig = create_cosine_network_figure(embeddings, labels)
>>> fig.show()

Le résultat est une scène 3D navigable :
- sphères (taille optionnellement proportionnelle à une métrique)
- couleur des sphères par cluster
- liens colorés et épaissis selon la similarité
- slider pour filtrer dynamiquement les liens par seuil de similarité.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


@dataclass
class CosineNetworkConfig:
    """Paramètres de création du graphe 3D.

    Attributes:
        threshold_values: Liste des seuils de similarité utilisés pour filtrer les arêtes.
        default_threshold: Seuil appliqué pour l'affichage initial.
        node_scale: Multiplicateur appliqué aux tailles de nœuds après normalisation.
        layout_seed: Graine pour la reproductibilité du layout 3D.
    """

    threshold_values: Sequence[float] = (0.4, 0.5, 0.6, 0.7, 0.8)
    default_threshold: float = 0.5
    node_scale: float = 24.0
    layout_seed: int | None = 42

    def __post_init__(self) -> None:
        if self.default_threshold not in self.threshold_values:
            raise ValueError("default_threshold doit appartenir à threshold_values")


def compute_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Renvoie la matrice de similarité cosinus.

    Args:
        embeddings: Matrice (n_samples, n_features) des embeddings.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings doit être une matrice 2D")
    return cosine_similarity(embeddings)


def build_similarity_graph(
    similarities: np.ndarray, labels: Sequence[str], threshold: float
) -> nx.Graph:
    """Construit un graphe pondéré à partir de la matrice de similarité."""

    if similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities doit être carrée")
    if len(labels) != similarities.shape[0]:
        raise ValueError("labels doit avoir la même longueur que similarities")

    graph = nx.Graph()
    for idx, label in enumerate(labels):
        graph.add_node(idx, label=label)

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(similarities[i, j])
            if weight >= threshold:
                graph.add_edge(i, j, weight=weight)
    return graph


def compute_layout_3d(graph: nx.Graph, seed: int | None = 42) -> dict[int, np.ndarray]:
    """Calcule un layout 3D spring-layout pondéré."""

    return nx.spring_layout(graph, dim=3, weight="weight", seed=seed)


def _scale_values(values: Iterable[float], factor: float) -> list[float]:
    scaler = MinMaxScaler(feature_range=(0.4 * factor, factor))
    arr = np.asarray(list(values)).reshape(-1, 1)
    if arr.size == 0:
        return []
    scaled = scaler.fit_transform(arr).flatten()
    return scaled.tolist()


def _make_edge_trace(graph: nx.Graph, positions: dict[int, np.ndarray], color_scale: str = "Viridis") -> go.Scatter3d:
    x_edges, y_edges, z_edges, weights = [], [], [], []
    for u, v, data in graph.edges(data=True):
        x_edges += [positions[u][0], positions[v][0], None]
        y_edges += [positions[u][1], positions[v][1], None]
        z_edges += [positions[u][2], positions[v][2], None]
        weights.append(data.get("weight", 0.0))

    line_widths = _scale_values(weights, factor=8.0) if weights else []
    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line=dict(color=weights if weights else "lightgray", colorscale=color_scale, width=line_widths),
        hoverinfo="none",
    )


def _make_node_trace(
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
    labels: Sequence[str],
    performances: Sequence[float] | None,
    cluster_labels: Sequence[str] | None,
    node_scale: float,
    color_scale: str = "Turbo",
) -> go.Scatter3d:
    x_nodes, y_nodes, z_nodes = [], [], []
    for idx in graph.nodes:
        pos = positions[idx]
        x_nodes.append(pos[0])
        y_nodes.append(pos[1])
        z_nodes.append(pos[2])

    sizes = _scale_values(performances if performances is not None else [1.0] * len(graph), factor=node_scale)

    if cluster_labels is not None and len(cluster_labels) == len(graph):
        node_colors = cluster_labels
        showscale = False
    else:
        node_colors = performances if performances is not None else [1.0] * len(graph)
        showscale = True

    hover_text = []
    for idx, label in enumerate(labels):
        perf_info = "" if performances is None else f"<br>métrique: {performances[idx]:.3f}"
        cluster_info = "" if cluster_labels is None else f"<br>cluster: {cluster_labels[idx]}"
        hover_text.append(f"{label}{perf_info}{cluster_info}")

    return go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=node_colors,
            colorscale=color_scale,
            showscale=showscale,
            colorbar=dict(title="Similarité/score"),
            opacity=0.9,
            line=dict(width=1.5, color="black"),
        ),
        text=labels,
        textposition="top center",
        hoverinfo="text",
    )


def create_cosine_network_figure(
    embeddings: np.ndarray,
    labels: Sequence[str],
    performances: Sequence[float] | None = None,
    cluster_labels: Sequence[str] | None = None,
    config: CosineNetworkConfig | None = None,
) -> go.Figure:
    """Construit la figure Plotly interactive pour explorer les similarités cosinus.

    Args:
        embeddings: Matrice d'embeddings (n, d).
        labels: Liste des noms de modèles.
        performances: Valeurs numériques optionnelles pour dimensionner les sphères.
        cluster_labels: Étiquettes de cluster pour colorer les sphères.
        config: Paramètres (seuils, taille des nœuds, etc.).
    """

    cfg = config or CosineNetworkConfig()
    similarities = compute_cosine_similarities(np.asarray(embeddings))

    # Graphe et layout basés sur le seuil minimal pour conserver la structure.
    base_threshold = min(cfg.threshold_values)
    base_graph = build_similarity_graph(similarities, labels, threshold=base_threshold)
    positions = compute_layout_3d(base_graph, seed=cfg.layout_seed)

    node_trace = _make_node_trace(
        base_graph,
        positions,
        labels,
        performances,
        cluster_labels,
        node_scale=cfg.node_scale,
    )

    frames = []
    for threshold in cfg.threshold_values:
        graph = build_similarity_graph(similarities, labels, threshold=threshold)
        edge_trace = _make_edge_trace(graph, positions)
        frames.append(
            go.Frame(name=f"≥ {threshold:.2f}", data=[edge_trace, node_trace])
        )

    default_index = cfg.threshold_values.index(cfg.default_threshold)
    fig = go.Figure(data=frames[default_index].data, frames=frames)

    steps = []
    for i, threshold in enumerate(cfg.threshold_values):
        step = dict(
            method="animate",
            args=[[frames[i].name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
            label=f"≥ {threshold:.2f}",
        )
        steps.append(step)

    fig.update_layout(
        title="Réseau 3D des similarités cosinus",
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            bgcolor="#f8f9fb",
        ),
        showlegend=False,
        sliders=[
            {
                "active": default_index,
                "currentvalue": {"prefix": "Seuil : "},
                "pad": {"t": 30},
                "steps": steps,
            }
        ],
    )

    return fig


if __name__ == "__main__":
    # Exemple reproductible pour tester rapidement dans un notebook ou en local.
    rng = np.random.default_rng(1234)
    n_models = 8
    embedding_dim = 12
    demo_embeddings = rng.normal(size=(n_models, embedding_dim))
    demo_labels = [f"modèle_{i}" for i in range(n_models)]
    demo_performance = rng.uniform(0.6, 0.95, size=n_models)
    demo_clusters = [f"C{c}" for c in rng.integers(0, 3, size=n_models)]

    figure = create_cosine_network_figure(
        demo_embeddings,
        demo_labels,
        performances=demo_performance,
        cluster_labels=demo_clusters,
    )
    figure.show()
