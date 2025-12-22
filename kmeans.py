"""Clustering K-means sur les connecteurs après filtrage des segments.

L'objectif est identique à l'AFC : reconstruire la matrice des occurrences de
connecteurs sur le sous-corpus filtré puis appliquer un K-means classique.
Le prétraitement inclut une standardisation creuse (sans recentrage) pour
éviter de favoriser les connecteurs les plus fréquents.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from afc import build_connector_matrix


def run_kmeans(
    dataframe: pd.DataFrame,
    connectors: Dict[str, str],
    modality_filters: Optional[Mapping[str, Iterable[str]]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.Series:
    """Calculer les étiquettes de cluster sur les segments filtrés.

    La fonction retourne une ``Series`` indexée comme le DataFrame filtré,
    ce qui permet de rattacher chaque cluster aux variables/modalités d'origine.
    """

    matrix = build_connector_matrix(dataframe, connectors, modality_filters)
    if matrix.empty:
        return pd.Series(dtype="int64")

    pipeline = make_pipeline(
        StandardScaler(with_mean=False),
        KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state),
    )

    labels = pipeline.fit_predict(matrix)
    return pd.Series(labels, index=matrix.index, name="cluster")


__all__ = ["run_kmeans"]
