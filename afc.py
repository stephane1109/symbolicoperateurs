"""Analyse factorielle des correspondances sur les connecteurs filtrés.

Ce module illustre la logique décrite dans l'application Streamlit :
1. filtrer les segments en fonction des variables issues des marqueurs IRaMuTeQ,
2. reconstruire un tableau de contingence connecteur × document,
3. appliquer une AFC pour obtenir des coordonnées factorielles.

L'implémentation reste volontairement légère pour être importable dans un
notebook ou un script sans dépendre de l'interface Streamlit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from analyses import count_connectors


def _apply_modality_filters(
    dataframe: pd.DataFrame, modality_filters: Optional[Mapping[str, Iterable[str]]]
) -> pd.DataFrame:
    """Filtrer le DataFrame selon des couples variable/modalités.

    ``modality_filters`` doit suivre la structure ``{"variable": ["mod1", "mod2"]}``.
    Les variables absentes du DataFrame sont ignorées silencieusement.
    """

    if not modality_filters:
        return dataframe

    filtered = dataframe.copy()

    for variable, modalities in modality_filters.items():
        if variable in filtered.columns and modalities:
            filtered = filtered[filtered[variable].isin(modalities)]

    return filtered


def build_connector_matrix(
    dataframe: pd.DataFrame,
    connectors: Dict[str, str],
    modality_filters: Optional[Mapping[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """Construire une matrice documents × connecteurs après filtrage.

    Chaque ligne correspond à un segment filtré, identifié par son index d'origine.
    Les colonnes représentent les connecteurs sélectionnés, les valeurs le nombre
    d'occurrences dans le segment.
    """

    filtered_df = _apply_modality_filters(dataframe, modality_filters)
    if filtered_df.empty:
        return pd.DataFrame()

    connector_names = sorted({name for name in connectors if name})
    if not connector_names:
        return pd.DataFrame()

    rows = []
    index_labels = []

    for row_index, text in filtered_df["texte"].fillna("").items():
        counts = count_connectors(text, connectors).set_index("connecteur")[
            "occurrences"
        ]
        rows.append([int(counts.get(name, 0)) for name in connector_names])
        index_labels.append(row_index)

    matrix = pd.DataFrame(rows, columns=connector_names, index=index_labels)

    # Retirer les colonnes et lignes entièrement nulles pour éviter des divisions par
    # zéro lors du calcul de l'AFC.
    matrix = matrix.loc[:, matrix.sum() > 0]
    matrix = matrix.loc[matrix.sum(axis=1) > 0]

    return matrix


def run_afc(
    dataframe: pd.DataFrame,
    connectors: Dict[str, str],
    modality_filters: Optional[Mapping[str, Iterable[str]]] = None,
    n_components: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Appliquer une AFC et retourner les coordonnées lignes/colonnes.

    Parameters
    ----------
    dataframe:
        Table contenant au minimum une colonne ``texte`` et les variables
        issues des marqueurs IRaMuTeQ.
    connectors:
        Dictionnaire ``connecteur -> label`` tel que fourni par l'application.
    modality_filters:
        Filtres sur les variables/modalités avant de recalculer la matrice.
    n_components:
        Nombre d'axes factoriels à conserver.
    """

    matrix = build_connector_matrix(dataframe, connectors, modality_filters)
    if matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    total = matrix.to_numpy().sum()
    if total == 0:
        return pd.DataFrame(), pd.DataFrame()

    relative = matrix / total
    row_masses = relative.sum(axis=1).to_numpy()
    col_masses = relative.sum(axis=0).to_numpy()

    expected = np.outer(row_masses, col_masses)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = (relative.to_numpy() - expected) / np.sqrt(expected)
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(standardized)

    row_coords = pca.transform(standardized)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    row_df = pd.DataFrame(row_coords, index=matrix.index)
    col_df = pd.DataFrame(loadings, index=matrix.columns)

    return row_df, col_df


__all__ = [
    "build_connector_matrix",
    "run_afc",
]
