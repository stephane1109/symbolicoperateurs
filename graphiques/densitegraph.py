"""Graphiques pour l'onglet Densité."""

from __future__ import annotations

from typing import Iterable

import altair as alt
import pandas as pd


def build_density_chart(per_modality_df: pd.DataFrame) -> alt.Chart:
    """Créer le graphique de densité par modalité."""

    density_chart = (
        alt.Chart(per_modality_df)
        .mark_bar()
        .encode(
            x=alt.X("modalite:N", title="Modalité"),
            y=alt.Y("densite:Q", title="Densité"),
            color=alt.Color("modalite:N", title="Modalité"),
            tooltip=["modalite", "densite", "mots", "connecteurs"],
        )
    )

    density_norm_rule = (
        alt.Chart(pd.DataFrame({"norme": [1.37]}))
        .mark_rule(color="red", strokeDash=[6, 4])
        .encode(y=alt.Y("norme:Q"))
    )

    return (density_chart + density_norm_rule).properties(title="Graphique de densité")


def build_connector_density_chart(per_modality_label_df: pd.DataFrame) -> alt.Chart:
    """Créer le graphique de densité par connecteur et modalité."""

    return (
        alt.Chart(per_modality_label_df)
        .mark_bar()
        .encode(
            x=alt.X("modalite:N", title="Modalité"),
            xOffset="label",
            y=alt.Y("densite:Q", title="Densité"),
            color=alt.Color("label:N", title="Connecteur"),
            tooltip=["modalite", "label", "densite", "connecteurs", "mots"],
        )
        .properties(title="Densité par connecteur et modalité")
    )


def build_density_scatter_chart(
    scatter_df: pd.DataFrame,
    selected_x_label: str,
    selected_y_label: str,
    tooltip_fields: Iterable[str] | None = None,
) -> alt.Chart:
    """Créer la classification visuelle des densités sur deux axes."""

    return (
        alt.Chart(scatter_df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(
                "densite_x:Q",
                title=f"Densité {selected_x_label}",
            ),
            y=alt.Y(
                "densite_y:Q",
                title=f"Densité {selected_y_label}",
            ),
            size=alt.Size(
                "densite_totale:Q",
                title="Densité totale (taille du cercle)",
                scale=alt.Scale(range=[50, 1200]),
            ),
            color=alt.Color(
                "densite_totale:Q",
                title="Densité totale",
                scale=alt.Scale(scheme="oranges"),
            ),
            tooltip=list(tooltip_fields) if tooltip_fields is not None else list(scatter_df.columns),
        )
        .properties(height=500)
    )
