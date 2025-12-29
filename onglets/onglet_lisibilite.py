"""# Onglet Lisibilité

Ce module orchestre l'onglet Streamlit consacré au calcul des indicateurs de
lisibilité Flesch-Kincaid sur un corpus filtré selon les variables
saisies.

## Dépendances
- `densite.py` : construction du texte combiné à partir du DataFrame filtré.
- `test_lesch_Kincaid.py` : calcul des scores Flesch-Kincaid, bande de
  difficulté et messages d'interprétation.
- `fcts_utils.py` : affichage des connecteurs actifs pour guider
  l'utilisateur.
- Bibliothèques `streamlit` et `pandas` pour la sélection des modalités et la
  restitution des résultats.
"""
from __future__ import annotations

from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from densite import build_text_from_dataframe
from fcts_utils import render_connectors_reminder
from test_lesch_Kincaid import (
    READABILITY_SCALE,
    compute_flesch_kincaid_metrics,
    get_readability_band,
    interpret_reading_ease,
)


def rendu_lisibilite(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    st.subheader("Test de lisibilité (Flesch-Kincaid)")
    render_connectors_reminder(filtered_connectors)

    st.markdown("### Sélection des variables/modalités")

    readability_variables = [
        column for column in df.columns if column not in ("texte", "entete")
    ]
    readability_selected_variables = st.multiselect(
        "Variables disponibles pour la lisibilité",
        readability_variables,
        default=readability_variables,
        help="Choisissez les variables à filtrer pour le test de lisibilité.",
        key="readability_variables",
    )

    readability_filtered_df = df.copy()
    readability_modalities_selection: Dict[str, List[str]] = {}

    for variable in readability_selected_variables:
        modality_options = sorted(
            readability_filtered_df[variable].dropna().unique().tolist()
        )
        selected_modalities = st.multiselect(
            f"Modalités à inclure pour {variable}",
            modality_options,
            default=modality_options,
            help="Filtrer les textes utilisés pour le test de lisibilité.",
        )
        readability_modalities_selection[variable] = selected_modalities
        if selected_modalities:
            readability_filtered_df = readability_filtered_df[
                readability_filtered_df[variable].isin(selected_modalities)
            ]
        else:
            readability_filtered_df = readability_filtered_df.iloc[0:0]

    if readability_filtered_df.empty:
        st.info("Aucun texte ne correspond aux filtres sélectionnés.")
        return

    readability_text = build_text_from_dataframe(readability_filtered_df)
    if not readability_text:
        st.info("Aucun texte valide à analyser pour la lisibilité.")
        return

    readability_metrics = compute_flesch_kincaid_metrics(readability_text)

    st.markdown("### Résultats de lisibilité")

    if readability_metrics is None:
        st.info("Impossible de calculer les scores de lisibilité pour le texte fourni.")
        return

    ease_score = readability_metrics.get("reading_ease", 0.0)
    grade_score = readability_metrics.get("grade_level", 0.0)

    col1, col2 = st.columns(2)
    col1.metric(
        "Flesch Reading Ease",
        f"{ease_score:.2f}",
        help="Indice mesurant la facilité de lecture (plus il est élevé, plus le texte est facile).",
    )
    col2.metric(
        "Flesch-Kincaid Grade",
        f"{grade_score:.2f}",
        help="Niveau scolaire approximatif nécessaire pour comprendre le texte.",
    )

    readability_band = get_readability_band(ease_score)
    readability_description = interpret_reading_ease(ease_score)

    st.markdown(
        f"**Interprétation** : {readability_description} (échelle : {readability_band.get('range', '')})"
    )

    st.caption(
        "Les scores de lisibilité sont calculés sur la base du texte filtré, en utilisant les variables/modalités sélectionnées."
    )

    st.markdown("### Position sur l'échelle de lisibilité")
    readability_scale_df = pd.DataFrame(READABILITY_SCALE).sort_values(
        by="min", ascending=False
    )
    readability_scale_df["niveau_ordre"] = readability_scale_df["niveau"]

    scale_chart = alt.Chart(readability_scale_df).mark_bar().encode(
        y=alt.Y(
            "niveau_ordre:N",
            title="Niveau de lecture",
            sort=readability_scale_df["niveau"].tolist(),
        ),
        x=alt.X(
            "min:Q",
            title="Flesch Reading Ease",
            scale=alt.Scale(domain=[0, 100]),
        ),
        x2="max:Q",
        color=alt.Color("niveau:N", legend=None),
        tooltip=["niveau", "range", "description"],
    )

    score_rule = (
        alt.Chart(pd.DataFrame({"score": [ease_score]}))
        .mark_rule(color="red", strokeWidth=2)
        .encode(x="score:Q", tooltip=[alt.Tooltip("score:Q", format=".2f")])
    )

    st.altair_chart(scale_chart + score_rule, use_container_width=True)

    readability_per_modality: List[Dict[str, float | str]] = []

    for variable, selected_modalities in readability_modalities_selection.items():
        for modality in selected_modalities:
            modality_df = readability_filtered_df[
                readability_filtered_df[variable] == modality
            ]
            modality_text = build_text_from_dataframe(modality_df)
            if not modality_text:
                continue

            modality_metrics = compute_flesch_kincaid_metrics(modality_text)
            readability_per_modality.append(
                {
                    "variable": variable,
                    "modalite": modality,
                    "reading_ease": modality_metrics.get("reading_ease", 0.0),
                    "grade_level": modality_metrics.get("grade_level", 0.0),
                }
            )

    if readability_per_modality:
        st.markdown("### Score de lisibilité par modalité")
        modality_scores_df = pd.DataFrame(readability_per_modality)
        modality_scores_df = modality_scores_df.sort_values(
            by=["variable", "reading_ease"], ascending=[True, False]
        )

        display_df = modality_scores_df.rename(
            columns={
                "variable": "Variable",
                "modalite": "Modalité",
                "reading_ease": "Indice de lisibilité",
                "grade_level": "Niveau scolaire (grade)",
            }
        )
        display_df["Indice de lisibilité"] = display_df["Indice de lisibilité"].apply(
            lambda score: f"{score:.2f}"
        )
        display_df["Niveau scolaire (grade)"] = display_df[
            "Niveau scolaire (grade)"
        ].apply(
            lambda score: f"Niveau {max(round(score), 0)}eme"
        )

        st.dataframe(display_df, use_container_width=True)

        modality_chart = alt.Chart(modality_scores_df).mark_bar().encode(
            x=alt.X("reading_ease:Q", title="Flesch Reading Ease"),
            y=alt.Y("modalite:N", sort="-x", title="Modalité"),
            color=alt.Color("variable:N", title="Variable"),
            tooltip=[
                "variable",
                "modalite",
                alt.Tooltip("reading_ease:Q", title="Indice", format=".2f"),
                alt.Tooltip("grade_level:Q", title="Grade", format=".2f"),
            ],
        )

        st.altair_chart(
            modality_chart.facet(row=alt.Row("variable:N", title="Variable")),
            use_container_width=True,
        )
    else:
        st.info(
            "Aucun score modalité n'a pu être calculé avec la sélection actuelle."
        )
