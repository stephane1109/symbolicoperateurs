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

    st.markdown("### Paramètres utilisés")
    st.write(
        {
            "Mots": readability_metrics.get("word_count", 0),
            "Phrases": readability_metrics.get("sentence_count", 0),
            "Syllabes": readability_metrics.get("syllable_count", 0),
        }
    )

    st.caption(
        "Les scores de lisibilité sont calculés sur la base du texte filtré, en utilisant les variables/modalités sélectionnées."
    )
