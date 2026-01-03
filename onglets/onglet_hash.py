"""# Onglet Hash

Ce module propose l'onglet Streamlit permettant de calculer la Longueur
Moyenne des Segments (LMS) entre connecteurs et de comparer différents modes
de segmentation.

## Dépendances
- `hash.py` : calculs de longueur de segments, moyennes et modes de
  segmentation.
- `ecartype.py` : écart-type des longueurs pour enrichir les indicateurs.
- `simicosinus.py` : concaténation des textes avec entêtes pour alimenter les
  statistiques.
- `fcts_utils.py` : rappel des connecteurs sélectionnés dans l'interface.
- Bibliothèques `streamlit`, `pandas` et `altair` pour l'affichage et les
  graphiques.
"""
from __future__ import annotations

from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from ecartype import compute_length_standard_deviation, standard_deviation_by_modality
from fcts_utils import render_connectors_reminder
from hash import (
    ECART_TYPE_EXPLANATION,
    SegmentationMode,
    TokenizationMode,
    average_segment_length,
    average_segment_length_by_modality,
    compute_segment_word_lengths,
    segments_with_word_lengths,
)
from simicosinus import concatenate_texts_with_headers


def rendu_hash(
    tab,
    filtered_df: pd.DataFrame,
    filtered_connectors: Dict[str, str],
    combined_text: str,
) -> None:
    st.subheader("Hash (LMS entre connecteurs)")
    render_connectors_reminder(filtered_connectors)
    st.write(
        """
La "LMS" correspond à la Longueur Moyenne des Segments d'un texte. Vous pouvez choisir
un découpage basé uniquement sur les connecteurs sélectionnés, ou bien considérer qu'une
ponctuation forte (. / ? / ! / ; /:) ferme aussi le segment.
- Des segments courts signalent un texte plutôt "haché", saccadé, algorithmique.
- Des segments longs évoquent une prose plus fluide.
        """
    )
    segmentation_labels: Dict[str, SegmentationMode] = {
        "Entre connecteurs uniquement (ignore la ponctuation)": "connecteurs",
        "Connecteurs + ponctuation qui ferme le segment": "connecteurs_et_ponctuation",
    }
    segmentation_choice = st.radio(
        "Mode de calcul de la LMS",
        list(segmentation_labels.keys()),
        help=(
            "Le découpage peut se faire uniquement entre connecteurs, ou bien s'arrêter"
            " dès qu'un signe de ponctuation forte (., ?, !, ;, :) est rencontré."
        ),
    )
    segmentation_mode = segmentation_labels[segmentation_choice]

    tokenization_labels: Dict[str, TokenizationMode] = {
        "Regex (Tokenisation avec une 'regex \b\w+\b')": "regex",
        "spaCy (fr_core_news_md - Tokenisation plus précise)": "spacy",
    }
    tokenization_choice = st.radio(
        "Mode de tokenisation", list(tokenization_labels.keys()), help=(
            "Choisissez le mode de découpage des mots pour compter la longueur des segments :\n"
            "- Regex : tokens basés sur une expression régulière (regex \b\w+\b).\n"
            "- spaCy : tokens linguistiques du modèle français fr_core_news_md (plus précis)."
        ),
    )
    tokenization_mode = tokenization_labels[tokenization_choice]

    try:
        segment_lengths = compute_segment_word_lengths(
            combined_text, filtered_connectors, segmentation_mode, tokenization_mode
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    if not segment_lengths:
        st.info(
            "Impossible de calculer la LMS : aucun segment n'a été détecté entre connecteurs."
        )
        return

    st.subheader("Sélection des variables/modalités")
    hash_variables = [
        column for column in filtered_df.columns if column not in ("texte", "entete")
    ]

    if not hash_variables:
        st.info("Aucune variable n'a été trouvée dans le fichier importé.")
        return

    selected_hash_variables = st.multiselect(
        "Variables à filtrer pour la LMS",
        hash_variables,
        default=hash_variables,
        help=(
            "Sélectionnez les variables et modalités à inclure avant de calculer la "
            "LMS."
        ),
    )

    if not selected_hash_variables:
        st.info(
            "Sélectionnez au moins une variable pour calculer la LMS."
        )
        return

    hash_modality_filters: Dict[str, List[str]] = {}
    hash_filtered_df = filtered_df.copy()

    for variable in selected_hash_variables:
        modality_options = sorted(
            hash_filtered_df[variable].dropna().unique().tolist()
        )
        selected_modalities = st.multiselect(
            f"Modalités à inclure pour {variable}",
            modality_options,
            default=modality_options,
            help=(
                "Sélectionnez les modalités dont les textes seront pris en compte pour"
                " cette variable."
            ),
            key=f"modalites_{variable}",
        )
        hash_modality_filters[variable] = selected_modalities

        if selected_modalities:
            hash_filtered_df = hash_filtered_df[
                hash_filtered_df[variable].isin(selected_modalities)
            ]
        else:
            hash_filtered_df = hash_filtered_df.iloc[0:0]

    if hash_filtered_df.empty:
        st.info(
            "Aucun texte ne correspond aux filtres appliqués. Ajustez vos sélections pour"
            " continuer."
        )
        return

    hash_text = concatenate_texts_with_headers(
        hash_filtered_df, selected_hash_variables
    )

    if hash_text:
        st.download_button(
            label="Télécharger les textes concaténés",
            data=hash_text,
            file_name="textes_concatenation_hash.txt",
            mime="text/plain",
            help=(
                "Export des textes regroupés selon les variables et modalités choisies "
                "pour vérifier la composition de la LMS."
            ),
        )
    try:
        segment_lengths = compute_segment_word_lengths(
            hash_text, filtered_connectors, segmentation_mode, tokenization_mode
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    if not hash_text or not segment_lengths:
        st.info(
            "Impossible de calculer la LMS : aucun segment n'a été détecté dans les données"
            " filtrées."
        )
        return

    average_length = average_segment_length(
        hash_text, filtered_connectors, segmentation_mode, tokenization_mode
    )
    _, std_dev = compute_length_standard_deviation(
        hash_text, filtered_connectors, segmentation_mode, tokenization_mode
    )

    col1, col2 = st.columns(2)
    col1.metric(
        "Longueur moyenne des segments (LMS)", f"{average_length:.4f}"
    )
    col2.metric(
        "Écart-type des segments", f"{std_dev:.4f}"
    )

    st.caption(
        "Ces indicateurs permettent de quantifier la fluidité ou la segmentation du texte."
    )

    segment_entries = segments_with_word_lengths(
        hash_text, filtered_connectors, segmentation_mode, tokenization_mode
    )

    st.markdown("### Segments et longueurs")
    st.dataframe(pd.DataFrame(segment_entries), use_container_width=True)

    st.download_button(
        label="Exporter les segments (CSV)",
        data=pd.DataFrame(segment_entries).to_csv(index=False),
        file_name="segments_longueurs.csv",
        mime="text/csv",
    )

    for variable in selected_hash_variables:
        st.markdown(f"### Analyse par variable : {variable}")

        selected_modalities = hash_modality_filters.get(variable, [])
        per_modality_hash_df = average_segment_length_by_modality(
            hash_filtered_df,
            variable,
            filtered_connectors,
            selected_modalities or None,
            segmentation_mode,
            tokenization_mode,
        )

        if per_modality_hash_df.empty and not selected_modalities:
            st.info(
                "Aucune modalité n'a été trouvée pour cette variable dans les données sélectionnées."
            )
            continue

        if selected_modalities and per_modality_hash_df.empty:
            st.info(
                "Aucun texte ne correspond aux modalités choisies : impossible de calculer la LMS."
            )
            continue

        if not per_modality_hash_df.empty:
            st.subheader(f"Modalité(s) sélectionnée(s) de la variable : {variable}")
            st.dataframe(
                per_modality_hash_df.rename(
                    columns={
                        "modalite": "Modalité",
                        "segments": "Segments comptés",
                        "lms": "LMS",
                    }
                ),
                use_container_width=True,
            )

            lms_chart = (
                alt.Chart(per_modality_hash_df)
                .mark_bar()
                .encode(
                    x=alt.X("modalite:N", title="Modalité"),
                    y=alt.Y("lms:Q", title="LMS (mots)"),
                    color=alt.Color("modalite:N", title="Modalité"),
                    tooltip=[
                        alt.Tooltip("modalite:N", title="Modalité"),
                        alt.Tooltip("lms:Q", title="LMS", format=".4f"),
                        alt.Tooltip("segments:Q", title="Segments"),
                    ],
                )
            )

            st.altair_chart(lms_chart, use_container_width=True)

        std_by_modality_df = standard_deviation_by_modality(
            hash_filtered_df,
            variable,
            filtered_connectors,
            selected_modalities or None,
            segmentation_mode,
            tokenization_mode,
        )

        if not std_by_modality_df.empty:
            st.subheader(f"Ecart-type de la variable : {variable}")
            st.markdown(ECART_TYPE_EXPLANATION)
            st.dataframe(
                std_by_modality_df.rename(
                    columns={
                        "modalite": "Modalité",
                        "segments": "Segments comptés",
                        "lms": "LMS",
                        "ecart_type": "Écart-type",
                    }
                ),
                use_container_width=True,
            )

            std_chart = (
                alt.Chart(std_by_modality_df)
                .mark_bar()
                .encode(
                    x=alt.X("modalite:N", title="Modalité"),
                    y=alt.Y("ecart_type:Q", title="Écart-type (mots)"),
                    color=alt.Color("modalite:N", title="Modalité"),
                    tooltip=[
                        alt.Tooltip("modalite:N", title="Modalité"),
                        alt.Tooltip("ecart_type:Q", title="Écart-type", format=".4f"),
                        alt.Tooltip("segments:Q", title="Segments"),
                        alt.Tooltip("lms:Q", title="LMS", format=".4f"),
                    ],
                )
            )

            st.altair_chart(std_chart, use_container_width=True)

            st.markdown(
                "#### Dispersion des longueurs (moyenne ± écart-type)"
            )

            dispersion_chart = (
                alt.Chart(
                    std_by_modality_df.assign(
                        borne_inferieure=lambda df: (
                            df["lms"] - df["ecart_type"]
                        ).clip(lower=0),
                        borne_superieure=lambda df: df["lms"] + df["ecart_type"],
                    )
                )
                .mark_errorbar(orient="horizontal")
                .encode(
                    y=alt.Y("modalite:N", title="Modalité"),
                    x=alt.X("borne_inferieure:Q", title="Longueur (mots)"),
                    x2="borne_superieure:Q",
                    color=alt.Color("modalite:N", title="Modalité"),
                    tooltip=[
                        alt.Tooltip("modalite:N", title="Modalité"),
                        alt.Tooltip("lms:Q", title="LMS (moyenne)", format=".2f"),
                        alt.Tooltip("ecart_type:Q", title="Écart-type", format=".2f"),
                        alt.Tooltip("segments:Q", title="Segments comptés"),
                    ],
                )
            )

            lms_points = (
                alt.Chart(std_by_modality_df)
                .mark_point(size=70, filled=True)
                .encode(
                    y=alt.Y("modalite:N", title="Modalité"),
                    x=alt.X("lms:Q", title="Longueur (mots)"),
                    color=alt.Color("modalite:N", title="Modalité"),
                )
            )

            st.altair_chart(
                dispersion_chart + lms_points, use_container_width=True
            )
