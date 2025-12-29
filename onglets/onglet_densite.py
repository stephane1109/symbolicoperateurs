"""Onglet de calcul de densité des connecteurs."""
from __future__ import annotations

from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from densite import (
    build_text_from_dataframe,
    compute_density,
    compute_density_per_modality,
    compute_density_per_modality_by_label,
    compute_total_connectors,
    count_words,
    filter_dataframe_by_modalities,
)
from fcts_utils import render_connectors_reminder
from graphiques.densitegraph import build_connector_density_chart, build_density_chart


def rendu_densite(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    render_connectors_reminder(filtered_connectors)
    st.write(
        "Densité des textes analysés : La densité correspond au nombre de connecteurs "
        "ramené à une base de 1 000 mots. "
    )
    if not filtered_connectors:
        st.info("Sélectionnez au moins un connecteur pour calculer la densité.")
        return

    st.subheader("Sélection des variables/modalités")
    density_variables = [column for column in df.columns if column not in ("texte", "entete")]
    default_density_index = 0 if not density_variables else 1
    density_variable_choice = st.selectbox(
        "Variable à filtrer pour la densité",
        ["(Aucune)"] + density_variables,
        index=default_density_index,
        help="Choisissez une variable pour restreindre le calcul à certaines modalités.",
    )

    density_modalities: List[str] = []

    if density_variable_choice != "(Aucune)":
        modality_options = sorted(df[density_variable_choice].dropna().unique().tolist())
        density_modalities = st.multiselect(
            "Modalités à inclure",
            modality_options,
            default=modality_options,
            help="Sélectionnez une ou plusieurs modalités pour filtrer l'analyse de densité.",
        )

    density_filtered_df = filter_dataframe_by_modalities(
        df,
        None if density_variable_choice == "(Aucune)" else density_variable_choice,
        density_modalities or None,
    )

    density_text = build_text_from_dataframe(density_filtered_df)
    if not density_text:
        st.info("Aucun texte disponible avec les modalités sélectionnées pour calculer la densité.")
        return

    st.download_button(
        label="Télécharger le texte",
        data=density_text,
        file_name="texte_filtre_densite.txt",
        mime="text/plain",
        help="Récupérez le texte correspondant aux variables/modalités sélectionnées pour vérifier le filtrage.",
    )
    st.caption(
        "Utilisez ce bouton pour vérifier facilement le contenu exact retenu après votre sélection de variables et modalités."
    )

    base = 1000

    total_words = count_words(density_text)
    total_connectors = compute_total_connectors(density_text, filtered_connectors)
    density = compute_density(density_text, filtered_connectors, base=base)

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre total de mots", f"{total_words:,}".replace(",", " "))
    col2.metric("Occurrences de connecteurs", f"{total_connectors:,}".replace(",", " "))
    col3.metric(f"Densité pour {base:,} mots", f"{density:.2f}".replace(",", " "))

    if total_connectors == 0:
        st.info("Aucun connecteur détecté : la densité est nulle pour ce texte.")

    st.caption(
        "La densité correspond au nombre de connecteurs ramené à une base commune. "
        "Un score élevé signale un texte plus riche en connecteurs logiques."
    )

    per_modality_df = compute_density_per_modality(
        density_filtered_df,
        None if density_variable_choice == "(Aucune)" else density_variable_choice,
        filtered_connectors,
        base=base,
    )
    per_modality_label_df = compute_density_per_modality_by_label(
        density_filtered_df,
        None if density_variable_choice == "(Aucune)" else density_variable_choice,
        filtered_connectors,
        base=base,
    )

    if not per_modality_df.empty:
        st.subheader("Densité par modalité sélectionnée")
        modality_display_df = per_modality_df.copy()
        modality_display_df["densite"] = modality_display_df["densite"].apply(
            lambda value: f"{value:.2f}"
        )
        modality_display_df["mots"] = modality_display_df["mots"].apply(
            lambda value: f"{int(value)}"
        )
        modality_display_df["connecteurs"] = modality_display_df["connecteurs"].apply(
            lambda value: f"{int(value)}"
        )

        modality_display_df = modality_display_df.rename(
            columns={
                "modalite": "Modalité",
                "densite": "Densité",
                "mots": "Mots comptés",
                "connecteurs": "Connecteurs",
            }
        )

        st.dataframe(
            modality_display_df,
            use_container_width=True,
            column_config={
                "Densité": st.column_config.TextColumn("Densité"),
                "Mots comptés": st.column_config.TextColumn("Mots comptés"),
                "Connecteurs": st.column_config.TextColumn("Connecteurs"),
            },
        )

        st.markdown("#### Graphique de densité")
        st.altair_chart(
            build_density_chart(per_modality_df),
            use_container_width=True,
        )

        if not per_modality_label_df.empty:
            st.markdown("#### Densité par connecteur et modalité")
            modality_label_display_df = per_modality_label_df.copy()
            modality_label_display_df["densite"] = modality_label_display_df[
                "densite"
            ].apply(lambda value: f"{value:.2f}")
            modality_label_display_df["mots"] = modality_label_display_df[
                "mots"
            ].apply(lambda value: f"{int(value)}")
            modality_label_display_df["connecteurs"] = modality_label_display_df[
                "connecteurs"
            ].apply(lambda value: f"{int(value)}")

            modality_label_display_df = modality_label_display_df.rename(
                columns={
                    "modalite": "Modalité",
                    "label": "Connecteur",
                    "densite": "Densité",
                    "mots": "Mots comptés",
                    "connecteurs": "Connecteurs",
                }
            )

            st.dataframe(
                modality_label_display_df,
                use_container_width=True,
                column_config={
                    "Densité": st.column_config.TextColumn("Densité"),
                    "Mots comptés": st.column_config.TextColumn("Mots comptés"),
                    "Connecteurs": st.column_config.TextColumn("Connecteurs"),
                },
            )

            st.altair_chart(
                build_connector_density_chart(per_modality_label_df),
                use_container_width=True,
            )
