"""# Onglet Connecteurs

Ce module affiche l'onglet Streamlit permettant de charger le dictionnaire
de connecteurs, de sélectionner les étiquettes à conserver et de visualiser
leurs occurrences dans le corpus.

## Dépendances
- `analyses.py` : annotation HTML des connecteurs et génération des styles
  associés.
- `connecteurs.py` : accès au chemin du dictionnaire, sélection et filtrage
  des connecteurs et labels.
- `fcts_utils.py` : rappel visuel des connecteurs sélectionnés dans l'UI.
- Bibliothèque Streamlit et `json` pour l'affichage interactif du fichier
  `connecteurs.json`.
"""
from __future__ import annotations

import json
from typing import Dict

import streamlit as st

from analyses import annotate_connectors_html, build_label_style_block, count_connectors, generate_label_colors
from connecteurs import (
    get_connectors_path,
    get_selected_connectors,
    get_selected_labels,
    load_available_connectors,
    set_selected_connectors,
)
from fcts_utils import render_connectors_reminder


def rendu_connecteurs(tab) -> None:
    render_connectors_reminder(get_selected_connectors())
    connectors_path = get_connectors_path()

    with tab.expander("Afficher le contenu de connecteurs.json"):
        st.caption(f"Fichier chargé : `{connectors_path}`")
        try:
            with connectors_path.open(encoding="utf-8") as handle:
                st.json(json.load(handle))
        except FileNotFoundError:
            st.error(
                "Le fichier de connecteurs est introuvable. Vérifiez la présence de "
                "`dictionnaires/connecteurs.json`."
            )
        except json.JSONDecodeError:
            st.error(
                "Impossible de lire `connecteurs.json` : le fichier ne contient pas un JSON valide."
            )
    try:
        available_connectors = load_available_connectors(connectors_path)
    except FileNotFoundError:
        st.error(
            "Le fichier de connecteurs est introuvable. Vérifiez la présence de "
            "`dictionnaires/connecteurs.json`."
        )
        available_connectors = {}

    allowed_labels = {"ALTERNATIVE", "CONDITION", "ALORS", "AND", "RETOUR À LA LIGNE"}
    available_connectors = {
        connector: label
        for connector, label in available_connectors.items()
        if label in allowed_labels
    }

    if not available_connectors:
        st.warning(
            "Aucun connecteur valide disponible dans le dictionnaire fourni. "
            "Ajoutez des entrées ou ajustez les filtres pour continuer."
        )
        return

    all_labels = sorted(set(available_connectors.values()))
    previously_selected = get_selected_labels(
        get_selected_connectors().values()
    ) or all_labels

    selected_labels = st.multiselect(
        "Labels de connecteurs à inclure",
        all_labels,
        default=previously_selected,
        help="Les connecteurs des labels sélectionnés seront utilisés dans tous les onglets.",
        key="connectors_labels_multiselect",
    )

    filtered_connectors = {
        connector: label
        for connector, label in available_connectors.items()
        if label in selected_labels
    }

    set_selected_connectors(filtered_connectors)

    st.success(f"{len(filtered_connectors)} connecteurs sélectionnés pour les analyses.")
    render_connectors_reminder(filtered_connectors)

