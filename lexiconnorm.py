"""Composants Streamlit pour l'onglet "Lexicon norm"."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from densite import (
    build_text_from_dataframe,
    compute_density_per_modality_by_label,
    count_words,
    filter_dataframe_by_modalities,
)
from regexanalyse import RegexPattern, load_regex_rules


def load_norm_patterns(path: Path) -> List[RegexPattern]:
    """Charger les motifs regex utilisés comme normes depuis un fichier JSON."""

    if not path.exists():
        return []

    return load_regex_rules(path)


def compute_norm_densities(
    text: str, patterns: Sequence[RegexPattern], base: int
) -> pd.DataFrame:
    """Calculer la densité de chaque motif sélectionné pour un texte donné."""

    if not text or not patterns:
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    word_count = count_words(text)

    rows: List[Dict[str, float | int | str]] = []
    for pattern in patterns:
        occurrences = len(list(pattern.compiled.finditer(text)))
        density = 0.0

        if word_count and occurrences:
            density = (occurrences / word_count) * float(base)

        rows.append(
            {
                "label": pattern.label or pattern.pattern_id or "Motif",
                "densite": density,
                "occurrences": occurrences,
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def _select_variable_modalities(
    dataframe: pd.DataFrame, variable_choice: str
) -> Iterable[str]:
    """Retourner les modalités disponibles pour la variable choisie."""

    if variable_choice == "(Aucune)" or variable_choice not in dataframe.columns:
        return []

    return sorted(dataframe[variable_choice].dropna().unique().tolist())


def render_lexicon_norm_tab(
    filtered_df: pd.DataFrame, filtered_connectors: Dict[str, str]
) -> None:
    """Afficher l'onglet « Lexicon norm » avec les normes sélectionnées."""

    st.subheader("Lexicon norm")

    if filtered_df.empty:
        st.info("Aucune donnée disponible après filtrage.")
        return

    if not filtered_connectors:
        st.info("Sélectionnez au moins un connecteur pour afficher la densité.")
        return

    variables = [column for column in filtered_df.columns if column not in ("texte", "entete")]
    default_index = 0 if not variables else 1
    variable_choice = st.selectbox(
        "Variable à filtrer",
        ["(Aucune)"] + variables,
        index=default_index,
        help="Choisissez la variable utilisée pour séparer les modalités.",
    )

    modality_options = _select_variable_modalities(filtered_df, variable_choice)
    selected_modalities = st.multiselect(
        "Modalités incluses",
        modality_options,
        default=modality_options,
        help="Affiner le calcul des densités par modalité.",
    )

    base = st.number_input(
        "Base de normalisation (mots)",
        min_value=10,
        max_value=100_000,
        value=1000,
        step=10,
    )

    density_filtered_df = filter_dataframe_by_modalities(
        filtered_df,
        None if variable_choice == "(Aucune)" else variable_choice,
        selected_modalities or None,
    )

    per_modality_label_df = compute_density_per_modality_by_label(
        density_filtered_df,
        None if variable_choice == "(Aucune)" else variable_choice,
        filtered_connectors,
        base=int(base),
    )

    regex_norm_path = Path(__file__).parent / "dictionnaires" / "motifs_progr_regex.json"
    norm_patterns = load_norm_patterns(regex_norm_path)

    st.markdown("### Normes disponibles")

    selected_patterns: List[RegexPattern] = []
    if not norm_patterns:
        st.info("Aucune norme n'a été trouvée dans le fichier motifs_progr_regex.json.")
    else:
        for pattern in norm_patterns:
            checkbox_label = pattern.label or pattern.pattern_id or "Motif"
            if st.checkbox(
                checkbox_label,
                value=True,
                key=f"lexicon-norm-{pattern.pattern_id or checkbox_label}",
            ):
                selected_patterns.append(pattern)

    norm_density_df = compute_norm_densities(
        build_text_from_dataframe(density_filtered_df), selected_patterns, base=int(base)
    )

    st.markdown("### Densité par connecteur et modalités")

    if per_modality_label_df.empty:
        st.info("Aucune donnée de densité disponible pour les paramètres sélectionnés.")
        return

    bar_chart = (
        alt.Chart(per_modality_label_df)
        .mark_bar()
        .encode(
            x=alt.X("modalite:N", title="Modalité"),
            y=alt.Y("densite:Q", title=f"Densité pour {int(base)} mots"),
            color=alt.Color("label:N", title="Connecteur"),
            tooltip=["modalite", "label", "densite", "mots", "connecteurs"],
        )
    )

    layers: List[alt.Chart] = [bar_chart]

    if not norm_density_df.empty:
        rule_layer = (
            alt.Chart(norm_density_df)
            .mark_rule(color="#dc2626", strokeDash=[6, 3])
            .encode(
                y=alt.Y("densite:Q", title=None),
                tooltip=["label", "densite", "occurrences"],
            )
        )

        text_layer = (
            alt.Chart(norm_density_df)
            .mark_text(align="left", baseline="bottom", dx=6, dy=-2, color="#dc2626", fontWeight="bold")
            .encode(y="densite:Q", text="label")
        )

        layers.extend([rule_layer, text_layer])

    density_chart = alt.layer(*layers)
    st.altair_chart(density_chart, use_container_width=True)
