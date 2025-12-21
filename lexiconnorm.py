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
    compute_total_connectors,
    count_words,
    filter_dataframe_by_modalities,
)


def load_norms_from_lexicon(
    lexicon_path: Path, connectors: Dict[str, str]
) -> pd.DataFrame:
    """Charger les normes disponibles à partir du dictionnaire lexicographique."""

    if not lexicon_path.exists():
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    try:
        raw_df = pd.read_json(lexicon_path)
        lexicon_entries = raw_df.get("entries", pd.Series(dtype=object)).dropna().tolist()
        lexicon_df = pd.DataFrame(lexicon_entries)
    except (ValueError, TypeError):
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    if lexicon_df.empty or "ortho" not in lexicon_df.columns:
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    connector_tokens = {connector.lower() for connector in connectors}
    lexicon_df["ortho_lower"] = lexicon_df["ortho"].str.lower()
    lexicon_df = lexicon_df[lexicon_df["ortho_lower"].isin(connector_tokens)]

    norm_columns = [
        column
        for column in lexicon_df.columns
        if column not in {"ortho", "Lexique3__cgram", "ortho_lower"}
    ]

    rows: List[Dict[str, float | str]] = []

    for column in norm_columns:
        numeric_values = pd.to_numeric(lexicon_df[column], errors="coerce").dropna()

        if numeric_values.empty:
            continue

        rows.append(
            {
                "label": column,
                "densite": float(numeric_values.mean()),
                "occurrences": float(numeric_values.sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def compute_norm_densities_by_label(
    text: str, connectors: Dict[str, str], labels: Sequence[str], base: int
) -> pd.DataFrame:
    """Calculer la densité de normes par label sur un texte donné."""

    if not text or not connectors or not labels:
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    word_count = count_words(text)
    if word_count == 0:
        return pd.DataFrame(columns=["label", "densite", "occurrences"])

    rows: List[Dict[str, float | int | str]] = []

    for label in labels:
        label_connectors = {
            connector: connector_label
            for connector, connector_label in connectors.items()
            if connector_label == label
        }
        occurrences = compute_total_connectors(text, label_connectors)
        density = 0.0

        if occurrences:
            density = (occurrences / word_count) * float(base)

        rows.append({"label": label.lower(), "densite": density, "occurrences": occurrences})

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

    allowed_labels = {"CONDITION", "ALORS", "ALTERNATIVE", "AND"}
    normalized_connectors = {
        connector: label
        for connector, label in filtered_connectors.items()
        if label in allowed_labels
    }

    if not normalized_connectors:
        st.info(
            "Sélectionnez au moins un connecteur de type condition, alors, alternative ou addition pour afficher la densité."
        )
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

    base = 1000
    st.markdown(
        "Les densités affichées sont systématiquement normalisées sur 1 000 mots afin de "
        "pouvoir comparer les corpus entre eux, quelle que soit leur longueur."
    )

    density_filtered_df = filter_dataframe_by_modalities(
        filtered_df,
        None if variable_choice == "(Aucune)" else variable_choice,
        selected_modalities or None,
    )

    per_modality_label_df = compute_density_per_modality_by_label(
        density_filtered_df,
        None if variable_choice == "(Aucune)" else variable_choice,
        normalized_connectors,
        base=int(base),
    )

    st.markdown("### Normes disponibles")

    norm_density_df = load_norms_from_lexicon(
        Path(__file__).parent / "dictionnaires" / "lexicon.json", normalized_connectors
    )

    if norm_density_df.empty:
        st.info("Aucune norme disponible pour les connecteurs sélectionnés.")
    else:
        # Les fréquences Lexique sont exprimées pour un million de mots ;
        # on les ramène à une base de 1 000 mots pour aligner le graphique.
        norm_density_df["densite"] = norm_density_df["densite"] / 1000.0

        selected_labels = []
        for _, row in norm_density_df.iterrows():
            checkbox_label = row["label"]
            if st.checkbox(
                checkbox_label,
                value=True,
                key=f"lexicon-norm-{row['label']}",
            ):
                selected_labels.append(row)

        norm_density_df = pd.DataFrame(selected_labels)

    st.markdown("### Densité par connecteur et modalités")

    if per_modality_label_df.empty:
        st.info("Aucune donnée de densité disponible pour les paramètres sélectionnés.")
        return

    per_modality_label_df["label"] = per_modality_label_df["label"].str.lower()

    bar_chart = (
        alt.Chart(per_modality_label_df)
        .mark_bar()
        .encode(
            x=alt.X("modalite:N", title="Modalité"),
            xOffset="label",
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
