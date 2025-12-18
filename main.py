from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from analyses import (
    annotate_connectors_html,
    build_label_style_block,
    count_connectors,
    count_connectors_by_label,
    generate_label_colors,
    load_connectors,
)


def parse_iramuteq(content: str) -> List[Dict[str, str]]:
    """Analyser un fichier texte de type IRaMuTeQ en une liste d'enregistrements."""

    lines = content.splitlines()
    records: List[Dict[str, str]] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if line.startswith("****"):
            tokens = line[4:].strip().split()
            variables: Dict[str, str] = {}

            for token in tokens:
                if token.startswith("*") and "_" in token:
                    name, modality = token[1:].split("_", maxsplit=1)
                    variables[name.strip()] = modality.strip()

            index += 1
            text_lines: List[str] = []

            while index < len(lines) and not lines[index].strip().startswith("****"):
                text_lines.append(lines[index])
                index += 1

            records.append(
                {
                    **variables,
                    "entete": line,
                    "texte": "\n".join(text_lines).strip(),
                }
            )
        else:
            index += 1

    return records


def build_dataframe(records: List[Dict[str, str]]) -> pd.DataFrame:
    """Créer un DataFrame avec les variables, les modalités et le texte."""

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def build_variable_stats(
    dataframe: pd.DataFrame,
    variables: List[str],
    connectors: Dict[str, str],
    labels: List[str],
) -> pd.DataFrame:
    """Construire un tableau des occurrences par variable, modalité et label."""

    rows: List[Dict[str, str | int]] = []

    for variable in variables:
        if variable not in dataframe.columns:
            continue

        for modality, subset in dataframe.dropna(subset=[variable]).groupby(variable):
            modality_text = " ".join(subset["texte"].dropna())
            label_counts = count_connectors_by_label(modality_text, connectors)

            for label in labels:
                rows.append(
                    {
                        "variable": variable,
                        "modalite": modality,
                        "label": label,
                        "occurrences": label_counts.get(label, 0),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Symbolic Connectors", layout="wide")

    st.title("Symbolic Connectors")
    st.write(
        "Téléversez un fichier texte IRaMuTeQ. Chaque article doit démarrer par "
        "une ligne de variables, par exemple `**** *model_gpt *prompt_1`."
    )

    uploaded_file = st.file_uploader("Fichier IRaMuTeQ", type=["txt"])  # type: ignore[assignment]

    if not uploaded_file:
        return

    content = uploaded_file.read().decode("utf-8")
    records = parse_iramuteq(content)

    if not records:
        st.warning("Aucune entrée valide trouvée dans le fichier fourni.")
        return

    df = build_dataframe(records)
    tabs = st.tabs(["Import", "Analyse stats"])

    with tabs[0]:
        st.subheader("Données importées")
        st.dataframe(df)

    with tabs[1]:
        variable_names = [column for column in df.columns if column not in ("texte", "entete")]
        st.subheader("Filtrer par variables")
        selected_variables = st.multiselect(
            "Variables disponibles", variable_names, default=variable_names
        )

        modality_filters: Dict[str, List[str]] = {}
        filtered_df = df.copy()

        for variable in selected_variables:
            options = sorted(filtered_df[variable].dropna().unique().tolist())
            selected_modalities = st.multiselect(
                f"Modalités pour {variable}", options, default=options
            )
            modality_filters[variable] = selected_modalities
            filtered_df = filtered_df[filtered_df[variable].isin(selected_modalities)]

        combined_text_parts: List[str] = []

        for _, row in filtered_df.iterrows():
            header = str(row.get("entete", "")).strip()
            body = str(row.get("texte", "")).strip()

            if header and body:
                combined_text_parts.append(f"{header}\n{body}")
            elif body:
                combined_text_parts.append(body)
            elif header:
                combined_text_parts.append(header)

        combined_text = "\n\n".join(part for part in combined_text_parts if part).strip()

        st.subheader("Texte combiné")
        if combined_text:
            st.text_area("", combined_text, height=200)
        else:
            st.info("Aucun texte ne correspond aux filtres sélectionnés.")
            return

        connectors = load_connectors(Path(__file__).parent / "dictionnaires" / "connecteurs.json")
        allowed_labels = {"ALTERNATIVE", "CONDITION", "ALORS"}
        connectors = {
            connector: label for connector, label in connectors.items() if label in allowed_labels
        }

        if not connectors:
            st.warning("Aucun connecteur valide disponible dans le dictionnaire fourni.")
            return

        connector_names = sorted(connectors.keys())
        selected_connector_names = st.multiselect(
            "Connecteurs à annoter",
            connector_names,
            default=connector_names,
            help="Sélectionnez les connecteurs à mettre en surbrillance dans le texte.",
        )

        filtered_connectors = {
            connector: label
            for connector, label in connectors.items()
            if connector in selected_connector_names
        }

        label_colors = generate_label_colors(filtered_connectors.values())
        label_style_block = build_label_style_block(label_colors)
        annotated_html = annotate_connectors_html(combined_text, filtered_connectors)

        st.subheader("Texte annoté par connecteurs")
        annotation_style = f"""
        <style>
        .connector-annotation {{
            background-color: #eef3ff;
            border-radius: 4px;
            padding: 2px 6px;
            margin: 0 2px;
            display: inline-block;
            border: 1px solid #c3d4ff;
        }}
        .connector-label {{
            color: #1a56db;
            font-weight: 700;
            margin-right: 6px;
        }}
        .connector-text {{
            color: #111827;
            font-weight: 500;
        }}
        .annotated-container {{
            line-height: 1.6;
            font-size: 15px;
        }}
        {label_style_block}
        </style>
        """

        st.markdown(annotation_style, unsafe_allow_html=True)
        st.markdown(
            f"<div class='annotated-container'>{annotated_html}</div>",
            unsafe_allow_html=True,
        )

        downloadable_html = f"""<!DOCTYPE html>
        <html lang=\"fr\">
        <head>
        <meta charset=\"utf-8\" />
        {annotation_style}
        </head>
        <body>
        <div class='annotated-container'>{annotated_html}</div>
        </body>
        </html>"""

        st.download_button(
            label="Télécharger le texte annoté (HTML)",
            data=downloadable_html,
            file_name="texte_annote.html",
            mime="text/html",
        )

        if not filtered_connectors:
            st.info("Sélectionnez au moins un connecteur pour afficher les statistiques.")
            return

        stats_df = count_connectors(combined_text, filtered_connectors)

        st.subheader("Statistiques des connecteurs")
        if stats_df.empty:
            st.info("Aucun connecteur trouvé dans le texte sélectionné.")
            return

        st.dataframe(stats_df)

        chart = (
            alt.Chart(stats_df)
            .mark_bar()
            .encode(
                x=alt.X("connecteur", sort="-y", title="Connecteur"),
                y=alt.Y("occurrences", title="Occurrences"),
                color=alt.Color("label", title="Label"),
                tooltip=["connecteur", "label", "occurrences"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Statistiques par variables")

        label_counts_overall = count_connectors_by_label(combined_text, filtered_connectors)
        selected_labels = sorted(
            label_counts_overall,
            key=label_counts_overall.get,
            reverse=True,
        )[:3]

        if not selected_labels:
            selected_labels = sorted(set(filtered_connectors.values()))[:3]

        variable_stats_df = build_variable_stats(
            filtered_df, selected_variables, filtered_connectors, selected_labels
        )

        if variable_stats_df.empty:
            st.info("Aucune donnée disponible pour les statistiques par variables.")
            return

        variable_chart = (
            alt.Chart(variable_stats_df)
            .mark_bar()
            .encode(
                x=alt.X("modalite:N", title="Modalité"),
                xOffset="label",
                y=alt.Y("occurrences:Q", title="Occurrences"),
                color=alt.Color("label:N", title="Connecteur"),
                column=alt.Column("variable:N", title="Variable"),
                tooltip=["variable", "modalite", "label", "occurrences"],
            )
            .properties(spacing=20)
        )

        st.altair_chart(variable_chart, use_container_width=True)


if __name__ == "__main__":
    main()
