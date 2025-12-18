from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from analyses import (
    annotate_connectors_html,
    count_connectors,
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

            records.append({**variables, "texte": "\n".join(text_lines).strip()})
        else:
            index += 1

    return records


def build_dataframe(records: List[Dict[str, str]]) -> pd.DataFrame:
    """Créer un DataFrame avec les variables, les modalités et le texte."""

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def main() -> None:
    st.title("Analyse de connecteurs dans un corpus IRaMuTeQ")
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
    st.subheader("Données importées")
    st.dataframe(df)

    variable_names = [column for column in df.columns if column != "texte"]
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

    combined_text = " ".join(filtered_df["texte"].dropna()).strip()

    st.subheader("Texte combiné")
    if combined_text:
        st.text_area("", combined_text, height=200)
    else:
        st.info("Aucun texte ne correspond aux filtres sélectionnés.")
        return

    connectors = load_connectors(Path(__file__).parent / "dictionnaires" / "connecteurs.json")
    annotated_html = annotate_connectors_html(combined_text, connectors)

    st.subheader("Texte annoté par connecteurs")
    annotation_style = """
    <style>
    .connector-annotation {
        background-color: #eef3ff;
        border-radius: 4px;
        padding: 2px 6px;
        margin: 0 2px;
        display: inline-block;
    }
    .connector-label {
        color: #1a56db;
        font-weight: 700;
        margin-right: 6px;
    }
    .connector-text {
        color: #111827;
        font-weight: 500;
    }
    .annotated-container {
        line-height: 1.6;
        font-size: 15px;
    }
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

    stats_df = count_connectors(combined_text, connectors)

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


if __name__ == "__main__":
    main()
