"""# Onglet Regex motifs

Ce module gère l'onglet Streamlit dédié à l'analyse de motifs regex complexes
dans le corpus, en chargeant un dictionnaire JSON et en affichant les
annotations et statistiques associées.

## Dépendances
- `regexanalyse.py` : chargement des règles regex, segmentation du texte et
  calcul des statistiques de correspondance.
- `analyses.py` : génération des couleurs et styles de labels pour les
  annotations.
- `fcts_utils.py` : construction du bloc de styles pour l'affichage HTML.
- Bibliothèques `streamlit`, `pandas`, `altair` et `pathlib` pour la gestion
  de l'interface, des données et du chemin vers les ressources.
"""
from __future__ import annotations

from html import escape
import re
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from analyses import (
    annotate_connectors_html,
    build_label_style_block,
    generate_label_colors,
)
from fcts_utils import build_annotation_style_block
from regexanalyse import (
    count_segments_by_pattern,
    highlight_matches_html,
    load_regex_rules,
    split_segments,
    summarize_matches_by_segment,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def _build_plain_annotated_text(
    text: str, connectors: Dict[str, str]
) -> str:
    """Annoter un texte en insérant les labels des connecteurs en clair.

    Les connecteurs détectés sont préfixés par leur label entre crochets afin de
    conserver l'information d'annotation dans un format texte simple.
    """

    cleaned_connectors = {key: value for key, value in connectors.items() if key and value}

    if not text or not cleaned_connectors:
        return text

    sorted_keys = sorted(cleaned_connectors.keys(), key=len, reverse=True)
    escaped_connectors = "|".join(re.escape(key) for key in sorted_keys)
    pattern = re.compile(rf"\b({escaped_connectors})\b", re.IGNORECASE)
    label_lookup = {key.lower(): value for key, value in cleaned_connectors.items()}

    def _replacer(match: re.Match[str]) -> str:
        connector = match.group(0)
        label = label_lookup.get(connector.lower(), "")

        return f"[{label}] {connector}" if label else connector

    return pattern.sub(_replacer, text)


def rendu_regex_motifs(tab, combined_text: str, filtered_connectors: Dict[str, str]) -> None:
    st.subheader("Regex motifs")

    texte_html = f"""<!DOCTYPE html>
    <html lang=\"fr\">
    <head>
    <meta charset=\"utf-8\" />
    </head>
    <body>
    <pre>{escape(combined_text)}</pre>
    </body>
    </html>"""

    col_html, col_txt = st.columns(2)

    with col_html:
        st.download_button(
            label="Télécharger le texte (HTML)",
            data=texte_html,
            file_name="corpus_combine.html",
            mime="text/html",
            key="download-combined-html",
        )

    with col_txt:
        st.download_button(
            label="Télécharger le texte (TXT)",
            data=combined_text,
            file_name="corpus_combine.txt",
            mime="text/plain",
            key="download-combined-txt",
        )

    connector_label_colors = generate_label_colors(filtered_connectors.values())
    connector_label_style = build_label_style_block(connector_label_colors)
    connector_annotation_style = build_annotation_style_block(connector_label_style)
    annotated_connectors_html = annotate_connectors_html(combined_text, filtered_connectors)
    annotated_connectors_text = _build_plain_annotated_text(
        combined_text, filtered_connectors
    )

    annotated_connectors_doc = f"""<!DOCTYPE html>
    <html lang=\"fr\">
    <head>
    <meta charset=\"utf-8\" />
    {connector_annotation_style}
    </head>
    <body>
    <div class='annotated-container'>{annotated_connectors_html}</div>
    </body>
    </html>"""

    download_connect_html, download_connect_txt = st.columns(2)

    with download_connect_html:
        st.download_button(
            label="Télécharger le texte annoté (HTML)",
            data=annotated_connectors_doc,
            file_name="texte_annote_connecteurs.html",
            mime="text/html",
            key="download-annotated-connectors-html",
        )

    with download_connect_txt:
        st.download_button(
            label="Télécharger le texte annoté (TXT)",
            data=annotated_connectors_text,
            file_name="texte_annote_connecteurs.txt",
            mime="text/plain",
            key="download-annotated-connectors-txt",
        )

    st.markdown(
        """
        Dans cet onglet, les motifs regex repèrent des structures combinées
        (ex : si…alors, si…sinon) dans les segments. La recherche est bornée par la ponctuation
        du texte (. ! ? ; : ou retour ligne) garantissant que les connecteurs sont détectés dans
        une unité lexicale (la phrase).
        """
    )

    regex_rules_path = BASE_DIR / "dictionnaires" / "motifs_progr_regex.json"
    regex_patterns = load_regex_rules(regex_rules_path)

    if not regex_patterns:
        st.info("Aucun motif regex n'a pu être chargé depuis le dictionnaire fourni.")
        return

    regex_label_colors = generate_label_colors([pattern.label for pattern in regex_patterns])
    regex_label_style = build_label_style_block(regex_label_colors)
    regex_annotation_style = build_annotation_style_block(regex_label_style)

    st.markdown(regex_annotation_style, unsafe_allow_html=True)

    highlighted_corpus = highlight_matches_html(combined_text, regex_patterns)
    st.markdown("Corpus annoté (motifs regex)")
    st.markdown(
        f"<div class='annotated-container'>{highlighted_corpus}</div>",
        unsafe_allow_html=True,
    )

    downloadable_regex_html = f"""<!DOCTYPE html>
    <html lang=\"fr\">
    <head>
    <meta charset=\"utf-8\" />
    {regex_annotation_style}
    </head>
    <body>
    <div class='annotated-container'>{highlighted_corpus}</div>
    </body>
    </html>"""

    st.download_button(
        label="Télécharger le corpus annoté (HTML)",
        data=downloadable_regex_html,
        file_name="corpus_regex_annote.html",
        mime="text/html",
        key="download-regex-annotated-html",
    )

    segments = split_segments(combined_text)
    segment_rows = summarize_matches_by_segment(segments, regex_patterns)

    st.markdown("---")
    st.subheader("Segments contenant au moins un motif")

    if not segment_rows:
        st.info("Aucun motif regex détecté dans le corpus fourni.")
        return

    table_rows = []

    for row in segment_rows:
        motif_details = "; ".join(
            f"{motif['label']} ({motif['occurrences']})" for motif in row["motifs"]
        )
        table_rows.append(
            {
                "Segment": row["segment_id"],
                "Texte": row["segment"],
                "Motifs détectés": motif_details,
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    segment_counts = count_segments_by_pattern(segment_rows)

    if segment_counts:
        st.subheader("Nombre de segments matchés par motif")
        counts_df = pd.DataFrame(
            [
                {"motif": motif, "segments": count}
                for motif, count in segment_counts.items()
            ]
        ).sort_values("segments", ascending=False)

        alt_counts_chart = (
            alt.Chart(counts_df)
            .mark_bar()
            .encode(
                x=alt.X("motif:N", sort="-y", title="Motif"),
                y=alt.Y("segments:Q", title="Segments matchés"),
                tooltip=["motif", "segments"],
            )
            .properties(title="Nombre de segments matchés par motif")
        )

        st.altair_chart(alt_counts_chart, use_container_width=True)
