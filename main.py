from __future__ import annotations

import sys
import io
import re
import json
import functools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords as nltk_stopwords

APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from analyses import (
    annotate_connectors_html,
    build_label_style_block,
    count_connectors,
    count_connectors_by_label,
    generate_label_colors,
)
from connecteurs import (
    get_connectors_path,
    get_selected_connectors,
    get_selected_labels,
    load_available_connectors,
    set_selected_connectors,
)
from lexiconnorm import render_lexicon_norm_tab
from densite import (
    compute_density,
    compute_density_by_label,
    compute_total_connectors,
    count_words,
    build_text_from_dataframe,
    compute_density_per_modality,
    filter_dataframe_by_modalities,
    compute_density_per_modality_by_label,
)
from ecartype import (
    compute_length_standard_deviation,
    standard_deviation_by_modality,
)
from hash import (
    ECART_TYPE_EXPLANATION,
    average_segment_length,
    average_segment_length_by_modality,
    compute_segment_word_lengths,
    segments_with_word_lengths,
)
from regexanalyse import (
    count_segments_by_pattern,
    highlight_matches_html,
    load_regex_rules,
    split_segments,
    summarize_matches_by_segment,
)
from test_lesch_Kincaid import (
    READABILITY_SCALE,
    compute_flesch_kincaid_metrics,
    get_readability_band,
    interpret_reading_ease,
)
from souscorpus import build_subcorpus


def display_centered_image(image_buffer: io.BytesIO, caption: str, width: int = 1200) -> None:
    """Afficher une image centrée avec une largeur et une légende cohérentes."""

    center_col = st.columns([1, 2, 1])[1]
    center_col.image(image_buffer, width=width, caption=caption)


def _normalize_words(words: list[str]) -> set[str]:
    return {word.strip().lower() for word in words if isinstance(word, str) and word.strip()}


@functools.lru_cache(maxsize=1)
def load_stopwords(path: Path | None = None) -> set[str]:
    """Charger une liste de mots vides en combinant NLTK et un fichier JSON optionnel."""

    stopword_set: set[str] = set()

    try:
        stopword_set.update(_normalize_words(nltk_stopwords.words("french")))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stopword_set.update(_normalize_words(nltk_stopwords.words("french")))

    if path and path.exists():
        with path.open(encoding="utf-8") as handle:
            words = json.load(handle)
        stopword_set.update(_normalize_words(words))

    return stopword_set


def build_annotation_style_block(label_style_block: str) -> str:
    """Créer un bloc de style commun pour l'affichage des annotations HTML."""

    return f"""
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
        color: #6b7280;
        font-weight: 500;
    }}
    .annotated-container {{
        line-height: 1.6;
        font-size: 15px;
    }}
    {label_style_block}
    </style>
    """


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


def render_connectors_reminder(connectors: Dict[str, str]) -> None:
    """Afficher une phrase rappelant les connecteurs sélectionnés."""

    if not connectors:
        st.info(
            "Aucun connecteur sélectionné pour les analyses. Rendez-vous dans l'onglet « Connecteurs » pour en choisir."
        )
        return

    connectors_by_label: Dict[str, List[str]] = {}
    for connector, label in connectors.items():
        connectors_by_label.setdefault(label, []).append(connector)

    label_summaries = [
        f"{label} ({len(names)})" for label, names in sorted(connectors_by_label.items())
    ]
    st.caption(
        "Connecteurs sélectionnés par catégorie — "
        + ", ".join(label_summaries)
        + f" — Total : {len(connectors)}"
    )


def main() -> None:
    st.set_page_config(page_title="Symbolic Connectors", layout="wide")

    st.title("Symbolic Connectors")
    st.markdown(
        "<span style='color: white'>"
        "Symbolic => fait référence au courant symbolique de l'ia (analogie à la "
        "machine/programme) vs connexionnisme (analogie avec le cerveau).<br>"
        "Connectors => l'idée de rechercher dans un corpus des marqueurs pouvant "
        "révéler un langage machine (si, alors, sinon, et, ou...)</span>",
        unsafe_allow_html=True,
    )
    st.caption("[www.codeandcortex.fr](https://www.codeandcortex.fr)")
    st.markdown("---")
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
    tabs = st.tabs(
        [
            "Import",
            "Connecteurs",
            "Données brutes",
            "Sous corpus",
            "Densité",
            "Lexicon norm",
            "Hash",
            "Regex motifs",
            "Test de lisibilité",
        ]
    )

    with tabs[1]:
        st.subheader("Choisir les connecteurs à analyser")
        render_connectors_reminder(get_selected_connectors())
        connectors_path = get_connectors_path()
        try:
            available_connectors = load_available_connectors(connectors_path)
        except FileNotFoundError:
            st.error(
                "Le fichier de connecteurs est introuvable. Vérifiez la présence de "
                "`dictionnaires/connecteurs.json`."
            )
            available_connectors = {}

        allowed_labels = {"ALTERNATIVE", "CONDITION", "ALORS", "AND"}
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
        else:
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

            st.success(
                f"{len(filtered_connectors)} connecteurs sélectionnés pour les analyses."
            )

            render_connectors_reminder(filtered_connectors)

    filtered_connectors = get_selected_connectors()

    with tabs[0]:
        st.subheader("Données importées")
        render_connectors_reminder(filtered_connectors)
        st.dataframe(df, use_container_width=True)

    with tabs[2]:
        variable_names = [column for column in df.columns if column not in ("texte", "entete")]
        st.subheader("Filtrer par variables")
        render_connectors_reminder(filtered_connectors)
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

        combined_text = build_text_from_dataframe(filtered_df)

        st.subheader("Texte combiné")
        if combined_text:
            st.text_area("", combined_text, height=200)
        else:
            st.info("Aucun texte ne correspond aux filtres sélectionnés.")
            return

        if not filtered_connectors:
            st.info("Choisissez des connecteurs dans l'onglet « Connecteurs » pour poursuivre.")
            return

        selected_labels = get_selected_labels(filtered_connectors.values())

        label_colors = generate_label_colors(filtered_connectors.values())
        label_style_block = build_label_style_block(label_colors)
        annotated_html = annotate_connectors_html(combined_text, filtered_connectors)

        st.subheader("Texte annoté par connecteurs")
        annotation_style = build_annotation_style_block(label_style_block)

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
            stats_df = pd.DataFrame(columns=["connecteur", "label", "occurrences"])
        else:
            stats_df = count_connectors(combined_text, filtered_connectors)

        st.subheader("Statistiques des connecteurs")
        if stats_df.empty:
            st.info("Aucun connecteur trouvé dans le texte sélectionné.")
        else:
            st.dataframe(stats_df, use_container_width=True)

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
        selected_labels = sorted(set(filtered_connectors.values()))

        variable_stats_df = build_variable_stats(
            filtered_df, selected_variables, filtered_connectors, selected_labels
        )

        if variable_stats_df.empty:
            st.info("Aucune donnée disponible pour les statistiques par variables.")
        else:
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

    with tabs[3]:
        st.subheader("Sous corpus")
        render_connectors_reminder(filtered_connectors)
        st.write(
            "Extraction automatique des segments dont la première ligne contient les marqueurs "
            "IRaMuTeQ (encodage commençant par `**** *`). Le sous-corpus peut être copié, "
            "téléchargé au format texte ou réutilisé pour d'autres analyses."
        )

        subcorpus_segments = build_subcorpus(records, filtered_connectors)

        if not subcorpus_segments:
            st.info(
                "Aucun segment avec encodage `**** *` n'a été trouvé dans le fichier téléversé."
            )
        else:
            subcorpus_text = "\n\n".join(subcorpus_segments)
            st.text_area(
                "Segments du sous-corpus", subcorpus_text, height=260, key="subcorpus_text"
            )

            st.download_button(
                label="Télécharger le sous-corpus (TXT)",
                data=subcorpus_text,
                file_name="sous_corpus.txt",
                mime="text/plain",
            )

    with tabs[4]:
        st.subheader("Densité des connecteurs")
        render_connectors_reminder(filtered_connectors)
        st.write(
            "Densité des textes analysés : La densité, c'est simplement le nombre de connecteurs "
            "ramené à une base (pour 1 000 mots). C'est ce qui permet de dire par exemple : "
            '"Ce texte est 3 fois plus \'logique\' que l\'autre".'
        )
        if not filtered_connectors:
            st.info("Sélectionnez au moins un connecteur pour calculer la densité.")
        else:
            st.subheader("Sélection des variables/modalités")
            density_variables = [column for column in filtered_df.columns if column not in ("texte", "entete")]
            default_density_index = 0 if not density_variables else 1
            density_variable_choice = st.selectbox(
                "Variable à filtrer pour la densité",
                ["(Aucune)"] + density_variables,
                index=default_density_index,
                help="Choisissez une variable pour restreindre le calcul à certaines modalités.",
            )

            density_modalities: List[str] = []

            if density_variable_choice != "(Aucune)":
                modality_options = sorted(
                    filtered_df[density_variable_choice].dropna().unique().tolist()
                )
                density_modalities = st.multiselect(
                    "Modalités à inclure",
                    modality_options,
                    default=modality_options,
                    help="Sélectionnez une ou plusieurs modalités pour filtrer l'analyse de densité.",
                )

            density_filtered_df = filter_dataframe_by_modalities(
                filtered_df,
                None if density_variable_choice == "(Aucune)" else density_variable_choice,
                density_modalities or None,
            )

            density_text = build_text_from_dataframe(density_filtered_df)
            if not density_text:
                st.info("Aucun texte disponible avec les modalités sélectionnées pour calculer la densité.")
            else:

                base = st.number_input(
                    "Base de normalisation (mots)",
                    min_value=10,
                    max_value=100_000,
                    value=1000,
                    step=10,
                )

                total_words = count_words(density_text)
                total_connectors = compute_total_connectors(density_text, filtered_connectors)
                density = compute_density(density_text, filtered_connectors, base=int(base))

                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre total de mots", f"{total_words:,}".replace(",", " "))
                col2.metric("Occurrences de connecteurs", f"{total_connectors:,}".replace(",", " "))
                col3.metric(f"Densité pour {int(base):,} mots", f"{density:.2f}".replace(",", " "))

                if total_connectors == 0:
                    st.info("Aucun connecteur détecté : la densité est nulle pour ce texte.")

                st.caption(
                    "La densité correspond au nombre de connecteurs ramené à une base commune. "
                    "Un score élevé signale un texte plus riche en articulations logiques."
                )

                per_modality_df = compute_density_per_modality(
                    density_filtered_df,
                    None if density_variable_choice == "(Aucune)" else density_variable_choice,
                    filtered_connectors,
                    base=int(base),
                )
                per_modality_label_df = compute_density_per_modality_by_label(
                    density_filtered_df,
                    None if density_variable_choice == "(Aucune)" else density_variable_choice,
                    filtered_connectors,
                    base=int(base),
                )

                if not per_modality_df.empty:
                    st.subheader("Densité par modalité sélectionnée")
                    st.dataframe(
                        per_modality_df.rename(
                            columns={
                                "modalite": "Modalité",
                                "densite": "Densité",
                                "mots": "Mots comptés",
                                "connecteurs": "Connecteurs",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.markdown("#### Graphique de densité")

                    density_chart = (
                        alt.Chart(per_modality_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("modalite:N", title="Modalité"),
                            y=alt.Y("densite:Q", title="Densité"),
                            color=alt.Color("modalite:N", title="Modalité"),
                            tooltip=["modalite", "densite", "mots", "connecteurs"],
                        )
                    )

                    density_norm_rule = (
                        alt.Chart(pd.DataFrame({"norme": [1.37]}))
                        .mark_rule(color="red", strokeDash=[6, 4])
                        .encode(y=alt.Y("norme:Q"))
                    )

                    density_chart = (
                        (density_chart + density_norm_rule)
                        .properties(title="Graphique de densité")
                    )
                    st.altair_chart(density_chart, use_container_width=True)

                    if not per_modality_label_df.empty:
                        st.markdown("#### Densité par connecteur et modalité")
                        st.dataframe(
                            per_modality_label_df.rename(
                                columns={
                                    "modalite": "Modalité",
                                    "label": "Connecteur",
                                    "densite": "Densité",
                                    "mots": "Mots comptés",
                                    "connecteurs": "Connecteurs",
                                }
                            ),
                            use_container_width=True,
                        )

                        connector_density_chart = (
                            alt.Chart(per_modality_label_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("modalite:N", title="Modalité"),
                                xOffset="label",
                                y=alt.Y("densite:Q", title="Densité"),
                                color=alt.Color("label:N", title="Connecteur"),
                                tooltip=["modalite", "label", "densite", "connecteurs", "mots"],
                            )
                            .properties(title="Densité par connecteur et modalité")
                        )

                        st.altair_chart(connector_density_chart, use_container_width=True)

                density_labels = sorted(set(filtered_connectors.values()))

                if not density_labels:
                    st.info("Aucun label de connecteur disponible pour le graphique de classification.")
                else:
                    default_x_index = density_labels.index("ALTERNATIVE") if "ALTERNATIVE" in density_labels else 0
                    default_y_index = density_labels.index("CONDITION") if "CONDITION" in density_labels else min(1, len(density_labels) - 1)

                    st.subheader("Classification visuelle (X-Y)")
                    col_x, col_y = st.columns(2)
                    selected_x_label = col_x.selectbox(
                        "Marqueur pour l'axe horizontal",
                        density_labels,
                        index=default_x_index,
                        help="Choisissez le connecteur dont la densité sera placée sur l'axe horizontal.",
                    )
                    selected_y_label = col_y.selectbox(
                        "Marqueur pour l'axe vertical",
                        density_labels,
                        index=default_y_index,
                        help="Choisissez le connecteur dont la densité sera placée sur l'axe vertical.",
                    )

                    scatter_rows: List[Dict[str, float | str]] = []

                    for idx, row in density_filtered_df.iterrows():
                        text_value = str(row.get("texte", "") or "")
                        densities = compute_density_by_label(
                            text_value,
                            filtered_connectors,
                            base=int(base),
                        )
                        scatter_rows.append(
                            {
                                "entree": (str(row.get("entete", "")).strip() or f"Entrée {idx + 1}"),
                                "densite_x": densities.get(selected_x_label, 0.0),
                                "densite_y": densities.get(selected_y_label, 0.0),
                                "densite_totale": compute_density(
                                    text_value, filtered_connectors, base=int(base)
                                ),
                                **{
                                    variable: str(row.get(variable, ""))
                                    for variable in selected_variables
                                    if variable in density_filtered_df.columns
                                },
                            }
                        )

                    scatter_df = pd.DataFrame(scatter_rows)

                    if scatter_df.empty:
                        st.info("Aucune donnée disponible pour générer le graphique de densité.")
                    else:
                        tooltip_fields = ["entree", "densite_x", "densite_y", "densite_totale"] + [
                            variable for variable in selected_variables if variable in scatter_df.columns
                        ]

                        scatter_chart = (
                            alt.Chart(scatter_df)
                            .mark_circle(opacity=0.7)
                            .encode(
                                x=alt.X(
                                    "densite_x:Q",
                                    title=f"Densité {selected_x_label}",
                                ),
                                y=alt.Y(
                                    "densite_y:Q",
                                    title=f"Densité {selected_y_label}",
                                ),
                                size=alt.Size(
                                    "densite_totale:Q",
                                    title="Densité totale (taille du cercle)",
                                    scale=alt.Scale(range=[50, 1200]),
                                ),
                                color=alt.Color(
                                    "densite_totale:Q",
                                    title="Densité totale",
                                    scale=alt.Scale(scheme="oranges"),
                                ),
                                tooltip=tooltip_fields,
                            )
                            .properties(height=500)
                        )

                        st.altair_chart(scatter_chart, use_container_width=True)

    with tabs[5]:
        render_connectors_reminder(filtered_connectors)
        render_lexicon_norm_tab(filtered_df, filtered_connectors)

    with tabs[6]:
        st.subheader("Hash (LMS entre connecteurs)")
        render_connectors_reminder(filtered_connectors)
        st.write(
            """
La "LMS" correspond à la Longueur Moyenne des Segments d'un texte, délimités ici par un
point (ou !, ?), ou par un retour à la ligne. Hypothèse :
- Des segments courts signalent un texte "haché", saccadé, algorithmique.
- Des segments longs évoquent une prose fluide, narrative ou explicative.
            """
        )
        segment_lengths = compute_segment_word_lengths(combined_text, filtered_connectors)

        if not segment_lengths:
            st.info(
                "Impossible de calculer la LMS : aucun segment n'a été détecté (ponctuation/retours à la ligne)."
            )
        else:
            st.subheader("Sélection des variables/modalités")
            hash_variables = [column for column in filtered_df.columns if column not in ("texte", "entete")]
            default_hash_index = 0 if not hash_variables else 1
            hash_variable_choice = st.selectbox(
                "Variable à filtrer pour la LMS",
                ["(Aucune)"] + hash_variables,
                index=default_hash_index,
                help="Restreindre le calcul de la LMS à certaines modalités.",
            )

            hash_modalities: List[str] = []

            if hash_variable_choice != "(Aucune)":
                modality_options = sorted(
                    filtered_df[hash_variable_choice].dropna().unique().tolist()
                )
                hash_modalities = st.multiselect(
                    "Modalités à inclure",
                    modality_options,
                    default=modality_options,
                    help="Choisissez les modalités dont les textes seront pris en compte.",
                )

            hash_filtered_df = filter_dataframe_by_modalities(
                filtered_df,
                None if hash_variable_choice == "(Aucune)" else hash_variable_choice,
                hash_modalities or None,
            )

            hash_text = build_text_from_dataframe(hash_filtered_df)
            segment_lengths = compute_segment_word_lengths(hash_text, filtered_connectors)

            if not hash_text or not segment_lengths:
                st.info(
                    "Impossible de calculer la LMS : aucun segment n'a été détecté entre connecteurs."
                )
            else:
                segment_entries = segments_with_word_lengths(hash_text, filtered_connectors)
                segment_lengths = [entry["longueur"] for entry in segment_entries]
                average_length = average_segment_length(hash_text, filtered_connectors)
                _, std_dev = compute_length_standard_deviation(hash_text, filtered_connectors)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Segments comptabilisés", str(len(segment_lengths)))
                col2.metric("LMS (mots)", f"{average_length:.2f}")
                col3.metric("Écart-type (mots)", f"{std_dev:.2f}")
                col4.metric("Segments min / max", f"{min(segment_lengths)} / {max(segment_lengths)}")

                distribution_df = pd.DataFrame(segment_entries)[
                    [
                        "segment_avec_marqueurs",
                        "connecteur_precedent",
                        "connecteur_suivant",
                        "longueur",
                    ]
                ]

                chart = (
                    alt.Chart(distribution_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("longueur:Q", bin=True, title="Longueur des segments (mots)"),
                        y=alt.Y("count()", title="Nombre de segments"),
                        tooltip=["count()", "longueur"]
                    )
                )

                st.altair_chart(chart, use_container_width=True)
                st.dataframe(
                    distribution_df.rename(
                        columns={
                            "segment_avec_marqueurs": "Segment",
                            "longueur": "Longueur",
                            "connecteur_precedent": "Connecteur précédent",
                            "connecteur_suivant": "Connecteur suivant",
                        }
                    ),
                    use_container_width=True,
                )

                per_modality_hash_df = average_segment_length_by_modality(
                    hash_filtered_df,
                    None if hash_variable_choice == "(Aucune)" else hash_variable_choice,
                    filtered_connectors,
                    hash_modalities or None,
                )

                if not per_modality_hash_df.empty:
                    st.subheader("LMS par modalité sélectionnée")
                    st.dataframe(
                        per_modality_hash_df.rename(
                            columns={
                                "modalite": "Modalité",
                                "segments": "Segments comptés",
                                "lms": "LMS",
                                "min": "Min",
                                "max": "Max",
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
                            tooltip=["modalite", "lms", "segments", "min", "max"],
                        )
                    )

                    st.altair_chart(lms_chart, use_container_width=True)

                std_by_modality_df = standard_deviation_by_modality(
                    hash_filtered_df,
                    None if hash_variable_choice == "(Aucune)" else hash_variable_choice,
                    filtered_connectors,
                    hash_modalities or None,
                )

                if not std_by_modality_df.empty:
                    st.subheader("Ecart-type")
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
                            tooltip=["modalite", "ecart_type", "segments", "lms"],
                        )
                    )

                    st.altair_chart(std_chart, use_container_width=True)

                    st.markdown("#### Dispersion des longueurs (moyenne ± écart-type)")

                    dispersion_chart = (
                        alt.Chart(
                            std_by_modality_df.assign(
                                borne_inferieure=lambda df: (df["lms"] - df["ecart_type"]).clip(lower=0),
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

                    st.altair_chart(dispersion_chart + lms_points, use_container_width=True)

    with tabs[7]:
        st.subheader("Regex motifs")

        st.markdown(
            """
            Dans cet onglet, les motifs regex repèrent des structures combinées "programmation"
            (ex : si…alors, si…sinon) dans les segments. La recherche est bornée par la ponctuation
            du texte (. ! ? ; : ou retour ligne) garantissant que les connecteurs sont détectés dans
            une unité lexicale (la phrase).
            """
        )

        regex_rules_path = Path(__file__).parent / "dictionnaires" / "motifs_progr_regex.json"
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
        )

        segments = split_segments(combined_text)
        segment_rows = summarize_matches_by_segment(segments, regex_patterns)

        st.markdown("---")
        st.subheader("Segments contenant au moins un motif")

        if not segment_rows:
            st.info("Aucun motif regex détecté dans le corpus fourni.")
        else:
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

    with tabs[8]:
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
                f"Modalités pour {variable}",
                modality_options,
                default=modality_options,
                help="Sélectionnez les modalités à inclure dans le calcul du score.",
                key=f"readability_modalities_{variable}",
            )
            readability_filtered_df = readability_filtered_df[
                readability_filtered_df[variable].isin(selected_modalities)
            ]

        readability_text = build_text_from_dataframe(readability_filtered_df)

        if not readability_text:
            st.info("Aucun texte disponible pour les variables/modalités sélectionnées.")
            return

        st.markdown(
            """
            Cet onglet calcule automatiquement le score de lisibilité Flesch-Kincaid
            sur le texte filtré selon les variables et modalités sélectionnées ci-dessus.
            Le score combine le nombre de phrases, de mots et de syllabes pour fournir
            un indicateur de difficulté de lecture.
            """
        )

        readability_metrics = compute_flesch_kincaid_metrics(readability_text)

        if readability_metrics["words"] == 0:
            st.info("Aucun mot n'a été détecté dans le texte à analyser.")
            return

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Phrases", f"{readability_metrics['sentences']:,}".replace(",", " "))
        col_b.metric("Mots", f"{readability_metrics['words']:,}".replace(",", " "))
        col_c.metric(
            "Syllabes estimées",
            f"{readability_metrics['syllables']:,}".replace(",", " "),
        )

        st.markdown("---")

        ease_score = readability_metrics["reading_ease"]
        grade_level = readability_metrics["grade_level"]

        col1, col2 = st.columns(2)
        col1.metric("Score Flesch-Kincaid", f"{ease_score:.1f}")
        col2.metric("Niveau scolaire estimé", f"{grade_level:.1f}")

        band = get_readability_band(ease_score)
        st.success(interpret_reading_ease(ease_score))

        scale_df = pd.DataFrame(READABILITY_SCALE)
        scale_order = scale_df.sort_values("min", ascending=False)["niveau"].tolist()
        score_df = pd.DataFrame({"score": [ease_score]})

        st.markdown("#### Position sur l'échelle Flesch-Kincaid")

        scale_chart = (
            alt.Chart(scale_df)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("min:Q", title="Score Flesch-Kincaid (0-100)"),
                x2="max:Q",
                y=alt.Y(
                    "niveau:N",
                    title="Niveau scolaire estimé",
                    sort=scale_order,
                ),
                color=alt.Color("niveau:N", legend=None),
                tooltip=["range:N", "niveau:N", "description:N"],
            )
            .properties(title="Échelle générale du test de lisibilité")
        )

        score_rule = (
            alt.Chart(score_df)
            .mark_rule(color="black", strokeDash=[6, 4], size=2)
            .encode(x="score:Q")
        )

        score_label = (
            alt.Chart(score_df)
            .mark_text(dx=6, dy=-6, fontWeight="bold", color="black")
            .encode(x="score:Q", y=alt.value(0), text=alt.Text("score:Q", format=".1f"))
        )

        st.altair_chart(scale_chart + score_rule + score_label, use_container_width=True)

        st.caption(
            f"Votre texte se situe dans la plage « {band['range']} » correspondant au niveau "
            f"{band['niveau']}."
        )

        st.markdown("#### Tableau de référence des niveaux de lisibilité")
        st.table(
            scale_df.sort_values("min", ascending=False)[["range", "niveau", "description"]]
            .rename(
                columns={
                    "range": "Score",
                    "niveau": "Niveau scolaire",
                    "description": "Interprétation",
                }
            )
        )

        readability_chart_rows = []

        for variable in readability_selected_variables:
            if variable not in readability_filtered_df.columns:
                continue

            variable_subset = readability_filtered_df.dropna(subset=[variable])

            for modality, subset in variable_subset.groupby(variable):
                modality_text = build_text_from_dataframe(subset)

                if not modality_text:
                    continue

                metrics = compute_flesch_kincaid_metrics(modality_text)

                if metrics["words"] == 0:
                    continue

                readability_chart_rows.append(
                    {
                        "variable": variable,
                        "modalite": modality,
                        "reading_ease": metrics["reading_ease"],
                        "sentences": metrics["sentences"],
                        "words": metrics["words"],
                    }
                )

        if readability_chart_rows:
            readability_scores_df = pd.DataFrame(readability_chart_rows)

            st.markdown("#### Comparaison par variable et modalité")
            st.dataframe(
                readability_scores_df.rename(
                    columns={
                        "variable": "Variable",
                        "modalite": "Modalité",
                        "reading_ease": "Score Flesch-Kincaid",
                        "sentences": "Phrases",
                        "words": "Mots",
                    }
                ),
                use_container_width=True,
            )

            readability_chart = (
                alt.Chart(readability_scores_df)
                .mark_bar()
                .encode(
                    x=alt.X("modalite:N", title="Modalité"),
                    y=alt.Y("reading_ease:Q", title="Score Flesch-Kincaid"),
                    color=alt.Color("variable:N", title="Variable"),
                    column=alt.Column("variable:N", title="Variable"),
                    tooltip=[
                        alt.Tooltip("variable:N", title="Variable"),
                        alt.Tooltip("modalite:N", title="Modalité"),
                        alt.Tooltip("reading_ease:Q", title="Score Flesch-Kincaid", format=".1f"),
                        alt.Tooltip("words:Q", title="Mots"),
                        alt.Tooltip("sentences:Q", title="Phrases"),
                    ],
                )
                .properties(title="Comparaison de lisibilité par modalité", spacing=12)
            )

            st.altair_chart(readability_chart, use_container_width=True)
        else:
            st.info(
                "Aucune donnée disponible pour comparer les variables/modalités sélectionnées."
            )

        st.caption(
            "La formule originale (206.835 − 1.015 × mots/phrases − 84.6 × syllabes/mot) "
            "a été conservée pour ce calcul. Les syllabes sont estimées par comptage des "
            "groupes de voyelles ; les résultats restent indicatifs pour le français."
        )


if __name__ == "__main__":
    main()
