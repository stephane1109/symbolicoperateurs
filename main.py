from __future__ import annotations

import sys
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

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
from fcts_utils import (
    build_annotation_style_block,
    build_dataframe,
    build_variable_stats,
    display_centered_chart,
    parse_iramuteq,
    render_connectors_reminder,
)
from connecteurs import (
    get_connectors_path,
    get_selected_connectors,
    get_selected_labels,
    load_available_connectors,
    set_selected_connectors,
)
from lexiconnorm import render_lexicon_norm_tab
from ngram import build_ngram_pattern, compute_ngram_statistics
from densite import (
    compute_density,
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
    SegmentationMode,
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
from simicosinus import (
    aggregate_texts_by_variable,
    compute_cosine_similarity_by_variable,
    get_french_stopwords,
)
from tf_idf import render_tfidf_tab
from graphiques.densitegraph import (
    build_connector_density_chart,
    build_density_chart,
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
    st.write("Pour des raisons d’interopérabilité entre applications, le corpus doit être formaté selon les exigences"
             "d’IRaMuTeQ : chaque texte commence par une ligne d’en-tête du type **** *variable_modalité.\n\n"
             "Pour le moment, l’application fonctionne avec un fichier dictionnaire.json, que vous pouvez consulter dans l’onglet « Connecteurs », "
             "ainsi qu’avec des règles regex. \n"
             "À l’avenir, une réflexion sera menée pour y associer une bibliothèque NLP (comme spaCy et/ou BERT), "
             "ce qui rendrait l’approche moins rigide que des règles regex. Toutefois, je suis en partie limité par le fait que l’application "
             "est hébergée sur Streamlit Cloud (gratuit), avec des ressources restreintes.\n\n" 
             "les stopwords sont toutefois filtrés avec la librairie NLP NLTK, la plus légère."
            )
    st.caption("[www.codeandcortex.fr](https://www.codeandcortex.fr)")
    st.markdown("---")
    st.write(
        "Téléversez un fichier texte IRaMuTeQ. Chaque article doit démarrer par "
        "une ligne de variables, par exemple `**** *model_gpt *prompt_1`."
    )

    uploaded_file = st.file_uploader("Fichier IRaMuTeQ", type=["txt"])  # type: ignore[assignment]

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
            "N-gram",
            "TF-IDF",
            "Simi cosinus",
        ]
    )

    if not uploaded_file:
        upload_message = (
            "Téléversez un fichier texte IRaMuTeQ pour accéder aux analyses disponibles dans les onglets."
        )

        with tabs[0]:
            st.subheader("Données importées")
            st.info(upload_message)

        for tab in tabs[1:]:
            with tab:
                st.info(upload_message)

        return

    content = uploaded_file.read().decode("utf-8")
    records = parse_iramuteq(content)

    if not records:
        st.warning("Aucune entrée valide trouvée dans le fichier fourni.")
        return

    df = build_dataframe(records)

    with tabs[1]:
        st.subheader("Choisir les connecteurs à analyser")
        st.markdown(
        "Dans cet onglet, vous devez sélectionner les connecteurs logiques qui auront un impact sur les analyses.\n\n"
        "Données brutes, Sous-corpus, Densité, Norme Lexicon et Hash.\n\n"
        "Vous pouvez à tout moment relancer vos analyses en sélectionnant ou en supprimant des connecteurs dans la section Connecteurs."
                    )

        render_connectors_reminder(get_selected_connectors())
        connectors_path = get_connectors_path()

        with st.expander("Afficher le contenu de connecteurs.json"):
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
            label="Télécharger le texte annoté",
            data=downloadable_html,
            file_name="texte_brut_connecteurs.html",
            mime="text/html",
        )

        st.markdown(annotation_style, unsafe_allow_html=True)
        st.markdown(
            f"<div class='annotated-container'>{annotated_html}</div>",
            unsafe_allow_html=True,
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
            
            st.subheader("Fréquences des connecteurs")
            
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
            "téléchargé au format texte pour être réutilisé pour d'autres analyses."
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
            "Densité des textes analysés : La densité correspond au nombre de connecteurs "
            "ramené à une base de 1 000 mots. "
        )
        if not filtered_connectors:
            st.info("Sélectionnez au moins un connecteur pour calculer la densité.")
        else:
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
                modality_options = sorted(
                    df[density_variable_choice].dropna().unique().tolist()
                )
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
                    "Un score élevé signale un texte plus riche en connecteurs logiques."
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

    with tabs[5]:
        render_connectors_reminder(filtered_connectors)
        render_lexicon_norm_tab(df, filtered_connectors)

    with tabs[6]:
        st.subheader("Hash (LMS entre connecteurs)")
        render_connectors_reminder(filtered_connectors)
        st.write(
            """
La "LMS" correspond à la Longueur Moyenne des Segments d'un texte. Vous pouvez choisir
un découpage basé uniquement sur les connecteurs sélectionnés, ou bien considérer qu'une
ponctuation forte (., ?, !, ;, :) ferme aussi le segment. Hypothèse :
- Des segments courts signalent un texte "haché", saccadé, algorithmique.
- Des segments longs évoquent une prose fluide, narrative ou explicative.
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

        segment_lengths = compute_segment_word_lengths(
            combined_text, filtered_connectors, segmentation_mode
        )

        if not segment_lengths:
            st.info(
                "Impossible de calculer la LMS : aucun segment n'a été détecté entre connecteurs."
            )
        else:
            st.subheader("Sélection des variables/modalités")
            hash_variables = [
                column for column in filtered_df.columns if column not in ("texte", "entete")
            ]

            selected_hash_variables = st.multiselect(
                "Variables à filtrer pour la LMS",
                hash_variables,
                default=hash_variables,
                help=(
                    "Choisissez les variables à filtrer. Laisser vide pour utiliser l'ensemble"
                    " du corpus actuellement chargé."
                ),
            )

            if not selected_hash_variables:
                st.info(
                    "Aucune variable sélectionnée : la LMS sera calculée sur tous les textes "
                    "affichés dans l'onglet « Données brutes »"
                )

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

            hash_text = build_text_from_dataframe(hash_filtered_df)
            segment_lengths = compute_segment_word_lengths(
                hash_text, filtered_connectors, segmentation_mode
            )

            if not hash_text or not segment_lengths:
                st.info(
                    "Impossible de calculer la LMS : aucun segment n'a été détecté entre connecteurs."
                )
            else:
                segment_entries = segments_with_word_lengths(
                    hash_text, filtered_connectors, segmentation_mode
                )
                segment_lengths = [entry["longueur"] for entry in segment_entries]
                average_length = average_segment_length(
                    hash_text, filtered_connectors, segmentation_mode
                )
                _, std_dev = compute_length_standard_deviation(
                    hash_text, filtered_connectors, segmentation_mode
                )

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

                for variable, selected_modalities in hash_modality_filters.items():
                    per_modality_hash_df = average_segment_length_by_modality(
                        hash_filtered_df,
                        variable,
                        filtered_connectors,
                        selected_modalities or None,
                        segmentation_mode,
                    )

                    if not per_modality_hash_df.empty:
                        st.subheader(f"LMS par modalité sélectionnée ({variable})")
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
                    )

                    if not std_by_modality_df.empty:
                        st.subheader(f"Ecart-type ({variable})")
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

    with tabs[7]:
        st.subheader("Regex motifs")

        st.markdown(
            """
            Dans cet onglet, les motifs regex repèrent des structures combinées
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
            "a été conservée pour ce calcul."
        )

    with tabs[9]:
        st.subheader("N-gram (3 à 6 mots)")
        st.markdown(
            """
            - Un n-gramme est une séquence contiguë de *n* unités (mots) dans un texte.
            - Ils capturent l'ordre et la continuité : un trigramme (n=3) reflète par exemple trois mots successifs.
            """
        )
        st.write(
            "Extraction des N-grams les plus fréquents sur l'intégralité du texte, "
            "avec la répartition des occurrences par variables/modalités lorsqu'elles "
            "sont présentes."
        )

        if df.empty:
            st.info("Aucun texte disponible pour extraire des N-grams.")
            return

        with st.expander("Options d'affichage et filtres"):
            col_a, col_b, col_c = st.columns(3)
            min_frequency = col_a.number_input(
                "Fréquence minimale",
                min_value=1,
                value=1,
                step=1,
                help="Exclut les N-grams dont la fréquence est inférieure au seuil.",
            )

            sort_choice = col_b.selectbox(
                "Tri",
                ["Fréquence décroissante", "Ordre alphabétique"],
                help="Choisissez l'ordre d'affichage des N-grams.",
            )

            exclude_stopwords = col_c.checkbox(
                "Exclure les stopwords français (NLTK)",
                help="Supprimer les stopwords avant le calcul des N-grams.",
            )

            search_pattern = st.text_input(
                "Rechercher un motif dans les N-grams",
                help="Surligner ou filtrer les N-grams qui contiennent le motif (expression régulière acceptée).",
            )

            hide_non_matches = st.checkbox(
                "Masquer les N-grams qui ne correspondent pas au motif",
                help="Lorsque le motif est renseigné, seuls les N-grams correspondants sont affichés.",
            )

        stop_words = get_french_stopwords() if exclude_stopwords else None

        results_by_size = {
            size: compute_ngram_statistics(
                df,
                min_n=size,
                max_n=size,
                top_k=10,
                min_frequency=min_frequency,
                exclude_stopwords=exclude_stopwords,
                stop_words=stop_words,
                sort_by="alphabetical"
                if sort_choice == "Ordre alphabétique"
                else "frequency",
            )
            for size in range(3, 7)
        }

        if all(result.empty for result in results_by_size.values()):
            st.info("Aucun N-gram n'a pu être calculé à partir du texte fourni.")
        else:
            def build_ngram_download_html(results: dict[int, pd.DataFrame]) -> str:
                annotation_style = build_annotation_style_block("")

                def _format_context_block(context_entry: dict[str, object], ngram_value: str) -> str:
                    raw_text = str(
                        context_entry.get("texte_complet")
                        or context_entry.get("contexte")
                        or ""
                    )

                    if not raw_text.strip():
                        return ""

                    pattern = build_ngram_pattern(ngram_value.split())
                    highlighted = pattern.sub(
                        lambda match: (
                            "<span class=\"connector-annotation\">"
                            f"<span class=\"connector-text\">{match.group(0)}</span>"
                            "</span>"
                        ),
                        raw_text,
                    )

                    header_parts: list[str] = []
                    entete = str(context_entry.get("entete", "") or "").strip()
                    if entete:
                        header_parts.append(entete)

                    modalities = context_entry.get("modalites", []) or []
                    if modalities:
                        header_parts.append(
                            ", ".join(str(modality) for modality in modalities)
                        )

                    header_text = " • ".join(header_parts) or "Texte"

                    return "\n".join(
                        [
                            "<div class=\"context-block\">",
                            f"<div class=\"context-header\">{header_text}</div>",
                            f"<div class=\"context-body\">{highlighted}</div>",
                            "</div>",
                        ]
                    )

                sections: list[str] = [
                    "<!DOCTYPE html>",
                    "<html lang=\"fr\">",
                    "<head>",
                    "<meta charset=\"utf-8\" />",
                    annotation_style,
                    "<style>",
                    "body { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; padding: 24px; background: #f8fafc; color: #111827; }",
                    "h1, h2 { color: #0f172a; }",
                    ".ngram-section { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 20px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06); }",
                    ".ngram-entry { margin: 12px 0; padding: 12px 14px; border-radius: 10px; background: #f9fafb; border: 1px solid #e5e7eb; }",
                    ".ngram-title { font-size: 17px; font-weight: 700; color: #0ea5e9; margin-bottom: 6px; }",
                    ".ngram-frequency { color: #475569; font-size: 14px; margin-bottom: 8px; }",
                    ".context-block { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 10px; padding: 10px 12px; margin: 10px 0; }",
                    ".context-header { font-weight: 700; color: #312e81; margin-bottom: 6px; }",
                    ".context-body { line-height: 1.6; }",
                    "</style>",
                    "</head>",
                    "<body>",
                    "<h1>Occurrences des N-grams</h1>",
                ]

                for size in range(3, 7):
                    ngram_df = results.get(size)
                    if ngram_df is None or ngram_df.empty:
                        continue

                    sections.append(
                        f"<div class=\"ngram-section\"><h2>N-grams de {size} mots</h2>"
                    )

                    for _, row in ngram_df.iterrows():
                        ngram_value = row.get("N-gram", "")
                        frequency_value = row.get("Fréquence", 0)

                        sections.append(
                            "\n".join(
                                [
                                    "<div class=\"ngram-entry\">",
                                    f"<div class=\"ngram-title\">{ngram_value}</div>",
                                    f"<div class=\"ngram-frequency\">{frequency_value} occurrence(s)</div>",
                                ]
                            )
                        )

                        detailed_contexts = row.get("Occurrences détaillées") or []

                        if not detailed_contexts and "Contexte" in row:
                            context_text = row.get("Contexte", "")
                            if context_text:
                                detailed_contexts = [
                                    {
                                        "contexte": context_text,
                                        "modalites": [],
                                        "entete": "",
                                        "texte_complet": context_text,
                                    }
                                ]

                        if not detailed_contexts:
                            sections.append("<p>Aucun contexte disponible.</p></div>")
                            continue

                        for context_entry in detailed_contexts:
                            block_html = _format_context_block(context_entry, ngram_value)
                            if block_html:
                                sections.append(block_html)

                        sections.append("</div>")

                    sections.append("</div>")

                sections.extend(["</body>", "</html>"])
                return "\n".join(sections)

            downloadable_ngram_html = build_ngram_download_html(results_by_size)

            st.download_button(
                label="Tout télécharger", 
                data=downloadable_ngram_html,
                file_name="ngrams.html",
                mime="text/html",
                help="Télécharger tous les N-grams et leurs contextes au format HTML.",
            )

            for size in range(3, 7):
                st.markdown(f"### N-grams de {size} mots")
                ngram_results = results_by_size[size]

                if ngram_results.empty:
                    st.info(
                        "Aucun N-gram n'a été trouvé pour cette taille avec les filtres actuels."
                    )
                    continue

                display_df = ngram_results.copy()

                full_context = display_df.get("Contexte", pd.Series(dtype=str))
                display_df["Contexte (aperçu)"] = full_context.fillna("").apply(
                    lambda value: value if len(value) <= 140 else value[:140].rstrip() + "…"
                )
                display_df = display_df.fillna("")

                st.caption(
                    "Le champ « Modalités associées » affiche combien de fois le N-gram "
                    "apparaît pour chaque modalité (ex. « model=gpt (27) » signifie 27 occurrences "
                    "dans les textes où model=gpt)."
                )

                if search_pattern.strip():
                    try:
                        match_mask = display_df["N-gram"].str.contains(
                            search_pattern, case=False, regex=True
                        )
                    except re.error:
                        match_mask = display_df["N-gram"].str.contains(
                            re.escape(search_pattern), case=False, regex=True
                        )

                    if hide_non_matches:
                        display_df = display_df[match_mask].copy()
                        match_mask = match_mask.reindex(display_df.index).fillna(False)
                else:
                    match_mask = pd.Series(False, index=display_df.index)

                if display_df.empty:
                    st.info(
                        "Aucun N-gram ne correspond au motif recherché pour cette taille."
                    )
                    continue

                if "Contexte" in display_df.columns:
                    display_df = display_df.drop(columns=["Contexte"])
                if "Occurrences détaillées" in display_df.columns:
                    display_df = display_df.drop(columns=["Occurrences détaillées"])

                if search_pattern.strip():
                    display_df.insert(0, "Correspond au motif", match_mask.values)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                context_map = (
                    ngram_results.set_index("N-gram")["Contexte"].to_dict()
                    if "Contexte" in ngram_results.columns
                    else {}
                )
                context_details_map = (
                    ngram_results.set_index("N-gram")["Occurrences détaillées"].to_dict()
                    if "Occurrences détaillées" in ngram_results.columns
                    else {}
                )
                frequency_map = (
                    ngram_results.set_index("N-gram")["Fréquence"].to_dict()
                    if "Fréquence" in ngram_results.columns
                    else {}
                )

                if context_map:
                    st.markdown("#### Contextes des N-grams")
                    st.markdown(
                        build_annotation_style_block(""),
                        unsafe_allow_html=True,
                    )

                    def _highlight_ngram(context_text: str, ngram_value: str) -> str:
                        if not context_text:
                            return ""

                        pattern = build_ngram_pattern(ngram_value.split())
                        return pattern.sub(
                            lambda match: (
                                "<span class=\"connector-annotation\">"
                                f"<span class=\"connector-text\">{match.group(0)}</span>"
                                "</span>"
                            ),
                            context_text,
                        )

                    for _, row in display_df.iterrows():
                        ngram_value = row.get("N-gram", "")
                        detailed_contexts = context_details_map.get(ngram_value, []) or []

                        if not detailed_contexts and ngram_value in context_map:
                            detailed_contexts = [
                                {
                                    "contexte": context_map.get(ngram_value, ""),
                                    "modalites": [],
                                    "entete": "",
                                    "texte_complet": context_map.get(ngram_value, ""),
                                }
                            ]

                        if not detailed_contexts:
                            continue

                        occurrence_total = frequency_map.get(
                            ngram_value, len(detailed_contexts)
                        )

                        expander_label = f"{ngram_value} – {occurrence_total} occurrence(s)"
                        with st.expander(expander_label):
                            for index, context_entry in enumerate(
                                detailed_contexts, start=1
                            ):
                                context_text = context_entry.get("contexte", "")
                                highlighted_context = _highlight_ngram(
                                    context_text, ngram_value
                                )
                                header_parts: list[str] = []

                                entete = (
                                    str(context_entry.get("entete", "") or "").strip()
                                )
                                if entete:
                                    header_parts.append(entete)

                                modalities = context_entry.get("modalites", []) or []
                                if modalities:
                                    header_parts.append(
                                        ", ".join(str(modality) for modality in modalities)
                                    )

                                header_text = " • ".join(header_parts) or "Texte"

                                st.markdown(f"**Occurrence {index} – {header_text}**")
                                st.markdown(highlighted_context, unsafe_allow_html=True)

                                full_text = str(context_entry.get("texte_complet", "") or "")
                                if full_text and full_text.strip() != context_text.strip():
                                    st.markdown("**Texte complet**")
                                    st.markdown(
                                        _highlight_ngram(full_text, ngram_value),
                                        unsafe_allow_html=True,
                                    )

    with tabs[10]:
        render_tfidf_tab(df)

    with tabs[11]:
        st.subheader("Similarité cosinus")
        st.write(
            "Comparer la similarité cosinus entre les variables en "
            "concaténant l'intégralité des textes par modalité."
        )

        model_variables = [column for column in df.columns if column not in ("texte", "entete")]

        if not model_variables:
            st.info("Aucune variable n'a été trouvée dans le fichier importé.")
            return

        model_variable_choice = st.selectbox(
            "Variable à comparer",
            model_variables,
            help="Les textes seront regroupés par modalité de cette variable avant le calcul TF-IDF.",
        )

        modality_options = sorted(df[model_variable_choice].dropna().unique().tolist())

        if not modality_options:
            st.info("Aucune modalité disponible pour la variable sélectionnée.")
            return

        selected_modalities = st.multiselect(
            "Modalités à inclure",
            modality_options,
            default=modality_options,
            help="Choisissez les modalités.",
        )

        if not selected_modalities:
            st.info("Sélectionnez au moins une modalité pour lancer le calcul.")
            return

        cosine_df = df[df[model_variable_choice].isin(selected_modalities)]

        apply_stopwords = st.checkbox(
            "Appliquer les stopwords français (NLTK) avant le calcul",
            value=False,
            help=(
                "Supprime les mots vides français fournis par NLTK avant de construire"
                " la matrice TF-IDF."
            ),
        )

        aggregated_texts = aggregate_texts_by_variable(cosine_df, model_variable_choice)

        if len(aggregated_texts) < 2:
            st.info(
                "Au moins deux modalités doivent contenir du texte pour calculer la similarité cosinus."
            )
            return

        texts_summary = pd.DataFrame(
            {
                "Modalité": list(aggregated_texts.keys()),
                "Mots": [len(text.split()) for text in aggregated_texts.values()],
            }
        ).sort_values("Modalité")

        st.markdown("### Textes regroupés")
        st.dataframe(texts_summary, use_container_width=True)

        similarity_df = compute_cosine_similarity_by_variable(
            cosine_df, model_variable_choice, use_stopwords=apply_stopwords
        )

        if similarity_df.empty:
            st.info("Impossible de calculer la matrice de similarité cosinus avec les données fournies.")
            return

        st.markdown("### Matrice de similarité cosinus")
        st.dataframe(similarity_df.style.format("{:.4f}"), use_container_width=True)

        similarity_long = (
            similarity_df.reset_index()
            .rename(columns={"index": "Modalité"})
            .melt(id_vars="Modalité", var_name="Comparée à", value_name="Similarité")
        )

        modalities_order = similarity_df.index.tolist()

        heatmap = (
            alt.Chart(similarity_long)
            .mark_rect()
            .encode(
                x=alt.X("Modalité:N", sort=modalities_order),
                y=alt.Y("Comparée à:N", sort=modalities_order),
                color=alt.Color(
                    "Similarité:Q",
                    scale=alt.Scale(
                        domain=[0, 0.5, 1],
                        range=["#f7fbff", "#4292c6", "#08306b"],
                    ),
                    title="Cosinus",
                ),
                tooltip=["Modalité", "Comparée à", alt.Tooltip("Similarité:Q", format=".4f")],
            )
            .properties(
                title="Carte de chaleur des similarités",
                width=alt.Step(80),
                height=alt.Step(80),
            )
        )

        display_centered_chart(heatmap)


if __name__ == "__main__":
    main()
