"""Logique des onglets Streamlit."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from analyses import (
    annotate_connectors_html,
    build_label_style_block,
    count_connectors,
    generate_label_colors,
)
from connecteurs import (
    get_connectors_path,
    get_selected_connectors,
    get_selected_labels,
    load_available_connectors,
    set_selected_connectors,
)
from densite import (
    build_text_from_dataframe,
    compute_density,
    compute_density_per_modality,
    compute_density_per_modality_by_label,
    compute_total_connectors,
    count_words,
    filter_dataframe_by_modalities,
)
from ecartype import compute_length_standard_deviation, standard_deviation_by_modality
from fcts_utils import (
    build_annotation_style_block,
    build_dataframe,
    build_variable_stats,
    display_centered_chart,
    parse_iramuteq,
    render_connectors_reminder,
)
from graphiques.densitegraph import build_connector_density_chart, build_density_chart
from hash import (
    ECART_TYPE_EXPLANATION,
    SegmentationMode,
    average_segment_length,
    average_segment_length_by_modality,
    compute_segment_word_lengths,
    segments_with_word_lengths,
)
from lexiconnorm import render_lexicon_norm_tab
from ngram import build_ngram_pattern, compute_ngram_statistics
from pattern import annotate_user_pattern_html, find_pattern_segments
from regexanalyse import (
    count_segments_by_pattern,
    highlight_matches_html,
    load_regex_rules,
    split_segments,
    summarize_matches_by_segment,
)
from simicosinus import (
    aggregate_texts_by_variables,
    concatenate_texts_with_headers,
    compute_cosine_similarity_matrix,
    get_french_stopwords,
)
from souscorpus import build_subcorpus
from test_lesch_Kincaid import (
    READABILITY_SCALE,
    compute_flesch_kincaid_metrics,
    get_readability_band,
    interpret_reading_ease,
)
from tf_idf import render_tfidf_tab

# Dossier racine de l'application (un niveau au-dessus du répertoire des onglets)
BASE_DIR = Path(__file__).resolve().parent.parent


def parse_upload(content: str) -> Tuple[List[dict], pd.DataFrame]:
    records = parse_iramuteq(content)
    return records, build_dataframe(records)


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


def rendu_donnees_importees(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    st.subheader("Données importées")
    render_connectors_reminder(filtered_connectors)
    st.dataframe(df, use_container_width=True)


def rendu_donnees_brutes(
    tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]
) -> Optional[Tuple[pd.DataFrame, List[str], str]]:
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
        return None

    if not filtered_connectors:
        st.info("Choisissez des connecteurs dans l'onglet « Connecteurs » pour poursuivre.")
        return None

    selected_labels = get_selected_labels(filtered_connectors.values())

    label_colors = generate_label_colors(filtered_connectors.values())
    label_style_block = build_label_style_block(label_colors)
    annotated_html = annotate_connectors_html(combined_text, filtered_connectors)

    st.markdown(label_style_block, unsafe_allow_html=True)
    st.subheader("Connecteurs annotés")
    st.markdown(
        f"<div class='annotated-container'>{annotated_html}</div>",
        unsafe_allow_html=True,
    )
    downloadable_html = f"""<!DOCTYPE html>
    <html lang=\"fr\">
    <head>
    <meta charset=\"utf-8\" />
    {label_style_block}
    </head>
    <body>
    <div class='annotated-container'>{annotated_html}</div>
    </body>
    </html>"""

    st.download_button(
        label="Télécharger le texte annoté (HTML)",
        data=downloadable_html,
        file_name="texte_annote_connecteurs.html",
        mime="text/html",
    )

    st.markdown("---")
    st.subheader("Statistiques des connecteurs")

    stats_df = count_connectors(combined_text, filtered_connectors)

    if stats_df.empty:
        st.info("Aucun connecteur trouvé dans le texte sélectionné.")
    else:
        stats_df = (
            stats_df.sort_values("occurrences", ascending=False)
            .reset_index(drop=True)
        )

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

    return filtered_df, selected_variables, combined_text


def rendu_sous_corpus(
    tab, records: List[dict], filtered_connectors: Dict[str, str]
) -> None:
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
        return

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


def rendu_openlexicon(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    render_connectors_reminder(filtered_connectors)
    render_lexicon_norm_tab(df, filtered_connectors)


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
ponctuation forte (., ?, !, ;, :) ferme aussi le segment.
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

    segment_lengths = compute_segment_word_lengths(
        combined_text, filtered_connectors, segmentation_mode
    )

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
    segment_lengths = compute_segment_word_lengths(
        hash_text, filtered_connectors, segmentation_mode
    )

    if not hash_text or not segment_lengths:
        st.info(
            "Impossible de calculer la LMS : aucun segment n'a été détecté entre connecteurs."
        )
        return

    st.download_button(
        label="Télécharger le texte",
        data=hash_text,
        file_name="texte_filtre_hash.txt",
        mime="text/plain",
        help=(
            "Récupérez le texte concaténé par variables/modalités sélectionnées, au même"
            " format que dans l'onglet 'Similarité cosinus'."
        ),
    )
    st.caption(
        "Utilisez ce bouton pour contrôler le contenu exact retenu après votre sélection"
        " de variables et modalités, avec l'entête IRaMuTeQ reconstruit."
    )

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
            tooltip=["count()", "longueur"],
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


def rendu_regex_motifs(tab, combined_text: str) -> None:
    st.subheader("Regex motifs")

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


def format_modalities_for_row(row: pd.Series, variables: List[str]) -> str:
    parts: List[str] = []

    for variable in variables:
        value = row.get(variable, "")

        if pd.isna(value) or value == "":
            continue

        parts.append(f"{variable} = {value}")

    header = str(row.get("entete", "")).strip()

    if parts:
        return " | ".join(parts)

    return header or "Non spécifié"


def rendu_patterns(
    tab,
    filtered_df: pd.DataFrame,
    combined_text: str,
    selected_variables: List[str],
) -> None:
    st.subheader("Patterns (motifs)")
    st.markdown(
        "Saisissez un motif (mot, expression ou signe tel que « ? ») pour identifier les segments qui le contiennent."
    )

    pattern_query = st.text_input(
        "Motif ou signe à rechercher", placeholder="?", key="simple_pattern_query"
    )

    if not pattern_query:
        return

    pattern_annotation_style = build_annotation_style_block("")
    annotated_pattern_html = annotate_user_pattern_html(combined_text, pattern_query)

    st.subheader("Texte annoté par motif")
    st.markdown(pattern_annotation_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='annotated-container'>{annotated_pattern_html}</div>",
        unsafe_allow_html=True,
    )

    annotated_download = f"""<!DOCTYPE html>
    <html lang=\"fr\">
    <head>
    <meta charset=\"utf-8\" />
    {pattern_annotation_style}
    </head>
    <body>
    <div class='annotated-container'>{annotated_pattern_html}</div>
    </body>
    </html>"""

    st.download_button(
        label="Télécharger le texte annoté par motif",
        data=annotated_download,
        file_name="texte_annotes_motif.html",
        mime="text/html",
    )

    enriched_segments: List[dict] = []
    segment_counter = 1

    for _, row in filtered_df.iterrows():
        row_text = build_text_from_dataframe(pd.DataFrame([row]))

        if not row_text:
            continue

        modalities_label = format_modalities_for_row(row, selected_variables)

        row_segments = find_pattern_segments(row_text, pattern_query)

        for segment in row_segments:
            enriched_segments.append(
                {
                    "modalites": modalities_label,
                    "segment_id": segment_counter,
                    "segment": segment.get("segment"),
                    "occurrences": segment.get("occurrences", 0),
                }
            )
            segment_counter += 1

    if not enriched_segments:
        st.info("Aucun segment ne contient ce motif dans le texte filtré.")
        return

    segments_df = pd.DataFrame(enriched_segments)[
        ["modalites", "segment_id", "segment", "occurrences"]
    ].rename(
        columns={
            "modalites": "Variables/modalités",
            "segment_id": "Segment",
            "segment": "Texte",
            "occurrences": "Occurrences",
        }
    )

    st.markdown("Segments contenant le motif")
    st.dataframe(segments_df, use_container_width=True)

    chart_df = segments_df.rename(
        columns={
            "Variables/modalités": "modalite",
            "Occurrences": "Occurrences",
        }
    )

    occurrences_by_modality = (
        chart_df.groupby("modalite", as_index=False)["Occurrences"].sum()
    )

    if occurrences_by_modality.empty:
        st.info("Aucune répartition par variables/modalités n'est disponible.")
        return

    chart = (
        alt.Chart(occurrences_by_modality)
        .mark_bar()
        .encode(
            x=alt.X("modalite:N", sort="-y", title="Variable / modalité"),
            y=alt.Y("Occurrences:Q", title="Occurrences du motif"),
            color=alt.Color("modalite:N", title="Variable / modalité"),
            tooltip=["modalite", "Occurrences"],
        )
        .properties(title="Occurrences du motif par variables/modalités")
    )

    st.altair_chart(chart, use_container_width=True)


def rendu_lisibilite(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
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
            f"Modalités à inclure pour {variable}",
            modality_options,
            default=modality_options,
            help="Filtrer les textes utilisés pour le test de lisibilité.",
        )
        if selected_modalities:
            readability_filtered_df = readability_filtered_df[
                readability_filtered_df[variable].isin(selected_modalities)
            ]
        else:
            readability_filtered_df = readability_filtered_df.iloc[0:0]

    if readability_filtered_df.empty:
        st.info("Aucun texte ne correspond aux filtres sélectionnés.")
        return

    readability_text = build_text_from_dataframe(readability_filtered_df)
    if not readability_text:
        st.info("Aucun texte valide à analyser pour la lisibilité.")
        return

    readability_metrics = compute_flesch_kincaid_metrics(readability_text)

    st.markdown("### Résultats de lisibilité")

    if readability_metrics is None:
        st.info("Impossible de calculer les scores de lisibilité pour le texte fourni.")
        return

    ease_score = readability_metrics.get("reading_ease", 0.0)
    grade_score = readability_metrics.get("grade_level", 0.0)

    col1, col2 = st.columns(2)
    col1.metric(
        "Flesch Reading Ease",
        f"{ease_score:.2f}",
        help="Indice mesurant la facilité de lecture (plus il est élevé, plus le texte est facile).",
    )
    col2.metric(
        "Flesch-Kincaid Grade",
        f"{grade_score:.2f}",
        help="Niveau scolaire approximatif nécessaire pour comprendre le texte.",
    )

    readability_band = get_readability_band(ease_score)
    readability_description = interpret_reading_ease(ease_score)

    st.markdown(
        f"**Interprétation** : {readability_description} (échelle : {READABILITY_SCALE.get(readability_band, '')})"
    )

    st.markdown("### Paramètres utilisés")
    st.write(
        {
            "Mots": readability_metrics.get("word_count", 0),
            "Phrases": readability_metrics.get("sentence_count", 0),
            "Syllabes": readability_metrics.get("syllable_count", 0),
        }
    )

    st.caption(
        "Les scores de lisibilité sont calculés sur la base du texte filtré, en utilisant les variables/modalités sélectionnées."
    )


def rendu_ngram(tab, filtered_df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    st.subheader("N-gram")

    ngram_variables = [column for column in filtered_df.columns if column not in ("texte", "entete")]
    selected_ngram_variables = st.multiselect(
        "Variables à filtrer pour les N-grams",
        ngram_variables,
        default=ngram_variables,
        help="Choisissez les variables à utiliser pour filtrer les N-grams.",
    )

    ngram_filtered_df = filtered_df.copy()

    for variable in selected_ngram_variables:
        modality_options = sorted(ngram_filtered_df[variable].dropna().unique().tolist())
        selected_modalities = st.multiselect(
            f"Modalités à inclure pour {variable}",
            modality_options,
            default=modality_options,
            help="Modalités retenues pour calculer les N-grams.",
        )

        if selected_modalities:
            ngram_filtered_df = ngram_filtered_df[
                ngram_filtered_df[variable].isin(selected_modalities)
            ]
        else:
            ngram_filtered_df = ngram_filtered_df.iloc[0:0]

    if ngram_filtered_df.empty:
        st.info("Aucun texte ne correspond aux filtres sélectionnés pour les N-grams.")
        return

    search_pattern = st.text_input(
        "Filtrer les N-grams par motif (regex)",
        help="Filtrer les N-grams affichés en fonction d'un motif regex (optionnel).",
    )
    hide_non_matches = st.checkbox(
        "Masquer les N-grams qui ne correspondent pas au motif",
        value=False,
    )

    results_by_size = compute_ngram_statistics(
        ngram_filtered_df,
        variables=selected_ngram_variables,
        search_pattern=search_pattern,
        hide_non_matches=hide_non_matches,
    )

    def _format_context_block(context_entry: dict, ngram_value: str) -> str:
        context_text = str(context_entry.get("contexte", "")).strip()
        if not context_text:
            return ""

        highlighted = _highlight_context(context_text, ngram_value)
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

    def _highlight_context(context_text: str, ngram_value: str) -> str:
        pattern = build_ngram_pattern(ngram_value.split())
        return pattern.sub(
            lambda match: (
                "<span class=\"connector-annotation\">"
                f"<span class=\"connector-text\">{match.group(0)}</span>"
                "</span>"
            ),
            escape(context_text),
        )

    def build_ngram_download_html(results: dict[int, pd.DataFrame]) -> str:
        annotation_style = build_annotation_style_block("")

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

        if search_pattern.strip():
            try:
                match_mask = display_df["N-gram"].str.contains(
                    search_pattern, case=False, regex=True
                )
            except Exception:
                match_mask = display_df["N-gram"].str.contains(
                    search_pattern, case=False, regex=True
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
                        highlighted_context = _highlight_context(
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
                        st.markdown(
                            f"<p><strong>Contexte {index} ({header_text})</strong></p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='annotated-container'>{highlighted_context}</div>",
                            unsafe_allow_html=True,
                        )


def rendu_tfidf(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    render_tfidf_tab(tab, df, filtered_connectors)


def rendu_simi_cosinus(tab, df: pd.DataFrame) -> None:
    st.subheader("Simi cosinus")

    selected_cosine_variables = [
        column for column in df.columns if column not in ("texte", "entete")
    ]

    cosine_filtered_df = df.copy()
    for variable in selected_cosine_variables:
        modality_options = sorted(
            cosine_filtered_df[variable].dropna().unique().tolist()
        )
        selected_modalities = st.multiselect(
            f"Modalités à inclure pour {variable}",
            modality_options,
            default=modality_options,
            help="Choisissez les modalités dont les textes seront pris en compte.",
        )

        if selected_modalities:
            cosine_filtered_df = cosine_filtered_df[
                cosine_filtered_df[variable].isin(selected_modalities)
            ]
        else:
            cosine_filtered_df = cosine_filtered_df.iloc[0:0]

    if cosine_filtered_df.empty:
        st.info(
            "Aucun texte ne correspond aux filtres appliqués. Ajustez vos sélections pour "
            "poursuivre."
        )
        return

    cosine_df = cosine_filtered_df

    apply_stopwords = st.checkbox(
        "Appliquer les stopwords français (NLTK) avant le calcul",
        value=False,
        help=(
            "Supprime les mots vides français fournis par NLTK avant de construire"
            " la matrice TF-IDF."
        ),
    )

    aggregated_texts = aggregate_texts_by_variables(
        cosine_df, selected_cosine_variables
    )

    aggregated_export_text = concatenate_texts_with_headers(
        cosine_filtered_df, selected_cosine_variables
    )

    if aggregated_export_text:
        st.download_button(
            label="Télécharger les textes concaténés par sélection",
            data=aggregated_export_text,
            file_name="textes_par_modalite.txt",
            mime="text/plain",
            help=(
                "Export des textes regroupés selon les variables et modalités choisies "
                "pour vérifier la composition de la matrice TF-IDF."
            ),
        )

    if len(aggregated_texts) < 2:
        st.info(
            "Au moins deux groupes de modalités doivent contenir du texte pour calculer la similarité cosinus."
        )
        return

    group_labels = sorted(aggregated_texts.keys())
    ordered_texts = {label: aggregated_texts[label] for label in group_labels}
    texts_summary = pd.DataFrame(
        {
            "Groupe": group_labels,
            "Mots": [len(aggregated_texts[label].split()) for label in group_labels],
        }
    )

    st.markdown("### Textes regroupés")
    st.dataframe(texts_summary, use_container_width=True)

    stop_words = get_french_stopwords() if apply_stopwords else None

    similarity_df = compute_cosine_similarity_matrix(
        ordered_texts, stop_words=stop_words
    )

    if similarity_df.empty:
        st.info("Impossible de calculer la matrice de similarité cosinus avec les données fournies.")
        return

    st.markdown("### Matrice de similarité cosinus")
    st.dataframe(similarity_df.style.format("{:.4f}"), use_container_width=True)

    similarity_long = (
        similarity_df.reset_index()
        .rename(columns={"index": "Groupe"})
        .melt(id_vars="Groupe", var_name="Comparé à", value_name="Similarité")
    )

    modalities_order = similarity_df.index.tolist()

    heatmap = (
        alt.Chart(similarity_long)
        .mark_rect()
        .encode(
            x=alt.X("Groupe:N", sort=modalities_order),
            y=alt.Y("Comparé à:N", sort=modalities_order),
            color=alt.Color(
                "Similarité:Q",
                scale=alt.Scale(
                    domain=[0, 0.5, 1],
                    range=["#f7fbff", "#4292c6", "#08306b"],
                ),
                title="Cosinus",
            ),
            tooltip=["Groupe", "Comparé à", alt.Tooltip("Similarité:Q", format=".4f")],
        )
        .properties(
            title="Carte de chaleur des similarités",
            width=alt.Step(80),
            height=alt.Step(80),
        )
    )

    display_centered_chart(heatmap)
