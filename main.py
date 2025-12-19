from __future__ import annotations

import sys
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
    load_connectors,
)
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
from hash import (
    average_segment_length,
    average_segment_length_by_modality,
    compute_segment_word_lengths,
    segments_with_word_lengths,
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
    tabs = st.tabs(["Import", "Analyse stats", "Densité", "Hash"])

    with tabs[0]:
        st.subheader("Données importées")
        st.dataframe(df, use_container_width=True)

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

        combined_text = build_text_from_dataframe(filtered_df)

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

        connector_labels = sorted(set(connectors.values()))
        selected_labels = [
            label
            for label in connector_labels
            if st.checkbox(
                f"Annoter les connecteurs « {label} »",
                value=True,
                help="Sélectionnez un ou plusieurs types de connecteurs à mettre en surbrillance.",
            )
        ]

        if not selected_labels:
            st.info("Sélectionnez au moins un type de connecteur pour lancer l'annotation.")
            return

        filtered_connectors = {
            connector: label for connector, label in connectors.items() if label in selected_labels
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

    with tabs[2]:
        st.subheader("Densité des connecteurs")
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

    with tabs[3]:
        st.subheader("Hash (LMS entre connecteurs)")
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

                col1, col2, col3 = st.columns(3)
                col1.metric("Segments comptabilisés", str(len(segment_lengths)))
                col2.metric("LMS (mots)", f"{average_length:.2f}")
                col3.metric("Segments min / max", f"{min(segment_lengths)} / {max(segment_lengths)}")

                distribution_df = pd.DataFrame(segment_entries)

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
                    distribution_df.rename(columns={"segment": "Segment", "longueur": "Longueur"}),
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


if __name__ == "__main__":
    main()
