"""### Application Streamlit des opérateurs symboliques

Ce fichier gère l'interface utilisateur : chargement des données,
assemblage des onglets et préparation des données partagées.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from connecteurs import get_selected_connectors  # noqa: E402
from onglets import (  # noqa: E402
    parse_upload,
    rendu_connecteurs,
    rendu_densite,
    rendu_donnees_brutes,
    rendu_donnees_importees,
    rendu_hash,
    rendu_lisibilite,
    rendu_ngram,
    rendu_openlexicon,
    rendu_patterns,
    rendu_regex_motifs,
    rendu_simi_cosinus,
    rendu_sous_corpus,
    rendu_tfidf,
)


def main() -> None:
    st.set_page_config(page_title="Symbolic Connectors", layout="wide")

    st.title("Symbolic Connectors")
    st.markdown(
        """Symbolic Connectors : ce titre renvoie au courant symbolique en IA, qui s’appuie sur une logique de programme et de règles (par opposition au connexionnisme, plus proche de l’analogie avec le cerveau). Le nom convient bien à l’objectif de cette application : repérer, dans les réponses des grands modèles de langage (LLM), des traces de “langage machine”, en particulier des structures linguistiques proches de la programmation. (logique conditionnelle - si, alors, sinon, ou, et…).

L’application a été développée pour explorer une hypothèse liée aux réponses des LLM face à une crise suicidaire (article 1). L’idée est que, dans ces situations, les modèles peuvent produire une surfréquence de connecteurs logiques (si, alors, ou, et, sinon…). Or, ce type d’écriture — très procédurale — pourrait augmenter la charge cognitive chez une personne déjà en détresse.

Pour des raisons d’interopérabilité, le corpus doit être formaté selon les exigences d’IRaMuTeQ : chaque texte commence par une ligne d’en-tête du type **** *variable_modalité.

Pour l’instant, l’application repose sur un fichier dictionnaire.json (visible dans l’onglet « Connecteurs ») et sur des règles regex. À terme, l’idéal serait de généraliser l’approche avec une bibliothèque NLP (par exemple spaCy et/ou BERT), afin de rendre la détection moins rigide que des motifs regex. Mais je suis limité par l’hébergement sur Streamlit Cloud (version gratuite), qui impose des ressources restreintes. Toutefois, les stopwords sont filtrés avec NLTK (léger), et l’onglet « patterns » s’appuie sur spaCy."""
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
            "OpenLexicon",
            "Hash",
            "Regex motifs",
            "Patterns",
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
    records, df = parse_upload(content)

    if not records:
        st.warning("Aucune entrée valide trouvée dans le fichier fourni.")
        return

    with tabs[1]:
        rendu_connecteurs(tabs[1])

    filtered_connectors = get_selected_connectors()

    with tabs[0]:
        rendu_donnees_importees(tabs[0], df, filtered_connectors)

    with tabs[2]:
        donnees_brutes = rendu_donnees_brutes(tabs[2], df, filtered_connectors)

    if donnees_brutes is None:
        return

    filtered_df, selected_variables, combined_text = donnees_brutes

    with tabs[3]:
        rendu_sous_corpus(tabs[3], records, filtered_connectors)

    with tabs[4]:
        rendu_densite(tabs[4], df, filtered_connectors)

    with tabs[5]:
        rendu_openlexicon(tabs[5], df, filtered_connectors)

    with tabs[6]:
        rendu_hash(tabs[6], filtered_df, filtered_connectors, combined_text)

    with tabs[7]:
        rendu_regex_motifs(tabs[7], combined_text)

    with tabs[8]:
        rendu_patterns(tabs[8], filtered_df, combined_text, selected_variables)

    with tabs[9]:
        rendu_lisibilite(tabs[9], df, filtered_connectors)

    with tabs[10]:
        rendu_ngram(tabs[10], filtered_df, filtered_connectors)

    with tabs[11]:
        rendu_tfidf(tabs[11], df, filtered_connectors)

    with tabs[12]:
        rendu_simi_cosinus(tabs[12], df)


if __name__ == "__main__":
    main()
