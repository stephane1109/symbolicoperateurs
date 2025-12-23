from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def aggregate_texts_by_variable(dataframe: pd.DataFrame, variable: str) -> Dict[str, str]:
    """Assembler les textes par modalité pour une variable donnée.

    Seules les lignes où la variable est définie sont conservées. Les textes vides
    ou manquants sont ignorés afin de ne calculer la similarité qu'à partir de
    contenus existants.
    """

    if variable not in dataframe.columns:
        raise KeyError(f"La variable '{variable}' est absente du tableau fourni.")

    aggregated_texts: Dict[str, str] = {}

    for modality, subset in dataframe.dropna(subset=[variable]).groupby(variable):
        texts = subset["texte"].dropna().astype(str).tolist()
        combined_text = " ".join(texts).strip()

        if combined_text:
            aggregated_texts[str(modality)] = combined_text

    return aggregated_texts


def get_french_stopwords() -> List[str]:
    """Retourner la liste des stopwords français fournie par NLTK.

    Le téléchargement des stopwords est effectué à la volée si nécessaire.
    """

    try:
        return stopwords.words("french")
    except LookupError:
        download("stopwords")
        return stopwords.words("french")


def compute_cosine_similarity_matrix(
    texts_by_group: Dict[str, str], stop_words: Iterable[str] | None = None
) -> pd.DataFrame:
    """Calculer une matrice de similarité cosinus à partir de textes regroupés."""

    if len(texts_by_group) < 2:
        return pd.DataFrame()

    labels = list(texts_by_group.keys())
    corpus = list(texts_by_group.values())

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return pd.DataFrame(similarity_matrix, index=labels, columns=labels)


def compute_cosine_similarity_by_variable(
    dataframe: pd.DataFrame, variable: str, use_stopwords: bool = False
) -> pd.DataFrame:
    """Retourner une matrice de similarité cosinus entre modalités d'une variable."""

    aggregated_texts = aggregate_texts_by_variable(dataframe, variable)

    if len(aggregated_texts) < 2:
        return pd.DataFrame()

    stop_words = get_french_stopwords() if use_stopwords else None

    return compute_cosine_similarity_matrix(aggregated_texts, stop_words=stop_words)
