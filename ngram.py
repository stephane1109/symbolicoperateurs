from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, List

import pandas as pd

TOKEN_PATTERN = re.compile(r"[\wÀ-ÖØ-öø-ÿ'-]+", re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """Convertir un texte en liste de mots normalisés."""

    return TOKEN_PATTERN.findall(text.lower())


def iter_ngrams(words: List[str], n: int) -> Iterable[tuple[str, ...]]:
    """Générer les n-grams pour une liste de mots donnée."""

    if n <= 0:
        return []

    return zip(*(words[i:] for i in range(n)))


def compute_ngram_statistics(
    dataframe: pd.DataFrame,
    min_n: int = 3,
    max_n: int = 6,
    top_k: int = 10,
) -> pd.DataFrame:
    """Calculer les N-grams les plus fréquents et leur distribution par modalités.

    Parameters
    ----------
    dataframe:
        Table contenant au minimum une colonne ``texte`` et éventuellement des colonnes
        de variables/modalités.
    min_n / max_n:
        Tailles minimale et maximale des n-grams à considérer.
    top_k:
        Nombre d'entrées à conserver dans le classement final.
    """

    variable_columns = [column for column in dataframe.columns if column not in ("texte", "entete")]
    overall_counts: Counter[str] = Counter()
    modality_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in dataframe.iterrows():
        text_value = str(row.get("texte", "") or "")
        words = tokenize_text(text_value)

        if not words:
            continue

        modalities = [
            f"{column}={row[column]}" if pd.notna(row[column]) else f"{column}=Non défini"
            for column in variable_columns
        ] or ["Modalité non spécifiée"]

        for n in range(min_n, max_n + 1):
            for ngram_tokens in iter_ngrams(words, n):
                ngram = " ".join(ngram_tokens)
                overall_counts[ngram] += 1

                for modality in modalities:
                    modality_counts[ngram][modality] += 1

    top_entries = overall_counts.most_common(top_k)
    rows = []

    for ngram, frequency in top_entries:
        modalities_summary = modality_counts.get(ngram)
        top_modalities = (
            ", ".join(f"{modality} ({count})" for modality, count in modalities_summary.most_common(3))
            if modalities_summary
            else "N/A"
        )

        rows.append(
            {
                "N-gram": ngram,
                "Taille": len(ngram.split()),
                "Fréquence": frequency,
                "Modalités associées": top_modalities,
            }
        )

    return pd.DataFrame(rows)
