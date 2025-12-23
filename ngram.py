from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence

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
    specific_n: int | None = None,
    top_modalities: int = 3,
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
    specific_n:
        Si fourni, ne calcule que pour cette taille d'``n``. Ignorée si en dehors
        de l'intervalle ``min_n`` / ``max_n``.
    top_modalities:
        Nombre de modalités à afficher pour chaque n-gram.
    """

    variable_columns = [column for column in dataframe.columns if column not in ("texte", "entete")]

    if specific_n is not None and specific_n <= 0:
        raise ValueError("specific_n doit être supérieur ou égal à 1")

    requested_sizes: Sequence[int] = (
        [specific_n]
        if specific_n is not None and min_n <= specific_n <= max_n
        else list(range(min_n, max_n + 1))
    )

    counts_by_size: dict[int, Counter[str]] = {size: Counter() for size in requested_sizes}
    modality_counts_by_size: dict[int, dict[str, Counter[str]]] = {
        size: defaultdict(Counter) for size in requested_sizes
    }

    for _, row in dataframe.iterrows():
        text_value = str(row.get("texte", "") or "")
        words = tokenize_text(text_value)

        if not words:
            continue

        modalities = [
            f"{column}={row[column]}" if pd.notna(row[column]) else f"{column}=Non défini"
            for column in variable_columns
        ] or ["Modalité non spécifiée"]

        for n in requested_sizes:
            for ngram_tokens in iter_ngrams(words, n):
                ngram = " ".join(ngram_tokens)
                counts_by_size[n][ngram] += 1

                for modality in modalities:
                    modality_counts_by_size[n][ngram][modality] += 1

    rows = []

    for n in requested_sizes:
        top_entries = counts_by_size[n].most_common(top_k)

        for ngram, frequency in top_entries:
            modalities_summary = modality_counts_by_size[n].get(ngram)
            modalities_display = (
                ", ".join(
                    f"{modality} ({count})"
                    for modality, count in modalities_summary.most_common(top_modalities)
                )
                if modalities_summary
                else "N/A"
            )

            rows.append(
                {
                    "N-gram": ngram,
                    "Taille": len(ngram.split()),
                    "Fréquence": frequency,
                    "Modalités associées": modalities_display,
                }
            )

    return pd.DataFrame(rows)
