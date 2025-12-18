from __future__ import annotations

"""
Le "hachage" du texte est calculé à partir de la Longueur Moyenne des Segments (LMS)
de texte séparés par une ponctuation forte (point, point d'exclamation, point
d'interrogation) ou un retour à la ligne. L'idée :
- Des segments courts signalent un texte haché, saccadé, algorithmique.
- Des segments longs évoquent une prose fluide, narrative ou explicative.
"""

import re
from statistics import mean
from typing import Dict, Iterable, List, Optional

import pandas as pd

from densite import build_text_from_dataframe, filter_dataframe_by_modalities


SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?]+|\n+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def split_segments_by_connectors(text: str, connectors: Dict[str, str]) -> List[str]:
    """Découper le texte en segments en se basant sur la ponctuation ou un retour à la ligne.

    Le paramètre ``connectors`` est conservé pour compatibilité de signature,
    mais les segments sont désormais définis par des bornes de phrase (., !, ?, \n).
    """

    if not text:
        return []

    # Connectors conservés pour compatibilité d'appel.
    _ = connectors

    segments: List[str] = []
    last_end = 0

    for match in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        segment = text[last_end: match.start()]

        if segment.strip():
            segments.append(segment)

        last_end = match.end()

    trailing = text[last_end:]

    if trailing.strip():
        segments.append(trailing)

    return segments


def compute_segment_word_lengths(text: str, connectors: Dict[str, str]) -> List[int]:
    """Obtenir la longueur (en mots) de chaque segment entre connecteurs."""

    segments = split_segments_by_connectors(text, connectors)
    lengths = []

    for segment in segments:
        tokens = _tokenize(segment)

        if tokens:
            lengths.append(len(tokens))

    return lengths


def segments_with_word_lengths(
    text: str, connectors: Dict[str, str]
) -> List[Dict[str, str | int]]:
    """Retourner chaque segment avec sa longueur en mots."""

    segments = split_segments_by_connectors(text, connectors)
    entries: List[Dict[str, str | int]] = []

    for segment in segments:
        tokens = _tokenize(segment)

        if tokens:
            entries.append({"segment": segment.strip(), "longueur": len(tokens)})

    return entries


def average_segment_length(text: str, connectors: Dict[str, str]) -> float:
    """Calculer la Longueur Moyenne des Segments (LMS) entre connecteurs."""

    lengths = compute_segment_word_lengths(text, connectors)

    if not lengths:
        return 0.0

    return float(mean(lengths))


def average_segment_length_by_modality(
    dataframe: pd.DataFrame,
    variable: Optional[str],
    connectors: Dict[str, str],
    modalities: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Calculer la LMS par modalité pour une variable donnée."""

    if dataframe.empty:
        return pd.DataFrame(columns=["modalite", "segments", "lms", "min", "max"])

    filtered_df = filter_dataframe_by_modalities(dataframe, variable, modalities)

    if not variable or variable not in filtered_df.columns or filtered_df.empty:
        return pd.DataFrame(columns=["modalite", "segments", "lms", "min", "max"])

    rows: List[Dict[str, float | int | str]] = []

    for modality, subset in filtered_df.groupby(variable):
        text_value = build_text_from_dataframe(subset)
        lengths = compute_segment_word_lengths(text_value, connectors)
        lms_value = float(mean(lengths)) if lengths else 0.0

        rows.append(
            {
                "modalite": modality,
                "segments": len(lengths),
                "lms": lms_value,
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
            }
        )

    return pd.DataFrame(rows).sort_values("modalite").reset_index(drop=True)
