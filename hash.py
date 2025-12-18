from __future__ import annotations

"""
Le "hachage du texte est calculé à partir de la Longueur Moyenne des Segments (LMS) de texte entre deux connecteurs.
Hypothèse : Plus les segments sont courts, plus le texte est "haché", saccadé, algorithmique.
Plus les segments sont longs, plus le texte est fluide, narratif, explicatif.
Visualisation (exemple) :
Texte "Saturant" :
"Appelle le 15 [OU] le 112 [SI] tu es en danger [MAIS] [SI] tu es seul [ALORS] sors."
5 connecteurs en 20 mots. Les segments sont dans cet exemple petits (3-4 mots).
"""

import re
from statistics import mean
from typing import Dict, Iterable, List, Optional

import pandas as pd

from densite import build_text_from_dataframe, filter_dataframe_by_modalities


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str]:
    """Construire un motif regex pour capturer tous les connecteurs."""

    cleaned = [key for key in connectors if key]
    sorted_keys = sorted(cleaned, key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def split_segments_by_connectors(text: str, connectors: Dict[str, str]) -> List[str]:
    """Découper le texte en segments séparés par les connecteurs détectés."""

    if not text:
        return []

    cleaned_connectors = {key: value for key, value in connectors.items() if key}

    if not cleaned_connectors:
        return []

    pattern = _build_connector_pattern(cleaned_connectors)
    segments: List[str] = []
    last_end = 0

    for match in pattern.finditer(text):
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
