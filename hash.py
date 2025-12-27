from __future__ import annotations

"""
Le "hachage" du texte est calculé à partir de la Longueur Moyenne des Segments (LMS)
de texte délimités par les connecteurs détectés dans le texte. L'idée :
- Des segments courts signalent un texte haché, saccadé, algorithmique.
- Des segments longs évoquent une prose fluide, narrative ou explicative.
"""

import re
from statistics import mean
from typing import Dict, Iterable, List, Optional

import pandas as pd

from densite import build_text_from_dataframe, filter_dataframe_by_modalities


METADATA_LINE_PATTERN = re.compile(r"^\*{4}\s+\*model_gpt\s+\*prompt_1\s*$", re.IGNORECASE)

ECART_TYPE_EXPLANATION = """L'écart-type est une mesure de dispersion. Le rapport entre l'écart-type et la longueur moyenne des segments (LMS) agit comme un indicateur de stabilité : une dispersion faible signale une fluidité de lecture, tandis qu'une dispersion forte révèle une structure hachée et imprévisible.
Tant que l'écart-type est inférieur à la moyenne, la série est considérée comme relativement "cohérente".
Dès que l'écart-type dépasse la moyenne on bascule dans une instabilité. Cela signifie que la variation est plus grande que la mesure elle-même.
"""


def _remove_metadata_first_line(text: str) -> str:
    """Retirer une éventuelle ligne de métadonnées en début de texte."""

    lines = text.splitlines()

    if not lines:
        return text

    if METADATA_LINE_PATTERN.match(lines[0].strip()):
        return "\n".join(lines[1:]).lstrip()

    return text


def _format_segment_with_markers(
    segment: str, previous_connector: Optional[str], next_connector: Optional[str]
) -> str:
    """Afficher le segment avec ses bornes encadrées par des crochets."""

    parts: List[str] = []

    if previous_connector:
        parts.append(f"[{previous_connector}]")

    parts.append(segment.strip())

    if next_connector:
        parts.append(f"[{next_connector}]")

    return " ".join(parts)


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str] | None:
    """Construire un motif regex sécurisé pour tous les connecteurs fournis."""

    cleaned = [key for key in connectors if key]

    if not cleaned:
        return None

    sorted_keys = sorted(cleaned, key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def _segments_with_boundaries(
    text: str, pattern: re.Pattern[str]
) -> List[tuple[str, Optional[str], Optional[str]]]:
    """Retourner les segments associés à leurs connecteurs de borne."""

    segments: List[tuple[str, Optional[str], Optional[str]]] = []
    last_end = 0
    previous_connector: Optional[str] = None

    for match in pattern.finditer(text):
        segment = text[last_end: match.start()]

        if segment.strip():
            segments.append((segment, previous_connector, match.group(0)))

        previous_connector = match.group(0)
        last_end = match.end()

    trailing = text[last_end:]

    if trailing.strip():
        segments.append((trailing, previous_connector, None))

    return segments


def split_segments_by_connectors(text: str, connectors: Dict[str, str]) -> List[str]:
    """Découper le texte en segments entre les connecteurs fournis."""

    if not text:
        return []

    text = _remove_metadata_first_line(text)

    pattern = _build_connector_pattern(connectors)

    if pattern is None:
        return []

    segments_with_boundaries = _segments_with_boundaries(text, pattern)

    return [segment for segment, _, _ in segments_with_boundaries]


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

    if not text:
        return []

    text = _remove_metadata_first_line(text)

    pattern = _build_connector_pattern(connectors)

    if pattern is None:
        return []

    segments = _segments_with_boundaries(text, pattern)
    entries: List[Dict[str, str | int]] = []

    for segment, previous_connector, next_connector in segments:
        tokens = _tokenize(segment)

        if tokens:
            entries.append(
                {
                    "segment": segment.strip(),
                    "segment_avec_marqueurs": _format_segment_with_markers(
                        segment, previous_connector, next_connector
                    ),
                    "longueur": len(tokens),
                    "connecteur_precedent": (previous_connector or ""),
                    "connecteur_suivant": (next_connector or ""),
                }
            )

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
        return pd.DataFrame(columns=["modalite", "segments", "lms"])

    filtered_df = filter_dataframe_by_modalities(dataframe, variable, modalities)

    if not variable or variable not in filtered_df.columns or filtered_df.empty:
        return pd.DataFrame(columns=["modalite", "segments", "lms"])

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
            }
        )

    return pd.DataFrame(rows).sort_values("modalite").reset_index(drop=True)
