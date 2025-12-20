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


def _split_records(text: str) -> List[tuple[str, str]]:
    """Diviser le corpus en enregistrements à partir des lignes d'en-tête IRaMuTeQ."""

    if not text:
        return []

    lines = text.splitlines()
    records: List[tuple[str, str]] = []
    current_header: str | None = None
    current_body: List[str] = []

    def push_record() -> None:
        if current_header is None and not current_body:
            return

        body_text = "\n".join(current_body).strip()
        records.append((current_header or "", body_text))

    for line in lines:
        if line.strip().startswith("****"):
            push_record()
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line)

    push_record()

    if not records and text.strip():
        return [("", text.strip())]

    return records


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


def _segments_with_headers(
    text: str, pattern: re.Pattern[str]
) -> List[tuple[str, Optional[str], Optional[str]]]:
    """Retourner les segments en conservant les en-têtes comme segments dédiés."""

    segments: List[tuple[str, Optional[str], Optional[str]]] = []

    for header, body in _split_records(text):
        if header.strip():
            segments.append((header, None, None))

        if body:
            segments.extend(_segments_with_boundaries(body, pattern))

    return segments


def split_segments_by_connectors(text: str, connectors: Dict[str, str]) -> List[str]:
    """Découper le texte en segments entre les connecteurs fournis."""

    if not text:
        return []

    pattern = _build_connector_pattern(connectors)

    if pattern is None:
        return []

    segments_with_boundaries = _segments_with_headers(text, pattern)

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

    pattern = _build_connector_pattern(connectors)

    if pattern is None:
        return []

    segments = _segments_with_headers(text, pattern)
    entries: List[Dict[str, str | int]] = []

    for segment, previous_connector, next_connector in segments:
        tokens = _tokenize(segment)

        if tokens:
            entries.append(
                {
                    "segment": segment.strip(),
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
