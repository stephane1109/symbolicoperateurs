from __future__ import annotations

"""
Le "hachage" du texte est calculé à partir de la Longueur Moyenne des Segments (LMS)
de texte délimités par les connecteurs détectés dans le texte. L'idée :
- Des segments courts signalent un texte haché, saccadé, algorithmique.
- Des segments longs évoquent une prose fluide, narrative ou explicative.
"""

import re
from statistics import mean
from typing import Dict, Iterable, List, Literal, Optional

import pandas as pd

from densite import build_text_from_dataframe, filter_dataframe_by_modalities


SegmentationMode = Literal["connecteurs", "connecteurs_et_ponctuation"]


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


def _build_boundary_pattern(
    connectors: Dict[str, str],
    include_punctuation: bool,
    connector_pattern: re.Pattern[str] | None = None,
) -> re.Pattern[str] | None:
    """Construire un motif pour les bornes de segment (connecteurs, ponctuation)."""

    connector_pattern = connector_pattern or _build_connector_pattern(connectors)
    punctuation_pattern = r"[\.!?;:]+" if include_punctuation else None

    if connector_pattern is None and not punctuation_pattern:
        return None

    if connector_pattern is None:
        return re.compile(punctuation_pattern, re.IGNORECASE)

    if not punctuation_pattern:
        return connector_pattern

    return re.compile(
        rf"{connector_pattern.pattern}|{punctuation_pattern}",
        re.IGNORECASE,
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def _is_connector(boundary: str | None, connector_pattern: re.Pattern[str] | None) -> bool:
    """Vérifier si une borne correspond à un connecteur (et non à de la ponctuation)."""

    if not boundary or connector_pattern is None:
        return False

    return connector_pattern.fullmatch(boundary.strip()) is not None


def _segments_with_boundaries(
    text: str,
    pattern: re.Pattern[str],
    connector_pattern: re.Pattern[str] | None,
) -> List[tuple[str, Optional[str], Optional[str]]]:
    """Retourner uniquement les segments bornés par au moins un connecteur."""

    segments: List[tuple[str, Optional[str], Optional[str]]] = []
    last_end = 0
    previous_connector: Optional[str] = None

    for match in pattern.finditer(text):
        segment = text[last_end: match.start()]
        next_connector = match.group(0)

        previous_is_connector = _is_connector(previous_connector, connector_pattern)
        next_is_connector = _is_connector(next_connector, connector_pattern)

        if segment.strip() and (previous_is_connector or next_is_connector):
            segments.append((segment, previous_connector, next_connector))

        previous_connector = next_connector
        last_end = match.end()

    trailing = text[last_end:]

    if trailing.strip() and _is_connector(previous_connector, connector_pattern):
        segments.append((trailing, previous_connector, None))

    return segments


def split_segments_by_connectors(
    text: str, connectors: Dict[str, str], segmentation_mode: SegmentationMode = "connecteurs"
) -> List[str]:
    """Découper le texte en segments entre les connecteurs ou ponctuations choisies."""

    if not text:
        return []

    text = _remove_metadata_first_line(text)

    connector_pattern = _build_connector_pattern(connectors)

    if connector_pattern is None:
        return []

    connector_found = connector_pattern.search(text)

    if connector_found is None:
        return [text]

    include_punctuation = segmentation_mode == "connecteurs_et_ponctuation"

    pattern = _build_boundary_pattern(
        connectors, include_punctuation, connector_pattern=connector_pattern
    )

    if pattern is None:
        return []

    segments_with_boundaries = _segments_with_boundaries(
        text, pattern, connector_pattern
    )

    return [segment for segment, _, _ in segments_with_boundaries]


def compute_segment_word_lengths(
    text: str, connectors: Dict[str, str], segmentation_mode: SegmentationMode = "connecteurs"
) -> List[int]:
    """Obtenir la longueur (en mots) de chaque segment selon le mode de segmentation."""

    segments = split_segments_by_connectors(text, connectors, segmentation_mode)
    lengths = []

    for segment in segments:
        tokens = _tokenize(segment)

        if tokens:
            lengths.append(len(tokens))

    return lengths


def segments_with_word_lengths(
    text: str, connectors: Dict[str, str], segmentation_mode: SegmentationMode = "connecteurs"
) -> List[Dict[str, str | int]]:
    """Retourner chaque segment avec sa longueur en mots."""

    if not text:
        return []

    text = _remove_metadata_first_line(text)

    connector_pattern = _build_connector_pattern(connectors)

    if connector_pattern is None:
        return []

    connector_found = connector_pattern.search(text)

    if connector_found is None:
        tokens = _tokenize(text)
        if tokens:
            return [
                {
                    "segment": text.strip(),
                    "segment_avec_marqueurs": text.strip(),
                    "longueur": len(tokens),
                    "connecteur_precedent": "",
                    "connecteur_suivant": "",
                }
            ]
        return []

    include_punctuation = segmentation_mode == "connecteurs_et_ponctuation"

    pattern = _build_boundary_pattern(
        connectors, include_punctuation, connector_pattern=connector_pattern
    )

    if pattern is None:
        return []

    segments = _segments_with_boundaries(text, pattern, connector_pattern)
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


def average_segment_length(
    text: str, connectors: Dict[str, str], segmentation_mode: SegmentationMode = "connecteurs"
) -> float:
    """Calculer la Longueur Moyenne des Segments (LMS)."""

    lengths = compute_segment_word_lengths(text, connectors, segmentation_mode)

    if not lengths:
        return 0.0

    return float(mean(lengths))


def average_segment_length_by_modality(
    dataframe: pd.DataFrame,
    variable: Optional[str],
    connectors: Dict[str, str],
    modalities: Optional[Iterable[str]] = None,
    segmentation_mode: SegmentationMode = "connecteurs",
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
        lengths = compute_segment_word_lengths(text_value, connectors, segmentation_mode)
        lms_value = float(mean(lengths)) if lengths else 0.0

        rows.append(
            {
                "modalite": modality,
                "segments": len(lengths),
                "lms": lms_value,
            }
        )

    return pd.DataFrame(rows).sort_values("modalite").reset_index(drop=True)
