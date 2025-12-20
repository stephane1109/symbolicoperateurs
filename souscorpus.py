from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from analyses import load_connectors


def has_header_markers(record: Dict[str, str]) -> bool:
    """Vérifie la présence de marqueurs (tokens commençant par « * ») dans l'entête."""

    entete = record.get("entete", "")
    tokens = entete.split()
    return entete.startswith("****") and any(token.startswith("*") for token in tokens)


def build_subcorpus(records: List[Dict[str, str]]) -> List[str]:
    """Construit la liste des segments du sous-corpus à partir des enregistrements IRaMuTeQ.

    Seuls les segments contenant au moins un connecteur sont conservés afin de ne
    pas exporter de textes sans pertinence pour l'analyse.
    """

    connectors = load_connectors(Path(__file__).parent / "dictionnaires" / "connecteurs.json")
    connector_pattern = _build_connector_pattern(connectors)

    if connector_pattern is None:
        return []

    subcorpus_segments: List[str] = []

    for record in records:
        if not has_header_markers(record):
            continue

        entete = record.get("entete", "").strip()
        texte = record.get("texte", "").strip()
        segment = f"{entete}\n{texte}".strip()

        if connector_pattern.search(segment):
            subcorpus_segments.append(segment)

    return subcorpus_segments


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str] | None:
    """Construire un motif regex qui identifie les connecteurs présents dans le texte."""

    cleaned_connectors = [key.strip() for key in connectors if key.strip()]

    if not cleaned_connectors:
        return None

    sorted_keys = sorted(cleaned_connectors, key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)
