from __future__ import annotations

from typing import Dict, List


def has_header_markers(record: Dict[str, str]) -> bool:
    """Vérifie la présence de marqueurs (tokens commençant par « * ») dans l'entête."""

    entete = record.get("entete", "")
    tokens = entete.split()
    return entete.startswith("****") and any(token.startswith("*") for token in tokens)


def build_subcorpus(records: List[Dict[str, str]]) -> List[str]:
    """Construit la liste des segments du sous-corpus à partir des enregistrements IRaMuTeQ."""

    subcorpus_segments: List[str] = []

    for record in records:
        if not has_header_markers(record):
            continue

        entete = record.get("entete", "").strip()
        texte = record.get("texte", "").strip()
        subcorpus_segments.append(f"{entete}\n{texte}".strip())

    return subcorpus_segments
