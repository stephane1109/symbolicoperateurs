from __future__ import annotations

import re
from typing import Dict


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str]:
    """Construire un motif regex sécurisé pour tous les connecteurs fournis."""

    cleaned = [key for key in connectors if key]
    sorted_keys = sorted(cleaned, key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def count_words(text: str) -> int:
    """Compter le nombre de mots dans un texte donné."""

    if not text:
        return 0

    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def compute_total_connectors(text: str, connectors: Dict[str, str]) -> int:
    """Compter toutes les occurrences des connecteurs dans le texte."""

    if not text:
        return 0

    cleaned_connectors = {key: value for key, value in connectors.items() if key}

    if not cleaned_connectors:
        return 0

    pattern = _build_connector_pattern(cleaned_connectors)
    return len(list(pattern.finditer(text)))


def compute_density(text: str, connectors: Dict[str, str], base: int = 1000) -> float:
    """Calculer la densité de connecteurs ramenée à ``base`` mots."""

    word_count = count_words(text)

    if word_count == 0:
        return 0.0

    total_connectors = compute_total_connectors(text, connectors)

    if total_connectors == 0:
        return 0.0

    return (total_connectors / word_count) * float(base)


def compute_density_by_label(text: str, connectors: Dict[str, str], base: int = 1000) -> Dict[str, float]:
    """Calculer la densité ramenée à ``base`` mots pour chaque label présent."""

    word_count = count_words(text)

    if word_count == 0:
        return {}

    per_label: Dict[str, float] = {}

    for label in sorted(set(connectors.values())):
        label_connectors = {
            connector: connector_label
            for connector, connector_label in connectors.items()
            if connector_label == label
        }
        total = compute_total_connectors(text, label_connectors)

        if total:
            per_label[label] = (total / word_count) * float(base)
        else:
            per_label[label] = 0.0

    return per_label
