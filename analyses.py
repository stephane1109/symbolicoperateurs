from __future__ import annotations

import hashlib
import json
import re
from html import escape
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def load_connectors(path: Path) -> Dict[str, str]:
    """Charger le dictionnaire de connecteurs depuis un fichier JSON.

    Paramètres
    ----------
    path:
        Chemin vers le fichier ``connecteurs.json``.
    """

    with path.open(encoding="utf-8") as handle:
        connectors = json.load(handle)

    return {key.strip(): value for key, value in connectors.items() if key.strip()}


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str]:
    """Construire un motif regex qui capture chaque connecteur."""
    sorted_keys = sorted(connectors.keys(), key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def build_label_color_map(labels: Iterable[str]) -> Dict[str, str]:
    """Attribuer une couleur pastel déterministe à chaque label."""

    palette = [
        "#fef3c7",
        "#e0f2fe",
        "#dcfce7",
        "#ede9fe",
        "#ffe4e6",
        "#cffafe",
        "#fce7f3",
        "#e5e7eb",
    ]

    unique_labels = {label for label in labels if label}
    colors: Dict[str, str] = {}

    for label in unique_labels:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        index = int(digest, 16) % len(palette)
        colors[label] = palette[index]

    return colors


def label_to_class(label: str) -> str:
    """Convertir un label en nom de classe CSS sûr."""

    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    slug = slug or "connecteur"
    return f"connector-label-{slug}"


def annotate_connectors(text: str, connectors: Dict[str, str]) -> str:
    """Insérer les étiquettes des connecteurs avant chaque occurrence dans le texte."""

    if not text:
        return ""

    cleaned_connectors = {key: value for key, value in connectors.items() if key}
    if not cleaned_connectors:
        return text

    pattern = _build_connector_pattern(cleaned_connectors)
    lower_map = {key.lower(): value for key, value in cleaned_connectors.items()}

    def _replacer(match: re.Match[str]) -> str:
        matched_connector = match.group(0)
        label = lower_map.get(matched_connector.lower(), "")
        return f"[{label}] {matched_connector}"

    return pattern.sub(_replacer, text)


def annotate_connectors_html(
    text: str,
    connectors: Dict[str, str],
    label_colors: Dict[str, str] | None = None,
) -> str:
    """Retourner une version HTML du texte annoté avec les labels des connecteurs.

    Chaque connecteur est entouré d'un conteneur HTML incluant une étiquette de
    label visible. Les caractères spéciaux du texte source sont échappés afin de
    garantir une sortie sécurisée. Les couleurs peuvent être personnalisées via
    ``label_colors`` pour différencier visuellement les catégories.
    """

    if not text:
        return ""

    cleaned_connectors = {key: value for key, value in connectors.items() if key}
    resolved_colors = label_colors or build_label_color_map(cleaned_connectors.values())
    if not cleaned_connectors:
        return escape(text)

    pattern = _build_connector_pattern(cleaned_connectors)
    lower_map = {key.lower(): value for key, value in cleaned_connectors.items()}

    def _replacer(match: re.Match[str]) -> str:
        matched_connector = match.group(0)
        label = lower_map.get(matched_connector.lower(), "")
        safe_label = escape(label)
        safe_connector = escape(matched_connector)
        color = resolved_colors.get(label, "#eef3ff")
        class_name = label_to_class(label)

        return (
            f'<span class="connector-annotation {class_name}" '
            f'style="background-color: {color};">'
            f'<span class="connector-label">{safe_label}</span>'
            f'<span class="connector-text">{safe_connector}</span>'
            "</span>"
        )

    escaped_text = escape(text)
    return pattern.sub(_replacer, escaped_text)


def count_connectors(text: str, connectors: Dict[str, str]) -> pd.DataFrame:
    """Compter le nombre d'occurrences de chaque connecteur dans le texte."""

    cleaned_connectors = {key: value for key, value in connectors.items() if key}
    rows = []

    for connector, label in cleaned_connectors.items():
        regex = re.compile(rf"\b{re.escape(connector)}\b", re.IGNORECASE)
        occurrences = len(regex.findall(text))

        if occurrences:
            rows.append(
                {
                    "connecteur": connector,
                    "label": label,
                    "occurrences": occurrences,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["connecteur", "label", "occurrences"])

    return (
        pd.DataFrame(rows)
        .sort_values(["label", "connecteur"])
        .reset_index(drop=True)
    )
