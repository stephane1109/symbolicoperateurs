from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd


def load_connectors(path: Path) -> Dict[str, str]:
    """Load the connector dictionary from JSON.

    Parameters
    ----------
    path:
        Path to the ``connecteurs.json`` file.
    """

    with path.open(encoding="utf-8") as handle:
        connectors = json.load(handle)

    return {key.strip(): value for key, value in connectors.items() if key.strip()}


def _build_connector_pattern(connectors: Dict[str, str]) -> re.Pattern[str]:
    sorted_keys = sorted(connectors.keys(), key=len, reverse=True)
    escaped = [re.escape(key) for key in sorted_keys]
    pattern = "|".join(escaped)

    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def annotate_connectors(text: str, connectors: Dict[str, str]) -> str:
    """Insert connector labels directly before each match in the text."""

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


def count_connectors(text: str, connectors: Dict[str, str]) -> pd.DataFrame:
    """Count how many times each connector appears in the text."""

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

