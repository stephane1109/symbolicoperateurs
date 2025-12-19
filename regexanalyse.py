from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class RegexPattern:
    """Représente une règle regex issue du dictionnaire."""

    pattern_id: str
    label: str
    category: str
    regex: str
    compiled: re.Pattern[str]


def load_regex_rules(path: Path) -> List[RegexPattern]:
    """Charger les règles regex depuis un fichier JSON."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    patterns: List[RegexPattern] = []

    for entry in payload.get("patterns", []):
        compiled = re.compile(entry["regex"], re.IGNORECASE)
        patterns.append(
            RegexPattern(
                pattern_id=entry.get("id", ""),
                label=entry.get("label", ""),
                category=entry.get("category", ""),
                regex=entry.get("regex", ""),
                compiled=compiled,
            )
        )

    return patterns


def split_segments(text: str) -> List[str]:
    """Découper un texte en segments en fonction de la ponctuation ou des retours à la ligne."""

    raw_segments = re.split(r"(?<=[.!?;:\n])\s+", text)
    return [segment.strip() for segment in raw_segments if segment and segment.strip()]


def _slugify_identifier(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "motif"


def highlight_matches_html(text: str, patterns: Sequence[RegexPattern]) -> str:
    """Retourner le texte en HTML avec surlignage des motifs détectés."""

    matches = []
    for pattern in patterns:
        for match in pattern.compiled.finditer(text):
            matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "label": pattern.label,
                    "identifier": pattern.pattern_id or pattern.label,
                    "match_text": match.group(0),
                }
            )

    if not matches:
        return escape(text).replace("\n", "<br />\n")

    matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

    highlighted_parts: List[str] = []
    cursor = 0

    for match in matches:
        if match["start"] < cursor:
            continue

        highlighted_parts.append(escape(text[cursor : match["start"]]))
        label_class = _slugify_identifier(match["identifier"])
        highlighted_parts.append(
            "<span class=\"regex-annotation regex-"
            f"{label_class}\"><span class=\"regex-label\">{escape(match['label'])}</span>"
            f"<span class=\"regex-text\">{escape(match['match_text'])}</span></span>"
        )
        cursor = match["end"]

    highlighted_parts.append(escape(text[cursor:]))

    return "".join(highlighted_parts).replace("\n", "<br />\n")


def summarize_matches_by_segment(
    segments: Sequence[str], patterns: Sequence[RegexPattern]
) -> List[Dict[str, object]]:
    """Retourner les segments contenant au moins un motif regex et leurs détails."""

    rows: List[Dict[str, object]] = []

    for index, segment in enumerate(segments, start=1):
        matches = []

        for pattern in patterns:
            occurrences = list(pattern.compiled.finditer(segment))
            if occurrences:
                matches.append(
                    {
                        "id": pattern.pattern_id,
                        "label": pattern.label,
                        "occurrences": len(occurrences),
                    }
                )

        if matches:
            rows.append(
                {
                    "segment_id": index,
                    "segment": segment,
                    "motifs": matches,
                }
            )

    return rows


def count_segments_by_pattern(segment_rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Compter le nombre de segments matchés par motif."""

    counts: Dict[str, int] = {}

    for row in segment_rows:
        for motif in row.get("motifs", []):
            identifier = motif.get("label") or motif.get("id")
            if identifier:
                counts[identifier] = counts.get(identifier, 0) + 1

    return counts


def build_regex_style_block(labels: Iterable[str]) -> str:
    """Construire un bloc CSS pour les annotations de motifs regex."""

    palette = [
        "#0EA5E9",
        "#22C55E",
        "#A855F7",
        "#F97316",
        "#EF4444",
        "#14B8A6",
        "#8B5CF6",
    ]

    unique_labels = [label for label in labels if label]
    styles: List[str] = []

    for index, label in enumerate(unique_labels):
        color = palette[index % len(palette)]
        label_class = _slugify_identifier(label)
        styles.append(
            f".regex-annotation.regex-{label_class} {{ background-color: {color}1a; "
            f"border: 1px solid {color}; border-radius: 4px; padding: 2px 4px; "
            "margin: 0 1px; }}"
        )
        styles.append(
            f".regex-annotation.regex-{label_class} .regex-label {{ color: {color}; "
            "font-weight: 700; margin-right: 4px; }}"
        )
        styles.append(
            ".regex-annotation .regex-text { color: #374151; font-weight: 500; }"
        )

    return "\n".join(styles)
