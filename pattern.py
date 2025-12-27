from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

DEFAULT_RULES_PATH = Path("dictionnaires/motifs_progr_regex.json")


@dataclass(frozen=True)
class LogicalPattern:
    """Représente une règle de combinaison de connecteurs logiques."""

    name: str
    label: str
    category: str
    interpretation: str
    regex: str
    token_patterns: List[List[dict]]


def _load_json_rules(path: Path = DEFAULT_RULES_PATH) -> Sequence[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("patterns", [])


def _extract_connectors(regex: str) -> List[tuple[list[str], bool]]:
    cleaned = regex.replace("\\b", " ")
    raw_tokens = [
        token
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", cleaned)
        if len(token) > 1
    ]

    connectors: List[tuple[list[str], bool]] = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]

        if f"(?![^)]*\\b{token}\\b" in regex:
            index += 1
            continue

        optional = f"\\b{token}\\b)?" in regex

        # Gestion des groupes explicites (ex: tu\s+peux)
        if index + 1 < len(raw_tokens):
            next_token = raw_tokens[index + 1]
            combined = f"\\b{token}\\s+{next_token}\\b"
            if combined in regex:
                optional = f"{combined})?" in regex
                connectors.append(([token, next_token], optional))
                index += 2
                continue

        # Gestion spécifique de d'abord
        if token == "abord" and "d[’']?\\s*abord" in regex:
            connectors.append((['d\'', "abord"], optional))
            index += 1
            continue

        connectors.append(([token], optional))
        index += 1

    return connectors


def _build_token_patterns(connectors: Sequence[tuple[list[str], bool]]) -> List[List[dict]]:
    patterns: List[List[dict]] = [[]]

    for tokens, optional in connectors:
        expanded: List[List[dict]] = []
        for base in patterns:
            if optional:
                expanded.append(base)

            current = list(base)
            if current:
                current.append({"OP": "*"})

            current.extend({"LOWER": token} for token in tokens)
            expanded.append(current)

        patterns = [pattern for pattern in expanded if pattern]

    return patterns


def load_logical_patterns(path: Path = DEFAULT_RULES_PATH) -> List[LogicalPattern]:
    rules = _load_json_rules(path)
    logical_patterns: List[LogicalPattern] = []

    for index, rule in enumerate(rules):
        regex = rule.get("regex", "")
        connectors = _extract_connectors(regex)
        token_patterns = _build_token_patterns(connectors)
        logical_patterns.append(
            LogicalPattern(
                name=f"LOGICAL_{index}",
                label=rule.get("label", ""),
                category=rule.get("category", ""),
                interpretation=rule.get("interpretation", ""),
                regex=regex,
                token_patterns=token_patterns,
            )
        )

    return logical_patterns


def load_spacy_model(model: str = "fr_core_news_sm") -> Language:
    return spacy.load(model)


def build_matcher(nlp: Language, patterns: Iterable[LogicalPattern]) -> Matcher:
    matcher = Matcher(nlp.vocab)

    for pattern in patterns:
        matcher.add(pattern.name, pattern.token_patterns)

    return matcher


def find_logical_patterns(text: str, nlp: Language | None = None) -> List[dict]:
    nlp = nlp or load_spacy_model()
    logical_patterns = load_logical_patterns()
    matcher = build_matcher(nlp, logical_patterns)

    doc: Doc = nlp(text)
    matches = matcher(doc)

    pattern_map = {pattern.name: pattern for pattern in logical_patterns}
    results: List[dict] = []

    for match_id, start, end in matches:
        name = nlp.vocab.strings[match_id]
        span: Span = doc[start:end]
        logical = pattern_map.get(name)
        if logical is None:
            continue

        results.append(
            {
                "name": logical.name,
                "label": logical.label,
                "category": logical.category,
                "interpretation": logical.interpretation,
                "regex": logical.regex,
                "span": span.text,
                "start": span.start_char,
                "end": span.end_char,
            }
        )

    return results
