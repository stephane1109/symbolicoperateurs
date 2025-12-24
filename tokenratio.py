"""Diversité Lexicale : Le TTR (Type-Token Ratio)

Cet onglet propose un indicateur simple pour mesurer la richesse du vocabulaire
: le *Type-Token Ratio*. Le TTR divise le nombre de mots uniques (types) par le
nombre total de mots (tokens) dans un texte.

Méthode
-------
TTR = Nombre de mots uniques (vocabulaire) / Nombre total de mots du texte

Exemple
-------
Si un texte de 100 mots n'utilise que 10 mots différents répétés 10 fois,
le TTR est de 0,10 (très pauvre/répétitif).
"""

from __future__ import annotations

import re
from typing import Iterable, List


_WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_words(text: str) -> List[str]:
    """Découper un texte en une liste de mots normalisés.

    - Les mots vides ou absents renvoient une liste vide.
    - La casse est neutralisée pour que ``Chat`` et ``chat`` soient comptés comme
      le même type.
    """

    if not text:
        return []

    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]


def compute_ttr(text: str) -> float:
    """Calculer le Type-Token Ratio (TTR) d'un texte.

    La valeur renvoyée est comprise entre 0 et 1. Un résultat de 0 signifie un
    texte vide ou sans mots détectés. Un résultat plus élevé indique une plus
    grande diversité lexicale.
    """

    tokens = tokenize_words(text)

    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)


def compute_ttr_for_corpus(texts: Iterable[str]) -> float:
    """Calculer le TTR global pour un ensemble de textes.

    Cette fonction fusionne tous les textes fournis avant de mesurer la
    diversité lexicale. Elle est pratique lorsqu'on souhaite évaluer un corpus
    complet plutôt que des documents isolés.
    """

    all_tokens: List[str] = []

    for text in texts:
        all_tokens.extend(tokenize_words(text))

    if not all_tokens:
        return 0.0

    unique_tokens = set(all_tokens)
    return len(unique_tokens) / len(all_tokens)
