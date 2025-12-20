"""Outils pour calculer la lisibilité d'un texte avec la formule Flesch-Kincaid."""

from __future__ import annotations

import re
from typing import Dict, List

VOWELS = "aeiouyàâäéèêëîïôöùûüÿœ"


def split_sentences(text: str) -> List[str]:
    """Découper un texte en phrases approximatives en se basant sur la ponctuation."""

    sentences = re.split(r"[.!?;:]+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> List[str]:
    """Extraire les mots du texte en ignorant la ponctuation."""

    return re.findall(r"[\w']+", text.lower())


def count_syllables_in_word(word: str) -> int:
    """Estimer le nombre de syllabes dans un mot en comptant les groupes de voyelles."""

    cleaned = re.sub(r"[^a-zàâäéèêëîïôöùûüÿœç-]", "", word.lower())
    if not cleaned:
        return 0

    vowel_groups = re.findall(rf"[{VOWELS}]+", cleaned)
    syllables = len(vowel_groups)

    if cleaned.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


def count_syllables(text: str) -> int:
    """Compter les syllabes dans un texte en agrégeant le total des mots."""

    return sum(count_syllables_in_word(word) for word in tokenize_words(text))


def compute_flesch_kincaid_metrics(text: str) -> Dict[str, float]:
    """Calculer les indicateurs de lisibilité Flesch-Kincaid pour un texte."""

    sentences = split_sentences(text)
    words = tokenize_words(text)

    sentence_count = max(len(sentences), 1)
    word_count = len(words)
    syllable_count = count_syllables(text)

    if word_count == 0:
        return {
            "sentences": 0,
            "words": 0,
            "syllables": 0,
            "reading_ease": 0.0,
            "grade_level": 0.0,
        }

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count if word_count else 0

    reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade_level = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

    return {
        "sentences": sentence_count,
        "words": word_count,
        "syllables": syllable_count,
        "reading_ease": reading_ease,
        "grade_level": grade_level,
    }


def interpret_reading_ease(score: float) -> str:
    """Fournir une interprétation qualitative du score de lisibilité."""

    if score >= 90:
        return "Très facile à lire (niveau primaire)."
    if score >= 70:
        return "Assez facile à lire (collège)."
    if score >= 50:
        return "Lisibilité moyenne (lycée)."
    if score >= 30:
        return "Difficile (études supérieures)."
    return "Très difficile (niveau académique avancé)."
