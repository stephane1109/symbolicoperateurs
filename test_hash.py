from __future__ import annotations

import hash as hash_module


def test_punctuation_ignored_without_connectors():
    connectors = {"mais": "adversatif"}
    text = (
        "Le fait que tu en parles, que tu mettes des mots dessus, c’est déjà un pas important."
        " Il n'y a pas de connecteur logique explicite dans cet exemple."
    )

    segments = hash_module.split_segments_by_connectors(
        text, connectors, segmentation_mode="connecteurs_et_ponctuation"
    )

    assert [segment.strip() for segment in segments] == [text.strip()]


def test_punctuation_applied_when_connectors_present():
    connectors = {"mais": "adversatif", "pourtant": "adversatif"}
    text = "Il avance mais il hésite. Pourtant il continue"

    segments = hash_module.split_segments_by_connectors(
        text, connectors, segmentation_mode="connecteurs_et_ponctuation"
    )

    assert [segment.strip() for segment in segments] == [
        "Il avance",
        "il hésite",
        "il continue",
    ]


def test_punctuation_segments_must_touch_connectors():
    connectors = {"si": "condition"}
    text = "Bonjour. Ensuite si tu veux. Merci."

    segments = hash_module.split_segments_by_connectors(
        text, connectors, segmentation_mode="connecteurs_et_ponctuation"
    )

    assert [segment.strip() for segment in segments] == [
        "Ensuite",
        "tu veux",
    ]
