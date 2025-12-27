import pytest

pytest.importorskip("spacy")
from pattern import _build_token_patterns


def test_pos_hint_applied_to_conjunction():
    patterns = _build_token_patterns([(["si"], False)])
    assert patterns == [[{"LOWER": "si", "POS": "SCONJ"}]]


def test_pos_hint_preserved_with_optionality():
    patterns = _build_token_patterns([(["si"], True)])
    # Optional connector should keep the POS hint in the generated pattern.
    assert {"LOWER": "si", "POS": "SCONJ"} in patterns[0]
