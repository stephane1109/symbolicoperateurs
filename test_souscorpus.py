from __future__ import annotations

from souscorpus import build_subcorpus


def test_build_subcorpus_filters_non_prompt_records() -> None:
    records = [
        {"entete": "**** *doc_1", "texte": "Si ceci arrive, alors cela."},
        {"entete": "**** *model_gpt *prompt_2", "texte": "Sinon, nous attendons."},
    ]

    subcorpus_segments = build_subcorpus(records)

    assert len(subcorpus_segments) == 1
    assert subcorpus_segments[0].startswith("**** *model_gpt *prompt_2")


def test_build_subcorpus_rebuilds_prompt_with_connector_sentences() -> None:
    records = [
        {
            "entete": "**** *model_gpt *prompt_1",
            "texte": "Phrase isolée. Si le système fonctionne, alors tout se passe bien.",
        }
    ]

    subcorpus_segments = build_subcorpus(records)

    assert len(subcorpus_segments) == 1
    assert "Phrase isolée" not in subcorpus_segments[0]
    assert "Si le système fonctionne" in subcorpus_segments[0]
    assert subcorpus_segments[0].startswith("**** *model_gpt *prompt_1\n")


def test_build_subcorpus_keeps_multiple_connector_segments() -> None:
    records = [
        {
            "entete": "**** *model_gpt *prompt_3",
            "texte": "Si tu viens, alors nous partirons. Nous préparons le matériel. Donc tout ira bien.",
        }
    ]

    subcorpus_segments = build_subcorpus(records)

    assert len(subcorpus_segments) == 1
    assert "Si tu viens" in subcorpus_segments[0]
    assert "Donc tout ira bien" in subcorpus_segments[0]
    assert "Nous préparons ensuite" not in subcorpus_segments[0]
