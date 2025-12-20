from __future__ import annotations

from souscorpus import build_subcorpus


def test_build_subcorpus_filters_segments_without_connectors() -> None:
    records = [
        {"entete": "**** *doc_1", "texte": "Texte sans connecteur explicite."},
        {
            "entete": "**** *doc_2",
            "texte": "Si le système fonctionne, alors tout se passe bien.",
        },
    ]

    subcorpus_segments = build_subcorpus(records)

    assert len(subcorpus_segments) == 1
    assert "doc_2" in subcorpus_segments[0]
