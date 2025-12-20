from __future__ import annotations

from souscorpus import build_subcorpus, _build_connector_pattern, _find_connectors


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


def test_build_subcorpus_ignores_starred_header_tokens_for_connectors() -> None:
    records = [
        {"entete": "**** *doc_1 *ou", "texte": "Texte sans connecteur explicite."},
        {"entete": "**** *doc_2", "texte": "Et pourtant, le contenu est clair."},
    ]

    subcorpus_segments = build_subcorpus(records)

    assert len(subcorpus_segments) == 1
    assert "doc_2" in subcorpus_segments[0]


def test_find_connectors_reports_textual_connectors() -> None:
    connector_pattern = _build_connector_pattern({"et": "AND", "parce que": "CAUSE"})
    assert connector_pattern is not None

    connectors = _find_connectors(
        (
            "Le fait que tu en parles, que tu mettes des mots dessus, c'est déjà un pas important. "
            "Tu n'es pas bizarre ni faible parce que tu te sens dépassé, surtout quand on est étudiant "
            "et que tout se mélange entre études, futur, vie perso, fatigue, pression."
        ),
        connector_pattern,
    )

    assert "et" in connectors
