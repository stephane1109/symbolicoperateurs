"""Tests autour du chargement et de la gestion des connecteurs."""

from pathlib import Path

import json

from analyses import load_connectors


def test_load_connectors_preserves_newline_entries(tmp_path: Path):
    connectors_file = tmp_path / "connecteurs.json"
    raw_connectors = {
        " si": "CONDITION",
        "\n": "RETOUR À LA LIGNE",
        "\r\n": "RETOUR À LA LIGNE",
        "": "IGNORED",
    }

    connectors_file.write_text(json.dumps(raw_connectors), encoding="utf-8")

    loaded = load_connectors(connectors_file)

    assert loaded == {
        "si": "CONDITION",
        "\n": "RETOUR À LA LIGNE",
        "\r\n": "RETOUR À LA LIGNE",
    }
