import pandas as pd

from afc import build_connector_matrix

def test_build_connector_matrix_removes_empty_rows():
    df = pd.DataFrame(
        {
            "texte": ["", "Le connecteur apparait"],
            "marker": ["A", "B"],
        }
    )
    connectors = {"connecteur": "label"}

    matrix = build_connector_matrix(df, connectors)

    assert matrix.shape == (1, 1)
    assert matrix.index.tolist() == [1]
    assert matrix.iloc[0, 0] == 1
