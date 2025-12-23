import pandas as pd

from ngram import compute_ngram_statistics


def test_compute_specific_trigrams_topk_and_modalities():
    df = pd.DataFrame(
        {
            "texte": [
                "alpha beta gamma alpha beta gamma",
                "alpha beta delta",
            ],
            "groupe": ["A", "B"],
        }
    )

    result = compute_ngram_statistics(
        df,
        min_n=2,
        max_n=4,
        top_k=2,
        specific_n=3,
        top_modalities=2,
    )

    assert not result.empty
    assert set(result["Taille"]) == {3}
    # the trigram "alpha beta gamma" should appear twice across group A
    first_row = result.iloc[0]
    assert first_row["N-gram"] == "alpha beta gamma"
    assert first_row["Fréquence"] == 2
    assert "groupe=A" in first_row["Modalités associées"]
