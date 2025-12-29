"""Onglet TF-IDF."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from tf_idf import render_tfidf_tab


def rendu_tfidf(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    render_tfidf_tab(tab, df, filtered_connectors)
