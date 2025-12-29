"""Onglet d'import et affichage des données importées."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from fcts_utils import build_dataframe, parse_iramuteq, render_connectors_reminder


def parse_upload(content: str) -> Tuple[List[dict], pd.DataFrame]:
    records = parse_iramuteq(content)
    return records, build_dataframe(records)


def rendu_donnees_importees(
    tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]
) -> None:
    st.subheader("Données importées")
    render_connectors_reminder(filtered_connectors)
    st.dataframe(df, use_container_width=True)
