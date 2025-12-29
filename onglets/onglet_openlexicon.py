"""Onglet OpenLexicon."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from fcts_utils import render_connectors_reminder
from lexiconnorm import render_lexicon_norm_tab


def rendu_openlexicon(tab, df: pd.DataFrame, filtered_connectors: Dict[str, str]) -> None:
    render_connectors_reminder(filtered_connectors)
    render_lexicon_norm_tab(df, filtered_connectors)
