"""Interface d'import des fonctions liées aux onglets Streamlit."""
from .onglets import (
    parse_upload,
    rendu_connecteurs,
    rendu_densite,
    rendu_donnees_brutes,
    rendu_donnees_importees,
    rendu_hash,
    rendu_lisibilite,
    rendu_ngram,
    rendu_openlexicon,
    rendu_patterns,
    rendu_regex_motifs,
    rendu_simi_cosinus,
    rendu_sous_corpus,
    rendu_tfidf,
)

__all__ = [
    "parse_upload",
    "rendu_connecteurs",
    "rendu_densite",
    "rendu_donnees_brutes",
    "rendu_donnees_importees",
    "rendu_hash",
    "rendu_lisibilite",
    "rendu_ngram",
    "rendu_openlexicon",
    "rendu_patterns",
    "rendu_regex_motifs",
    "rendu_simi_cosinus",
    "rendu_sous_corpus",
    "rendu_tfidf",
]
