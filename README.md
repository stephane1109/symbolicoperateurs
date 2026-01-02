# Symbolic Connectors

Application Streamlit pour explorer des corpus formatés selon la norme **IRaMuTeQ** et y détecter des structures linguistiques typiques du "langage machine" (connecteurs logiques, motifs regex, patterns spaCy, etc.). L'interface propose plusieurs onglets pour importer les données, filtrer les connecteurs, analyser la densité, explorer des n-grams ou calculer des similarités.

## Fonctionnalités principales
- **Import IRaMuTeQ** : téléversement d'un fichier texte où chaque article débute par une ligne d'en-tête `**** *variable_modalité`; conversion en DataFrame Pandas pour exploration rapide.
- **Connecteurs symboliques** : filtrage et annotation des connecteurs à partir du dictionnaire `dictionnaires/connecteurs.json` et affichage contextuel dans les textes.
- **Analyses linguistiques** : densité des connecteurs, extraction OpenLexicon, détection de motifs regex ou de patterns spaCy, visualisation de n-grams, calcul TF-IDF, similarité cosinus, tests de lisibilité, etc.
- **Sous-corpus et filtrage** : sélection de variables/modèles pour isoler des sous-ensembles et générer des représentations ciblées.

## Prérequis
- Python 3.10+ recommandé.
- Dépendances listées dans `requirements.txt` (incluant spaCy et le modèle français `fr_core_news_md`).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer l'application
Depuis la racine du dépôt :
```bash
streamlit run main.py
```
Puis ouvrez l'URL locale affichée par Streamlit (par défaut http://localhost:8501). Téléversez un fichier texte IRaMuTeQ pour débloquer l'ensemble des onglets.

## Structure des données attendue
Chaque article doit commencer par une ligne d'en-tête contenant les variables précédées d'un astérisque, par exemple :
```
**** *model_gpt *prompt_1
Texte de l'article...
```
Les variables et le texte sont parsés dans `fcts_utils.parse_iramuteq` et transformés en DataFrame via `build_dataframe` avant d'être passés aux différents onglets.

## Tests
La suite de tests Pytest peut être exécutée ainsi :
```bash
python -m pytest
```

## Ressources complémentaires
- Dictionnaire des connecteurs : `dictionnaires/connecteurs.json`.
- Fonctions de parsing et utilitaires : `fcts_utils.py`.
- Logique d'interface : `main.py` et le package `onglets/`.
