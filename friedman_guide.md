# Guide rapide : tester automatiquement l'effet « consigne » ou « modèle »

Le module `friedman.py` s'utilise désormais directement sur vos DataFrames
chargés en mémoire (onglets LMS, densité, etc.). Il construit le tableau croisé
pertinent puis applique **le test non paramétrique adapté automatiquement** :

- **Friedman** pour l'effet de la consigne/prompt sur un même ensemble de
  modèles (conditions répétées) ;
- **Kruskal–Wallis** pour l'effet "modèle" quand on veut comparer plusieurs
  modèles entre eux (groupes indépendants).

## Quelles métriques fournir ?
Les deux tests attendent une **métrique numérique** observée dans vos onglets
pour chaque combinaison `(modèle, consigne)` (Friedman) ou par modèle (Kruskal–
Wallis). Vous pouvez utiliser, par exemple :

- une densité ou probabilité estimée (onglet « densité ») ;
- un score de similarité ou de compréhension (onglet « LMS ») ;
- toute autre mesure quantitative comparable entre consignes.

Chaque ligne doit représenter une observation élémentaire (un item) avec au
moins :

- `modele` : le nom ou identifiant du modèle évalué ;
- `consigne` : la consigne/prompt appliquée ;
- `score` : la métrique numérique à analyser.

Le test de Friedman nécessite **au moins deux modèles** et **au moins trois
consignes**, avec une valeur présente pour chaque combinaison modèle × consigne
(les lignes incomplètes sont éliminées automatiquement). Le test de
Kruskal–Wallis nécessite **au moins deux modèles** avec au moins un score chacun.

## Utilisation sur un DataFrame existant
```python
from friedman import analyser_effet

# df est votre DataFrame déjà chargé (onglet LMS, densité, etc.)
nom_test, tableau, statistique, pvalue = analyser_effet(
    df,
    effet="consigne",         # ou "modele" pour Kruskal–Wallis
    colonne_modele="modele",
    colonne_consigne="consigne",
    colonne_score="score",    # votre métrique numérique
    fonction_agregation="mean",  # moyenne par défaut
)

print(tableau)
print(nom_test, statistique, pvalue)
```

## Usage en ligne de commande (optionnel)
Si vous voulez tout de même partir d'un CSV :

```bash
python friedman.py --csv evaluations.csv --effet consigne \
  --colonne-modele modele --colonne-consigne consigne --colonne-score score
```

Le script affichera le tableau croisé puis les résultats du test choisi
automatiquement.
