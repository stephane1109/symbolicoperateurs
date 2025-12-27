"""Outils pour réaliser automatiquement un test non paramétrique sur des
tableaux/données déjà chargés en mémoire.

Le module ne dépend plus d'un CSV en entrée : vous lui transmettez directement
vos :class:`pandas.DataFrame` (onglets LMS, densité, etc.) et il construit le
tableau de contingence adéquat avant de choisir le test à appliquer :

- **Friedman** pour mesurer l'effet d'une consigne/prompt (conditions
  répétées sur les mêmes modèles) ;
- **Kruskal–Wallis** pour mesurer l'effet "modèle" (plusieurs groupes
  indépendants, un par modèle).

L'utilisateur fournit simplement les noms des colonnes à croiser (modèle,
condition et métrique numérique). Le reste est automatisé : tableau croisé,
filtrage des cas complets, choix du test, affichage des résultats.

Données nécessaires
-------------------
Le test de Friedman compare une mesure numérique observée pour chaque
combinaison d'un *sujet* (ici le modèle) et d'une *condition* (la consigne/
prompt). Il faut donc fournir :

- une colonne identifiant le modèle (ou système) évalué ;
- une colonne identifiant la consigne / le prompt (ne pas la nommer « prompt »
  si vous préférez, via ``--colonne-consigne``) ;
- une colonne numérique contenant la métrique à analyser (densité, score LMS,
  etc.).

Chaque ligne représente une observation élémentaire : le score obtenu par un
modèle pour une consigne sur une instance donnée. Le test exige au moins deux
modèles et trois consignes, avec une observation pour chaque combinaison (la
fonction :func:`filtrer_cas_complets` élimine les paires incomplètes).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal


TableauCroise = pd.DataFrame


def charger_donnees(chemin: Path, separateur: str = ",", decimal: str = ".") -> pd.DataFrame:
    """Option utilitaire : charger un CSV si vous souhaitez tout de même le faire.

    L'API principale fonctionne sur des DataFrames déjà en mémoire ; cette
    fonction ne sert qu'à faciliter un usage en ligne de commande.
    """

    return pd.read_csv(chemin, sep=separateur, decimal=decimal)


def construire_tableau_croise(
    donnees: pd.DataFrame,
    colonne_modele: str,
    colonne_consigne: str,
    colonne_score: str,
    fonction_agregation: str | callable = "mean",
) -> TableauCroise:
    """Construire un tableau croisé ``modèle x consigne``.

    Les valeurs sont agrégées par défaut par la moyenne, mais l'utilisateur peut
    fournir n'importe quelle fonction d'agrégation compatible avec
    :func:`pandas.DataFrame.pivot_table`.
    """

    tableau = (
        donnees.pivot_table(
            index=colonne_modele,
            columns=colonne_consigne,
            values=colonne_score,
            aggfunc=fonction_agregation,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    return tableau


def verifier_score_numerique(donnees: pd.DataFrame, colonne_score: str) -> pd.DataFrame:
    """S'assurer que la colonne de score est numérique.

    Cette conversion explicite évite les erreurs silencieuses lorsqu'on
    utilise des colonnes importées comme chaînes (par exemple si le CSV
    contient des virgules décimales ou des espaces). Une erreur explicite
    explique à l'utilisateur quelles données sont attendues pour appliquer le
    test de Friedman.
    """

    converti = donnees.copy()
    converti[colonne_score] = pd.to_numeric(converti[colonne_score], errors="coerce")
    if converti[colonne_score].isna().any():
        raise ValueError(
            "La métrique fournie doit être numérique (ex. densité, LMS). "
            "Vérifiez le séparateur décimal ou le nom de la colonne, puis réessayez."
        )
    return converti


def filtrer_cas_complets(tableau: TableauCroise) -> TableauCroise:
    """Conserver uniquement les lignes sans valeurs manquantes."""

    return tableau.dropna(axis=0, how="any")


def construire_tableau_pour_effet(
    donnees: pd.DataFrame,
    effet: str,
    colonne_modele: str,
    colonne_consigne: str,
    colonne_score: str,
    fonction_agregation: str | callable = "mean",
) -> TableauCroise:
    """Construire automatiquement le tableau croisé adapté à l'effet testé.

    - effet="consigne"  -> tableau « modèle × consigne » (pour Friedman)
    - effet="modele"    -> tableau « consigne × modèle » (pour Kruskal–Wallis)
    """

    if effet == "consigne":
        return construire_tableau_croise(
            donnees,
            colonne_modele=colonne_modele,
            colonne_consigne=colonne_consigne,
            colonne_score=colonne_score,
            fonction_agregation=fonction_agregation,
        )

    if effet == "modele":
        return construire_tableau_croise(
            donnees,
            colonne_modele=colonne_consigne,
            colonne_consigne=colonne_modele,
            colonne_score=colonne_score,
            fonction_agregation=fonction_agregation,
        )

    raise ValueError("L'effet doit être 'consigne' ou 'modele'.")


def appliquer_friedman(tableau: TableauCroise) -> Tuple[float, float]:
    """Appliquer le test de Friedman sur un tableau croisé complet.

    Le tableau doit contenir au moins trois colonnes (consignes) et deux lignes
    (modèles). Chaque colonne est interprétée comme une condition et chaque
    ligne comme un sujet.
    """

    if tableau.shape[1] < 3:
        raise ValueError(
            "Le test de Friedman nécessite au moins trois consignes (colonnes)."
        )

    if tableau.shape[0] < 2:
        raise ValueError(
            "Le tableau doit contenir au moins deux systèmes pour comparer les consignes."
        )

    colonnes = tableau.columns.tolist()
    series_conditions: Iterable[np.ndarray] = [tableau[col].to_numpy() for col in colonnes]

    resultat = friedmanchisquare(*series_conditions)
    return float(resultat.statistic), float(resultat.pvalue)


def appliquer_kruskal(tableau: TableauCroise) -> Tuple[float, float]:
    """Appliquer le test de Kruskal–Wallis pour comparer les modèles.

    Les colonnes du tableau représentent les modèles (groupes indépendants) et
    les lignes les consignes/items. Chaque colonne doit contenir au moins une
    valeur pour pouvoir estimer l'effet du modèle.
    """

    groupes = [tableau[col].dropna().to_numpy() for col in tableau.columns]
    if len(groupes) < 2:
        raise ValueError(
            "Le test de Kruskal–Wallis nécessite au moins deux modèles (colonnes)."
        )

    resultat = kruskal(*groupes)
    return float(resultat.statistic), float(resultat.pvalue)


def afficher_tableau(tableau: TableauCroise) -> None:
    """Afficher le tableau croisé sous forme Markdown pour faciliter la lecture."""

    markdown = tableau.to_markdown(tablefmt="github", floatfmt=".3f")
    print("\nTableau croisé modèle x consigne :\n")
    print(markdown)


def afficher_resultats_friedman(statistique: float, pvalue: float) -> None:
    """Afficher de façon concise les résultats du test de Friedman."""

    print("\nRésultats du test de Friedman :")
    print(f"- Statistique : {statistique:.4f}")
    print(f"- p-value     : {pvalue:.6f}")


def afficher_resultats_kruskal(statistique: float, pvalue: float) -> None:
    """Afficher les résultats du test de Kruskal–Wallis."""

    print("\nRésultats du test de Kruskal–Wallis :")
    print(f"- Statistique : {statistique:.4f}")
    print(f"- p-value     : {pvalue:.6f}")


def analyser_effet(
    donnees: pd.DataFrame,
    effet: str,
    colonne_modele: str,
    colonne_consigne: str,
    colonne_score: str,
    fonction_agregation: str | callable = "mean",
) -> tuple[str, TableauCroise, float, float]:
    """Construire le tableau croisé et appliquer le test adapté.

    Parameters
    ----------
    effet:
        "consigne" pour tester l'effet du prompt (Friedman) ou "modele" pour
        tester l'effet du modèle (Kruskal–Wallis).
    """

    donnees = verifier_score_numerique(donnees, colonne_score)
    tableau = construire_tableau_pour_effet(
        donnees,
        effet=effet,
        colonne_modele=colonne_modele,
        colonne_consigne=colonne_consigne,
        colonne_score=colonne_score,
        fonction_agregation=fonction_agregation,
    )

    if effet == "consigne":
        tableau_complet = filtrer_cas_complets(tableau)
        if tableau_complet.empty:
            raise ValueError(
                "Aucune combinaison complète modèle/consigne trouvée après filtrage des valeurs manquantes."
            )

        statistique, pvalue = appliquer_friedman(tableau_complet)
        return "friedman", tableau_complet, statistique, pvalue

    tableau_filtre = tableau.dropna(axis=0, how="all")
    statistique, pvalue = appliquer_kruskal(tableau_filtre)
    return "kruskal-wallis", tableau_filtre, statistique, pvalue



def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyser l'effet d'une variable (consigne ou modèle) sur une métrique "
            "en construisant automatiquement le tableau croisé puis en choisissant "
            "le test adapté (Friedman ou Kruskal–Wallis)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Chemin du fichier CSV (optionnel). Si absent, importez le module et "
            "appelez analyser_effet avec vos DataFrames en mémoire."
        ),
    )
    parser.add_argument(
        "--effet",
        choices=["consigne", "modele"],
        default="consigne",
        help="Variable dont on teste l'effet : consigne/prompt ou modèle.",
    )
    parser.add_argument(
        "--colonne-modele",
        default="modele",
        help="Nom de la colonne représentant le modèle évalué.",
    )
    parser.add_argument(
        "--colonne-consigne",
        default="consigne",
        help="Nom de la colonne représentant la consigne (variations testées).",
    )
    parser.add_argument(
        "--colonne-score",
        default="score",
        help="Nom de la colonne contenant la mesure à comparer.",
    )
    parser.add_argument(
        "--separateur",
        default=",",
        help="Séparateur utilisé dans le fichier CSV (par défaut: ',').",
    )
    parser.add_argument(
        "--decimal",
        default=".",
        help="Symbole de décimale utilisé dans le fichier (par défaut: '.').",
    )
    parser.add_argument(
        "--aggregation",
        default="mean",
        help=(
            "Fonction d'agrégation pour construire le tableau croisé. "
            "Exemples: mean, median, max, min."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parser_arguments()
    if args.csv is None:
        raise SystemExit(
            "Aucun CSV fourni. Importez le module et appelez analyser_effet() avec vos DataFrames en mémoire."
        )

    donnees = charger_donnees(args.csv, separateur=args.separateur, decimal=args.decimal)
    nom_test, tableau, statistique, pvalue = analyser_effet(
        donnees,
        effet=args.effet,
        colonne_modele=args.colonne_modele,
        colonne_consigne=args.colonne_consigne,
        colonne_score=args.colonne_score,
        fonction_agregation=args.aggregation,
    )

    afficher_tableau(tableau)
    if nom_test == "friedman":
        afficher_resultats_friedman(statistique, pvalue)
    else:
        afficher_resultats_kruskal(statistique, pvalue)


if __name__ == "__main__":
    main()
