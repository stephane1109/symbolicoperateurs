# Normalisation de la densité (onglet « Densité »)

L’onglet « Densité » présente la fréquence des connecteurs dans le corpus pour chaque variable et modalité choisie, ramenée à une base commune de 1 000 mots. Cette normalisation permet de comparer des sous-corpus de tailles différentes en évitant que les textes plus longs ou plus courts ne biaisent la lecture des résultats.

## Étapes du calcul

1. **Sélection du sous-corpus** :
   - Le DataFrame est éventuellement filtré pour ne retenir que les modalités demandées (`filter_dataframe_by_modalities`).
   - Pour chaque modalité, les champs `entete` et `texte` des lignes associées sont concaténés afin de former un texte unique (`build_text_from_dataframe`).

2. **Comptage des unités** :
   - Les mots sont comptés via une expression régulière qui retient toutes les suites alphanumériques (`count_words`).
   - Les connecteurs sont repérés à l’aide d’un motif regex sécurisé, construit à partir du dictionnaire de connecteurs fourni. Chaque occurrence correspondante est comptée (`compute_total_connectors`).

3. **Calcul de la densité globale** :
   - La densité correspond au ratio `connecteurs / mots`, multiplié par une base (1 000 par défaut). Formellement, `densité = (total_connecteurs / total_mots) * 1000` (`compute_density`).
   - Si le texte contient zéro mot ou zéro connecteur, la densité renvoyée est 0 afin d’éviter les divisions par zéro et de signaler l’absence de données exploitables.

4. **Calcul par modalité** :
   - Pour chaque modalité, la densité, le nombre total de mots et le nombre total de connecteurs sont calculés et restitués dans un tableau (`compute_density_per_modality`).

5. **Calcul par label de connecteur** (si appliqué) :
   - Lorsqu’un connecteur est associé à un label (catégorie), la densité peut être ventilée par label en reprenant la même formule de normalisation sur 1 000 mots pour chaque groupe (`compute_density_by_label`, `compute_density_per_modality_by_label`).

## Pourquoi une base 1 000 ?

Normaliser sur 1 000 mots fournit une mesure intuitive « pour mille » qui facilite la comparaison entre sous-corpus de tailles différentes. Cette base est paramétrable, mais l’interface la fixe à 1 000 pour conserver une échelle cohérente entre les tableaux et les graphiques de l’onglet.

## Résultat affiché dans l’onglet

L’onglet « Densité » affiche donc, pour chaque modalité (et éventuellement pour chaque label de connecteur) :

- la densité normalisée sur 1 000 mots ;
- le nombre total de mots du sous-corpus ;
- le nombre total de connecteurs trouvés.

Ces valeurs permettent de comparer directement les modalités, même lorsque leur volume textuel diffère fortement, en s’appuyant sur une métrique homogène et normalisée.
