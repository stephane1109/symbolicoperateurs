# Calcul de l'écart-type dans l'onglet « hash »

## Étapes de segmentation
- Le texte est d'abord nettoyé des éventuelles lignes de métadonnées puis découpé en segments à l'aide d'une regex qui repère les connecteurs (et, selon le mode choisi, la ponctuation forte). Seuls les segments réellement bordés par un connecteur sont conservés. Chaque segment est ensuite tokenisé soit par une regex `\b\w+\b` (comportement de base), soit par le tokenizer spaCy du modèle `fr_core_news_md` pour ignorer automatiquement ponctuation et espaces.【F:hash.py†L18-L74】【F:hash.py†L202-L244】
- La Longueur Moyenne des Segments (LMS) correspond à la moyenne simple des longueurs en mots de ces segments. Elle est affichée avec l'écart-type dans l'onglet « hash ».【F:hash.py†L288-L317】

## Calcul statistique
- La liste des longueurs de segments est passée à `_mean_and_std`, qui la convertit en tableau NumPy de flottants puis renvoie la moyenne et l'écart-type via `np.mean` et `np.std` (écart-type de population). Pour une liste vide, la fonction retourne `(0.0, 0.0)`.【F:ecartype.py†L14-L29】
- Le calcul final de l'onglet « hash » appelle `compute_length_standard_deviation`, lequel orchestre l'extraction des longueurs et le calcul statistique en une seule étape pour fournir la paire (LMS, écart-type).【F:ecartype.py†L22-L29】
