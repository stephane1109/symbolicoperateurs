# N-grammes vs cooccurrences

Ce document résume les différences entre l'extraction d'**n-grammes** et l'analyse de **cooccurrences**, deux approches courantes pour modéliser la proximité entre mots.

## N-grammes
- Un n-gramme est une séquence contiguë de *n* unités (souvent des mots) dans un texte.
- Ils capturent l'ordre et la continuité : un trigramme (n=3) reflète par exemple trois mots successifs.
- L'extraction parcourt les phrases en conservant les frontières exactes ; chaque n-gramme est donc un « instantané » local.
- Les mesures classiques incluent la fréquence brute, la probabilité conditionnelle ou la vraisemblance (ex. modèles de Markov).

### Forces
- Très adaptés aux tâches sensibles à l'ordre (prédiction de texte, détection de collocations figées).
- Faciles à intégrer dans des modèles probabilistes ou des réseaux de neurones comme contraintes de contexte local.

### Limites
- Explosion combinatoire quand *n* augmente : plus de paramètres et données nécessaires.
- Sensibles aux variations superficielles (pluriels, flexions, ponctuation) si le prétraitement est minimal.
- Peu robustes aux réarrangements syntaxiques ou aux dépendances longues.

## Cooccurrences
- Les cooccurrences comptent combien de fois deux unités apparaissent dans un contexte partagé, sans exiger de contiguïté.
- Le contexte peut être une fenêtre glissante (ex. ±5 mots) ou une unité linguistique (phrase, paragraphe, document).
- Elles mesurent une association « lâche » : deux mots peuvent coapparaître même séparés par d'autres termes.
- Les scores courants incluent PMI, log-likelihood ratio, chi² ou t-score, qui pondèrent la force d'association.

### Forces
- Capturent des relations sémantiques ou thématiques au-delà de l'ordre strict (synonymie, champs lexicaux).
- Plus robustes aux permutations de mots et aux dépendances longues.
- Produisent des matrices mot×mot utilisables pour des embeddings distributionnels ou des analyses de graphes.

### Limites
- La définition de la fenêtre de contexte influence fortement les résultats (trop étroite : bruit, trop large : dilution).
- Perdent l'information d'ordre exact, ce qui peut gêner les tâches nécessitant une syntaxe précise.
- Les mots fréquents dominent si l'on ne normalise pas (d'où l'intérêt de PMI, etc.).

## Quand utiliser l'un ou l'autre ?
- **Tâches sensibles à la syntaxe ou aux expressions figées** : privilégier des n-grammes de longueur adaptée.
- **Exploration sémantique, clustering, recommandations ou similarité** : les cooccurrences offrent une vue plus souple des associations lexicales.
- Dans la pratique, on combine souvent les deux : les n-grammes pour la précision locale, les cooccurrences pour la structure globale du vocabulaire.
