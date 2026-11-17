# Protocole d'évaluation

Ce paquet sépare la vérité terrain des observations fournies à un détecteur.
Il n'entraîne aucun modèle. Il prépare des perturbations reproductibles et
mesure les alertes ou les estimations de reconstruction sur une référence
connue.

## Jeu de référence requis

`make_benchmark` attend une série **journalière, complète et validée** avec les
colonnes `station`, `date` et les six variables météo canoniques. Les dates
doivent être uniques par station et toutes les valeurs de référence doivent
être connues. Ce contrat strict est nécessaire pour savoir quelles valeurs ont
été modifiées et pour compter les vrais négatifs. Une observation terrain
réellement manquante ne peut pas servir de valeur de vérité pour la
reconstruction.

```python
from evaluation.benchmark import make_benchmark, evaluate_detection
from quality.control import run_quality_checks

benchmark = make_benchmark(reference_df, seed=2026)
alerts = run_quality_checks(benchmark.observations)
scores = evaluate_detection(benchmark, alerts, level="cell")
```

`benchmark.observations` est la seule entrée à fournir au contrôle ou au
modèle. `benchmark.truth` et `benchmark.reference` sont réservées au calcul des
scores et ne doivent pas entrer dans les features. Les types de perturbation
incluent les cellules manquantes, plages manquantes, journées retirées, pics,
capteurs figés et dérives progressives. La graine fixe rend les mêmes défauts
reproductibles pour la même référence et la même version du code.

## Indicateurs

La détection fournit précision, rappel, F1, spécificité et exactitude équilibrée
au niveau global, par variable, station et saison indicative. Le rappel est
aussi ventilé par famille de défaut. Utiliser `level="cell"` pour des règles
qui indiquent la variable en cause ; utiliser `level="row"` pour un modèle qui
ne signale qu'une journée entière. Ne pas comparer ces deux niveaux comme s'ils
avaient la même unité d'évaluation.

La reconstruction est comparée uniquement aux cellules perturbées dont la
valeur originale est connue. Les indicateurs incluent MAE, RMSE, biais moyen et,
si la méthode fournit une incertitude, couverture et largeur des intervalles.
Les résultats sont ventilés par variable, station, type de défaut, saison et
longueur de lacune.

`chronological_split` sépare l'entraînement, une période d'embargo et le test.
La durée d'embargo doit couvrir au moins la plus longue fenêtre de features ou
de contexte utilisée par la méthode évaluée. Toute calibration de seuil doit
être faite sur une période de validation distincte du test final.

L'entraînement depuis le tableau de bord utilise `model_selection_split` pour
créer des périodes chronologiques d'entraînement, validation et test. Les
anomalies explicitement étiquetées ne participent pas à l'ajustement du modèle.
Le seuil est choisi sur les labels de validation quand ils sont suffisamment
nombreux ; sinon, un quantile correspondant au taux de contamination sert de
solution de repli. Les métriques du test sont enregistrées avec la version du
modèle. Les modèles et scalers sont isolés par station et stockés dans un bundle
versionné.

## Référence terrain du service météo

Le fichier `reference_labels_template.csv` est un modèle de travail pour
consigner les labels et décisions expertes. Pour chaque valeur candidate,
conserver la mesure originale, la station, la date, la variable, le type de
problème, l'identifiant du réviseur et un commentaire. Les désaccords entre
réviseurs doivent être arbitrés et conservés dans l'historique.

## Seuils d'acceptation

Aucun seuil universel de précision, rappel ou erreur de reconstruction n'est
fixé dans le code. Le service doit choisir les seuils par variable, type de
problème, saison et coût opérationnel d'une fausse alerte ou d'une mauvaise
correction. Comparer au minimum les règles qualité, une méthode simple de
reconstruction et la méthode proposée, sur un test temporel tenu à l'écart.

Les regroupements saisonniers fournis (`saison_seche_fraiche`,
`saison_chaude`, `hivernage`, `transition`) sont uniquement des découpages de
rapport. Leur calendrier doit être adapté aux stations et aux zones climatiques
du Tchad avant interprétation.
