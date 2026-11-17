# Contrôle qualité des observations météorologiques

Prototype de plateforme pour importer des observations météo, repérer des
anomalies et préparer leur contrôle par les exploitants. La reconstruction
automatique et sa validation métier restent à développer.

## Démarrage

Depuis la racine du dépôt :

```bash
python -m pip install -r requirements.txt
streamlit run main.py
```

L'application utilise SQLite dans `data/meteo_sahelien.db`. Les lots importés
sont archivés dans leur forme reçue, tandis que `donnees_meteo` sert de vue
courante par station et date. Les modèles et
scalers sont écrits sous `models/saved/`, les rapports sous `reports/output/`.
Ces répertoires sont créés au moment de l'écriture, pas lors du simple import de
la configuration.

## Parcours actuellement couvert

1. Générer des séries synthétiques journalières ou importer un CSV.
2. Contrôler les observations par règles explicables et enregistrer la revue des alertes.
3. Entraîner l'ensemble Isolation Forest, One-Class SVM et Z-score par station.
4. Détecter et visualiser des observations suspectes.
5. Exporter un rapport CSV et un résumé textuel.

Les contrôles qualité repèrent les jours et valeurs manquants, les valeurs hors
bornes physiques configurées, les variations journalières atypiques par rapport
à la série de la station et les longues séquences identiques pouvant signaler
un capteur figé. Les alertes sont revues comme « À examiner », « Confirmée » ou
« Faux positif ». L'analyse ne corrige pas les observations ; les seuils restent
à valider par les spécialistes météo avant usage opérationnel.

L'entraînement sépare les périodes dans l'ordre temporel, calibre le seuil sur
la validation et réserve un test distinct. Si des labels sont présents, ils
servent à sélectionner le seuil sur validation et à mesurer le test ; les
anomalies labellisées sont exclues de l'ajustement des modèles. Sans labels, le
seuil suit le taux de contamination configuré. Chaque station conserve ses
versions de modèle et son pipeline de features. Les scores restent à évaluer
sur des observations de référence avant usage climatologique. Aucune
reconstruction ni validation d'une correction par un expert n'est encore
proposée.

## Contrat d'import actuel

Le CSV doit contenir `date` et au moins une variable reconnue parmi :
`temperature`, `humidite`, `pression`, `pluie`, `vent`, `rayonnement`.
Les colonnes météo absentes sont conservées comme manquantes. Les colonnes
anglaises usuelles (`humidity`, `pressure`, `precipitation`, `wind_speed`,
`solar_radiation`, `timestamp`, `station_id`) sont normalisées à l'import.

Les colonnes de référence facultatives `est_anomalie` et `type_ref` servent aux
données étiquetées. La station peut être fournie par l'interface ou par une
colonne `station`. Les dates illisibles, mesures non numériques et dates
dupliquées dans un même fichier bloquent l'import ; les valeurs inhabituelles
restent importables pour être examinées par les contrôles qualité.

Les six unités configurées sont °C, %, hPa, mm, m/s et MJ/m². La définition
temporelle exacte des mesures (par exemple température minimale/maximale,
observation instantanée ou cumul de pluie) doit être précisée par source avant
toute utilisation métier. Le format actuel traite les dates comme des jours
calendaires sans fuseau horaire.

## Architecture actuellement utilisée

- `main.py` : interface Streamlit.
- `config.py` : variables, seuils, stations et chemins.
- `data/contract.py` : schéma canonique et validation d'import.
- `data/synthetic_generator.py` : séries synthétiques de démonstration.
- `database/db.py` : SQLite et migration additive du schéma d'observations.
- `preprocessing/pipeline.py` : features temporelles et mise à l'échelle.
- `models/ensemble_detector.py` : détecteur expérimental.
- `evaluation/benchmark.py` : défauts reproductibles, découpage temporel et métriques.
- `visualization/plots.py`, `reports/reporter.py` : restitution.

`data/generator.py`, `detection/preprocessor.py` et `database/db_manager.py`
sont des implémentations historiques non utilisées par le parcours Streamlit.
Leur retrait ou leur réintégration sera décidé après stabilisation du flux
principal.

## Dépendances

Les dépendances Python directes sont listées dans `requirements.txt`. Les
versions ne sont pas encore verrouillées ; un fichier de verrouillage sera
nécessaire pour préparer un déploiement reproductible.
