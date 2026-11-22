**ML 1 — contrat d’intégration provisoire, version 1 — 15 novembre 2026**

ML 1 fournit une probabilité de crise de SPY pour l’optimiseur du membre 4 et les travaux de backtesting du membre 5. Il ne calcule aucune allocation, stratégie ou recommandation d’investissement.

**Sens de la sortie**

`p_crisis` est un Python `float` fini dans `[0,1]` : probabilité prédite qu’au moins une des 21 prochaines clôtures SPY ajustées soit **strictement inférieure à 93 %** de la clôture ajustée à la date de décision. Ce n’est ni un rendement attendu, ni une probabilité en pourcentage, ni une garantie.

Politique fixée par instruction utilisateur : `model=logistic_regression`, `feature_variant=core`, `calibration=none`, `status=provisional`, `horizon=21_spy_sessions`. Aucun Platt, Isotonic, VIX ou macro. La baseline historique n’est pas une calibration implicite. Les probabilités brutes ne sont pas certifiées calibrées ; l’étape 6 a sélectionné seulement un candidat provisoire, dont le Brier était moins bon que la baseline historique.

**Préparation et dépendances**

Utiliser l’environnement `.venv` Python 3.12 existant. Les versions exactes de NumPy, pandas, pyarrow, scikit-learn, joblib et exchange-calendars sont inscrites et vérifiées dans `model_manifest.json`. Ne pas modifier les versions ou bibliothèques globales pour charger un artefact ancien ; reconstruire explicitement après une migration autorisée. Aucune connexion réseau nécessaire.

Les données brutes partagées restent en lecture seule : `src/data/financial_dataset/raw/SPY_raw.csv` et `IEF_raw.csv`. Les sorties ML 1 sont ignorées par Git et doivent être régénérées ou transmises séparément par une source de confiance.

Pour un nouveau clone où les sorties core sont absentes :

```bash
.venv/bin/python -B -m src.models.ml1.pipeline
```

Si le dossier existe déjà, choisir un dossier neuf sans écraser l’ancien :

```bash
.venv/bin/python -B -m src.models.ml1.pipeline --output-dir data/processed/ml1/rebuild_001
```

Le chargeur de développement impose les dix colonnes core ordonnées et les labels dont la date et la fin de cible sont antérieures à 2023. L’artefact courant est prévu pour 4 329 observations, dernière décision train le 30/11/2022, dernière fin de cible le 30/12/2022. Un contrat de données différent doit être revu.

**Prédictions OOF requises**

La reconstruction complète nécessite aussi `data/processed/ml1_training/ml_models_oof_predictions.parquet` et `training_metadata.json`. Ces fichiers ignorés par Git proviennent de l’étape 5. Leur schéma, leurs empreintes et leurs huit folds sont contrôlés sans réentraîner les modèles.

Si ces fichiers sont absents, les récupérer depuis une sauvegarde de confiance. Leur régénération par la commande historique de l’étape 5 constitue une opération distincte de 24 entraînements, à autoriser séparément ; elle n’a pas été exécutée pendant cette livraison. Après autorisation seulement, la commande existante est :

```bash
.venv/bin/python -B -m src.models.ml1.train_ml1
```

Cette commande refuse son dossier par défaut s’il existe déjà. Pour une reconstruction dans d’autres dossiers, utiliser ses options `--dataset-dir` et `--output-dir`, puis passer les chemins correspondants au constructeur d’inférence. Un artefact OOF peut être reproductible numériquement sans reproduire octet par octet un ancien JSON d’environnement ; la nouvelle provenance sera conservée.

**Construction du modèle figé et des exports**

Sur `feature/ml1-crisis-detection`, avec les sources actuelles :

```bash
.venv/bin/python -B -m src.models.ml1.inference
```

Un seul fit figé est réalisé : `StandardScaler` + `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)` sur les observations admissibles pré-2023. Aucun ajustement du scaler ou des paramètres sur 2023+, aucune calibration, aucun calcul de performance du holdout.

Le dossier neuf `data/processed/ml1_inference_release/` reçoit :

- `logistic_regression_final.joblib` : pipeline figé ; son nom de fichier ne transforme pas son statut provisoire en validation financière.
- `model_manifest.json` : identité, configuration, versions, dates, empreintes, scaler et provenance.
- `monthly_oof_probabilities.csv` : références historiques OOF, sans labels.
- `monthly_blind_probabilities.csv` : inférences aveugles mensuelles, sans labels ni performances.
- `integration_contract.json` : sens des champs et limites d’intégration.

Le dossier existant est refusé avant entraînement. Pour une reconstruction ultérieure autorisée :

```bash
.venv/bin/python -B -m src.models.ml1.inference --output-dir data/processed/ml1_inference_release_rebuild_001
```

Autres options : `--dataset-dir`, `--raw-dir`, `--training-dir`. Les sorties du dépôt doivent rester sous un dossier ignoré `data/processed/ml1_inference_release*`. La reconstruction n’est pas lancée automatiquement lors d’une prédiction.

**Chargement explicite et appel simple**

Les artefacts joblib peuvent exécuter du code lors du chargement. Les charger exclusivement depuis une source de confiance. Les SHA-256 contrôlent l’intégrité, pas l’authenticité d’un manifeste potentiellement remplacé.

```python
from src.models.ml1.inference import (
    configure_inference,
    predict_crisis,
    predict_crisis_details,
)

# Accepter explicitement la confiance dans cet artefact local.
configure_inference(trusted=True)

p = predict_crisis("2022-11-30")  # OOF enregistrée, Python float.
details = predict_crisis_details("2022-11-30")
```

Sans configuration préalable, l’appel explique comment charger l’artefact. Une source, un modèle ou une version manquante/incompatible provoque une erreur explicite, jamais un nouveau fit ou téléchargement.

Pour un objet indépendant, sans configuration globale :

```python
from src.models.ml1.inference import load_inference

engine = load_inference("data/processed/ml1_inference_release", trusted=True)
p = engine.predict_crisis("2023-01-31")
details = engine.predict_details("2023-01-31")
# details est une dataclass ; dataclasses.asdict(details) donne un dictionnaire.
```

Cet appel 2023+ produit seulement une inférence aveugle. Aucun label n’est lu et aucune qualité prédictive n’est déduite de la valeur.

**Structure de détail**

Champs principaux : `prediction_date`, `p_crisis`, `horizon_sessions`, `model_name`, `model_version`, `feature_variant`, `calibration_method`, `prediction_origin`, `training_cutoff`, `decision_time`, `status`.

Champs supplémentaires : `training_label_end`, `decision_timezone`, `publication_status`, `performance_evaluated`.

`training_cutoff` est la dernière date des features/décisions utilisées pour entraîner le modèle concerné. `training_label_end` est la dernière date de fin de ses cibles. La convention de décision est **17 h America/New_York**, avec décalage été/hiver dans l’horodatage retourné. `performance_evaluated=False` identifie les inférences aveugles ; les OOF ont été évaluées lors de la validation historique, ce qui ne valide pas l’artefact figé.

Une prédiction calculée rétrospectivement n’est pas une preuve qu’elle avait été publiée à la date affichée : `publication_status=retrospective_not_contemporaneously_published`.

**Règles de routage et de refus**

- 2015–2022 : uniquement la probabilité OOF enregistrée du fold annuel. Le modèle figé n’est jamais appelé pour fabriquer une prédiction historique hors échantillon.
- Dates historiques sans OOF, notamment avant 2015 ou décembre 2022 : refus explicite, même si des prix/features existent.
- 2023+ : modèle figé pré-2023 et features de la date exacte ; origine `blind_holdout_inference`, sans évaluation.
- Date invalide, date sans séance SPY observée, absence de features, NaN/infini, schéma différent ou date future sans données locales : refus. Aucun report automatique au dernier cours disponible.

Pour chaque date aveugle, les features sont recalculées à partir de préfixes SPY/IEF s’arrêtant à cette date et comparées aux features enregistrées (`rtol=1e-10`, `atol=1e-12`). Les prix ultérieurs ne sont jamais fournis à cette transformation. Cette propriété n’annule pas les éventuelles révisions historiques de l’instantané ajusté.

**Consommation par le membre 4**

Lire `p_crisis` comme une entrée numérique provisoire, avec son origine, sa date, son horizon et son statut. Conserver `calibration_method=none` et les réserves dans les résultats de l’optimiseur. Aucune conversion en allocation, seuil de décision ou rééquilibrage n’est définie par ML 1.

La donnée de clôture est utilisée pour une décision à 17 h New York. Un composant aval ne doit pas traiter cette information comme disponible avant la clôture ni supposer une exécution rétrospective au même cours de clôture.

**Consommation par le membre 5**

`monthly_oof_probabilities.csv` contient 95 mois, janvier 2015 à novembre 2022 : dernière séance OOF admissible de chaque mois. Décembre 2022 n’est pas inventé. Aucun label joint.

Le tableau aveugle reste séparé. La fonction `monthly_blind_probabilities(engine)` exige que toutes les séances du mois selon le calendrier ARCX/XNYS soient couvertes par les features locales. Un mois partiel est exclu ; aucune date mensuelle n’est extrapolée.

```python
from src.models.ml1.inference import monthly_oof_probabilities, monthly_blind_probabilities

historical = monthly_oof_probabilities(engine)
blind = monthly_blind_probabilities(engine)
```

Les deux politiques ne sont pas interchangeables : les OOF proviennent de modèles réentraînés annuellement avec purge, tandis que les inférences aveugles utilisent un unique modèle figé pré-2023. Conserver leurs origines et versions lors de tout assemblage ultérieur. Aucun backtest ni rapport de performance 2023+ n’est réalisé ici.

**Limites et vérifications**

Le modèle est provisoire et sélectionné sur les validations 2015–2022, avec risque de biais de sélection. Il n’a pas démontré de supériorité statistique ni de calibration fiable. Les cibles voisines de 21 séances se chevauchent ; les modèles et calendriers reposent sur les conventions documentées aux étapes précédentes.

Les prix ajustés sont des instantanés pouvant être révisés ; la disponibilité historique avant 17 h est supposée, sans journal de réception. Le calendrier de bibliothèque ARCX résout vers XNYS et ne constitue pas un audit officiel de marché. L’inférence aveugle reste non évaluée et ne doit pas être présentée comme validée pour une utilisation financière réelle.

```bash
.venv/bin/python -B -m pytest tests/ml1/test_inference.py -q -p no:cacheprovider
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Ces tests utilisent des données synthétiques pour les nouveaux fits. Aucun code de l’optimiseur, du backtesting ou d’un autre membre n’est modifié par cette intégration.
