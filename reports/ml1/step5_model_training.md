**Rapport ML 1 — étape 5 : entraînement des modèles — 15 novembre 2026**

Les trois configurations fixes ont produit leurs probabilités hors échantillon sur les huit folds existants : **24 entraînements réussis sur 24 et 5 979 prédictions**. Aucune métrique de performance des nouveaux modèles, sélection, optimisation ou calibration n’a été réalisée.

**1. Branche et état Git initial**

Dépôt : `/Users/mac/Allocation ML/asset-allocation-ml`.

Branche initiale et finale : `feature/ml1-crisis-detection`. Commit de référence inchangé : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`, message `feat(ml1): add crisis data pipeline, temporal validation and baselines`.

L’arbre Git était propre au début de cette étape. Une empreinte SHA-256 des 75 fichiers préexistants contrôlés a été conservée : fichiers suivis, résultats de baselines, rapports précédents et données générées core/core_vix. Leur comparaison après entraînement confirme leur préservation. Aucun travail sur main, commit, push ou merge.

**2. Données et features utilisées**

Dataset exclusivement `data/processed/ml1/`, variante `core`. Développement admissible : **4 329 observations**, du **21/09/2005 au 30/11/2022**, dont la fenêtre cible se termine intégralement avant 2023. Les 21 dates de décision de décembre 2022 dont la cible atteint 2023 restent exclues par le chargeur existant.

Cible inchangée : `y_crisis_21d = 1` si au moins un prix ajusté de SPY dans les 21 séances suivantes descend strictement de plus de 7 % sous le prix ajusté à la décision ; zéro sinon. La décision est conventionnellement prise à 17 h, heure de New York, après clôture. `y_stress_regime` n’est pas utilisée.

Dix colonnes, dans cet ordre exact :

```text
SPY_ret_1d
SPY_ret_5d
SPY_ret_21d
SPY_ret_63d
SPY_vol_5d
SPY_vol_21d
SPY_vol_63d
SPY_drawdown_252d
SPY_vs_ma200
SPY_IEF_corr_63d
```

Aucun VIX, indicateur macro, prix futur, label, diagnostic ou contrôle de fin de cible n’entre dans X. Le schéma doit être exact, y compris l’ordre. Les NaN et infinis sont refusés avant le fit, sans imputation implicite ni suppression de lignes.

**3. Configurations exactes des trois modèles**

Les configurations demandées sont fixes et déclarées avant l’entraînement. Les paramètres supplémentaires non explicités conservent les valeurs par défaut des versions installées ; leurs valeurs résolues sont enregistrées dans `training_metadata.json`.

Logistic Regression : `Pipeline([StandardScaler(), LogisticRegression(...)])`. Le scaler utilise ses paramètres par défaut, centrage et réduction activés, et est ajusté exclusivement sur X_train du fold.

```text
C=1.0
max_iter=2000
solver=lbfgs
class_weight=None
```

Identifiant : `logistic_regression_102c776acc68dede`.

Random Forest : `RandomForestClassifier`.

```text
n_estimators=300
min_samples_leaf=10
max_features=sqrt
random_state=42
n_jobs=1
class_weight=None
```

Identifiant : `random_forest_c3d09bf77ca0b0b7`.

XGBoost : `XGBClassifier`.

```text
n_estimators=300
max_depth=3
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
objective=binary:logistic
eval_metric=logloss
tree_method=hist
random_state=42
n_jobs=1
scale_pos_weight=1
```

Identifiant : `xgboost_ed397cc469cc15e0`.

`eval_metric=logloss` configure XGBoost ; aucun `eval_set`, early stopping ou score de validation n’est fourni ou calculé. Aucun rééquilibrage des classes, SMOTE, suréchantillonnage, recherche d’hyperparamètres ou calibration. Le bootstrap interne par défaut de Random Forest et l’échantillonnage interne explicitement demandé pour XGBoost restent propres aux estimateurs ; ils ne mélangent pas les ensembles chronologiques train/validation. La politique `resampling=False` des métadonnées désigne l’absence de rééchantillonnage externe des classes.

**4. Vérification XGBoost / OpenMP**

Environnement : macOS 26.5.2 ARM64, Python **3.12.13**. Versions utilisées : NumPy **2.5.4**, pandas **3.1.1**, pyarrow **26.0.1**, scikit-learn **1.9.1**, XGBoost **3.4.1**, pytest **9.2.0**, exchange-calendars **4.13.3**.

L’import, un fit synthétique sur 32 lignes et `predict_proba` sur quatre lignes réussissent avant les entraînements réels. Le contrôle reproductible est exécuté dans un sous-processus : une erreur d’import, un signal natif ou un délai dépassé est enregistré sans désactiver les autres modèles. Le précontrôle de l’exécution réelle retourne `passed`, une matrice `(4, 2)` et aucun message sur stderr.

Les deux bibliothèques locales existent :

```text
.venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
.venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib
```

Empreintes SHA-256 :

```text
libxgboost.dylib : 652d0e2c5c6891c1887b913450f93274d50baaca81e71a9ff96ab2200bbac8c1
libomp.dylib     : 7fe04d06d824817d39e4c99ec44e837fd6b5f0f75f2cc3e77a384c298f3f1709
```

`otool` confirme la dépendance à `@rpath/libomp.dylib` et le chemin local déjà ajouté `@loader_path/../../sklearn/.dylibs`. `threadpoolctl` identifie un runtime OpenMP chargé, celui de scikit-learn. Son état global annonce huit threads ; les estimateurs RF/XGBoost conservent bien `n_jobs=1`.

Aucune erreur native observée. Aucune installation, modification de bibliothèque, nouvelle signature native ou modification Python globale effectuée. Les adaptations de l’étape 1 sont simplement réutilisées.

**5. Nombre réel d’entraînements réussis**

| Modèle | Entraînements réels réussis | Folds en échec |
| --- | ---: | ---: |
| logistic_regression | 8 | 0 |
| random_forest | 8 | 0 |
| xgboost | 8 | 0 |
| Total | 24 | 0 |

Les 24 fits utilisent chacun un nouvel estimateur et les deux classes dans y_train. Aucun avertissement d’estimateur ou de convergence n’a été enregistré pendant ces fits. Les smoke tests et fits synthétiques des tests sont distincts de ces 24 entraînements réels.

Aucun entraînement final sur tout 2005–2022, aucun modèle final sérialisé. L’interface `fit_predict_fold` retourne néanmoins l’estimateur du fold en mémoire et ses probabilités, pour permettre une réutilisation ultérieure sous protocole explicitement autorisé.

**6. Huit folds et effectifs**

Les fonctions réutilisées sans modification sont `load_development_data`, `build_walk_forward_folds`, `assert_fold_is_safe` et `get_fold_xy` dans `time_validation.py`. Les API d’`indicator_validation.py`, `baselines.py` et `evaluate_baselines.py` ont également été inspectées. Le moteur core suffit ici : aucun chargement de la variante VIX pour entraîner les modèles.

Tous les entraînements commencent le 21/09/2005. Les effectifs ci-dessous s’appliquent à chacun des trois modèles.

| Fold | Dernière décision train | Fin maximale de cible train | N train | Purge SPY | Validation | N validation |
| --- | --- | --- | ---: | ---: | --- | ---: |
| 2015 | 2014-12-01 | 2014-12-31 | 2 315 | 21 | 2015-01-02 → 2015-12-31 | 252 |
| 2016 | 2015-12-01 | 2015-12-31 | 2 567 | 21 | 2016-01-04 → 2016-12-30 | 252 |
| 2017 | 2016-11-30 | 2016-12-30 | 2 819 | 21 | 2017-01-03 → 2017-12-29 | 251 |
| 2018 | 2017-11-29 | 2017-12-29 | 3 070 | 21 | 2018-01-02 → 2018-12-31 | 251 |
| 2019 | 2018-11-28 | 2018-12-31 | 3 321 | 21 | 2019-01-02 → 2019-12-31 | 252 |
| 2020 | 2019-11-29 | 2019-12-31 | 3 573 | 21 | 2020-01-02 → 2020-12-31 | 253 |
| 2021 | 2020-12-01 | 2020-12-31 | 3 826 | 21 | 2021-01-04 → 2021-12-31 | 252 |
| 2022 | 2021-12-01 | 2021-12-31 | 4 078 | 21 | 2022-01-03 → 2022-11-30 | 230 |

La purge est celle de 21 séances existante. L’embargo post-validation configuré ne retire aucune observation supplémentaire dans cette topologie d’entraînement strictement antérieur à la validation. Aucun fold, index ou ordre chronologique modifié.

**7. Nombre de prédictions générées**

| Modèle | Dates distinctes | Lignes exportées |
| --- | ---: | ---: |
| logistic_regression | 1 993 | 1 993 |
| random_forest | 1 993 | 1 993 |
| xgboost | 1 993 | 1 993 |
| Total | 1 993 communes | 5 979 |

Les dates vont du 02/01/2015 au 30/11/2022. Les ensembles de dates des trois modèles sont identiques à ceux des baselines enregistrées. Cette vérification ne lit que les identifiants de baseline et les dates de leur Parquet, sans comparer les performances.

Chaque ligne exportée contient : `prediction_date`, `validation_year`, `fold`, `model`, `feature_variant`, `predicted_probability`, `configuration_id`, `train_start`, `train_end`, `train_target_end_max`, `latest_train_label_known_at`, `first_validation_decision`, `n_train`, `n_purged`.

Les fins de cible et dates de disponibilité exportées sont des informations de provenance de l’entraînement, jamais des features. Aucun label, prix futur ou diagnostic exporté avec les prédictions. L’évaluation ultérieure pourra joindre les cibles pré-2023 par date via le chargeur protégé existant.

**8. Vérification des probabilités**

Toutes les probabilités sont finies, comprises dans `[0, 1]`, avec exactement une ligne par modèle et date. La classe positive est identifiée via `classes_`, pas uniquement par convention de position. La matrice binaire de `predict_proba` est aussi contrôlée avant extraction, notamment la somme des deux colonnes.

| Modèle | Minimum observé | Maximum observé |
| --- | ---: | ---: |
| logistic_regression | 0.007759414907 | 0.860731977027 |
| random_forest | 0.000000000000 | 0.664952473111 |
| xgboost | 0.000486067263 | 0.824963927269 |

Ces bornes contrôlent le format des sorties ; elles ne constituent pas une évaluation de performance ou de calibration. La probabilité zéro de Random Forest est valide et n’a pas été écrêtée.

**9. Contrôles de data leakage**

Avant chaque fit : schéma exact core, index alignés et chronologiques, séparation stricte des dates, purge en séances SPY, fin de chaque cible train antérieure à la première validation et disponibilité du label avant la première décision de validation.

Le fit reçoit uniquement X_train et y_train. L’interface d’entraînement ne prend aucun argument y_validation ; la quatrième sortie de `get_fold_xy` est écartée. Aucun `eval_set`, transformation ajustée sur validation ou early stopping. Le `StandardScaler` est reconstruit dans chaque pipeline et ajusté pendant le fit de ce seul train.

Les tests vérifient l’invariance des probabilités lorsque seuls les labels de validation du fold changent, ainsi que les moyennes, variances et effectifs réellement appris par le scaler. Les fréquences de classes ne servent à aucune adaptation des configurations.

Dans les folds ultérieurs, des dates de validation antérieures peuvent devenir des observations d’entraînement lorsque leurs labels sont disponibles, conformément au moteur croissant existant. Aucun shuffle des ensembles ni validation croisée aléatoire.

**10. Protection du test final 2023+**

Le chargeur existant lit y avec les filtres Parquet `Date < 2023-01-01` et `target_end_session < 2023-01-01`, ainsi que la borne initiale 2005. Aucun label 2023+ n’est chargé ou exporté. Les contrôles de calendrier consultent les dates et métadonnées temporelles nécessaires, sans utiliser les résultats du test final.

Les tests surveillent les appels réels à `read_parquet` pour vérifier ces filtres ; un test à labels sentinelles confirme le rejet de dates 2023+ avant inspection de leurs valeurs. Les entrées d’estimateur et exports disposent aussi de gardes explicites avant 2023.

**11. Tests pytest**

Les **168 tests existants sont préservés et réussissent**. Les **48 nouveaux tests** couvrent les interfaces sklearn et paramètres fixes, les probabilités, la reproductibilité, la séparation temporelle, le scaler train-only, l’invariance à y_validation, les features futures/VIX interdites, NaN/infinis, classes uniques et labels invalides, la purge, les erreurs natives, XGBoost indisponible, les résultats partiels, la convergence, les exports sans écrasement et la préservation des composants précédents.

Les tests d’intégration sur les huit folds réels utilisent des estimateurs espions pour contrôler les interfaces et accès aux données, sans répéter les 24 entraînements réels ni calculer de scores des nouveaux modèles.

Résultat complet :

```text
216 passed, 108 warnings in 13.24s
```

Répartition des avertissements : pipeline 64, validation temporelle 16, indicateurs 12, baselines 8, nouveaux tests d’entraînement 8. Ils correspondent au `Pandas4Warning` déjà connu sur `Series.values` avec fuseau horaire ; aucun avertissement n’a été masqué. Aucun test en échec ou ignoré dans l’exécution finale.

Un premier test d’export avait détecté le paramètre par défaut XGBoost `missing=np.nan`, incompatible avec un JSON strict. Le nouveau module encode désormais explicitement cette valeur dans les métadonnées des paramètres sous `{"nonfinite_float": "NaN"}`. Les données et probabilités restent soumises au refus des NaN. Cette correction ne modifie aucun hyperparamètre ni fichier préexistant.

**12. Fichiers créés ou modifiés**

Trois nouveaux fichiers destinés au suivi Git :

- `src/models/ml1/train_ml1.py` : construction, précontrôle natif, fit par fold, probabilités, validations, métadonnées et CLI.
- `tests/ml1/test_training.py` : les 48 nouveaux tests.
- `reports/ml1/step5_model_training.md` : ce rapport.

Deux nouvelles sorties dans le dossier ignoré `data/processed/ml1_training/`, absent avant l’exécution :

| Fichier | Taille | Contenu |
| --- | ---: | --- |
| `ml_models_oof_predictions.parquet` | 90 013 octets | 5 979 lignes, 14 colonnes |
| `training_metadata.json` | 72 077 octets | Configurations, 24 résultats de fit, folds, versions, empreintes et preuve native |

SHA-256 des sorties :

```text
ml_models_oof_predictions.parquet
2990ee5d8f2bef48c266e7d46236676a4f8f643d21490ad2a7045e931c3b371a

training_metadata.json
9e1d78b125010d99e98ba68ae888a03292d71d1a4fbf4fab482ca91b0090c665
```

Les métadonnées enregistrent les versions, configurations et paramètres résolus, graines, dix colonnes ordonnées, dates de validation exactes, empreintes des indices, données pré-2023, sources brutes et modules, ainsi que la branche et le commit de référence.

Aucun fichier préexistant modifié, aucune donnée brute ou partagée régénérée, aucun résultat de baseline changé, aucune dépendance installée et aucun téléchargement. Les empreintes des 75 fichiers protégés concordent.

**13. Commandes de reproduction**

Depuis la racine du dépôt et la branche `feature/ml1-crisis-detection` :

```bash
# Tests de cette étape
.venv/bin/python -B -m pytest tests/ml1/test_training.py -q -p no:cacheprovider

# Suite complète ML 1
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider

# Reproduction ultérieure : choisir impérativement un dossier encore absent
.venv/bin/python -B -m src.models.ml1.train_ml1 --output-dir data/processed/ml1_training_rebuild_001
```

La commande exécutée pour cette livraison était `.venv/bin/python -B -m src.models.ml1.train_ml1`. Son dossier par défaut existe maintenant : toute relance vers ce même dossier est refusée avant entraînement. Les exports sont préparés dans un dossier temporaire neuf puis publiés ensemble ; aucun écrasement silencieux.

L’API `build_estimator` construit une configuration fixe, `fit_predict_fold` fournit l’estimateur du fold et ses probabilités, `train_walk_forward` fournit un `TrainingRun`, et `export_training` contrôle puis enregistre les résultats. Aucun appel de scoring dans ce module. Les commandes de reproduction sont documentées pour une exécution ultérieure, pas lancées en plus de la livraison.

**14. Problèmes non résolus**

Aucun blocage d’entraînement ou d’export observé. Les avertissements pandas restent à traiter lors d’une future maintenance autorisée. L’adaptation OpenMP locale préexistante doit être revalidée après une mise à jour de l’environnement ; ses empreintes sont enregistrées.

Les réserves du dataset core sur les prix ajustés et leur disponibilité historique restent celles des étapes précédentes. Les fenêtres de cible se chevauchent et les folds partagent une partie de leurs entraînements ; cela devra être pris en compte lors de l’évaluation ultérieure.

Les probabilités ne sont pas calibrées et aucune performance n’a été mesurée ici. Aucun modèle n’est sélectionné, déclaré supérieur ou validé pour une utilisation financière. Les trois configurations restent des points de départ fixes.

Les deux sorties de données sont volontairement ignorées par Git. Leur conservation sur disque reste nécessaire pour les prochaines étapes ; elles n’ont pas été ajoutées de force à Git.

**15. État Git final**

Branche : `feature/ml1-crisis-detection`. HEAD inchangé : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`.

```text
?? reports/ml1/step5_model_training.md
?? src/models/ml1/train_ml1.py
?? tests/ml1/test_training.py
```

Aucun fichier suivi modifié ni changement à l’index. Les deux nouvelles sorties sous `data/processed/ml1_training/` sont ignorées par la règle existante `data/*`. Aucun commit, push, merge ou Pull Request.

**16. Verdict : PRÊT AVEC RÉSERVES**

L’étape 5 est terminée : trois configurations fixes core, huit folds préservés, 24 fits réussis, 5 979 probabilités OOF contrôlées et 216 tests réussis. Les réserves concernent les hypothèses historiques du dataset, l’environnement natif local et les avertissements connus, sans blocage pour une évaluation ultérieure autorisée.

Le test final 2023+ reste fermé. Aucune métrique de performance des nouveaux modèles, comparaison aux baselines, optimisation, calibration, sélection ou entraînement final réalisé. Toute étape suivante attend la validation de l’utilisateur.
