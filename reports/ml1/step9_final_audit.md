**ML 1 — étape 9 : audit final et préparation de revue — 15 novembre 2026**

Le module est prêt pour une revue technique du membre 1, **avec réserves** sur la portabilité de l’environnement, les gardes de branche des CLI et les limites méthodologiques. Les sorties validées sont préservées. Deux reconstructions isolées réussissent, dont une sans lecture des anciens artefacts ignorés. Aucun commit, push, merge, PR ou transmission.

**1. État Git initial**

Dépôt : `/Users/mac/Allocation ML/asset-allocation-ml`. Branche : `feature/ml1-crisis-detection`. Dernier commit : `42f36c10250bb234e6ce0422b6dcece1020f7b6f`.

Aucun fichier suivi modifié. Les quatre fichiers non suivis de l’étape 8 sont présents et lisibles : `src/models/ml1/inference.py`, `tests/ml1/test_inference.py`, `reports/ml1/integration_guide.md`, `reports/ml1/step8_integration.md`.

Référence locale `main` et `origin/main` : `8d9ff5ff66c161181f55a1da56d59cae02394375`. La branche est en avance de deux commits, sans retard sur ce main local. Son diff contient **46 ajouts**, tous sous `src/models/ml1/`, `tests/ml1/`, `reports/ml1/`, sans modification ni suppression d’un fichier de main. Aucun composant d’un autre membre dans le diff.

Aucun fetch : la fraîcheur distante n’est pas vérifiée. Le membre 1 devra actualiser et revoir cette comparaison avant la PR réelle. Un inventaire complet de **110 fichiers préexistants** est conservé ; leurs empreintes sont vérifiées en fin d’audit. Les données générées et `.venv` restent ignorées.

**2. Architecture finale**

Le core calcule dix features SPY/IEF historiques et une cible de baisse de clôture ajustée strictement >7 % dans les 21 séances SPY suivantes. Le calendrier et les horizons sont contrôlés ; les séries macro/VIX ne participent pas à l’inférence core.

Les couches sont : transformations et données (`core`, `pipeline`), validation chronologique (`time_validation`), variante VIX expérimentale (`market_indicators`, `indicator_validation`), baselines et trois modèles fixes (`baselines`, `train_ml1`, évaluateurs), calibration exploratoire et SHAP (`calibration`, `explain_ml1`), interface figée/OOF (`inference`).

La documentation de livraison est `README_HANDOFF.md`, complétée par `integration_guide.md`, les rapports 1–8, le lock ML 1 et `PR_DESCRIPTION.md`. Aucune allocation, stratégie de backtest ou intégration par modification des composants des membres 4/5.

**3. Résultats des tests**

Commande complète exécutée au début de l’audit, puis de nouveau après création des documents :

```bash
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Résultat attendu et confirmé : **326 passed, 148 warnings**, sans échec ni test ignoré. La première exécution dure 35,89 s. Le contrôle final utilise la même suite après les ajouts documentaires ; aucun test n’est modifié pour réussir.

Les entraînements des tests et des deux reconstructions isolées sont explicitement autorisés par cette mission. Aucune sortie validée d’étape 5 ou 8 n’est remplacée.

**4. Analyse des avertissements**

Les 148 avertissements sont des `Pandas4Warning`. Origine vérifiée par une sonde qui les transforme temporairement en erreurs : `load_development_data` → `check_calendar` → création du calendrier exchange-calendars → `exchange_calendar.py:405–408`, quatre accès `.values` sur les séries de calendrier avec fuseau horaire (`open`, `break_start`, `break_end`, `close`).

Répartition : pipeline 64, validation temporelle 16, indicateurs 12, baselines 8, entraînement 8, calibration 8, explicabilité 4, inférence 28. Aucun avertissement masqué.

Ils ne bloquent pas l’environnement actuel, mais deviennent bloquants sous `warnings as errors` et signalent un risque de changement de comportement avec pandas 4. Une migration devra mettre à jour/revalider les dépendances et le calendrier. Aucun patch risqué des bibliothèques.

Un avertissement séparé de pip indique un cache non inscriptible ; `pip check` réussit avec `No broken requirements found`. Les caches graphiques des sondes sont temporaires. Aucun problème natif XGBoost observé sur cette machine.

**5. Audit de reproductibilité**

Python 3.12.13 et onze imports directs ML 1 sont vérifiés. Les contraintes installées sont cohérentes et `pip check` réussit. `reports/ml1/requirements-ml1.lock.txt` fixe **31 distributions** : dépendances runtime/tests de ML 1 et transitives actives sur ce Mac. Les fournisseurs de collecte yfinance/fredapi/python-dotenv sont exclus car inutilisés dans le parcours hors ligne ; les requirements partagés ne sont pas modifiés.

Les données autorisées SPY/IEF nécessaires sont suivies dans main. Les sorties ignorées ne sont pas un obstacle à la reconstruction : la seconde exécution construit les données, OOF et release sans lire les anciennes sorties, puis vérifie l’API et ses CSV nouvellement produits.

Limite explicite : aucune installation neuve, résolution réseau ou disponibilité de roues n’est testée. Le lock validé sur macOS ARM64 n’est ni un verrou multi-plateforme garanti ni un inventaire de dépendances système ; il ne comporte pas de hashes de roues.

**6. Résultats de reconstruction**

Deux dossiers neufs, conservés hors dépôt :

| Exécution | Dossier | Comparaison aux sorties existantes |
| --- | --- | --- |
| Avec référence | `/private/tmp/ml1-step9-rebuild-_mv_tp6f/` | Oui, égalité exacte vérifiée |
| Sans ancienne sortie | `/private/tmp/ml1-step9-source-only-jfh6xjy8/` | Non, mode sans référence explicitement activé |

Chaque exécution produit : **4 329 lignes supervisées pré-2023**, **5 314 features d’inférence**, huit folds purgés, **24 entraînements OOF réussis**, **5 979 prédictions**, un modèle figé pré-2023, **95 mois OOF** et **46 mois aveugles**. Identifiant figé reproduit : `ml1_lr_core_pre2023_86aa4c7f66463f90`.

Première exécution : X, y admissible et contrôles temporels identiques ; les 5 979 lignes OOF complètes sont identiques, de même que les coefficients/intercept du modèle figé et les lignes des deux CSV mensuels. La seconde prouve l’absence de dépendance aux anciennes sorties ignorées ; ses assertions de comparaison de référence sont explicitement non réalisées, pas présentées comme réussies.

**Adaptation de fermeture du holdout :** le pipeline standard construit aussi des targets après 2022. La construction isolée utilise donc son API avec un adaptateur en mémoire qui ne transmet à `build_targets` que les prix antérieurs à 2023, dernier prix le 30/12/2022. Les features sont calculées sur toute la plage locale, sans labels. Aucune source Python ni contrôle temporel n’est modifié sur disque ; l’adaptateur est explicite et conservé dans la recette de livraison.

Le core isolé ne reproduit pas la table supervisée complète de 5 293 lignes : seules ses 4 329 lignes admissibles sont matérialisées. Cette différence volontaire évite de générer un label holdout, tout en reproduisant exactement les entrées d’apprentissage et les prédictions validées.

Les CLI réelles de validation, entraînement OOF et inférence sont ensuite exécutées sur les chemins temporaires. Les logs, scripts et `reconstruction_summary.json` sont conservés. Au total hors tests : **48 fits OOF et deux fits figés**, uniquement pour ces reconstructions autorisées. Aucun réglage, remplacement d’artefact validé ou score 2023+.

**7. Contrat d’intégration**

Imports contrôlés : `configure_inference`, `predict_crisis`, `predict_crisis_details`. Confiance explicite `trusted=True` obligatoire avant joblib. Les empreintes contrôlent l’intégrité, pas l’authenticité d’un ensemble modèle/manifest remplacé.

`predict_crisis(date)` retourne un Python float fini dans `[0,1]`. Horizon : 21 séances SPY. Le détail fournit date, probabilité, modèle/version, variante core, calibration none, origine, bornes d’entraînement, décision horodatée, statut provisional et caractère rétrospectif.

Historique 2015–2022 : uniquement OOF enregistrée du fold correspondant ; dates sans OOF, notamment décembre 2022, refusées. Le modèle figé n’est jamais utilisé pour fabriquer une fausse OOF in-sample.

2023+ : `blind_holdout_inference`, non évaluée, modèle figé uniquement sur développement pré-2023. Date locale exacte, séance SPY et features finies exigées. Aucun report au dernier cours, téléchargement, label ou fit pendant l’inférence.

Le membre 4 peut consommer seulement `p_crisis` comme entrée numérique en conservant les réserves de contexte. Le membre 5 peut lire les CSV sans labels et garder séparées leurs origines. Aucun code de ces membres modifié.

**8. Contrôles des probabilités mensuelles**

Artefacts originaux de `data/processed/ml1_inference_release/` présents et empreintes vérifiées : joblib, manifest, CSV OOF, CSV aveugle, contrat JSON. Schémas et source/configurations contrôlés au chargement.

- OOF : **95 mois**, janvier 2015–novembre 2022 ; dates 30/01/2015–30/11/2022, sans doublon, aucune date de décembre inventée.
- Aveugle : **46 mois**, janvier 2023–octobre 2026 ; dates 31/01/2023–30/10/2026. Novembre est incomplet, dernière feature locale le 04/11/2026.

Aucune valeur manquante dans les tables, probabilités finies et bornées, dates/mois triés, aucun label ou prix futur exporté. Les dates `training_label_end` sont des métadonnées de provenance pré-2023. Concordance CSV/API vérifiée pour les 141 lignes. Aucune conclusion qualitative sur les valeurs aveugles et aucune métrique holdout.

Rapports historiques : neuf rapports d’étapes 1–8 plus le guide d’intégration lisibles. Trois CSV de métriques : 18 lignes baseline, 45 évaluation, 68 calibration. Sept PNG décodables et intacts. Les sorties de validation restent inchangées.

**9. Protection temporelle et data leakage**

Dix features : rendements SPY 1/5/21/63, volatilités 5/21/63, drawdown 252, écart moyenne mobile 200, corrélation SPY/IEF 63. Transformations historiques, aucun scaler ajusté globalement sur 2023+.

La cible porte strictement sur les clôtures t+1 à t+21 du calendrier SPY, avec comparaison stricte à 0,93 du prix à t. Le label de stress historique n’est pas la cible de crise. Les séances attendues sont contrôlées par ARCX résolu vers XNYS, sans remplissage ou suppression silencieuse.

Le développement admissible a 4 329 dates jusqu’au 30/11/2022, dernière fin de cible le 30/12/2022. Les 21 décisions de décembre débordant sur 2023 sont exclues de l’apprentissage. Les huit folds imposent purge de 21 séances et maturité des labels avant validation. Calibration seulement sur OOF antérieures et labels matures, aucune méthode sélectionnée.

Le modèle figé et son scaler n’apprennent que le développement pré-2023. Les features d’une date aveugle sont recalculées sur les préfixes de prix s’arrêtant à cette date puis comparées au fichier validé. Aucun accès aux targets pendant l’inférence ; les exports ne contiennent pas de labels.

Cela établit les gardes et l’absence de look-ahead algorithmique contrôlée, **pas une certification point-in-time** : prix ajustés révisables, réception historique à 17 h supposée, calendrier logiciel et publications non documentées. Le VIX reste expérimental et les macros exclues.

**10. Dépendances et environnement**

| Composant | Version validée |
| --- | --- |
| Python | 3.12.13 |
| pandas | 3.1.1 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| SHAP | 0.54.1 |
| pyarrow | 26.0.1 |
| joblib | 1.6.0 |
| exchange-calendars | 4.13.3 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |
| threadpoolctl | 3.8.0 |

Transitives, dont SciPy 1.18.1, verrouillées dans le fichier ML 1. Environnement validé : macOS 26.5.2 ARM64.

XGBoost import et fit/predict synthétique réussis. Le binaire local dépend de `@rpath/libomp.dylib` et réutilise le runtime `.venv/.../sklearn/.dylibs/libomp.dylib` via une adaptation locale préexistante de rpath/signature. Ce n’est pas une solution portable garantie. La procédure de diagnostic est dans le handoff ; aucune bibliothèque globale, wheel ou installation système n’est modifiée. Linux/Windows nécessitent la revue de leurs dépendances conditionnelles, notamment NCCL déclaré pour Linux par cette distribution XGBoost.

**11. Livrables pour les membres 1, 4 et 5**

Membre 1 : `README_HANDOFF.md`, lock ML 1, `PR_DESCRIPTION.md`, ce rapport, sources/tests/rapports précédents et recette stricte de reconstruction depuis SPY/IEF Git. Les artefacts ignorés sont reconstruisibles dans l’environnement validé ; une machine neuve reste à valider.

Membre 4 : float `p_crisis` et détail de provenance. Politique `logistic_regression/core/none/provisional`, sans règle d’allocation. Membre 5 : tableaux mensuels OOF/aveugles séparés, sans labels, avec versions et dates train distinctes.

Décision à **17 h America/New_York**, après clôture. Aucun backtest aval ne doit exécuter rétrospectivement au cours de clôture déjà connu à 17 h. Une prédiction rétrospective n’est pas une preuve de publication réelle à sa date.

**12. Fichiers à inclure dans la PR**

Périmètre proposé contre main local : **54 chemins**, dont 46 déjà enregistrés dans les deux commits locaux et huit encore non suivis. Tous sont dans les espaces ML 1. Liste exacte :

```text
reports/ml1/PR_DESCRIPTION.md
reports/ml1/README_HANDOFF.md
reports/ml1/baselines/baseline_comparison.md
reports/ml1/baselines/baseline_metrics.csv
reports/ml1/baselines/baseline_predictions.parquet
reports/ml1/calibration/brier_calibration_by_year.png
reports/ml1/calibration/calibrated_predictions.parquet
reports/ml1/calibration/calibration_comparison.md
reports/ml1/calibration/calibration_curves.png
reports/ml1/calibration/calibration_metrics.csv
reports/ml1/calibration/shap_feature_importance.csv
reports/ml1/calibration/shap_importance.png
reports/ml1/calibration/shap_summary.png
reports/ml1/evaluation/ap_by_year.png
reports/ml1/evaluation/brier_by_year.png
reports/ml1/evaluation/model_comparison.md
reports/ml1/evaluation/model_metrics.csv
reports/ml1/evaluation/precision_recall_curves.png
reports/ml1/integration_guide.md
reports/ml1/requirements-ml1.lock.txt
reports/ml1/step1_remediation.md
reports/ml1/step2_data_validation.md
reports/ml1/step3_walk_forward_validation.md
reports/ml1/step3bis_vix_macro.md
reports/ml1/step4_baselines.md
reports/ml1/step5_model_training.md
reports/ml1/step6_model_evaluation.md
reports/ml1/step7_calibration_shap.md
reports/ml1/step8_integration.md
reports/ml1/step9_final_audit.md
src/models/ml1/__init__.py
src/models/ml1/baselines.py
src/models/ml1/calibration.py
src/models/ml1/core.py
src/models/ml1/evaluate_baselines.py
src/models/ml1/evaluate_ml1.py
src/models/ml1/explain_ml1.py
src/models/ml1/indicator_validation.py
src/models/ml1/inference.py
src/models/ml1/market_indicators.py
src/models/ml1/pipeline.py
src/models/ml1/time_validation.py
src/models/ml1/train_ml1.py
tests/ml1/__init__.py
tests/ml1/test_baselines.py
tests/ml1/test_calibration.py
tests/ml1/test_core.py
tests/ml1/test_evaluation.py
tests/ml1/test_explainability.py
tests/ml1/test_inference.py
tests/ml1/test_market_indicators.py
tests/ml1/test_pipeline.py
tests/ml1/test_time_validation.py
tests/ml1/test_training.py
```

Les neuf binaires générés déjà suivis dans ces commits sont sept PNG et les Parquet de prédictions baseline/calibration sous `reports/ml1/`. Ils ne sont pas ajoutés de force pendant l’audit et ne sont pas supprimés. Le membre 1 doit revoir leur politique de stockage avant fusion. L’audit ne réécrit aucun commit.

**Liste exacte du futur commit encore nécessaire : huit fichiers texte uniquement** — aucune commande git add/commit n’a été exécutée :

```text
reports/ml1/PR_DESCRIPTION.md
reports/ml1/README_HANDOFF.md
reports/ml1/integration_guide.md
reports/ml1/requirements-ml1.lock.txt
reports/ml1/step8_integration.md
reports/ml1/step9_final_audit.md
src/models/ml1/inference.py
tests/ml1/test_inference.py
```

Aucun composant partagé, requirement commun, `.idea`, fichier personnel ou secret ajouté. Une recherche ciblée de motifs de clés privées/tokens connus ne relève aucun motif à forte confiance dans les candidats texte ; elle ne remplace pas une revue humaine complète.

**13. Fichiers ignorés et procédure de reconstruction**

Exclure de tout nouvel ajout : `.venv/`, caches, secrets `.env`/clés, `data/processed/ml1/` (5 fichiers), `data/processed/ml1_core_vix_experimental/` (7), `data/processed/ml1_training/` (2), `data/processed/ml1_inference_release/` (5), tout dossier temporaire de reconstruction. Ne pas utiliser `git add -f`.

Ces **19 fichiers générés ignorés** restent présents. Les sources brutes SPY/IEF déjà suivies dans main sont des entrées inchangées, pas des nouveaux fichiers de PR. Le README fournit la recette complète strictement isolée ; aucune collecte ni clé d’API n’est nécessaire pour core.

La variante VIX optionnelle n’a pas été reconstruite à cette étape ; elle reste présente et protégée. L’environnement `.venv` n’a pas été recréé. Les rapports/binaires historiques déjà suivis relèvent de la revue de stockage, pas d’un nouvel ajout forcé.

**14. Limites méthodologiques**

Logistic Regression est seulement candidate provisoire sur l’AP des validations. Son Brier brut reste moins bon que la référence historique ; aucune supériorité statistique. La sélection sur 2015–2022 peut introduire un biais. Les fenêtres de 21 séances se chevauchent et les dates positives ne sont pas autant de crises indépendantes.

Platt/Isotonic sont exploratoires, sans méthode retenue. VIX non certifié point-in-time. Les prix ajustés et disponibilités historiques restent sous hypothèses, sans journal de réception. Les OOF annuelles et l’inférence figée représentent des politiques différentes.

Aucune performance 2023+, stratégie, optimisation ou backtest. Aucun module validé pour un usage financier réel. Les inférences aveugles autorisées ne constituent pas une évaluation du holdout.

**15. Problèmes bloquants ou non bloquants**

Aucun blocage technique local : tests, imports, dépendances installées, reconstruction stricte et intégration réussissent. Aucun artefact absent ou corrompu détecté.

Réserves à examiner avant fusion/diffusion :

- Environnement neuf et portabilité OpenMP non validés ; lock spécifique à la machine, sans hashes de roues ni garantie multi-plateforme.
- CLI train/inference gardées sur `feature/ml1-crisis-detection` ; une utilisation directe sur main après fusion serait refusée. Interface importable sans cette garde, mais procédure de reconstruction sur main à revoir collectivement.
- CLI pipeline standard construit des targets 2023+ ; la recette stricte explicite est nécessaire sous fermeture actuelle du holdout. Aucune assertion de reconstruction standard non restreinte.
- Dépréciations pandas/exchange-calendars et migration future à tester.
- Politique des neuf binaires de rapports déjà suivis à confirmer.
- Références main distantes non actualisées ; validation finale du membre 1 et limites méthodologiques toujours requises.

Ces points ne sont pas contournés et aucun code existant n’est modifié pour les cacher.

**16. Commandes recommandées pour les tests**

```bash
.venv/bin/python -B -m pip check
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
.venv/bin/python -B -m pytest tests/ml1/test_inference.py -q -p no:cacheprovider
.venv/bin/python -B -c 'import json; from src.models.ml1.train_ml1 import xgboost_preflight; print(json.dumps(xgboost_preflight(), indent=2))'
```

Les CLI `pipeline`, `time_validation`, `train_ml1`, `inference` ont été contrôlées par `--help`. La recette complète du README exige un dossier neuf ; option `--without-reference` pour reconstruire sans lire les anciennes sorties. Les entraînements d’une telle reconstruction restent une action explicite autorisée, jamais déclenchée au chargement du modèle.

**17. Liste des nouveaux fichiers créés**

Quatre fichiers documentaires ML 1 :

```text
reports/ml1/requirements-ml1.lock.txt
reports/ml1/README_HANDOFF.md
reports/ml1/PR_DESCRIPTION.md
reports/ml1/step9_final_audit.md
```

Deux arborescences temporaires hors dépôt contiennent scripts, copies SPY/IEF, core strict, OOF, releases, logs et résumés de preuve. Elles sont conservées et exclues de Git. Aucun fichier préexistant du projet supprimé, aucun historique écrasé, aucun code existant ou requirement partagé modifié.

**18. État Git final**

Branche `feature/ml1-crisis-detection`, HEAD `42f36c10250bb234e6ce0422b6dcece1020f7b6f`, inchangés.

```text
?? reports/ml1/PR_DESCRIPTION.md
?? reports/ml1/README_HANDOFF.md
?? reports/ml1/integration_guide.md
?? reports/ml1/requirements-ml1.lock.txt
?? reports/ml1/step8_integration.md
?? reports/ml1/step9_final_audit.md
?? src/models/ml1/inference.py
?? tests/ml1/test_inference.py
```

Aucun fichier suivi modifié, aucun changement à l’index et aucun fichier ignoré ajouté. Les 110 fichiers préexistants protégés sont intacts. Aucun commit, push, merge ou Pull Request ; aucune transmission au membre 1.

**19. Verdict : PRÊT AVEC RÉSERVES**

Le module et sa documentation sont prêts pour une revue technique du membre 1 : 326 tests réussis, intégrité conservée, deux reconstructions isolées abouties et interfaces vérifiées. Les réserves de portabilité, reconstruction sur main, stockage des binaires et méthodologie doivent être discutées avant fusion ou diffusion.

Le texte de PR est préparé uniquement dans le dépôt. **Toute transmission au membre 1, création de commit/PR, push ou merge attend votre validation.**
