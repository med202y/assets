**Rapport ML 1 — étape 4 : baselines — 14 novembre 2026**

Deux baselines probabilistes ont été évaluées sur les validations walk-forward 2015–2022. Les résultats sont calculés sur les données locales, sans téléchargement, sans utilisation des labels 2023+ et sans entraîner Logistic Regression, Random Forest ou XGBoost.

**1. État Git initial**

Branche vérifiée avant intervention : `feature/ml1-crisis-detection`, commit `8d9ff5ff66c161181f55a1da56d59cae02394375`. Les quinze fichiers non suivis des étapes précédentes étaient présents ; aucun fichier suivi n’était modifié. Le dossier de résultats demandé n’existait pas.

Une empreinte des fichiers préexistants, des rapports et des deux jeux de données a été enregistrée avant intervention. La comparaison finale confirme leur préservation intégrale.

**2. Objectif des deux baselines**

`historical_frequency`, sur `core`, fournit une référence simple : probabilité constante égale au taux de crises dans l’entraînement purgé de chaque fold.

`vix_threshold_25`, sur `core_vix`, étudie une hypothèse expérimentale : les fréquences futures de crise peuvent différer selon que le niveau VIX retardé est inférieur à 25 ou supérieur ou égal à 25. Le seuil n’est jamais directement converti en une probabilité arbitraire de zéro ou un.

Les deux baselines utilisent la même cible : baisse de SPY strictement supérieure à 7 % au moins une fois dans les 21 prochaines séances. Elles ne sont ni ajustées ni sélectionnées sur le test final.

**3. Architecture développée**

- `src/models/ml1/baselines.py` : estimateurs `HistoricalFrequencyBaseline` et `VixThresholdBaseline`, avec `fit`, `predict_proba`, `classes_`, `get_params` et résumé des estimations d’entraînement. `predict_proba` retourne deux colonnes `[P(0), P(1)]`.
- `src/models/ml1/evaluate_baselines.py` : chargement protégé, plan de comparaison, estimation sur train, prédictions hors échantillon, métriques et exports.
- `tests/ml1/test_baselines.py` : tests unitaires, synthétiques et intégration réelle.

Les interfaces existantes sont réutilisées : `load_feature_variant`, `build_feature_variant_folds`, `get_feature_variant_fold_xy`, et le moteur `time_validation.py` inchangé. Le chargement VIX comporte explicitement `allow_experimental=True` et exige exactement le retard d’une séance SPY.

Les dates communes sont déterminées à partir des index avant estimation et calcul des scores. En cas de différence, le plan conserve les listes précises de dates propres à chaque variante et l’évaluation utilise leur intersection, sous garde d’effectif minimal. Les ensembles d’entraînement restent ceux des folds purgés respectifs.

Sur les données réelles, aucune date ne diffère : **1 993 dates de validation communes**, du 02/01/2015 au 30/11/2022. Chaque baseline produit une seule prédiction par date, soit **3 986 lignes** au total. Les validations de décembre 2022 dont les cibles atteignent 2023 restent exclues.

**4. Calcul de chaque baseline**

Fréquence historique, pour un entraînement de n dates et k labels positifs :

```text
p_global = k / n
p_validation(t) = p_global
```

Cette fréquence est recalculée dans chaque fold uniquement à partir de `y_train`. Aucun lissage n’est appliqué à cette référence.

VIX : partition de l’entraînement selon `VIX_level < 25` et `VIX_level >= 25`. Pour chaque groupe g de n_g dates et k_g positifs :

```text
si n_g >= 30 : p_g = (k_g + 1) / (n_g + 2)
sinon        : p_g = p_global
```

Le lissage correspond à la moyenne a posteriori sous une loi Beta(1,1), fixée avant les scores. Un groupe vide ou de moins de 30 observations utilise la fréquence globale non lissée de l’entraînement. Un entraînement entièrement dans une classe peut donc donner un repli global à zéro ou un ; ce serait une fréquence observée, pas une probabilité arbitraire issue du seuil.

Dans l’exécution réelle, les deux groupes dépassent toujours 30 observations : aucun repli n’a été nécessaire. Le comportement de repli et la frontière exacte 29/30 sont couverts par les tests.

Exemple réel du fold 2015 : fréquence globale `257/2315 = 0,111015`. Groupe bas : 132 positifs sur 1 824 dates, soit `(132+1)/(1824+2) = 0,072837`. Groupe haut : 125 positifs sur 491 dates, soit `(125+1)/(491+2) = 0,255578`.

À la prédiction, seule l’appartenance au groupe VIX est consultée. `y_validation` ne participe ni au seuil ni aux fréquences. Les paramètres et estimations de chaque entraînement sont conservés avec les prédictions.

**5. Justification du seuil VIX fixé à 25**

Le seuil 25 est une hypothèse expérimentale demandée par l’utilisateur et inscrite avant l’observation des scores. Il n’est pas présenté comme un optimum financier ou statistique. Le lissage Beta(1,1) et le support minimal de 30 dates ont également été déclarés avant les scores.

Le code refuse un autre seuil dans cette expérience. Aucune grille, optimisation, inversion a posteriori du classement, calibration Platt/isotonic ou adaptation aux résultats d’un fold n’est effectuée. Le résultat de validation d’une année peut devenir un label d’entraînement dans une année ultérieure, mais seulement quand les règles de disponibilité et de purge du moteur l’autorisent.

**6. Limites point-in-time du VIX**

La variante utilisée est exclusivement celle de l’étape 3 bis : VIXCLS local réindexé sur SPY puis retardé d’une séance. Aucun VIX contemporain brut n’est chargé pour calculer les probabilités de cette baseline.

Le retard ne certifie pas la date de disponibilité historique, ni l’absence de révisions du fournisseur. Le statut `experimental=True`, la variante `core_vix` et le retard d’une séance sont conservés dans les exports. Les réserves de disponibilité des clôtures du jeu `core` restent également applicables.

**7. Résultats réels par fold**

Les observations positives sont des dates dont la fenêtre future franchit le seuil de baisse, pas des épisodes indépendants de crise. Les taux et probabilités moyennes ci-dessous sont exprimés en pourcentage.

| Année | N train | N validation | Positifs | Négatifs | Taux réel | Moyenne F | Moyenne V |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 2315 | 252 | 32 | 220 | 12.70 % | 11.10 % | 8.30 % |
| 2016 | 2567 | 252 | 2 | 250 | 0.79 % | 10.79 % | 8.24 % |
| 2017 | 2819 | 251 | 0 | 251 | 0.00 % | 10.32 % | 7.25 % |
| 2018 | 3070 | 251 | 47 | 204 | 18.73 % | 9.48 % | 7.60 % |
| 2019 | 3321 | 252 | 0 | 252 | 0.00 % | 9.79 % | 7.31 % |
| 2020 | 3573 | 253 | 40 | 213 | 15.81 % | 9.46 % | 17.13 % |
| 2021 | 3826 | 252 | 6 | 246 | 2.38 % | 9.88 % | 8.63 % |
| 2022 | 4078 | 230 | 79 | 151 | 34.35 % | 9.27 % | 14.54 % |

Scores calculés sur exactement les mêmes dates pour les deux baselines ; F = fréquence historique, V = VIX expérimental :

| Année | AP F | AP V | Brier F | Brier V | ROC F | ROC V |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 0.126984 | 0.126984 | 0.111114 | 0.117125 | 0.500000 | 0.468182 |
| 2016 | 0.007937 | 0.007937 | 0.017868 | 0.014934 | 0.500000 | 0.476000 |
| 2017 | NaN | NaN | 0.010656 | 0.005254 | NaN | NaN |
| 2018 | 0.187251 | 0.187251 | 0.160737 | 0.170312 | 0.500000 | 0.463235 |
| 2019 | NaN | NaN | 0.009577 | 0.005571 | NaN | NaN |
| 2020 | 0.158103 | 0.139215 | 0.137139 | 0.148188 | 0.500000 | 0.403404 |
| 2021 | 0.023810 | 0.023810 | 0.028866 | 0.029038 | 0.500000 | 0.457317 |
| 2022 | 0.343478 | 0.295216 | 0.288394 | 0.293088 | 0.500000 | 0.300905 |

L’AP correspond à `average_precision_score`, moyenne des précisions pondérée par les incréments de rappel. Elle n’est pas l’intégrale trapézoïdale d’une courbe précision-rappel.

En 2017 et 2019, aucune date positive n’apparaît dans la validation. AP et ROC-AUC sont donc exportées en `NaN`, sans substituer un score arbitraire. Par politique prudente, le code indique aussi ces deux métriques en NaN dans le cas synthétique où il n’existe que des positifs ; l’AP mathématique de ce cas peut être triviale, mais n’évalue pas une discrimination entre deux classes. Le Brier reste défini dans les deux situations.

**8. Résultats agrégés hors échantillon**

Les prédictions des huit validations sont concaténées par baseline. Les métriques agrégées sont recalculées sur ces 1 993 prédictions ; ce ne sont pas des moyennes des huit scores annuels.

Composition commune : **206 dates positives et 1 787 négatives**, taux réel **10,3362 %**.

| Baseline | N | Probabilité moyenne | AP | Brier | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| historical_frequency | 1993 | 10.0192 % | 0.088309 | 0.093446 | 0.272661 |
| vix_threshold_25 | 1993 | 9.8290 % | 0.091453 | 0.095820 | 0.349705 |

Le Parquet exporté contient les 3 986 prédictions, dates, fold/année, baseline, variante, statut expérimental, probabilité, cible observée, paramètres, estimations d’entraînement et bornes temporelles de l’entraînement. Aucune prédiction d’évaluation 2023+.

**9. Comparaison AP / Brier / ROC-AUC**

La fréquence historique obtient un Brier agrégé inférieur : **0,093446 contre 0,095820**. Elle a aussi un meilleur Brier dans cinq validations sur huit. Le VIX a un Brier inférieur en 2016, 2017 et 2019.

Le VIX présente une AP agrégée légèrement supérieure, **0,091453 contre 0,088309**, et une ROC-AUC agrégée supérieure, **0,349705 contre 0,272661**. Ces résultats ne constituent pas une démonstration de capacité prédictive : les deux ROC-AUC agrégées sont inférieures à 0,5 et les AP inférieures au taux positif agrégé de 0,103362, repère mathématique d’un score constant sur cette population.

La fréquence est constante à l’intérieur d’un fold : sa ROC-AUC annuelle vaut 0,5 lorsque deux classes sont présentes. Elle change entre folds. Son classement agrégé peut donc différer de 0,5 et refléter l’association entre les fréquences d’entraînement passées et les régimes futurs. Il ne faut pas confondre cette agrégation avec une discrimination quotidienne dans chaque année.

Le VIX n’améliore pas l’AP dans les validations annuelles : il l’égale dans plusieurs périodes et la dégrade en 2020 et 2022. La petite différence d’AP agrégée ne prouve donc pas un gain stable d’une année à l’autre. Aucun test de significativité, intervalle de confiance ou réoptimisation n’est entrepris ici.

**10. Comportement pendant les années comportant davantage de labels de crise**

En 2015 : taux observé 12,70 %, probabilités moyennes 11,10 % pour F et 8,30 % pour V. Le Brier est meilleur pour F.

En 2018 : taux observé 18,73 %, moyennes 9,48 % et 7,60 %. Les deux baselines sous-estiment la fréquence réalisée ; le Brier VIX est plus élevé.

En 2020 : taux observé 15,81 %. La moyenne VIX, 17,13 %, est plus proche du taux réalisé que celle de F, 9,46 %, mais son Brier est néanmoins plus mauvais : **0,148188 contre 0,137139**. Une bonne moyenne annuelle ne garantit pas une bonne attribution des probabilités aux différentes dates.

En 2022, jusqu’au 30 novembre : taux observé 34,35 %, moyennes 9,27 % et 14,54 %. Les deux baselines sous-estiment fortement la fréquence réalisée. La hausse de moyenne de V ne s’accompagne pas d’un meilleur Brier ou classement : ROC-AUC VIX **0,300905**, Brier **0,293088**, contre **0,288394** pour F.

En 2017 et 2019, années sans label positif selon cette définition, le Brier plus faible du VIX reflète notamment des probabilités plus basses ; cela ne démontre pas une meilleure anticipation des baisses.

**11. Contrôles de data leakage**

Les huit folds proviennent du moteur existant, avec entraînement croissant, purge de 21 séances, `target_end_session < première séance de validation` et labels d’entraînement disponibles avant cette première décision. L’embargo post-validation ne crée aucun second gap dans cette topologie.

Le chargement de y impose dates et fins de cible antérieures à 2023. Les nouveaux estimateurs et fonctions de métriques rejettent aussi les dates 2023+ avant d’inspecter les valeurs des labels. Les sorties sont contrôlées avant export.

Les fréquences sont estimées uniquement sur train ; modifier `y_validation` seul dans un fold ne change aucune probabilité prédite. Les labels de validation ne sont utilisés que pour le calcul des métriques et pour vérifier la cohérence des cibles entre variantes.

Les dates de comparaison sont définies avant estimation, sans consulter les labels. Les listes de différences sont vides dans cette exécution ; aucune date de validation disponible n’a été retirée pour améliorer un score.

Aucun shuffle, ajustement sur validation, calibrateur, sélection de features ou réoptimisation du seuil n’a lieu. Les contrôles ne suppriment pas les réserves historiques de disponibilité/révision du VIX expérimental.

**12. Résultats pytest**

Les **130 tests préexistants** sont conservés et réussissent. **38 nouveaux tests** couvrent probabilités, calcul train-only, invariance à y_validation, protection 2023+, VIX retardé, seuil et égalité à 25, groupes vides/petits, lissage/repli, métriques et NaN, absence de shuffle, folds, dates communes, reproductibilité, refus d’écrasement et préservation des anciens modules.

Résultat final réel :

```text
168 passed, 100 warnings in 7.49s
```

Les avertissements sont les dépréciations pandas/exchange-calendars déjà identifiées : 64 dans le pipeline, 16 dans la validation temporelle, 12 dans les indicateurs et 8 dans les tests des baselines. Aucun test ignoré ou en échec dans cette exécution.

Vérifications supplémentaires des exports : chaque probabilité a été reproduite depuis son entraînement purgé, les cibles exportées correspondent aux validations, et l’AP agrégée a été recalculée indépendamment par seuils de scores ex aequo. Le Brier agrégé a été vérifié par moyenne des erreurs quadratiques.

**13. Fichiers créés ou modifiés**

Nouveaux modules et test :

- `src/models/ml1/baselines.py`
- `src/models/ml1/evaluate_baselines.py`
- `tests/ml1/test_baselines.py`

Nouveau rapport : `reports/ml1/step4_baselines.md`.

Résultats :

- `reports/ml1/baselines/baseline_metrics.csv` : 18 lignes, 16 évaluations annuelles et deux agrégations.
- `reports/ml1/baselines/baseline_predictions.parquet` : 3 986 lignes de prédictions.
- `reports/ml1/baselines/baseline_comparison.md` : protocole, alignement, tables et environnement d’exécution.

Aucun fichier préexistant modifié. Les données brutes, Parquet `core` et `core_vix`, pipelines, tests et rapports précédents sont conservés. Aucun composant d’un autre membre modifié ; aucune dépendance installée ou donnée téléchargée.

**14. Commandes de relance**

Depuis la racine du dépôt :

```bash
# Le dossier principal existe déjà : reconstruire dans un dossier absent
.venv/bin/python -B -m src.models.ml1.evaluate_baselines --output-dir reports/ml1/baselines_rebuild_001

# Nouveaux tests
.venv/bin/python -B -m pytest tests/ml1/test_baselines.py -q -p no:cacheprovider

# Tous les tests ML 1
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Sur un dossier principal encore absent, la commande par défaut est `.venv/bin/python -B -m src.models.ml1.evaluate_baselines`. Un dossier existant est refusé, sans écrasement silencieux. Les trois résultats sont préparés temporairement puis publiés ensemble.

Environnement utilisé : Python 3.12.13, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1 et pyarrow 26.0.1. Aucune connexion réseau nécessaire.

**15. Limites méthodologiques**

Les fenêtres futures de 21 séances se chevauchent : les dates positives ne sont pas autant d’épisodes indépendants. Les entraînements croissants se recoupent également. Les métriques ne doivent pas être interprétées comme provenant d’essais indépendants.

Le nombre de positifs varie fortement selon les années, dont seulement deux en 2016 et aucun en 2017/2019. Les métriques annuelles doivent être lues avec leurs effectifs. La validation 2022 s’arrête au 30/11 pour ne pas utiliser les résultats 2023.

Le VIX est un instantané retardé sous hypothèse expérimentale, non une source historiquement certifiée point-in-time. Le seuil 25, le lissage et le support minimal sont des hypothèses fixes, pas des paramètres démontrés optimaux. Les réserves du jeu `core` sur les prix ajustés et leur disponibilité demeurent.

Aucune calibration avancée ou analyse SHAP, aucun coût de transaction, stratégie d’allocation, backtest ou test final. Les scores observés ne démontrent pas une capacité prédictive exploitable et n’autorisent pas à inverser ou optimiser a posteriori les règles à cette étape.

**16. État Git final**

Branche toujours `feature/ml1-crisis-detection`, même commit `8d9ff5f`. Les quinze fichiers non suivis préexistants sont préservés ; les sept fichiers de cette étape s’y ajoutent. Les fichiers de résultats sont visibles comme nouveaux fichiers non suivis dans Git.

Aucun fichier suivi modifié, aucun ajout à l’index, commit, push ou merge. Les empreintes confirment la préservation de tous les fichiers préexistants et des sorties de données.

**17. Verdict : PRÊT AVEC RÉSERVES**

Les deux baselines, leur évaluation chronologique et les exports sont implémentés, exécutés et testés. La fréquence historique constitue désormais une référence chiffrée ; la baseline VIX reste expérimentale et n’apporte pas ici de gain constant sur les métriques.

Le test final 2023+ reste fermé. Aucun entraînement Logistic Regression, Random Forest ou XGBoost, aucune calibration avancée ni SHAP n’a commencé. Toute étape suivante attend la validation de l’utilisateur.
