**Rapport ML 1 — étape 6 : évaluation des modèles — 15 novembre 2026**

L’évaluation des probabilités OOF enregistrées désigne **Logistic Regression comme CANDIDAT PROVISOIRE**, selon l’AP agrégée prioritaire. Son Brier reste moins bon que celui des deux baselines et les résultats annuels sont variables : aucune supériorité statistique ou opérationnelle n’est établie. Aucun réentraînement de production, réglage ou calibration.

**1. État Git initial**

Dépôt : `/Users/mac/Allocation ML/asset-allocation-ml`. Branche : `feature/ml1-crisis-detection`. HEAD : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`.

Aucun fichier suivi modifié. Les trois fichiers non suivis de l’étape 5 étaient présents : `src/models/ml1/train_ml1.py`, `tests/ml1/test_training.py`, `reports/ml1/step5_model_training.md`. Ils ont été conservés intégralement. Les empreintes initiales portent sur 80 fichiers préexistants : fichiers suivis, ces trois fichiers et les données générées des étapes précédentes.

**2. Sources et prédictions utilisées**

Sources d’évaluation :

- `data/processed/ml1_training/ml_models_oof_predictions.parquet` : 5 979 lignes, 14 colonnes, trois modèles core.
- `data/processed/ml1_training/training_metadata.json` : configurations fixes, empreintes, 24 résultats de fit et huit folds.
- `reports/ml1/baselines/baseline_predictions.parquet` : 3 986 lignes, deux baselines.
- `reports/ml1/baselines/baseline_metrics.csv` : 18 lignes historiques de référence.
- `data/processed/ml1/` : labels admissibles exclusivement via `load_development_data`, chargeur protégé de l’étape 3.

Les API réelles de `time_validation.py`, `train_ml1.py` et `evaluate_baselines.py` ont été inspectées. Le nouveau module n’importe ni les fonctions d’entraînement ni l’évaluateur des baselines : ses métriques sont recalculées indépendamment depuis les sorties existantes.

Les trois modèles utilisent exclusivement les dix features core de l’étape 5. La baseline `vix_threshold_25` reste **EXPÉRIMENTALE**, non certifiée point-in-time ; aucune feature VIX n’est introduite dans les modèles ML.

**3. Vérification d’intégrité des données**

Les schémas Parquet correspondent aux schémas inspectés. L’empreinte du Parquet ML correspond aux métadonnées et au rapport de l’étape 5 ; l’empreinte du JSON correspond également à ce rapport. Les trois fichiers de baselines correspondent exactement à leurs contenus enregistrés dans HEAD.

Les empreintes des modules référencés par l’entraînement et des données core concordent. Les empreintes des DataFrames X, y pré-2023 et des contrôles temporels sont reproduites depuis le chargeur protégé. Les huit folds reconstruits correspondent aux métadonnées : dates exactes, indices, effectifs, purge de 21 séances et disponibilité des labels.

Identifiants de configuration contrôlés, y compris recalcul de leur empreinte JSON et cohérence des lignes/records :

```text
logistic_regression_102c776acc68dede
random_forest_c3d09bf77ca0b0b7
xgboost_ed397cc469cc15e0
```

Toutes les probabilités sont finies et dans `[0,1]`. Aucun doublon par approche/date. Les 80 fichiers préexistants contrôlés sont inchangés. Aucun résultat d’étape 4 ou 5 n’est réécrit.

**4. Alignement des cinq approches**

Les cinq approches couvrent exactement les mêmes **1 993 dates**, du **02/01/2015 au 30/11/2022** : 9 965 couples approche/date en mémoire pour l’évaluation. Aucun jour supprimé, ajouté, interpolé ou choisi d’après les scores.

Les prédictions respectent les huit années et identifiants de folds existants. Les labels des deux baselines correspondent exactement, date par date, aux labels récupérés via le chargeur protégé. Les trois modèles sont joints à ces mêmes labels.

Le module établit un diagnostic par approche avant tout score : effectif, dates manquantes et dates supplémentaires. Toute différence interrompt le calcul comparatif ; aucune intersection silencieuse ne réduit l’échantillon. Ici, toutes les listes de différences sont vides.

**5. Tableau des métriques annuelles**

Les effectifs et fréquences réelles ci-dessous sont communs aux cinq approches. Les positifs sont des dates dont la fenêtre future franchit le seuil, pas des épisodes indépendants de crise.

| Année | N par approche | Positifs | Négatifs | Taux réel (%) |
| --- | --- | --- | --- | --- |
| 2015 | 252 | 32 | 220 | 12.698413 |
| 2016 | 252 | 2 | 250 | 0.793651 |
| 2017 | 251 | 0 | 251 | 0.000000 |
| 2018 | 251 | 47 | 204 | 18.725100 |
| 2019 | 252 | 0 | 252 | 0.000000 |
| 2020 | 253 | 40 | 213 | 15.810277 |
| 2021 | 252 | 6 | 246 | 2.380952 |
| 2022 | 230 | 79 | 151 | 34.347826 |

AP = `average_precision_score`, Brier binaire = moyenne de `(y-p)²`, ROC-AUC complémentaire. Les probabilités moyennes sont exprimées en pourcentage. `vix_threshold_25` est EXPÉRIMENTALE dans toutes les lignes qui suivent.

| Année | Approche | AP | Brier | ROC-AUC | P moyenne (%) |
| --- | --- | --- | --- | --- | --- |
| 2015 | historical_frequency | 0.126984 | 0.111114 | 0.500000 | 11.101512 |
| 2015 | logistic_regression | 0.109532 | 0.118088 | 0.428693 | 7.803863 |
| 2015 | random_forest | 0.083307 | 0.138738 | 0.235369 | 9.051537 |
| 2015 | vix_threshold_25 | 0.126984 | 0.117125 | 0.468182 | 8.298910 |
| 2015 | xgboost | 0.086969 | 0.170073 | 0.280256 | 11.740198 |
| 2016 | historical_frequency | 0.007937 | 0.017868 | 0.500000 | 10.790806 |
| 2016 | logistic_regression | 0.016199 | 0.013463 | 0.644000 | 7.316462 |
| 2016 | random_forest | 0.017305 | 0.015965 | 0.668000 | 6.215772 |
| 2016 | vix_threshold_25 | 0.007937 | 0.014934 | 0.476000 | 8.243233 |
| 2016 | xgboost | 0.031830 | 0.016184 | 0.822000 | 4.956162 |
| 2017 | historical_frequency | NaN | 0.010656 | NaN | 10.322810 |
| 2017 | logistic_regression | NaN | 0.001307 | NaN | 3.534670 |
| 2017 | random_forest | NaN | 0.000540 | NaN | 1.933545 |
| 2017 | vix_threshold_25 | NaN | 0.005254 | NaN | 7.248264 |
| 2017 | xgboost | NaN | 0.000113 | NaN | 0.893369 |
| 2018 | historical_frequency | 0.187251 | 0.160737 | 0.500000 | 9.478827 |
| 2018 | logistic_regression | 0.244871 | 0.171339 | 0.419065 | 5.451555 |
| 2018 | random_forest | 0.166428 | 0.178052 | 0.469545 | 4.474391 |
| 2018 | vix_threshold_25 | 0.187251 | 0.170312 | 0.463235 | 7.596438 |
| 2018 | xgboost | 0.189936 | 0.181208 | 0.484251 | 2.441397 |
| 2019 | historical_frequency | NaN | 0.009577 | NaN | 9.786209 |
| 2019 | logistic_regression | NaN | 0.008600 | NaN | 8.539746 |
| 2019 | random_forest | NaN | 0.005549 | NaN | 5.192815 |
| 2019 | vix_threshold_25 | NaN | 0.005571 | NaN | 7.314257 |
| 2019 | xgboost | NaN | 0.002736 | NaN | 3.387736 |
| 2020 | historical_frequency | 0.158103 | 0.137139 | 0.500000 | 9.459838 |
| 2020 | logistic_regression | 0.174286 | 0.169045 | 0.525704 | 16.104531 |
| 2020 | random_forest | 0.147284 | 0.171551 | 0.470188 | 19.018992 |
| 2020 | vix_threshold_25 | 0.139215 | 0.148188 | 0.403404 | 17.127661 |
| 2020 | xgboost | 0.175694 | 0.183661 | 0.573474 | 13.603779 |
| 2021 | historical_frequency | 0.023810 | 0.028866 | 0.500000 | 9.879770 |
| 2021 | logistic_regression | 0.163362 | 0.022227 | 0.924797 | 2.478006 |
| 2021 | random_forest | 0.020475 | 0.036265 | 0.363144 | 10.491169 |
| 2021 | vix_threshold_25 | 0.023810 | 0.029038 | 0.457317 | 8.628727 |
| 2021 | xgboost | 0.061884 | 0.027569 | 0.789295 | 5.241884 |
| 2022 | historical_frequency | 0.343478 | 0.288394 | 0.500000 | 9.269250 |
| 2022 | logistic_regression | 0.292870 | 0.309998 | 0.436751 | 5.878472 |
| 2022 | random_forest | 0.238690 | 0.308477 | 0.268924 | 8.974364 |
| 2022 | vix_threshold_25 | 0.295216 | 0.293088 | 0.300905 | 14.537106 |
| 2022 | xgboost | 0.345236 | 0.335143 | 0.457624 | 1.252127 |

En 2017 et 2019, aucune classe positive : AP et ROC-AUC restent `NaN` pour les cinq approches, avec justification explicite dans le CSV. Le Brier reste défini. La politique applique également NaN aux métriques de classement d’une période ne contenant que des positifs ; ce cas n’est pas observé ici. Aucun remplacement par zéro ou 0,5.

**6. Tableau des métriques agrégées**

Ces scores sont recalculés sur les prédictions individuelles concaténées des huit validations, sans moyenne des scores annuels. Fréquence positive commune : **10,336177 %**.

| Approche | N | Positifs | Négatifs | P moyenne (%) | AP | Brier | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 1993 | 206 | 1787 | 7.159475 | 0.111118 | 0.099509 | 0.545417 |
| random_forest | 1993 | 206 | 1787 | 8.170610 | 0.097037 | 0.104717 | 0.499764 |
| xgboost | 1993 | 206 | 1787 | 5.493687 | 0.086914 | 0.112210 | 0.411413 |
| historical_frequency | 1993 | 206 | 1787 | 10.019151 | 0.088309 | 0.093446 | 0.272661 |
| vix_threshold_25 | 1993 | 206 | 1787 | 9.828954 | 0.091453 | 0.095820 | 0.349705 |

Le classement AP des modèles ML est Logistic Regression, Random Forest, XGBoost. Logistic Regression présente aussi le meilleur Brier et la meilleure ROC-AUC parmi ces trois modèles, mais le meilleur Brier des cinq approches reste celui de la fréquence historique.

**7. Comparaison aux baselines**

Recalcul indépendant des métriques à partir des prédictions enregistrées :

| Baseline | AP recalculée | Brier recalculé | ROC-AUC recalculée |
| --- | ---: | ---: | ---: |
| historical_frequency | 0.088309334928 | 0.093445925817 | 0.272660965658 |
| vix_threshold_25 — EXPÉRIMENTALE | 0.091452988253 | 0.095820039439 | 0.349704717458 |

Les 18 lignes historiques concordent à `1e-12`, y compris les effectifs, fréquences réelles et probabilités moyennes. Les six valeurs agrégées respectent la tolérance `1e-6` par rapport aux valeurs arrondies demandées ; l’écart maximal est inférieur à `3,35e-7`. Aucun désaccord nécessitant de bloquer la recommandation.

L’AP agrégée de Logistic Regression dépasse les deux baselines. Celle de Random Forest les dépasse aussi, avec un Brier plus élevé. XGBoost a ici une AP agrégée inférieure aux deux baselines et le Brier agrégé le plus élevé.

Les trois modèles ont un Brier agrégé moins bon que les deux baselines. Logistic Regression améliore le Brier de la fréquence historique dans quatre années sur huit, et celui du VIX dans trois années sur huit : le bénéfice n’est pas uniforme.

La fréquence historique est constante dans chaque fold, mais varie entre folds : sa ROC-AUC annuelle vaut 0,5 quand deux classes sont présentes, alors que sa ROC-AUC agrégée peut différer de 0,5. L’agrégation reflète aussi les différences entre régimes annuels.

**8. Analyse des années 2018, 2020 et 2022**

**2018 :** fréquence réelle 18,73 %. Logistic Regression obtient l’AP la plus élevée des cinq approches, **0,244871**, mais sa probabilité moyenne est seulement **5,45 %** et son Brier **0,171339**, contre **0,160737** pour la fréquence historique. Sa ROC-AUC est **0,419065**. L’AP et la ROC-AUC pondèrent différemment les positions du classement ; leurs indications peuvent diverger. Les moyennes RF et XGBoost, 4,47 % et 2,44 %, sont encore plus basses que la fréquence réalisée.

**2020 :** fréquence réelle 15,81 %. XGBoost et Logistic Regression ont des AP proches, **0,175694** et **0,174286** ; aucun avantage statistique ne peut être déduit de cet écart. La moyenne Logistic Regression, **16,10 %**, est proche de la fréquence annuelle, mais son Brier **0,169045** reste moins bon que celui de la fréquence historique **0,137139**. Une bonne moyenne ne garantit pas une bonne attribution quotidienne des probabilités. Random Forest prédit 19,02 % en moyenne, XGBoost 13,60 %.

**2022, jusqu’au 30 novembre :** fréquence réelle **34,35 %**. Moyennes ML : Logistic Regression **5,88 %**, Random Forest **8,97 %**, XGBoost **1,25 %**. Tous sous-estiment fortement la fréquence réalisée au niveau annuel. XGBoost obtient l’AP la plus élevée de cette année, **0,345236**, à peine supérieure à la fréquence historique **0,343478**, mais son Brier atteint **0,335143**, contre **0,288394** pour cette baseline. Sa ROC-AUC reste sous 0,5. Logistic Regression et Random Forest ont aussi un Brier plus élevé que les deux baselines en 2022.

**9. Analyse AP, Brier et ROC-AUC**

**Résultats calculés :** Logistic Regression a la meilleure AP agrégée ML (**0,111118**), un Brier de **0,099509** et une ROC-AUC de **0,545417**. Sa moyenne prédite est **7,1595 %**, contre **10,3362 %** de positifs. Random Forest et XGBoost prédisent respectivement **8,1706 %** et **5,4937 %** en moyenne.

**Interprétation :** l’AP favorise le classement utile pour retrouver les dates positives ; elle est prioritaire dans ce projet et dépend de leur fréquence. Le Brier évalue l’erreur globale des probabilités, sans isoler à lui seul la calibration. La ROC-AUC examine la discrimination positive/négative et reste complémentaire. Une AP élevée n’implique ni bonne calibration ni qualité uniforme sur tous les régimes.

La préférence AP de Logistic Regression est limitée : son AP dépasse modestement le repère d’un score constant sur cette population, soit la prévalence de 0,103362. Le meilleur Brier des baselines constitue un désaccord matériel avec ce classement.

Stabilité : parmi les trois ML, Logistic Regression arrive en tête d’AP en 2015, 2018 et 2021 ; XGBoost en 2016, 2020 et 2022 ; Random Forest dans aucune de ces six années à deux classes. Logistic Regression dépasse l’AP des deux baselines dans quatre de ces six années, mais pas en 2015 ou 2022. En 2021, son AP de 0,163362 et sa ROC-AUC de 0,924797 reposent sur seulement six dates positives potentiellement liées. En 2016, il n’y a que deux dates positives.

Bornes des probabilités : Logistic Regression `[0,007759 ; 0,860732]`, Random Forest `[0 ; 0,664952]`, XGBoost `[0,000486 ; 0,824964]`. Random Forest contient huit probabilités exactement nulles ; aucun modèle ne prédit exactement un. Ces sorties sont valides numériquement, mais leurs amplitudes ne démontrent pas une confiance justifiée. Aucun écrêtage ni transformation après observation des scores.

**10. Candidat provisoire et justification**

Méthode fixée avant le calcul : classer les trois modèles selon l’AP agrégée, examiner ensuite le Brier et la stabilité annuelle, puis comparer aux baselines. Un leader AP unique, au-delà d’une égalité numérique de `1e-12`, doit dépasser l’AP des deux baselines pour devenir candidat. Sinon, aucune sélection unique n’est forcée. Le seuil numérique traite les égalités ; il n’est pas un test de significativité.

**CANDIDAT PROVISOIRE pour l’étape 7 : Logistic Regression**, configuration `logistic_regression_102c776acc68dede`.

Justification : meilleure AP agrégée ML, devant Random Forest de **0,014080** ; AP supérieure aux deux baselines ; meilleur Brier parmi les trois ML. Réserves déterminantes : Brier moins bon que les deux baselines, résultats annuels variables, forte sous-estimation moyenne en 2022 et absence de démonstration statistique de supériorité.

Cette désignation identifie une piste pour la prochaine étape ; elle ne constitue pas un vainqueur définitif ni une validation d’usage. Les configurations et données restent inchangées après les scores. Le candidat et sa justification sont également enregistrés dans `model_comparison.md`.

**11. Limites statistiques et méthodologiques**

Les cibles futures de 21 séances se chevauchent : 206 dates positives ne correspondent pas à 206 crises indépendantes. Les entraînements croissants se recoupent aussi. Aucun test de significativité ni intervalle de confiance fondé sur l’indépendance des dates n’est calculé.

La sélection sur 2015–2022 introduit un biais de sélection potentiel. Les métriques présentées ne sont pas une estimation finale indépendante de cette sélection. La fréquence des labels varie fortement entre années ; la validation 2022 ne couvre que janvier à novembre.

Le VIX reste expérimental, sans preuve de disponibilité historique point-in-time. Les réserves antérieures sur les prix ajustés core et leur disponibilité sont conservées. Aucune conclusion de profitabilité, aucune stratégie, optimisation ou backtest.

Toute calibration ultérieure devra utiliser sa propre séparation temporelle. Le label d’une prédiction ne devra pas servir à calibrer cette même prédiction. Aucun modèle final n’est entraîné ou sauvegardé à cette étape.

**12. Vérification du data leakage**

L’évaluateur ne construit aucun estimateur et ne possède aucun appel de fit, réglage ou calibration. Il lit les probabilités existantes, vérifie leur provenance, joint les labels admissibles et calcule les scores.

Les bornes temporelles des lignes ML correspondent exactement aux folds existants : purge de 21 séances, fin de cible train avant validation et label disponible avant la première décision. Les identifiants de configuration ne sont pas modifiés. Les dates de comparaison sont vérifiées avant les métriques ; les labels des baselines sont contrôlés contre la même source de vérité.

Conformément à votre clarification, la suite complète de tests peut effectuer les fits nécessaires aux tests existants. Cette exception concerne uniquement les tests : aucun réentraînement de production ni régénération des probabilités livrées n’a eu lieu.

**13. Protection du test final**

Le seul chargeur de cibles de développement est `load_development_data`, avec filtres Parquet sur `Date >= 2005`, `Date < 2023-01-01` et `target_end_session < 2023-01-01`. Aucune lecture globale non filtrée des labels des targets.

Pour les baselines, les dates sont inspectées sans lire `observed_target` ; les labels enregistrés sont ensuite lus explicitement avec `prediction_date < 2023-01-01`. Toute prédiction 2023+ est rejetée. Le test final n’est utilisé ni pour calculer ces métriques ni pour départager les modèles.

Les tests surveillent les appels réels de lecture Parquet et utilisent des labels sentinelles pour vérifier le rejet des dates finales avant inspection de leurs valeurs.

**14. Graphiques et fichiers exportés**

Nouveau dossier `reports/ml1/evaluation/`, absent avant cette étape :

| Fichier | Taille | Contenu |
| --- | ---: | --- |
| `model_metrics.csv` | 8 880 octets | 45 lignes : 40 annuelles et cinq agrégées |
| `model_comparison.md` | 11 785 octets | Tables, candidat, justification, alignement, versions et empreintes |
| `precision_recall_curves.png` | 112 794 octets | Courbes PR OOF des cinq approches et prévalence |
| `brier_by_year.png` | 210 700 octets | Brier par année et approche |
| `ap_by_year.png` | 159 148 octets | AP annuelle avec interruptions pour les années non évaluables |

Les trois figures ont été ouvertes et inspectées : modèles identifiés, VIX marqué EXPÉRIMENTAL, période, fenêtres chevauchantes et fermeture du test final indiquées. Les courbes PR sont descriptives ; les AP sont calculées par `average_precision_score`, pas par intégration trapézoïdale des courbes.

Fontconfig a signalé des répertoires de cache non inscriptibles pendant la génération. La commande a néanmoins réussi et les trois PNG sont lisibles et correctement rendus ; aucune bibliothèque ou configuration globale n’a été modifiée.

Les sorties ont été préparées dans un dossier temporaire neuf puis publiées ensemble. Tout dossier de destination existant est refusé. Aucun graphique antérieur n’a été modifié.

**15. Résultats des tests pytest**

Les **216 tests précédents réussissent toujours**. **40 nouveaux tests** couvrent le chargement des sorties, schémas, empreintes et provenance, dates identiques, doublons, probabilités invalides, jointure des labels, protection 2023+, AP/Brier/ROC synthétiques, périodes à classe unique, concordance des baselines, VIX expérimental, recommandation reproductible, absence d’appels d’entraînement/tuning, figures, refus d’écrasement et conservation des modules.

Résultat réel de la suite complète, avec les fits de tests que vous avez explicitement autorisés :

```text
256 passed, 108 warnings in 32.48s
```

Aucun échec ni test ignoré. Les 108 avertissements sont les dépréciations pandas déjà connues : pipeline 64, validation temporelle 16, indicateurs 12, baselines 8, entraînement 8. Les nouveaux tests seuls ont aussi réussi : 40 tests, quatre avertissements de cette même dépréciation lors d’un chargement de calendrier non encore mis en cache. Aucun avertissement masqué.

Contrôle indépendant supplémentaire des **45 lignes exportées** : AP reconstruite par seuils de scores et traitement des ex æquo, Brier par erreur quadratique moyenne, ROC-AUC par rangs moyens. Écart numérique maximal **1,11 × 10⁻¹⁶** par rapport au CSV.

**16. Commandes de reproduction**

Depuis la racine, sur `feature/ml1-crisis-detection` :

```bash
# Tests de l’évaluation : aucun fit
.venv/bin/python -B -m pytest tests/ml1/test_evaluation.py -q -p no:cacheprovider

# Suite complète : comporte les fits de tests existants autorisés
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider

# Recalcul depuis les seules prédictions enregistrées, dans un dossier absent
.venv/bin/python -B -m src.models.ml1.evaluate_ml1 --output-dir reports/ml1/evaluation_rebuild_001
```

Commande exécutée pour la livraison : `.venv/bin/python -B -m src.models.ml1.evaluate_ml1`. Le dossier par défaut existe désormais ; une relance vers ce même dossier est refusée. Aucune commande d’entraînement de production n’a été lancée.

Environnement : Python 3.12.13, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, pyarrow 26.0.1, matplotlib 3.11.2 et pytest 9.2.0. Aucune dépendance installée ou donnée téléchargée.

**17. Fichiers créés ou modifiés**

Huit nouveaux fichiers :

- `src/models/ml1/evaluate_ml1.py`.
- `tests/ml1/test_evaluation.py`.
- `reports/ml1/step6_model_evaluation.md`.
- Les cinq fichiers de `reports/ml1/evaluation/` détaillés ci-dessus.

Aucun fichier préexistant modifié. Les trois fichiers non suivis de l’étape 5 sont préservés, ainsi que toutes les prédictions, métriques, données brutes et composants des autres membres. Les empreintes des 80 fichiers protégés concordent.

**18. État Git final**

Branche inchangée : `feature/ml1-crisis-detection`. HEAD inchangé : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`.

```text
?? reports/ml1/evaluation/
?? reports/ml1/step5_model_training.md
?? reports/ml1/step6_model_evaluation.md
?? src/models/ml1/evaluate_ml1.py
?? src/models/ml1/train_ml1.py
?? tests/ml1/test_evaluation.py
?? tests/ml1/test_training.py
```

Onze fichiers non suivis au total : les trois précédents et les huit nouveaux. Aucun fichier suivi modifié, aucun changement à l’index, commit, push ou merge. Aucun modèle final sauvegardé.

**19. Verdict : PRÊT AVEC RÉSERVES**

L’étape 6 est terminée : cinq approches évaluées sur les mêmes dates, baselines vérifiées, 45 lignes de métriques contrôlées indépendamment, trois figures inspectées et 256 tests réussis.

Logistic Regression est seulement le **CANDIDAT PROVISOIRE** fondé sur l’AP de validation. Le Brier moins favorable que les baselines, la variabilité annuelle et la sélection sur ces mêmes validations empêchent de conclure à une supériorité générale ou statistiquement établie.

Le test final 2023+ reste fermé. Aucun réentraînement de production, réglage, calibration ou backtest. **L’étape 7 n’a pas commencé ; toute suite attend votre validation.**
