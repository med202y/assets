**Rapport ML 1 — étape 7 : calibration chronologique et SHAP — 15 novembre 2026**

La calibration exploratoire et les explications SHAP sont terminées. Les deux méthodes améliorent le Brier agrégé du modèle brut, mais restent moins bonnes que la baseline historique sur ce critère. **Aucune méthode de calibration n’est sélectionnée.** La reconstruction de Logistic Regression pour SHAP reproduit exactement ses probabilités OOF 2022. **292 tests réussissent.**

**1. État Git initial**

Dépôt : `/Users/mac/Allocation ML/asset-allocation-ml`. Branche : `feature/ml1-crisis-detection`. HEAD : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`.

Aucun fichier suivi modifié. Les onze fichiers non suivis des étapes 5 et 6 étaient présents. Les empreintes SHA-256 de **88 fichiers préexistants**, incluant les fichiers suivis/non suivis, rapports et données générées, ont été conservées et comparées après intervention : aucun changement.

**2. Sources utilisées**

Les interfaces réelles de `time_validation.py`, `train_ml1.py`, `evaluate_ml1.py` et `baselines.py` ont été inspectées. Les modules précédents sont inchangés.

Sources : les probabilités OOF de `data/processed/ml1_training/ml_models_oof_predictions.parquet`, leurs métadonnées `training_metadata.json`, les métriques de `reports/ml1/evaluation/model_metrics.csv`, les baselines enregistrées et le dataset core via le chargeur protégé de l’étape 3.

Les schémas, empreintes, dates, identifiants de configuration et métriques enregistrées sont contrôlés grâce aux interfaces existantes. Seules les probabilités OOF de **Logistic Regression** servent aux calibrateurs. La baseline historique est une référence enregistrée, sans nouveau fit pour cette livraison. Les 5 979 prédictions des trois modèles restent intactes.

Configuration de base : `logistic_regression_102c776acc68dede`. Dix features core, dans l’ordre : `SPY_ret_1d`, `SPY_ret_5d`, `SPY_ret_21d`, `SPY_ret_63d`, `SPY_vol_5d`, `SPY_vol_21d`, `SPY_vol_63d`, `SPY_drawdown_252d`, `SPY_vs_ma200`, `SPY_IEF_corr_63d`. Aucun VIX ou macro dans les calibrateurs ou le modèle expliqué.

**3. Protocole temporel de calibration**

Pour l’année Y, sélectionner uniquement les prédictions OOF d’années strictement antérieures à Y. Exiger simultanément :

```text
prediction_date.year < Y
target_end_session < première date de validation de Y
label_known_at < première décision de validation de Y
```

La disponibilité du label est calculée selon la convention existante de 17 h, heure de New York, à la fin de sa cible. Les dates admissibles sont sélectionnées avant d’inspecter les valeurs des labels.

Platt et Isotonic utilisent exactement le même historique admissible, croissant depuis 2015. Les 21 dernières séances de l’année précédente sont exclues à chaque frontière de 2016 à 2022. Aucune calibration rétroactive d’une année sur ses propres labels.

En **2015**, aucun historique OOF : conserver exactement les probabilités brutes pour les deux méthodes, statut **`not_calibrated_no_history`**. Ce repli n’est jamais compté comme une calibration réussie.

Supports fixés avant tout score : au moins **200 observations, 10 positifs et 10 négatifs**. Sinon : probabilités brutes, statut `not_calibrated_insufficient_support`, aucun calibrateur ajusté. Un problème de convergence entraîne aussi un repli documenté, sans modifier les paramètres. Aucun repli de support ou de convergence n’a été nécessaire en 2016–2022.

**4. Effectifs historiques admissibles par année**

| Année | OOF antérieures | Exclues immatures | Admissibles | Positifs | Négatifs | Dernière date OOF | Fin cible maximale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 0 | 0 | 0 | 0 | 0 | — | — |
| 2016 | 252 | 21 | 231 | 20 | 211 | 2015-12-01 | 2015-12-31 |
| 2017 | 504 | 21 | 483 | 34 | 449 | 2016-11-30 | 2016-12-30 |
| 2018 | 755 | 21 | 734 | 34 | 700 | 2017-11-29 | 2017-12-29 |
| 2019 | 1006 | 21 | 985 | 68 | 917 | 2018-11-28 | 2018-12-31 |
| 2020 | 1258 | 21 | 1237 | 81 | 1156 | 2019-11-29 | 2019-12-31 |
| 2021 | 1511 | 21 | 1490 | 121 | 1369 | 2020-12-01 | 2020-12-31 |
| 2022 | 1763 | 21 | 1742 | 121 | 1621 | 2021-12-01 | 2021-12-31 |

L’historique commence le 02/01/2015 pour toutes les années effectivement calibrées. Par exemple, la calibration 2018 utilise 734 OOF admissibles de 2015–2017, jusqu’au 29/11/2017, dont les cibles se terminent au plus tard le 29/12/2017.

Pour cette livraison, hors tests : **sept calibrateurs Platt et sept calibrateurs Isotonic**, sans réentraînement des modèles de base pour la calibration. Les labels des validations courantes restent réservés à leur évaluation.

**5. Configurations Platt et Isotonic**

Platt : régression logistique sur le logit des probabilités OOF historiques.

```text
p_clip = clip(p, 1e-6, 1-1e-6)
z = log(p_clip / (1-p_clip))
LogisticRegression(C=1.0, solver=lbfgs, max_iter=2000, class_weight=None)
```

Isotonic :

```text
IsotonicRegression(
    increasing=True,
    y_min=0.0,
    y_max=1.0,
    out_of_bounds=clip
)
```

Aucune recherche de paramètres, contrainte de pente ajoutée après observation des résultats ou sélection de méthode. Les configurations et identifiants sont conservés dans les exports. Le clipping epsilon ne sert qu’au logit numérique de Platt ; les probabilités originales ne sont pas modifiées.

La pente Platt est libre, conformément à la configuration demandée. Elle est négative en **2016 (-0,001528), 2019 (-0,187518) et 2020 (-0,190145)** : le calibrateur inverse alors le classement de cette année. Cette instabilité est documentée, sans correction a posteriori. Isotonic est croissante, mais peut créer des ex æquo et des probabilités nulles.

**6. Résultats annuels**

Effectifs communs aux quatre approches :

| Année | N | Positifs | Négatifs | Taux réel (%) |
| --- | --- | --- | --- | --- |
| 2015 | 252 | 32 | 220 | 12.698413 |
| 2016 | 252 | 2 | 250 | 0.793651 |
| 2017 | 251 | 0 | 251 | 0.000000 |
| 2018 | 251 | 47 | 204 | 18.725100 |
| 2019 | 252 | 0 | 252 | 0.000000 |
| 2020 | 253 | 40 | 213 | 15.810277 |
| 2021 | 252 | 6 | 246 | 2.380952 |
| 2022 | 230 | 79 | 151 | 34.347826 |

`raw` = Logistic Regression brute ; `platt` et `isotonic` = méthodes exploratoires ; `historical_frequency` = baseline historique enregistrée. P moyenne en pourcentage.

| Année | Méthode | P moyenne (%) | Brier | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| 2015 | historical_frequency | 11.101512 | 0.111114 | 0.126984 | 0.500000 |
| 2015 | isotonic | 7.803863 | 0.118088 | 0.109532 | 0.428693 |
| 2015 | platt | 7.803863 | 0.118088 | 0.109532 | 0.428693 |
| 2015 | raw | 7.803863 | 0.118088 | 0.109532 | 0.428693 |
| 2016 | historical_frequency | 10.790806 | 0.017868 | 0.007937 | 0.500000 |
| 2016 | isotonic | 9.629630 | 0.018660 | 0.010989 | 0.640000 |
| 2016 | platt | 8.659489 | 0.014062 | 0.009095 | 0.356000 |
| 2016 | raw | 7.316462 | 0.013463 | 0.016199 | 0.644000 |
| 2017 | historical_frequency | 10.322810 | 0.010656 | NaN | NaN |
| 2017 | isotonic | 0.993191 | 0.000953 | NaN | NaN |
| 2017 | platt | 6.588027 | 0.004342 | NaN | NaN |
| 2017 | raw | 3.534670 | 0.001307 | NaN | NaN |
| 2018 | historical_frequency | 9.478827 | 0.160737 | 0.187251 | 0.500000 |
| 2018 | isotonic | 5.016969 | 0.176713 | 0.170607 | 0.431738 |
| 2018 | platt | 4.359669 | 0.173752 | 0.244871 | 0.419065 |
| 2018 | raw | 5.451555 | 0.171339 | 0.244871 | 0.419065 |
| 2019 | historical_frequency | 9.786209 | 0.009577 | NaN | NaN |
| 2019 | isotonic | 8.257271 | 0.006914 | NaN | NaN |
| 2019 | platt | 6.300369 | 0.003994 | NaN | NaN |
| 2019 | raw | 8.539746 | 0.008600 | NaN | NaN |
| 2020 | historical_frequency | 9.459838 | 0.137139 | 0.158103 | 0.500000 |
| 2020 | isotonic | 6.920784 | 0.140412 | 0.164256 | 0.533333 |
| 2020 | platt | 5.923138 | 0.143190 | 0.143248 | 0.474296 |
| 2020 | raw | 16.104531 | 0.169045 | 0.174286 | 0.525704 |
| 2021 | historical_frequency | 9.879770 | 0.028866 | 0.023810 | 0.500000 |
| 2021 | isotonic | 4.361546 | 0.022633 | 0.128573 | 0.886856 |
| 2021 | platt | 7.563135 | 0.025727 | 0.163362 | 0.924797 |
| 2021 | raw | 2.478006 | 0.022227 | 0.163362 | 0.924797 |
| 2022 | historical_frequency | 9.269250 | 0.288394 | 0.343478 | 0.500000 |
| 2022 | isotonic | 7.877684 | 0.295512 | 0.347089 | 0.507922 |
| 2022 | platt | 6.872275 | 0.301570 | 0.292870 | 0.436751 |
| 2022 | raw | 5.878472 | 0.309998 | 0.292870 | 0.436751 |

En 2015, les lignes Platt et Isotonic sont identiques à `raw` : absence d’historique, aucune calibration. Sur le périmètre effectivement calibré, les résultats annuels sont exactement ceux de 2016–2022 dans ce tableau ; 2015 est exclue.

AP et ROC-AUC restent `NaN` en 2017 et 2019, sans positifs ; justification explicite dans le CSV, Brier toujours défini. Aucun remplacement par zéro ou 0,5.

**7. Résultats agrégés sur périodes comparables**

Deux comparaisons séparées :

- `all_validation` : **1 993 dates**, 2015–30/11/2022, incluant le repli brut 2015 ; 206 positifs, 1 787 négatifs, fréquence réelle **10,3362 %**.
- `actually_calibrated_common` : **1 741 dates**, 2016–30/11/2022, où les deux méthodes ont effectivement été ajustées ; 174 positifs, 1 567 négatifs, fréquence réelle **9,9943 %**.

Les quatre approches sont évaluées sur exactement les mêmes dates dans chaque périmètre. Les agrégats sont recalculés à partir des prédictions individuelles, pas moyennés depuis les scores annuels.

| Périmètre | Méthode | N | Positifs | Négatifs | P moyenne (%) | Brier | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_validation | raw | 1993 | 206 | 1787 | 7.159475 | 0.099509 | 0.111118 | 0.545417 |
| all_validation | platt | 1993 | 206 | 1787 | 6.758362 | 0.095876 | 0.078383 | 0.394630 |
| all_validation | isotonic | 1993 | 206 | 1787 | 6.344485 | 0.095330 | 0.117440 | 0.539010 |
| all_validation | historical_frequency | 1993 | 206 | 1787 | 10.019151 | 0.093446 | 0.088309 | 0.272661 |
| actually_calibrated_common | raw | 1741 | 174 | 1567 | 7.066203 | 0.096820 | 0.111162 | 0.545988 |
| actually_calibrated_common | platt | 1741 | 174 | 1567 | 6.607032 | 0.092661 | 0.073679 | 0.374763 |
| actually_calibrated_common | isotonic | 1741 | 174 | 1567 | 6.133248 | 0.092036 | 0.130084 | 0.569657 |
| actually_calibrated_common | historical_frequency | 1741 | 174 | 1567 | 9.862485 | 0.090889 | 0.072096 | 0.171308 |

La comparaison limitée aux dates effectivement calibrées est indispensable pour ne pas attribuer à une calibration les résultats bruts de 2015. Aucun des deux périmètres n’est un test final indépendant.

**8. Comparaison au modèle brut et à la baseline**

Sur toutes les validations, Brier brut **0,099509**, Platt **0,095876**, Isotonic **0,095330**, baseline **0,093446**. Sur les seules dates effectivement calibrées : brut **0,096820**, Platt **0,092661**, Isotonic **0,092036**, baseline **0,090889**.

Les deux méthodes améliorent donc descriptivement le Brier agrégé du modèle brut, tout en restant moins bonnes que la baseline historique. L’AP de Platt se dégrade : **0,078383** sur toutes dates et **0,073679** sur les dates calibrées, contre **0,111118** et **0,111162** pour le brut.

Isotonic obtient ici des AP agrégées **0,117440** et **0,130084**. Ces résultats ne désignent pas un gagnant indépendant : transformations différentes selon les années, ex æquo et composition des régimes influencent le classement agrégé. Aucune méthode n’est promue pour la production après observation de ces scores.

**9. Amélioration ou dégradation du Brier**

Écarts de Brier, méthode moins référence ; un écart négatif signifie une erreur quadratique plus faible :

| Périmètre | Méthode | Écart au brut | Écart à la baseline |
| --- | --- | ---: | ---: |
| Toutes dates | Platt | -0.003633 | +0.002430 |
| Toutes dates | Isotonic | -0.004179 | +0.001884 |
| Dates effectivement calibrées | Platt | -0.004159 | +0.001773 |
| Dates effectivement calibrées | Isotonic | -0.004784 | +0.001147 |

Platt améliore le Brier brut en **2019, 2020 et 2022**, mais le dégrade en **2016, 2017, 2018 et 2021**. Isotonic l’améliore en **2017, 2019, 2020 et 2022**, mais le dégrade en **2016, 2018 et 2021**. Les gains ne sont donc pas uniformes.

En 2022, Brier brut **0,309998**, Platt **0,301570**, Isotonic **0,295512** : amélioration par rapport au brut, mais baseline encore meilleure à **0,288394**. Les probabilités moyennes restent très basses face aux **34,35 %** de labels positifs : brut **5,88 %**, Platt **6,87 %**, Isotonic **7,88 %**, baseline **9,27 %**.

Un Brier inférieur ne prouve pas à lui seul une calibration correcte : il dépend aussi de la discrimination et de la fréquence des événements. Aucun test de significativité n’est entrepris.

**10. Analyse des courbes de calibration**

Les deux panneaux utilisent dix intervalles uniformes fixés dans `[0,1]`, sans adapter les bins aux résultats : toutes dates à gauche, dates effectivement calibrées communes à droite. Les intervalles vides sont omis par la fonction de fiabilité.

Le brut présente une sous-estimation dans son intervalle `[0 ; 0,1]` : **1 620 dates**, probabilité moyenne **4,62 %**, fréquence observée **11,42 %**. Les intervalles de probabilités élevées sont peu peuplés et instables ; certains n’ont que deux ou trois observations, elles-mêmes potentiellement dépendantes. Leur tracé ne permet pas une conclusion statistique robuste.

Sur les dates effectivement calibrées, Platt comprime les probabilités entre **0,022096 et 0,096151** : un seul intervalle uniforme non vide. Sa moyenne de **6,6070 %** reste inférieure à la fréquence réelle de **9,9943 %** ; le panneau ne permet pas d’examiner finement la forme de sa calibration.

Isotonic produit des probabilités entre **0 et 0,133333** sur ces dates. Il existe **493 probabilités exactement nulles**, dont **26 correspondent à des labels positifs**. Cette observation limite fortement toute interprétation de zéro comme absence de risque. Aucun écrêtage ou nouvel ajustement n’a été introduit après ce constat.

La réduction des probabilités élevées peut améliorer le Brier agrégé tout en conservant une sous-estimation moyenne. Les courbes sont descriptives, sans intervalle de confiance et sans hypothèse d’indépendance des fenêtres.

**11. Résultats SHAP du fold 2022**

SHAP **0.54.1**, méthode réelle `shap.LinearExplainer` avec `shap.maskers.Independent`. Background : **4 078 observations train purgées**, du 21/09/2005 au 01/12/2021, transformées par le scaler appris sur ce même train. Le sous-échantillonnage par défaut à 100 lignes est désactivé : toutes les lignes train sont utilisées.

Validation expliquée : **230 séances**, du 03/01/2022 au 30/11/2022. Aucune observation ni label de validation ne sert au background ou à l’ajustement du scaler.

Les valeurs SHAP sont exprimées en **log-odds du modèle brut**, pas en variations directes de probabilité :

```text
decision_function(x) = expected_log_odds + somme_j SHAP_j(x)
expected_log_odds = -2.68200642401702
probabilité brute = sigmoid(decision_function(x))
```

Erreur maximale d’additivité : **4,44 × 10⁻¹⁶**. Importance descriptive moyenne sur les seules séances 2022 :

| Feature | Moyenne absolue SHAP (log-odds) | SHAP moyen signé (log-odds) |
| --- | --- | --- |
| SPY_IEF_corr_63d | 0.895648 | -0.891394 |
| SPY_vs_ma200 | 0.715496 | 0.693864 |
| SPY_vol_5d | 0.160069 | 0.141982 |
| SPY_ret_63d | 0.153987 | -0.134958 |
| SPY_ret_21d | 0.143592 | -0.057487 |
| SPY_ret_5d | 0.111034 | 0.021620 |
| SPY_vol_63d | 0.078496 | -0.074752 |
| SPY_drawdown_252d | 0.073685 | 0.070418 |
| SPY_vol_21d | 0.047431 | 0.046786 |
| SPY_ret_1d | 0.005524 | -0.000435 |

Pour ce modèle et cette période, `SPY_IEF_corr_63d` contribue en moyenne négativement aux log-odds, tandis que `SPY_vs_ma200` contribue en moyenne positivement par rapport au background. Ces directions décrivent les contributions du modèle appris ; elles ne constituent pas une relation causale ni une loi financière générale.

Les features SPY sont corrélées. L’approche interventional n’attribue pas les effets selon une distribution conditionnelle préservant toutes ces dépendances. Les couleurs du summary plot montrent les valeurs des features originales ; les contributions proviennent du modèle dans l’espace standardisé. Les résultats d’un seul fold ne démontrent aucune stabilité globale des importances.

**12. Vérification de reconstruction Logistic Regression**

Les estimateurs de l’étape 5 n’étant pas sauvegardés, **un seul refit historique hors tests** est réalisé pour l’interprétabilité : Logistic Regression du fold 2022, configuration originale `C=1`, `lbfgs`, `max_iter=2000`, `class_weight=None`, avec `StandardScaler` train-only.

Train : 4 078 dates jusqu’au 01/12/2021 ; cibles train terminées au plus tard le 31/12/2021. Purge de 21 séances inchangée. Aucun entraînement final destiné à la production.

Tolérance fixée avant reconstruction : `atol=1e-10`, `rtol=1e-8`, pour de faibles différences numériques éventuelles de bibliothèques. La concordance observée est **exacte : erreur maximale absolue 0,0** sur les 230 probabilités OOF enregistrées. Aucun avertissement de fit. Les probabilités originales restent intactes.

Une divergence au-delà de la tolérance interrompt l’interprétation avant SHAP ; ce comportement est testé. Ici, SHAP réussit : aucune substitution par des contributions de coefficients présentées abusivement comme SHAP.

**13. Contrôles de data leakage**

La calibration ne reçoit que les probabilités OOF antérieures et leurs labels matures. Les dates sont filtrées avant les valeurs de y. Platt et Isotonic ont les mêmes empreintes de dates historiques pour chaque année. Les labels de Y ne participent jamais au calibrateur de Y ; les changer seuls ne modifie pas ses probabilités, ce que vérifient les tests.

Les seuils de support, configurations et règles de repli sont fixés avant les scores. Aucun ajustement guidé par les validations, aucune optimisation des trois modèles de base. Le scaler et le background SHAP sont exclusivement issus du train purgé ; leurs effectifs, moyennes, variances et dates sont contrôlés.

Les labels d’une année passée peuvent servir à calibrer une année future lorsqu’ils sont disponibles. Les probabilités déjà évaluées ne sont jamais recalibrées rétroactivement avec leurs propres labels.

**14. Protection du test final**

Les vrais labels proviennent exclusivement du chargeur protégé de l’étape 3 : `Date >= 2005`, `Date < 2023-01-01`, `target_end_session < 2023-01-01`. Aucune lecture globale non filtrée des labels des targets.

Les prédictions OOF, résultats calibrés et observations expliquées restent strictement antérieurs à 2023. Les contrôles de schéma et de dates rejettent les entrées finales avant d’inspecter leurs labels. Aucun label 2023+ consulté, aucune prédiction 2023+ produite, aucun usage du test final pour choisir une méthode.

**15. Tests pytest et avertissements**

Les **256 tests précédents réussissent toujours**. **36 nouveaux tests** : 24 sur la calibration et 12 sur l’explicabilité. Ils couvrent maturité et frontières, labels indisponibles non inspectés, 2015 sans historique, supports minimaux, classes absentes, séparation des années, invariance aux labels courants, dates identiques, NaN/infinis, 2023+, reproductibilité, conservation des sources, background train-only, schéma core, reconstruction conforme ou divergente, additivité et refus d’écrasement.

Résultat complet réel :

```text
292 passed, 120 warnings in 31.85s
```

Aucun échec ni test ignoré. Les avertissements sont les `Pandas4Warning` connus sur `Series.values` avec fuseau horaire : pipeline 64, validation temporelle 16, indicateurs 12, baselines 8, entraînement 8, calibration 8, explicabilité 4. Aucun avertissement masqué.

Les fits synthétiques nécessaires aux tests existants et nouveaux sont autorisés par la demande. Hors tests, seuls les 14 calibrateurs historiques et le refit LR d’interprétabilité décrit plus haut sont ajustés. Aucune régénération des 5 979 OOF.

Les **68 lignes de métriques exportées** ont également été recalculées indépendamment depuis le nouveau Parquet et les labels protégés : AP par seuils et ex æquo, Brier par erreur quadratique, ROC-AUC par rangs moyens. Écart maximal : **1,11 × 10⁻¹⁶**. Aucun refit supplémentaire pour ce contrôle.

**16. Fichiers exportés**

Tous les fichiers attendus ont été créés dans le dossier neuf `reports/ml1/calibration/` :

| Fichier | Taille | Contenu |
| --- | ---: | --- |
| `calibrated_predictions.parquet` | 106 853 octets | 7 972 lignes, 23 colonnes, quatre approches × 1 993 dates |
| `calibration_metrics.csv` | 13 033 octets | 68 lignes : périmètres complet et effectivement calibré |
| `calibration_comparison.md` | 19 576 octets | Agrégats, historiques, paramètres appris, provenance et contrôles SHAP |
| `calibration_curves.png` | 117 721 octets | Fiabilité sur les deux périmètres |
| `brier_calibration_by_year.png` | 160 472 octets | Brier annuel des quatre approches |
| `shap_summary.png` | 168 435 octets | Contributions SHAP du fold 2022 |
| `shap_importance.png` | 75 611 octets | Importance moyenne absolue, fold 2022 |
| `shap_feature_importance.csv` | 815 octets | Dix features, importance absolue et signée, coefficients standardisés |

Le Parquet distingue méthode, statut et éligibilité. Il conserve la probabilité LR originale dans `raw_probability` pour comparaison ; `predicted_probability` contient la sortie de la méthode indiquée, y compris la baseline. Les dates et effectifs de l’historique sont des informations de contexte pour les références raw/baseline, pas la preuve d’un calibrateur ajusté sur ces références.

Les lignes contiennent aussi les identifiants de configuration, dates maximales de cible et de disponibilité, empreinte des dates historiques, exclusion 2023+ et réserve de disponibilité historique core. Aucun label n’est exporté dans ce Parquet.

Les quatre figures ont été ouvertes et inspectées : titres, légendes, périodes, échelles et réserves lisibles. Aucun fichier attendu impossible à produire. Les caches graphiques sont temporaires et inscriptibles, sans modification de configuration ou bibliothèque globale. Tous les exports passent par un dossier temporaire neuf ; une destination existante est refusée.

Nouveaux fichiers source/tests/rapport : `src/models/ml1/calibration.py`, `src/models/ml1/explain_ml1.py`, `tests/ml1/test_calibration.py`, `tests/ml1/test_explainability.py`, `reports/ml1/step7_calibration_shap.md`. Soit **13 nouveaux fichiers** avec les huit sorties. Aucun fichier préexistant modifié.

**17. Limites méthodologiques**

Logistic Regression a été choisie provisoirement sur les validations 2015–2022 à l’étape 6 : ce biais de sélection reste présent. Les nouvelles comparaisons exploratoires n’offrent pas une performance finale indépendante.

Le modèle de base évolue entre folds expanding-window. Le calibrateur apprend une relation entre probabilités et labels produits par des modèles antérieurs, relation pouvant changer pour le modèle de l’année suivante. Le changement de distribution et de régime n’est pas supprimé par le filtrage temporel.

Les supports minimaux ne garantissent pas 200 observations indépendantes : les cibles de 21 séances se chevauchent. Les positifs peuvent appartenir à quelques épisodes, et les historiques se recoupent entre années. Aucun test de significativité basé sur l’indépendance des dates.

Platt est instable selon les années et peut inverser le classement. Isotonic produit des plateaux et des zéros, y compris sur des dates positives ultérieures. Les améliorations agrégées du Brier ne suffisent pas à déclarer les probabilités fiables, notamment en 2022.

Les réserves sur les prix ajustés core et leur disponibilité historique à la décision demeurent. SHAP est descriptif et limité au modèle brut du fold 2022 ; il n’explique ni directement une probabilité calibrée ni une causalité. Aucun modèle final ou méthode de calibration définitive sélectionné.

**18. Commandes de reproduction**

Depuis la racine, sur `feature/ml1-crisis-detection` :

```bash
# Nouveaux tests : fits de calibrateurs et modèles synthétiques autorisés
.venv/bin/python -B -m pytest tests/ml1/test_calibration.py tests/ml1/test_explainability.py -q -p no:cacheprovider

# Suite complète ML 1
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider

# Reproduction ultérieure dans un dossier absent : calibration + un refit LR pour SHAP
.venv/bin/python -B -m src.models.ml1.calibration --output-dir reports/ml1/calibration_rebuild_001
```

Commande exécutée pour cette livraison : `.venv/bin/python -B -m src.models.ml1.calibration`. Le dossier par défaut existe maintenant : une relance vers ce même dossier est refusée avant les ajustements.

Environnement : Python 3.12.13, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, pyarrow 26.0.1, matplotlib 3.11.2, SHAP 0.54.1, pytest 9.2.0. Aucun téléchargement ni installation. Les valeurs SHAP utilisent tout le background train ; seul le jitter de visualisation utilise un générateur explicite de graine 42.

**19. État Git final**

Branche inchangée : `feature/ml1-crisis-detection`. HEAD inchangé : `f5ed847ef5986c67d7479185c2d8d82d4d4e7863`.

```text
?? reports/ml1/calibration/
?? reports/ml1/evaluation/
?? reports/ml1/step5_model_training.md
?? reports/ml1/step6_model_evaluation.md
?? reports/ml1/step7_calibration_shap.md
?? src/models/ml1/calibration.py
?? src/models/ml1/evaluate_ml1.py
?? src/models/ml1/explain_ml1.py
?? src/models/ml1/train_ml1.py
?? tests/ml1/test_calibration.py
?? tests/ml1/test_evaluation.py
?? tests/ml1/test_explainability.py
?? tests/ml1/test_training.py
```

Au total, 24 fichiers non suivis : les onze précédents et treize nouveaux. Aucun fichier suivi modifié, aucun ajout à l’index, commit, push ou merge. Les 88 fichiers protégés restent identiques. Aucun composant partagé ni travail d’un autre membre modifié.

**20. Verdict : PRÊT AVEC RÉSERVES**

L’étape 7 est terminée : calibration strictement chronologique, comparaison équitable sur deux périmètres, SHAP réel du fold 2022 après reconstruction conforme, huit sorties présentes et 292 tests réussis.

Les deux méthodes améliorent le Brier agrégé du brut sans battre la baseline historique. Leur instabilité et les limites de sélection empêchent d’en déclarer une supérieure de façon indépendante. **Aucune méthode n’est sélectionnée et aucun modèle final n’est sauvegardé.**

Le test final 2023+ reste fermé. Aucun backtest, optimisation de portefeuille, fonction `predict_crisis(date)` ou entraînement final de production. **L’étape 8 n’a pas commencé ; toute suite attend votre validation.**
