**Comparaison des baselines ML 1 — développement uniquement**

VIX expérimental, non certifié point-in-time. Aucun label ou score 2023+.

**Protocole fixé avant les scores**

VIX < 25 / VIX >= 25; Beta(1, 1); repli global si groupe < 30 observations. Fréquences calculées sur train uniquement.

**Alignement décidé avant estimation**

- 2015 : core 252, VIX 252, commun 252; dates core seules []; dates VIX seules []; train core/VIX 2315/2315; purge 21/21.
- 2016 : core 252, VIX 252, commun 252; dates core seules []; dates VIX seules []; train core/VIX 2567/2567; purge 21/21.
- 2017 : core 251, VIX 251, commun 251; dates core seules []; dates VIX seules []; train core/VIX 2819/2819; purge 21/21.
- 2018 : core 251, VIX 251, commun 251; dates core seules []; dates VIX seules []; train core/VIX 3070/3070; purge 21/21.
- 2019 : core 252, VIX 252, commun 252; dates core seules []; dates VIX seules []; train core/VIX 3321/3321; purge 21/21.
- 2020 : core 253, VIX 253, commun 253; dates core seules []; dates VIX seules []; train core/VIX 3573/3573; purge 21/21.
- 2021 : core 252, VIX 252, commun 252; dates core seules []; dates VIX seules []; train core/VIX 3826/3826; purge 21/21.
- 2022 : core 230, VIX 230, commun 230; dates core seules []; dates VIX seules []; train core/VIX 4078/4078; purge 21/21.

Les deux baselines sont évaluées sur les mêmes dates communes. Les entraînements restent ceux de leurs folds purgés.

**Résultats par fold**

| baseline | validation_year | n_observations | n_positive | n_negative | observed_crisis_rate | mean_predicted_probability | average_precision | brier_score | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_frequency | 2015 | 252 | 32 | 220 | 0.126984 | 0.111015 | 0.126984 | 0.111114 | 0.500000 |
| historical_frequency | 2016 | 252 | 2 | 250 | 0.007937 | 0.107908 | 0.007937 | 0.017868 | 0.500000 |
| historical_frequency | 2017 | 251 | 0 | 251 | 0.000000 | 0.103228 | NaN | 0.010656 | NaN |
| historical_frequency | 2018 | 251 | 47 | 204 | 0.187251 | 0.094788 | 0.187251 | 0.160737 | 0.500000 |
| historical_frequency | 2019 | 252 | 0 | 252 | 0.000000 | 0.097862 | NaN | 0.009577 | NaN |
| historical_frequency | 2020 | 253 | 40 | 213 | 0.158103 | 0.094598 | 0.158103 | 0.137139 | 0.500000 |
| historical_frequency | 2021 | 252 | 6 | 246 | 0.023810 | 0.098798 | 0.023810 | 0.028866 | 0.500000 |
| historical_frequency | 2022 | 230 | 79 | 151 | 0.343478 | 0.092692 | 0.343478 | 0.288394 | 0.500000 |
| vix_threshold_25 | 2015 | 252 | 32 | 220 | 0.126984 | 0.082989 | 0.126984 | 0.117125 | 0.468182 |
| vix_threshold_25 | 2016 | 252 | 2 | 250 | 0.007937 | 0.082432 | 0.007937 | 0.014934 | 0.476000 |
| vix_threshold_25 | 2017 | 251 | 0 | 251 | 0.000000 | 0.072483 | NaN | 0.005254 | NaN |
| vix_threshold_25 | 2018 | 251 | 47 | 204 | 0.187251 | 0.075964 | 0.187251 | 0.170312 | 0.463235 |
| vix_threshold_25 | 2019 | 252 | 0 | 252 | 0.000000 | 0.073143 | NaN | 0.005571 | NaN |
| vix_threshold_25 | 2020 | 253 | 40 | 213 | 0.158103 | 0.171277 | 0.139215 | 0.148188 | 0.403404 |
| vix_threshold_25 | 2021 | 252 | 6 | 246 | 0.023810 | 0.086287 | 0.023810 | 0.029038 | 0.457317 |
| vix_threshold_25 | 2022 | 230 | 79 | 151 | 0.343478 | 0.145371 | 0.295216 | 0.293088 | 0.300905 |

**Prédictions hors échantillon concaténées**

| baseline | n_observations | n_positive | n_negative | observed_crisis_rate | mean_predicted_probability | average_precision | brier_score | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_frequency | 1993 | 206 | 1787 | 0.103362 | 0.100192 | 0.088309 | 0.093446 | 0.272661 |
| vix_threshold_25 | 1993 | 206 | 1787 | 0.103362 | 0.098290 | 0.091453 | 0.095820 | 0.349705 |

AP est average_precision_score, pas une intégrale trapézoïdale. Brier plus faible = meilleur score probabiliste.
AP et ROC-AUC sont NaN si une classe manque, par politique prudente de comparaison ; Brier reste défini.
Les fenêtres de 21 séances se chevauchent : ni observations ni folds ne sont supposés indépendants.
La fréquence est constante à l’intérieur de chaque fold ; son classement agrégé peut refléter les différences de fréquence entre années.
Aucun ajustement sur validation, recherche de seuil, calibrateur, intervalle de confiance ou preuve de capacité prédictive.

**Traçabilité**

Évaluation 0.1.0; Python 3.12.13; pandas 3.1.1; NumPy 2.5.4; scikit-learn 1.9.1.

Les prédictions stockent paramètres fixes, estimations train, dates et références des folds.
