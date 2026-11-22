**ML1 — comparaison des probabilités OOF enregistrées, 2015–30 novembre 2022**

Aucun nouvel entraînement, tuning ou calibration. VIX : **EXPÉRIMENTAL**, non certifié point-in-time.

**Agrégation OOF recalculée sur les prédictions individuelles**

| approach | n_observations | n_positive | n_negative | observed_crisis_rate | mean_predicted_probability | average_precision | brier_score | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 1993 | 206 | 1787 | 0.103362 | 0.071595 | 0.111118 | 0.099509 | 0.545417 |
| random_forest | 1993 | 206 | 1787 | 0.103362 | 0.081706 | 0.097037 | 0.104717 | 0.499764 |
| xgboost | 1993 | 206 | 1787 | 0.103362 | 0.054937 | 0.086914 | 0.112210 | 0.411413 |
| historical_frequency | 1993 | 206 | 1787 | 0.103362 | 0.100192 | 0.088309 | 0.093446 | 0.272661 |
| vix_threshold_25 | 1993 | 206 | 1787 | 0.103362 | 0.098290 | 0.091453 | 0.095820 | 0.349705 |

**Résultats annuels**

| validation_year | approach | n_observations | n_positive | n_negative | observed_crisis_rate | mean_predicted_probability | average_precision | brier_score | roc_auc | metric_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015.000000 | logistic_regression | 252 | 32 | 220 | 0.126984 | 0.078039 | 0.109532 | 0.118088 | 0.428693 | both_classes |
| 2016.000000 | logistic_regression | 252 | 2 | 250 | 0.007937 | 0.073165 | 0.016199 | 0.013463 | 0.644000 | both_classes |
| 2017.000000 | logistic_regression | 251 | 0 | 251 | 0.000000 | 0.035347 | NaN | 0.001307 | NaN | single_class_no_positives |
| 2018.000000 | logistic_regression | 251 | 47 | 204 | 0.187251 | 0.054516 | 0.244871 | 0.171339 | 0.419065 | both_classes |
| 2019.000000 | logistic_regression | 252 | 0 | 252 | 0.000000 | 0.085397 | NaN | 0.008600 | NaN | single_class_no_positives |
| 2020.000000 | logistic_regression | 253 | 40 | 213 | 0.158103 | 0.161045 | 0.174286 | 0.169045 | 0.525704 | both_classes |
| 2021.000000 | logistic_regression | 252 | 6 | 246 | 0.023810 | 0.024780 | 0.163362 | 0.022227 | 0.924797 | both_classes |
| 2022.000000 | logistic_regression | 230 | 79 | 151 | 0.343478 | 0.058785 | 0.292870 | 0.309998 | 0.436751 | both_classes |
| 2015.000000 | random_forest | 252 | 32 | 220 | 0.126984 | 0.090515 | 0.083307 | 0.138738 | 0.235369 | both_classes |
| 2016.000000 | random_forest | 252 | 2 | 250 | 0.007937 | 0.062158 | 0.017305 | 0.015965 | 0.668000 | both_classes |
| 2017.000000 | random_forest | 251 | 0 | 251 | 0.000000 | 0.019335 | NaN | 0.000540 | NaN | single_class_no_positives |
| 2018.000000 | random_forest | 251 | 47 | 204 | 0.187251 | 0.044744 | 0.166428 | 0.178052 | 0.469545 | both_classes |
| 2019.000000 | random_forest | 252 | 0 | 252 | 0.000000 | 0.051928 | NaN | 0.005549 | NaN | single_class_no_positives |
| 2020.000000 | random_forest | 253 | 40 | 213 | 0.158103 | 0.190190 | 0.147284 | 0.171551 | 0.470188 | both_classes |
| 2021.000000 | random_forest | 252 | 6 | 246 | 0.023810 | 0.104912 | 0.020475 | 0.036265 | 0.363144 | both_classes |
| 2022.000000 | random_forest | 230 | 79 | 151 | 0.343478 | 0.089744 | 0.238690 | 0.308477 | 0.268924 | both_classes |
| 2015.000000 | xgboost | 252 | 32 | 220 | 0.126984 | 0.117402 | 0.086969 | 0.170073 | 0.280256 | both_classes |
| 2016.000000 | xgboost | 252 | 2 | 250 | 0.007937 | 0.049562 | 0.031830 | 0.016184 | 0.822000 | both_classes |
| 2017.000000 | xgboost | 251 | 0 | 251 | 0.000000 | 0.008934 | NaN | 0.000113 | NaN | single_class_no_positives |
| 2018.000000 | xgboost | 251 | 47 | 204 | 0.187251 | 0.024414 | 0.189936 | 0.181208 | 0.484251 | both_classes |
| 2019.000000 | xgboost | 252 | 0 | 252 | 0.000000 | 0.033877 | NaN | 0.002736 | NaN | single_class_no_positives |
| 2020.000000 | xgboost | 253 | 40 | 213 | 0.158103 | 0.136038 | 0.175694 | 0.183661 | 0.573474 | both_classes |
| 2021.000000 | xgboost | 252 | 6 | 246 | 0.023810 | 0.052419 | 0.061884 | 0.027569 | 0.789295 | both_classes |
| 2022.000000 | xgboost | 230 | 79 | 151 | 0.343478 | 0.012521 | 0.345236 | 0.335143 | 0.457624 | both_classes |
| 2015.000000 | historical_frequency | 252 | 32 | 220 | 0.126984 | 0.111015 | 0.126984 | 0.111114 | 0.500000 | both_classes |
| 2016.000000 | historical_frequency | 252 | 2 | 250 | 0.007937 | 0.107908 | 0.007937 | 0.017868 | 0.500000 | both_classes |
| 2017.000000 | historical_frequency | 251 | 0 | 251 | 0.000000 | 0.103228 | NaN | 0.010656 | NaN | single_class_no_positives |
| 2018.000000 | historical_frequency | 251 | 47 | 204 | 0.187251 | 0.094788 | 0.187251 | 0.160737 | 0.500000 | both_classes |
| 2019.000000 | historical_frequency | 252 | 0 | 252 | 0.000000 | 0.097862 | NaN | 0.009577 | NaN | single_class_no_positives |
| 2020.000000 | historical_frequency | 253 | 40 | 213 | 0.158103 | 0.094598 | 0.158103 | 0.137139 | 0.500000 | both_classes |
| 2021.000000 | historical_frequency | 252 | 6 | 246 | 0.023810 | 0.098798 | 0.023810 | 0.028866 | 0.500000 | both_classes |
| 2022.000000 | historical_frequency | 230 | 79 | 151 | 0.343478 | 0.092692 | 0.343478 | 0.288394 | 0.500000 | both_classes |
| 2015.000000 | vix_threshold_25 | 252 | 32 | 220 | 0.126984 | 0.082989 | 0.126984 | 0.117125 | 0.468182 | both_classes |
| 2016.000000 | vix_threshold_25 | 252 | 2 | 250 | 0.007937 | 0.082432 | 0.007937 | 0.014934 | 0.476000 | both_classes |
| 2017.000000 | vix_threshold_25 | 251 | 0 | 251 | 0.000000 | 0.072483 | NaN | 0.005254 | NaN | single_class_no_positives |
| 2018.000000 | vix_threshold_25 | 251 | 47 | 204 | 0.187251 | 0.075964 | 0.187251 | 0.170312 | 0.463235 | both_classes |
| 2019.000000 | vix_threshold_25 | 252 | 0 | 252 | 0.000000 | 0.073143 | NaN | 0.005571 | NaN | single_class_no_positives |
| 2020.000000 | vix_threshold_25 | 253 | 40 | 213 | 0.158103 | 0.171277 | 0.139215 | 0.148188 | 0.403404 | both_classes |
| 2021.000000 | vix_threshold_25 | 252 | 6 | 246 | 0.023810 | 0.086287 | 0.023810 | 0.029038 | 0.457317 | both_classes |
| 2022.000000 | vix_threshold_25 | 230 | 79 | 151 | 0.343478 | 0.145371 | 0.295216 | 0.293088 | 0.300905 | both_classes |

NaN AP/ROC : une seule classe, discrimination non évaluable ; aucun remplacement par zéro ou 0,5.

**Méthode et candidat provisoire**

Rank the three ML configurations by pooled AP; inspect Brier and annual stability. A unique AP leader exceeding both baseline APs is only a provisional candidate. AP ties within 1e-12 or failure to exceed both baselines yield no unique candidate. No statistical superiority, tuning, calibration or final-test tie-breaking.

{
  "policy": "Rank the three ML configurations by pooled AP; inspect Brier and annual stability. A unique AP leader exceeding both baseline APs is only a provisional candidate. AP ties within 1e-12 or failure to exceed both baselines yield no unique candidate. No statistical superiority, tuning, calibration or final-test tie-breaking.",
  "candidate": "logistic_regression",
  "status": "CANDIDAT PROVISOIRE",
  "ml_ranking_by_pooled_ap": [
    "logistic_regression",
    "random_forest",
    "xgboost"
  ],
  "pooled_ap_gap_to_second_ml": 0.014080396426135067,
  "best_ml_brier": "logistic_regression",
  "annual_ap_wins": {
    "logistic_regression": 3,
    "random_forest": 0,
    "xgboost": 3
  },
  "annual_two_class_periods": 6,
  "statistical_superiority_established": false,
  "selection_bias": "Selection uses 2015–2022 validation outcomes; this is not an unbiased final-test estimate."
}

**CANDIDAT PROVISOIRE pour l’étape 7 : logistic_regression.** Son AP agrégée (0.111118) est la plus élevée des trois modèles et dépasse les deux baselines. Son Brier (0.099509) doit être comparé à celui de la fréquence historique (0.093446) ; il est moins bon. ROC-AUC : 0.545417. Probabilité moyenne : 7.16%, contre une fréquence observée de 10.34%. Ce décalage moyen ne remplace pas un diagnostic de calibration. Victoires annuelles AP parmi les trois ML : {'logistic_regression': 3, 'random_forest': 0, 'xgboost': 3}, sur 6 années à deux classes. La préférence est limitée au critère AP de validation ; la stabilité annuelle et la qualité des probabilités ne permettent pas de déclarer une supériorité universelle ou statistiquement établie. Aucune configuration n’est modifiée après observation des résultats.

L’AP mesure le classement ; le Brier évalue l’erreur quadratique des probabilités et ne suffit pas à isoler leur calibration. La ROC-AUC est complémentaire. Une AP supérieure ne garantit pas des probabilités calibrées. Les fenêtres de 21 séances se chevauchent ; aucun test de significativité fondé sur leur indépendance. Le candidat est sélectionné sur les validations : biais de sélection possible. Toute calibration ultérieure doit utiliser une séparation temporelle propre, sans calibrer une prédiction sur son propre label. Test final 2023+ fermé ; aucun modèle final sauvegardé.

**Alignement et vérifications des baselines**

{
  "expected_dates": 1993,
  "approaches": {
    "logistic_regression": {
      "n_dates": 1993,
      "missing_dates": [],
      "extra_dates": []
    },
    "random_forest": {
      "n_dates": 1993,
      "missing_dates": [],
      "extra_dates": []
    },
    "xgboost": {
      "n_dates": 1993,
      "missing_dates": [],
      "extra_dates": []
    },
    "historical_frequency": {
      "n_dates": 1993,
      "missing_dates": [],
      "extra_dates": []
    },
    "vix_threshold_25": {
      "n_dates": 1993,
      "missing_dates": [],
      "extra_dates": []
    }
  }
}

| baseline | historical_csv_match | average_precision | brier_score | roc_auc |
| --- | --- | --- | --- | --- |
| historical_frequency | True | 0.000000 | 0.000000 | 0.000000 |
| vix_threshold_25 | True | 0.000000 | 0.000000 | 0.000000 |

**Provenance**

{
  "input_sha256": {
    "/Users/mac/Allocation ML/asset-allocation-ml/data/processed/ml1_training/ml_models_oof_predictions.parquet": "2990ee5d8f2bef48c266e7d46236676a4f8f643d21490ad2a7045e931c3b371a",
    "/Users/mac/Allocation ML/asset-allocation-ml/data/processed/ml1_training/training_metadata.json": "9e1d78b125010d99e98ba68ae888a03292d71d1a4fbf4fab482ca91b0090c665",
    "/Users/mac/Allocation ML/asset-allocation-ml/reports/ml1/baselines/baseline_predictions.parquet": "eaecfde127cfddf9d20a6c97058ba9dc2b69dafb57756d00fa70c44f97cb1914",
    "/Users/mac/Allocation ML/asset-allocation-ml/reports/ml1/baselines/baseline_metrics.csv": "e8124e5eaea75b61db963a2d1b06b73efab986a32aecfe668b57557cd72a8b71",
    "/Users/mac/Allocation ML/asset-allocation-ml/reports/ml1/baselines/baseline_comparison.md": "5882d22efada0b665988792498a6f5b9e53ba1936b4f38aef61123e65a4a2580",
    "/Users/mac/Allocation ML/asset-allocation-ml/data/processed/ml1/dataset_metadata.json": "156447065b261bf7233eb89ec8b8ffc768fc9aa42bdffb70cb98c9ad3ac6e74d"
  },
  "committed_baseline_sha256": {
    "baseline_predictions.parquet": "eaecfde127cfddf9d20a6c97058ba9dc2b69dafb57756d00fa70c44f97cb1914",
    "baseline_metrics.csv": "e8124e5eaea75b61db963a2d1b06b73efab986a32aecfe668b57557cd72a8b71",
    "baseline_comparison.md": "5882d22efada0b665988792498a6f5b9e53ba1936b4f38aef61123e65a4a2580"
  },
  "baseline_metrics_verified": true,
  "new_fits": 0,
  "feature_columns": [
    "SPY_ret_1d",
    "SPY_ret_5d",
    "SPY_ret_21d",
    "SPY_ret_63d",
    "SPY_vol_5d",
    "SPY_vol_21d",
    "SPY_vol_63d",
    "SPY_drawdown_252d",
    "SPY_vs_ma200",
    "SPY_IEF_corr_63d"
  ],
  "dependencies": {
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "pyarrow": "26.0.1",
    "scikit-learn": "1.9.1",
    "matplotlib": "3.11.2"
  },
  "configuration_ids": {
    "logistic_regression": "logistic_regression_102c776acc68dede",
    "random_forest": "random_forest_c3d09bf77ca0b0b7",
    "xgboost": "xgboost_ed397cc469cc15e0"
  }
}
