**Rapport ML 1 — étape 8 : interface d’inférence et intégration — 15 novembre 2026**

L’interface provisoire est créée, testée et documentée. Un modèle Logistic Regression brut est figé sur les seules données admissibles pré-2023. Les sorties comprennent **95 mois OOF historiques** et **46 mois d’inférence aveugle distincte**, sans aucun label ni calcul de performance 2023+. **326 tests réussissent.**

**1. État Git initial**

Dépôt : `/Users/mac/Allocation ML/asset-allocation-ml`. Branche : `feature/ml1-crisis-detection`. HEAD : `42f36c10250bb234e6ce0422b6dcece1020f7b6f`, message `feat(ml1): add ML training, evaluation, calibration and SHAP`.

L’arbre Git était propre. La préservation a été vérifiée sur **101 fichiers préexistants** : les 87 fichiers suivis comparés à HEAD et les 14 sorties de données ignorées comparées à leurs empreintes initiales. Aucun changement de leur contenu. Aucun travail sur main ni modification des composants des autres membres.

Les API de `pipeline.py`, `time_validation.py`, `train_ml1.py`, `evaluate_ml1.py` et `calibration.py` ainsi que les artefacts OOF ont été inspectés. Les schémas, configurations, dates, empreintes et plans des huit folds sont contrôlés avant construction de l’artefact.

**2. Contrat de sortie**

`predict_crisis(date)` retourne un **Python float fini dans `[0,1]`**, après chargement explicite de l’artefact de confiance.

Sens : probabilité prédite qu’au moins une clôture ajustée de SPY parmi les **21 séances SPY suivantes** soit **strictement inférieure à 93 %** de sa clôture ajustée à la date de décision. Aucun rendement attendu, allocation ou conseil d’investissement n’est calculé.

L’interface de détail expose :

| Champ | Sens |
| --- | --- |
| `prediction_date` | Date exacte de la séance demandée |
| `p_crisis` | Probabilité numérique dans `[0,1]` |
| `horizon_sessions` | 21 séances SPY futures |
| `model_name` | `logistic_regression` |
| `model_version` | Identifiant du fold OOF ou du modèle figé, selon l’origine |
| `feature_variant` | `core` |
| `calibration_method` | `none` |
| `prediction_origin` | `historical_oof` ou `blind_holdout_inference` |
| `training_cutoff` | Dernière date des features/décisions train du modèle concerné |
| `decision_time` | Horodatage de décision à 17 h New York, avec décalage été/hiver |
| `status` | `provisional` |
| `training_label_end` | Dernière fin de cible utilisée pour son entraînement |
| `decision_timezone` | `America/New_York` |
| `publication_status` | Reconstruction rétrospective, non preuve de publication contemporaine |
| `performance_evaluated` | Faux pour les inférences aveugles ; vrai pour les OOF déjà évaluées historiquement |

La date de fin des labels d’entraînement n’est pas un label exporté. Elle rend explicite la maturité des observations ayant servi au modèle.

**3. Architecture du module**

Nouveau module `src/models/ml1/inference.py` :

- `create_release` : audits, unique fit figé, manifest et exports dans un dossier neuf.
- `FeatureSource` : lecture label-free, contrôle du schéma et des empreintes, recalcul causal par préfixe de prix.
- `load_inference` : chargement explicite et contrôlé avec `trusted=True`.
- `ML1Predictor` : objet indépendant pour prédictions simples et détaillées.
- `configure_inference`, `predict_crisis`, `predict_crisis_details` : interface importable avec configuration initiale explicite.
- `monthly_oof_probabilities` et `monthly_blind_probabilities` : tables séparées, sans labels.

Le modèle n’est jamais réentraîné lors du chargement ou d’une prédiction. Aucun téléchargement, recherche de paramètres, calibrateur, optimiseur ou backtest. Les modules précédents restent inchangés.

La documentation destinée aux membres 1, 4 et 5 est dans `reports/ml1/integration_guide.md`.

**4. Politique du modèle et justification**

Politique explicitement imposée pour cette livraison :

```text
model = logistic_regression
feature_variant = core
calibration = none
status = provisional
horizon = 21_spy_sessions
```

Logistic Regression reste le candidat provisoire de l’étape 6 selon l’AP. Aucune calibration de l’étape 7 n’est sélectionnée. La baseline historique reste une référence de comparaison, jamais une calibration implicite.

Les probabilités brutes ne sont pas présentées comme parfaitement calibrées. L’étape 6 n’a pas établi de supériorité statistique et le Brier brut était moins bon que celui de la baseline. Aucun résultat 2023+ n’a servi à fixer cette politique.

**5. Entraînement du modèle figé**

**Un seul entraînement réel de construction**, hors fits synthétiques des tests :

```text
Pipeline : StandardScaler + LogisticRegression
C = 1.0
solver = lbfgs
max_iter = 2000
class_weight = None
calibration = none
```

Les autres paramètres conservent les valeurs par défaut des versions installées, enregistrées sous forme résolue dans le manifest. Aucun avertissement de fit ou de convergence.

Développement utilisé : **4 329 observations**, du **21/09/2005 au 30/11/2022**, cible `y_crisis_21d`, deux classes présentes. Le scaler est ajusté sur exactement ces 4 329 observations ; ses moyennes et variances ont été vérifiées contre ce seul sous-ensemble. Aucune donnée 2023+ n’adapte le scaler ou les paramètres.

Dix colonnes, dans l’ordre validé :

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

Identifiant : **`ml1_lr_core_pre2023_86aa4c7f66463f90`**. Il dépend notamment de la politique, configuration, versions, code d’entraînement et empreintes des données admissibles.

Cet artefact figé est une livraison d’intégration provisoire. Le nom `logistic_regression_final.joblib` ne constitue pas une validation pour un usage financier réel.

**6. Contrôle de la date limite des labels**

Le fit réutilise exclusivement `load_development_data`, avec filtres Parquet sur dates pré-2023 et `target_end_session < 2023-01-01`. Dernière fin de cible : **30/12/2022**. Dernier label d’entraînement connu selon la convention du projet : **30/12/2022 à 17 h New York**, soit `2022-12-30 22:00:00+00:00`.

Les 21 décisions de décembre 2022 dont la cible franchit 2023 restent exclues du développement. Le nombre attendu de 4 329 observations est vérifié ; un changement impose une revue du contrat plutôt qu’une adaptation silencieuse.

Pendant l’inférence, le chargeur de cibles n’est pas appelé : seuls les features, prix SPY/IEF, OOF et métadonnées nécessaires sont lus. Un contrôle supplémentaire a interdit toute lecture de fichier targets pendant chargement et prédictions quotidiennes/mensuelles : **zéro lecture de targets**.

**7. Fonction predict_crisis(date)**

Exemple d’appel après chargement de confiance :

```python
from src.models.ml1.inference import (
    configure_inference,
    predict_crisis,
    predict_crisis_details,
)

configure_inference(trusted=True)
p = predict_crisis("2022-11-30")
details = predict_crisis_details("2022-11-30")
```

Le premier appel sans configuration indique explicitement la procédure de chargement. Il n’installe rien et ne reconstruit aucun modèle.

Alternative sans état global :

```python
from src.models.ml1.inference import load_inference

engine = load_inference("data/processed/ml1_inference_release", trusted=True)
p = engine.predict_crisis("2023-01-31")
details = engine.predict_details("2023-01-31")
```

Le second exemple est exclusivement une inférence aveugle, sans label ni mesure de qualité. `predict_crisis_details` retourne un dictionnaire ; la méthode de l’objet retourne une dataclass convertible par `dataclasses.asdict`.

Dates acceptées : date ISO `YYYY-MM-DD` ou objet date naïf normalisé. Dates invalides, horodatages intraday/avec fuseau, séances non couvertes, features absentes ou non finies : refus explicite. Aucune substitution automatique par le dernier cours connu ni extrapolation vers une date future sans données.

**8. Séparation historique OOF / inférence aveugle**

Pour **2015–2022**, seule la probabilité OOF enregistrée est retournée. Le modèle figé n’est jamais appelé pour fabriquer une prédiction historique hors échantillon. Les dates et années de validation, identifiants de configuration et bornes d’entraînement sont vérifiés.

Les dates avant 2015 et les dates historiques sans OOF admissible, notamment décembre 2022, sont refusées même si des features existent.

Pour **2023+**, le modèle figé pré-2023 utilise les features de la date exacte. Origine `blind_holdout_inference`, `performance_evaluated=False`, statut provisoire et calibration absente. Aucun label de holdout n’est chargé.

Exemple de traçabilité : l’OOF du 30/11/2022 provient de `wf_2022`, dernière décision train le 01/12/2021 et dernière fin de cible le 31/12/2021. L’artefact figé utilisé après 2022 a pour bornes le 30/11/2022 et le 30/12/2022. Ces politiques d’entraînement sont différentes.

Toutes ces prédictions sont calculées ou récupérées rétrospectivement. Elles ne sont pas présentées comme ayant réellement été publiées aux dates affichées.

**9. Export mensuel historique**

`monthly_oof_probabilities.csv` contient **95 mois**, de **janvier 2015 à novembre 2022**. Première date sélectionnée : **30/01/2015** ; dernière : **30/11/2022**.

Pour chaque mois, la dernière séance OOF admissible est sélectionnée. Les probabilités viennent exclusivement du fichier OOF existant. Aucun fit, aucun label de la date utilisé pour produire sa probabilité et aucun label joint à l’export.

Décembre 2022 n’est pas inventé. Les mois et dates sont triés, sans doublon. Les 95 valeurs exportées concordent exactement avec l’interface et les probabilités OOF sources lors de la lecture numérique de référence.

Ce tableau peut préparer les travaux du membre 5 ; aucun backtest n’a été implémenté ou lancé.

**10. Export mensuel 2023+**

`monthly_blind_probabilities.csv` contient **46 mois entièrement couverts**, de **janvier 2023 à octobre 2026**. Première date : **31/01/2023** ; dernière : **30/10/2026**.

Les features locales s’arrêtent au **04/11/2026**. Novembre 2026 est incomplet et exclu. Le calendrier ARCX/XNYS sert à vérifier que toutes les séances attendues de chaque mois sont couvertes ; un mois avec une séance de features manquante est également exclu.

La fonction distincte `monthly_blind_probabilities(engine)` ne réentraîne rien, ne lit aucun label et ne calcule aucune performance. Les 46 sorties portent leur origine aveugle et restent séparées des OOF.

Aucune conclusion qualitative n’est tirée des valeurs du holdout : seuls le format, la provenance, les dates et la cohérence avec l’interface sont contrôlés.

**11. Artefacts et empreintes**

Nouveau dossier ignoré : `data/processed/ml1_inference_release/`, absent avant construction. Les cinq fichiers attendus sont présents :

| Fichier | Taille |
| --- | ---: |
| `logistic_regression_final.joblib` | 1 285 octets |
| `model_manifest.json` | 57 705 octets |
| `monthly_oof_probabilities.csv` | 25 010 octets |
| `monthly_blind_probabilities.csv` | 12 333 octets |
| `integration_contract.json` | 1 922 octets |

SHA-256 :

```text
logistic_regression_final.joblib
826f52284d462f2acbe036eb6719eaa7a327d5f4a0d8a45ff563d83363d4ed5f

model_manifest.json
4d01d417cf99540f3b8231ffe3e55ce0e60fc32225a84161b58399ec14c8c599

monthly_oof_probabilities.csv
2614d05c93962efcb9cf9e0d704a04190bf3f622ee5235350c503633d584710b

monthly_blind_probabilities.csv
a431a40aec2eafc40f94abe3214e2f4a979ab2947833586861b904a9f7502984

integration_contract.json
6623042a5fa5bec27e4251b532b36a919d412d4f826d75fada336ea6210ec53c
```

Le manifest conserve les dix features ordonnées, configuration demandée et paramètres résolus, versions/Python, bornes du train et de ses labels, scaler, identité, six empreintes de sources, treize empreintes de modules, quatre empreintes des autres artefacts, plans des folds et réserves temporelles. Le hash du manifest lui-même est donné ci-dessus ; aucune signature d’authenticité n’est revendiquée.

Avant désérialisation : confiance explicite, contrôle des empreintes des artefacts/sources, chemins de ressources, code, versions et politique. Après chargement : structure Pipeline, types, configuration et scaler contrôlés. Un fichier absent, modifié ou incompatible provoque une erreur, jamais un fit automatique.

**Joblib ne doit provenir que d’une source de confiance** : il peut exécuter du code au chargement. Les empreintes détectent des modifications, mais n’authentifient pas un ensemble modèle/manifest remplacé par un tiers.

**12. Tests pytest**

Les **292 tests précédents réussissent toujours**. **34 nouveaux tests** couvrent : float simple et détails, configuration de confiance, dates invalides/non couvertes, refus in-sample, historique OOF sans appel au modèle figé, absence de lecture des targets en inférence, scaler pré-2023, préfixes de prix, invariance aux prix ultérieurs en mémoire, schéma strict et VIX refusé, NaN, limites des labels, 95 mois OOF, mois incomplets, séparation des origines, cohérence CSV, empreintes, modèle manquant, non-écrasement et reproductibilité synthétique.

Résultat complet réel :

```text
326 passed, 148 warnings in 36.24s
```

Aucun échec ni test ignoré. Les avertissements sont les dépréciations `Pandas4Warning` déjà connues sur `Series.values` avec fuseau horaire : pipeline 64, validation temporelle 16, indicateurs 12, baselines 8, entraînement 8, calibration 8, explicabilité 4, inférence 28. Aucun avertissement masqué.

Les nouveaux tests réalisent leurs fits sur un jeu de prix entièrement synthétique dans un répertoire temporaire. Les OOF synthétiques de ce jeu sont produites par des estimateurs espions ; aucun réentraînement des 24 modèles réels de l’étape 5. L’unique fit réel figé est celui de cette livraison.

**13. Contrôles de data leakage**

Le fit n’accède qu’aux X/y admissibles du développement, avec fin de cible avant 2023. Les données ultérieures ne servent ni au scaler, ni à la sélection des paramètres, ni à un calibrateur.

Pour chaque séance aveugle demandée, `build_features` reçoit uniquement les préfixes SPY et IEF s’arrêtant à cette date. Les features recalculées sont comparées à celles enregistrées avec `rtol=1e-10`, `atol=1e-12`. Schéma exact, ordre, valeurs finies et séance SPY locale sont exigés. Aucun remplissage par une valeur future ou report au dernier prix.

Les tests espionnent les entrées de transformation et vérifient l’invariance aux modifications de prix ultérieurs en mémoire. Les empreintes protègent en parallèle l’identité des fichiers : modifier physiquement un instantané invalide sa provenance plutôt que de changer silencieusement la prédiction.

Les sources ajustées restent des instantanés susceptibles de révisions ; la causalité du calcul par préfixe ne certifie pas leur disponibilité historique réelle. Cette réserve est conservée dans le contrat.

Les exports aval ne contiennent aucun label, prix futur ou métrique de performance. Les dates de fin des labels d’entraînement servent uniquement à la traçabilité. Les observations futures ne sont jamais utilisées pour produire une fausse OOF historique.

**14. Résultats de reproductibilité**

Deux fits identiques sur le jeu synthétique reproduisent exactement les probabilités. Sur l’artefact réel, les chargements/appels répétés et les deux exports mensuels sont cohérents, sans second entraînement réel.

Les 95 valeurs OOF et 46 valeurs aveugles ont été revérifiées contre `predict_crisis` et les CSV. La sérialisation CSV à 17 chiffres significatifs conserve les flottants de référence ; un lecteur pandas peut introduire une différence d’arrondi minimale, couverte par une tolérance de `1e-15` dans les tests.

L’identité est déterministe pour les mêmes données, code, politique et versions. Le manifest comporte aussi une date réelle de génération : son contenu daté et son hash ne sont pas promis identiques entre reconstructions. Les versions différentes sont refusées lors du chargement ; une migration impose une reconstruction explicitement revue.

**15. Fichiers créés et modifiés**

Quatre nouveaux fichiers destinés au suivi Git :

- `src/models/ml1/inference.py`.
- `tests/ml1/test_inference.py`.
- `reports/ml1/integration_guide.md`.
- `reports/ml1/step8_integration.md`.

Cinq nouveaux artefacts dans le dossier ignoré détaillé en section 11. Soit neuf nouveaux fichiers au total. Aucun fichier préexistant modifié, aucune donnée brute changée, aucun pipeline partagé, fichier d’optimiseur ou composant de backtesting modifié.

Aucune dépendance installée, aucun téléchargement, aucune calibration choisie. Les sorties et prédictions des étapes précédentes restent intactes.

**16. Commandes de reconstruction**

Depuis la racine, sur `feature/ml1-crisis-detection` :

```bash
# Données core : uniquement si le dossier de destination est absent
.venv/bin/python -B -m src.models.ml1.pipeline

# Autre dossier neuf de données, sans écraser les sorties existantes
.venv/bin/python -B -m src.models.ml1.pipeline --output-dir data/processed/ml1/rebuild_001

# Modèle figé et exports : commande exécutée pour cette livraison
.venv/bin/python -B -m src.models.ml1.inference

# Reconstruction ultérieure autorisée dans un dossier neuf
.venv/bin/python -B -m src.models.ml1.inference --output-dir data/processed/ml1_inference_release_rebuild_001

# Tests
.venv/bin/python -B -m pytest tests/ml1/test_inference.py -q -p no:cacheprovider
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Le constructeur propose aussi `--dataset-dir`, `--raw-dir`, `--training-dir`. La documentation d’intégration explique les prérequis OOF ignorés par Git et leur récupération depuis une sauvegarde de confiance. Leur éventuelle régénération par les 24 entraînements de l’étape 5 est une opération distincte nécessitant une autorisation séparée ; elle n’a pas été exécutée ici.

Toute destination existante est refusée avant fit. Les cinq artefacts sont préparés dans un dossier temporaire neuf, puis publiés ensemble. Aucune relance automatique au chargement.

Environnement : **Python 3.12.13**, pandas **3.1.1**, NumPy **2.5.4**, pyarrow **26.0.1**, scikit-learn **1.9.1**, joblib **1.6.0**, exchange-calendars **4.13.3**, pytest **9.2.0**.

**17. Limites et éléments restant à finaliser**

Le module est techniquement exploitable pour préparer l’intégration, avec statut provisoire obligatoire. Aucune performance du modèle figé n’est mesurée sur 2023+. Aucune conclusion de qualité, rentabilité ou utilité financière n’est tirée des inférences aveugles.

La sélection du candidat sur 2015–2022 introduit un biais de sélection potentiel. Les probabilités brutes ne sont pas certifiées calibrées et le modèle n’a pas démontré de supériorité statistique. Le modèle figé ne se réadapte pas aux régimes postérieurs ; sa politique diffère des modèles OOF réentraînés annuellement.

Les prix ajustés peuvent être révisés ; leur disponibilité historique avant 17 h n’est pas attestée par un journal de réception. Le calendrier ARCX résolu vers XNYS est une référence logicielle, pas un audit officiel de marché. Les cibles de 21 séances se chevauchent.

L’optimiseur doit conserver les champs de provenance et réserves, sans assimiler `p_crisis` à une allocation. Le backtesting doit respecter la disponibilité à 17 h et ne pas supposer une exécution rétrospective au même cours de clôture. Aucun de ces composants aval n’est développé ici.

Les artefacts restent locaux et ignorés par Git : ils nécessitent une sauvegarde/transmission de confiance ou une reconstruction. Aucune diffusion, mise en production ou utilisation financière réelle n’est validée par cette étape.

**18. État Git final**

Branche inchangée : `feature/ml1-crisis-detection`. HEAD inchangé : `42f36c10250bb234e6ce0422b6dcece1020f7b6f`.

```text
?? reports/ml1/integration_guide.md
?? reports/ml1/step8_integration.md
?? src/models/ml1/inference.py
?? tests/ml1/test_inference.py
```

Les cinq artefacts de `data/processed/ml1_inference_release/` sont ignorés par la règle existante `data/*`, sans ajout forcé. Aucun fichier suivi modifié ni changement à l’index. Aucun commit, push, merge ou Pull Request. Les 101 fichiers préexistants contrôlés sont préservés.

**19. Verdict : PRÊT AVEC RÉSERVES**

L’étape 8 est terminée : interface simple et détaillée, routage OOF/aveugle strict, modèle figé pré-2023, traçabilité, chargement contrôlé, 95 mois OOF, 46 mois aveugles séparés, documentation d’intégration et 326 tests réussis.

Ce verdict porte sur la livraison technique provisoire. Les labels et performances 2023+ restent fermés ; les inférences aveugles produites à la demande ne constituent pas une évaluation. Aucun modèle, calibration ou usage financier réel n’est validé.

**L’étape 9 n’a pas commencé. Toute suite attend votre validation.**
