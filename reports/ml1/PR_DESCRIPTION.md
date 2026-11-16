**Titre proposé**

feat(ml1): ajouter la détection provisoire de crise SPY et son interface temporelle

**Objectif et comportement livré**

Cette branche ajoute un module ML 1 indépendant pour prédire la probabilité qu’au moins une clôture SPY ajustée des 21 prochaines séances soit strictement inférieure à 93 % du prix à la décision. Elle fournit une interface `predict_crisis(date) -> float`, accompagnée d’une provenance explicite pour l’optimiseur et les travaux de backtesting.

Sur 2015–2022, l’interface retourne exclusivement les OOF enregistrées du fold concerné ; elle refuse une date historique sans OOF, notamment décembre 2022. Sur 2023+, elle utilise une LR brute figée uniquement sur le développement pré-2023 et marque la sortie `blind_holdout_inference`, non évaluée. Aucune allocation, stratégie ou performance holdout.

**Résumé des changements**

- Pipeline core historique et cible de baisse strictement >7 %, indépendants des scripts partagés ; dix features SPY/IEF ordonnées, calendrier contrôlé et aucune imputation future.
- Huit folds croissants 2015–2022, purge de 21 séances et contrôle de maturité des labels ; chargeur protégé contre les labels 2023+.
- Baselines historique/VIX, configurations fixes LR/RF/XGBoost, 5 979 OOF et évaluation pré-2023. VIX seulement EXPÉRIMENTAL.
- Calibration Platt/Isotonic exploratoire sans méthode sélectionnée ; SHAP LR 2022 en log-odds après reconstruction concordante.
- Artefact LR figé pré-2023, manifest vérifié, chargement joblib de confiance explicite et tables mensuelles séparées : 95 mois OOF et 46 mois aveugles complets.
- Documentation d’intégration, guide de livraison, lock ML 1 de 31 distributions et rapports d’audit.

**Modules ajoutés**

Dans `src/models/ml1/` : `__init__.py`, `core.py`, `pipeline.py`, `time_validation.py`, `market_indicators.py`, `indicator_validation.py`, `baselines.py`, `evaluate_baselines.py`, `train_ml1.py`, `evaluate_ml1.py`, `calibration.py`, `explain_ml1.py`, `inference.py`.

Tests dans `tests/ml1/`. Rapports et sorties de revue dans `reports/ml1/`. Aucun composant, requirement partagé ou source brute d’un autre membre modifié.

**Validation**

```bash
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Résultat validé : **326 passed, 148 warnings**. Les avertissements proviennent des conversions `.values` avec fuseau horaire dans exchange-calendars 4.13.3 avec pandas 3.1.1 ; risque futur pandas 4, sans échec sur l’environnement constaté. `pip check` réussi. Imports et smoke test XGBoost/OpenMP réussis sur ce Mac.

Reconstruction isolée effectuée depuis SPY/IEF locaux, sans écraser les sorties : 4 329 lignes de développement, 5 314 features d’inférence, huit folds, 24 fits OOF, 5 979 lignes OOF identiques, coefficients et identifiant figés identiques, CSV mensuels identiques. Une exécution distincte vérifie le fonctionnement sans lecture des anciens artefacts ignorés. Targets calculés seulement avec des prix antérieurs à 2023 ; aucun label final généré ou consulté, aucune performance 2023+.

La construction core de ce test utilise un adaptateur temporaire en mémoire, documenté dans `README_HANDOFF.md`, car la CLI pipeline standard construit également des targets post-2022. Aucun fichier source n’est modifié et aucun contrôle n’est supprimé. Les CLI de validation, entraînement OOF et inférence sont ensuite exécutées sur les chemins temporaires explicites.

**Reproduction**

1. Lire `reports/ml1/README_HANDOFF.md` et préparer un environnement Python 3.12 neuf avec `reports/ml1/requirements-ml1.lock.txt` ; l’installation neuve et la portabilité ne sont pas certifiées.
2. Vérifier les imports, `pip check`, puis le précontrôle natif XGBoost. L’adaptation OpenMP locale existante n’est pas une recette portable.
3. Exécuter la suite ML 1.
4. Utiliser la recette de reconstruction stricte dans un dossier temporaire neuf. Elle ne matérialise aucun label 2023+ et ne remplace aucun résultat validé.
5. Charger explicitement les artefacts de confiance avec `trusted=True`.

Les CLI `train_ml1` et `inference` exigent actuellement `feature/ml1-crisis-detection`. Cette garde doit être revue collectivement pour un usage après fusion sur main ; aucun contournement silencieux dans cette livraison.

**Interface membre 4**

```python
from src.models.ml1.inference import configure_inference, predict_crisis, predict_crisis_details

configure_inference(trusted=True)
p_crisis = predict_crisis("2022-11-30")
details = predict_crisis_details("2022-11-30")
```

Le consommateur peut utiliser uniquement le float `p_crisis` comme entrée, mais doit conserver le statut provisoire, l’origine et la convention temporelle. Aucun seuil ni règle d’allocation proposé. Les probabilités ne sont pas certifiées calibrées.

**Interface membre 5**

`monthly_oof_probabilities.csv` : 95 mois janvier 2015–novembre 2022, dernière séance OOF admissible du mois, sans labels. `monthly_blind_probabilities.csv` : 46 mois complets janvier 2023–octobre 2026, origine aveugle distincte, sans labels ni performances. Novembre 2026 reste incomplet, dernière feature locale le 04/11/2026.

Les OOF annuellement réentraînées et les prédictions du modèle figé ne sont pas interchangeables. Décision à 17 h New York, après clôture : ne pas simuler une exécution rétrospective au même cours de clôture connu à 17 h. Une reconstruction historique n’est pas une preuve de publication contemporaine.

**Limites et risques**

Logistic Regression est seulement candidate provisoire sur l’AP de validation (0.111118). Son Brier brut (0.099509) reste moins bon que la fréquence historique (0.093446). Aucune supériorité statistique ; choix sur validation potentiellement biaisé. Fenêtres de 21 séances chevauchantes, périodes de crise peu nombreuses, variabilité annuelle.

Aucune calibration sélectionnée. VIX expérimental non certifié point-in-time. Prix ajustés susceptibles de révisions, disponibilité historique à 17 h supposée sans journal de réception, calendrier logiciel ARCX résolu vers XNYS. Aucun test final de performance 2023+, backtest ou validation financière réelle.

Les artefacts joblib nécessitent une source de confiance ; les SHA-256 ne remplacent pas l’authentification. Le lock décrit les versions validées sur macOS ARM64 et n’inclut pas toutes les dépendances conditionnelles d’autres plateformes. Une nouvelle machine doit être validée avant usage.

**Points de revue demandés au membre 1**

- Formule de cible, calendrier, maturité des labels et contrôle de la frontière 2023.
- Contrat des dix features, hypothèses de disponibilité des prix ajustés et source de vérité des données.
- Routage OOF/aveugle, métadonnées et fichiers sans labels pour les membres 4/5.
- Politique provisoire LR brute ; absence de calibration sélectionnée et de validation financière réelle.
- Recette stricte de reconstruction, environnement neuf/OpenMP et garde de branche des CLI après fusion.
- Politique des neuf binaires de rapport déjà suivis dans les commits locaux : conserver dans Git ou déplacer ultérieurement sous une autre politique d’artefacts. Ils n’ont été ni supprimés ni ajoutés de force dans cet audit.
- Fraîcheur de main distant à vérifier avant la vraie PR : l’audit compare uniquement les références locales, sans fetch/push.

**Exclus du futur ajout Git parce qu’ignorés ou hors périmètre**

`.venv/`, caches, `data/processed/ml1/`, `data/processed/ml1_core_vix_experimental/`, `data/processed/ml1_training/`, `data/processed/ml1_inference_release/`, dossiers temporaires de reconstruction, secrets et fichiers personnels. Ne pas employer `git add -f`. Les sources brutes SPY/IEF déjà présentes dans main ne sont pas ajoutées ni modifiées par cette PR.

La liste exacte des 54 chemins de la PR et des huit fichiers encore à enregistrer est dans `step9_final_audit.md`.

**Statut de ce document**

Texte préparatoire uniquement. Aucune Pull Request, transmission au membre 1, création de commit, push ou merge n’a été effectuée par l’étape 9. Revue et fusion seulement après validation utilisateur puis membre 1.
