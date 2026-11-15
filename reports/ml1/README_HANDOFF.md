**ML 1 — guide de livraison au membre 1 — 15 novembre 2026**

Le module fournit `p_crisis`, probabilité qu’au moins une clôture SPY ajustée dans les **21 prochaines séances SPY** soit **strictement inférieure à 93 %** de sa clôture à la décision. Aucun rendement attendu, allocation, ordre ou backtest. Statut obligatoire : **provisional**, calibration **none**.

**Fonctionnalités livrées et architecture**

- `core.py`, `pipeline.py` : transformations historiques, calendrier SPY, cible stricte, sources autorisées locales.
- `time_validation.py` : développement admissible pré-2023, huit folds croissants 2015–2022, purge de 21 séances, contrôle de maturité.
- `market_indicators.py`, `indicator_validation.py` : variante VIX facultative EXPÉRIMENTALE, jamais activée dans l’inférence core.
- `baselines.py`, `evaluate_baselines.py` : références historique et VIX expérimental.
- `train_ml1.py`, `evaluate_ml1.py` : configurations fixes LR/RF/XGBoost, probabilités OOF et évaluation pré-2023.
- `calibration.py`, `explain_ml1.py` : calibration chronologique exploratoire, SHAP historique 2022.
- `inference.py` : unique LR figée pré-2023, routage historique OOF/aveugle, manifest et exports mensuels.
- `tests/ml1/` : 326 tests ; rapports des étapes 1–8 et `integration_guide.md` conservés.

Entrées utiles : `src/data/financial_dataset/raw/SPY_raw.csv` et `IEF_raw.csv`, instantanés locaux en lecture seule. Aucun besoin de collecte, clé FRED ou téléchargement pour la chaîne core. Ne pas modifier les sources d’un autre membre.

Dix features, dans leur ordre strict :

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

**Résultats pré-2023 et politique provisoire**

AP est prioritaire, Brier secondaire, ROC-AUC complémentaire. Scores OOF agrégés 2015–30/11/2022 sur 1 993 dates communes :

| Approche | AP | Brier | ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression brute | 0.111118 | 0.099509 | 0.545417 |
| Random Forest | 0.097037 | 0.104717 | 0.499764 |
| XGBoost | 0.086914 | 0.112210 | 0.411413 |
| Fréquence historique | 0.088309 | 0.093446 | 0.272661 |
| VIX seuil 25 — EXPÉRIMENTAL | 0.091453 | 0.095820 | 0.349705 |

Logistic Regression est seulement un candidat provisoire sélectionné sur ces validations. Le Brier brut ne bat pas les baselines ; résultats annuels variables, biais de sélection et dépendance des cibles chevauchantes. Aucune supériorité statistique ou validation d’usage financier réel.

Platt et Isotonic restent exploratoires : sur 1 741 dates réellement calibrées 2016–2022, Brier brut 0.096820, Platt 0.092661, Isotonic 0.092036, fréquence historique 0.090889. Aucun calibrateur choisi. Pentes Platt parfois négatives ; Isotonic produit 493 probabilités nulles dont 26 dates positives. SHAP décrit uniquement la LR brute du fold 2022, en log-odds, sans causalité ni stabilité globale certifiée.

Le VIX est retardé mais non certifié point-in-time. Les macros sont exclues. L’inférence livrée utilise exclusivement core et LR brute.

**Environnement et installation sur une nouvelle machine**

Environnement constaté : Python **3.12.13**, macOS 26.5.2 ARM64. `requirements-ml1.lock.txt` fixe les **31 distributions** de la fermeture runtime/tests de ML 1, sans les fournisseurs de collecte inutilisés hors ligne. Le fichier partagé `requirements.txt` est inchangé.

Sur un clone neuf et dans un environnement neuf seulement, après autorisation locale d’installation :

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reports/ml1/requirements-ml1.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Aucune installation neuve, résolution réseau ou disponibilité de roues n’a été testée pendant l’audit. Le lock décrit la machine validée, sans hash de roue et sans garantir la portabilité système. Les dépendances conditionnelles Linux/Windows devront être revues ; XGBoost déclare notamment une dépendance NCCL conditionnelle Linux qui n’est pas installée sur ce Mac. Ne pas présenter ce fichier comme un verrou universel multi-plateforme.

**OpenMP XGBoost : vérification et dépannage contrôlés**

Le binaire XGBoost local charge `@rpath/libomp.dylib`. À l’étape 1, un rpath relatif vers le runtime déjà fourni dans `.venv/.../sklearn/.dylibs` a été ajouté au binaire local XGBoost et sa signature locale renouvelée. Cette adaptation n’est ni une installation portable garantie ni une instruction à reproduire automatiquement.

Sondes de lecture et smoke test synthétique :

```bash
.venv/bin/python -B -c 'import sys, platform; print(sys.executable, platform.machine())'
.venv/bin/python -B -c 'import json; from src.models.ml1.train_ml1 import xgboost_preflight; print(json.dumps(xgboost_preflight(), indent=2))'
# macOS uniquement : inspection, aucune modification
ls .venv/lib/python3.12/site-packages/xgboost/lib/
ls .venv/lib/python3.12/site-packages/sklearn/.dylibs/
otool -L .venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
otool -l .venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
```

En cas d’échec : conserver stdout/stderr et le code/signal natif, vérifier architecture Python/roue, versions et existence d’un runtime OpenMP compatible. Sur macOS, examiner l’installation système existante de libomp ; toute installation ou adaptation native doit être revue par le propriétaire de la machine. Sur Linux/Windows, diagnostiquer les bibliothèques et DLL propres à la plateforme. Ne pas copier une dylib Mac vers une autre plateforme, modifier silencieusement des bibliothèques globales ou appliquer un patch de signature sans revue. Valider import + fit/predict synthétique avant toute reconstruction OOF. L’inférence LR seule n’appelle pas XGBoost, mais les tests et la reconstruction des trois modèles le requièrent.

**Avertissements**

Les 148 `Pandas4Warning` proviennent des quatre accès `schedule.open/break_start/break_end/close.values` dans `exchange_calendars/exchange_calendar.py:405–408`, via `check_calendar`. Ils sont visibles et non masqués. Non bloquants sur la combinaison validée, ils deviennent bloquants avec `warnings as errors` et signalent un risque de rupture avec pandas 4. Réviser la dépendance, puis relancer les tests/calendriers après toute migration ; aucun patch de bibliothèque pendant cet audit.

Le message pip sur son cache non inscriptible est distinct : `pip check` réussit. Les caches graphiques des vérifications sont temporaires et n’exigent aucune modification globale.

**Reconstruction stricte du core jusqu’à l’inférence**

Attention : la CLI pipeline standard construit normalement aussi des targets sur 2023+. Pour cet audit fermé aux labels du holdout, elle n’a pas été exécutée sans restriction. La construction a utilisé son API existante avec un adaptateur **en mémoire**, visible ci-dessous, limitant les prix fournis à `build_targets` à avant 2023. Les features sont recalculées sur toute la plage locale sans labels. Aucune source Python du dépôt n’est modifiée.

Le core isolé contient ainsi 4 329 lignes supervisées pré-2023 et 5 314 lignes de features d’inférence. Il ne reproduit pas la table supervisée complète de 5 293 lignes, car il exclut volontairement tout label 2023+. En revanche, X/y/contrôles du développement sont vérifiés identiques ; les 5 979 OOF, coefficients figés et tables mensuelles reproduisent exactement les livraisons.

La validation temporelle, les 24 entraînements OOF et la construction figée sont ensuite exécutés par leurs CLI réelles avec chemins temporaires explicites. Les ressources existantes ne sont jamais remplacées. Aucun label final ni score holdout n’est lu/calculé.

Pour reproduire **avec autorisation d’entraîner dans ce test isolé** : copier le bloc Python suivant sous `reconstruct.py` dans un dossier neuf, puis lancer depuis la racine du dépôt :

```bash
# Exemple de destination neuve ; ne pas réutiliser un ancien dossier de résultats.
mkdir /private/tmp/ml1-review-rebuild-001
# Enregistrer le bloc ci-dessous : /private/tmp/ml1-review-rebuild-001/reconstruct.py
.venv/bin/python -B /private/tmp/ml1-review-rebuild-001/reconstruct.py
```

Le script refuse ses dossiers de sortie s’ils existent. Il conserve sources copiées, logs et preuves ; aucun nettoyage ou suppression n’est effectué par le script. La recette fonctionne avec les seuls fichiers Git et sources SPY/IEF : si les anciennes sorties sont absentes, elle les reconstruit sans comparaison de référence. Si elles sont présentes, elle vérifie en plus l’égalité. L’option --without-reference force le mode sans lecture des anciens artefacts ignorés ; les assertions de comparaison sont alors explicitement non réalisées.

```python
"""Isolated audit: compute targets only from pre-2023 prices, retain all features."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

repo = Path.cwd()
sys.path.insert(0, str(repo))
from src.models.ml1 import pipeline, inference
from src.models.ml1.time_validation import FINAL_TEST_START, LABEL, build_walk_forward_folds, load_development_data

compare_reference = '--without-reference' not in sys.argv and all(p.exists() for p in [pipeline.DEFAULT_OUTPUT/'dataset_metadata.json', inference.DEFAULT_TRAINING/'ml_models_oof_predictions.parquet', inference.DEFAULT_RELEASE/'model_manifest.json'])
root = Path(__file__).resolve().parent
raw, core, training, release = [root/name for name in ('raw', 'core', 'training', 'inference_release')]
if any(p.exists() for p in (raw, core, training, release)):
    raise FileExistsError('Use a completely new isolated directory; no overwrites')
raw.mkdir()
for name in ('SPY_raw.csv', 'IEF_raw.csv'):
    shutil.copyfile(pipeline.DEFAULT_RAW/name, raw/name)

original_builder = pipeline.build_targets
target_calls = []
def closed_target_builder(spy):
    historical = spy.loc[spy.index < FINAL_TEST_START]
    if historical.empty or historical.index.max() >= FINAL_TEST_START:
        raise AssertionError('Target builder cannot inspect 2023+ prices or labels')
    target_calls.append({'first_price': str(historical.index[0].date()), 'last_price': str(historical.index[-1].date())})
    result = original_builder(historical).reindex(spy.index)
    # Unknown future labels remain NA, never zero. Only the count metadata gets zero.
    result['future_session_count'] = result.future_session_count.fillna(0).astype('int16')
    return result

pipeline.build_targets = closed_target_builder
try:
    pipeline.build_dataset(raw, core)
finally:
    pipeline.build_targets = original_builder

reader = pd.read_parquet
label_reads = []
def guarded(path, *args, **kwargs):
    if Path(path).name == 'targets_ml1.parquet':
        columns = kwargs.get('columns')
        assert columns is not None
        if LABEL in columns:
            assert columns == [LABEL]
            assert ('Date','<',FINAL_TEST_START) in kwargs.get('filters', [])
            assert ('target_end_session','<',FINAL_TEST_START) in kwargs.get('filters', [])
            label_reads.append(str(path))
    return reader(path, *args, **kwargs)
pd.read_parquet = guarded

data = load_development_data(core, raw/'SPY_raw.csv')
if compare_reference:
    reference = load_development_data()
    pd.testing.assert_frame_equal(data.X, reference.X)
    pd.testing.assert_series_equal(data.y, reference.y)
    pd.testing.assert_frame_equal(data.target_info, reference.target_info)
folds = build_walk_forward_folds(data)
assert len(data.X) == 4329 and len(folds) == 8
assert all(len(f.purged_train_indices) == 21 for f in folds)
full_features = reader(core/'features_ml1_inference.parquet')
if compare_reference:
    pd.testing.assert_frame_equal(full_features, reader(pipeline.DEFAULT_OUTPUT/'features_ml1_inference.parquet'))

commands = [
    [sys.executable, '-B', '-m', 'src.models.ml1.time_validation', '--dataset-dir', str(core), '--spy-csv', str(raw/'SPY_raw.csv')],
    [sys.executable, '-B', '-m', 'src.models.ml1.train_ml1', '--dataset-dir', str(core), '--spy-csv', str(raw/'SPY_raw.csv'), '--output-dir', str(training)],
    [sys.executable, '-B', '-m', 'src.models.ml1.inference', '--dataset-dir', str(core), '--raw-dir', str(raw), '--training-dir', str(training), '--output-dir', str(release)],
]
for index, command in enumerate(commands, 1):
    with (root/f'command_{index}.log').open('x') as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Reconstruction command {index} failed; inspect command_{index}.log')

rebuilt = reader(training/'ml_models_oof_predictions.parquet')
if compare_reference:
    original = reader(inference.DEFAULT_TRAINING/'ml_models_oof_predictions.parquet')
    pd.testing.assert_frame_equal(rebuilt, original)
metadata = json.loads((training/'training_metadata.json').read_text())
assert metadata['successful_fits'] == 24

engine = inference.load_inference(release, trusted=True)
if compare_reference:
    old_engine = inference.load_inference(trusted=True)
    np.testing.assert_array_equal(engine.model.named_steps['classifier'].coef_, old_engine.model.named_steps['classifier'].coef_)
    np.testing.assert_array_equal(engine.model.named_steps['classifier'].intercept_, old_engine.model.named_steps['classifier'].intercept_)
    assert engine.manifest['model_id'] == old_engine.manifest['model_id']

# From here on no targets read, and no fit can be invoked.
def no_labels(path, *args, **kwargs):
    assert 'targets' not in Path(path).name
    return reader(path, *args, **kwargs)
pd.read_parquet = no_labels
inference.fit_frozen_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No inference refit'))
inference.configure_inference(release, trusted=True)
for filename, count, origin in [('monthly_oof_probabilities.csv',95,'historical_oof'), ('monthly_blind_probabilities.csv',46,'blind_holdout_inference')]:
    with (release/filename).open() as f:
        rows = list(csv.DictReader(f))
    if compare_reference:
        with (inference.DEFAULT_RELEASE/filename).open() as f:
            old_rows = list(csv.DictReader(f))
        assert rows == old_rows
    assert len(rows) == count
    for row in rows:
        p = inference.predict_crisis(row['prediction_date'])
        assert type(p) is float and 0 <= p <= 1 and p == float(row['p_crisis'])
        detail = inference.predict_crisis_details(row['prediction_date'])
        assert detail['prediction_origin'] == origin and detail['calibration_method'] == 'none'
        assert detail['status'] == 'provisional' and detail['horizon_sessions'] == 21
        assert not {'y_crisis_21d','observed_target','future_min_price_21'}.intersection(row)

summary = {
    'status': 'complete', 'temporary_root': str(root), 'dataset_rows': len(data.X),
    'inference_feature_rows': len(full_features), 'folds': len(folds), 'successful_fits': 24,
    'oof_prediction_rows': len(rebuilt), 'reference_comparison_performed': compare_reference, 'oof_full_frame_identical': True if compare_reference else None,
    'monthly_oof_rows': 95, 'monthly_blind_rows': 46, 'monthly_rows_identical': True if compare_reference else None,
    'frozen_coefficients_identical': True if compare_reference else None, 'model_id': engine.manifest['model_id'],
    'target_builder_calls': target_calls, 'protected_label_reads': label_reads,
    'holdout_labels_generated': False, 'holdout_labels_read': False, 'holdout_metrics_computed': False,
    'pipeline_mode': 'existing API with explicit in-memory pre-2023 target adapter; no source-file edit',
    'adapter_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'commands': commands,
}
with (root/'reconstruction_summary.json').open('x') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

```

La reconstruction avec comparaison est conservée dans `/private/tmp/ml1-step9-rebuild-_mv_tp6f/`. Une seconde reconstruction source-only est conservée dans `/private/tmp/ml1-step9-source-only-jfh6xjy8/`, avec `reconstruction_summary.json` et `command_1.log` à `command_3.log`. Ces preuves temporaires ne sont pas des fichiers à ajouter à Git. Une archive de revue peut être transmise ultérieurement seulement après validation utilisateur.

**Commandes et prérequis des interfaces existantes**

```bash
.venv/bin/python -B -m src.models.ml1.pipeline --help
.venv/bin/python -B -m src.models.ml1.time_validation --help
.venv/bin/python -B -m src.models.ml1.train_ml1 --help
.venv/bin/python -B -m src.models.ml1.inference --help
```

Options validées : pipeline `--raw-dir/--output-dir` ; validation `--dataset-dir/--spy-csv/--embargo-sessions` ; OOF `--dataset-dir/--spy-csv/--output-dir` ; inférence `--dataset-dir/--raw-dir/--training-dir/--output-dir`. Dossiers de destination neufs obligatoires. `train_ml1` et `inference` exigent la branche `feature/ml1-crisis-detection` ; ne pas annoncer leur exécution directe sur main après fusion sans une revue distincte de cette garde.

Les validateurs CLI d’évaluation et calibration restent disponibles pour leurs étapes respectives ; ils ne sont pas nécessaires pour consommer `p_crisis`. Une CLI non modifiée qui recrée des targets du holdout ne doit pas être utilisée sous l’exigence actuelle de ne consulter aucun label final ; employer la recette stricte explicitée ci-dessus.

**Utilisation par les membres 4 et 5**

```python
from src.models.ml1.inference import (
    configure_inference, predict_crisis, predict_crisis_details,
    load_inference, monthly_oof_probabilities, monthly_blind_probabilities,
)

configure_inference(trusted=True)
p = predict_crisis("2022-11-30")
details = predict_crisis_details("2022-11-30")

# Objet indépendant : exemple d’inférence aveugle, sans label ni performance.
engine = load_inference("data/processed/ml1_inference_release", trusted=True)
p_blind = engine.predict_crisis("2023-01-31")
historical = monthly_oof_probabilities(engine)
blind = monthly_blind_probabilities(engine)
```

Le membre 4 n’a besoin que du float `p_crisis` comme entrée numérique ; il doit conserver en contexte les réserves `status=provisional`, `calibration_method=none`, l’horizon et l’origine. Aucun seuil d’allocation n’est proposé.

Le membre 5 peut lire `monthly_oof_probabilities.csv` : 95 mois de janvier 2015 à novembre 2022, sans labels, dernière séance OOF admissible de chaque mois. Décembre 2022 est absent. La table distincte `monthly_blind_probabilities.csv` contient 46 mois complets de janvier 2023 à octobre 2026. Novembre 2026 est exclu, les features locales s’arrêtant au 04/11/2026. Aucun label dans ces CSV ; les champs `training_label_end` sont des dates de provenance.

Historique : `historical_oof`, modèles annuellement réentraînés et purgés ; dates sans OOF refusées, jamais inférées in-sample avec le modèle figé. Post-2022 : `blind_holdout_inference`, modèle figé sur 4 329 observations 2005–30/11/2022, dernière fin de cible 30/12/2022, sans réadaptation ni évaluation. Ces deux politiques ne sont pas interchangeables.

Le détail conserve date, probabilité, horizon, modèle/version, variante, calibration, origine, dernière décision train, dernière fin de cible train, décision locale et statut. Décision **17 h America/New_York**, après clôture et sous hypothèse de réception du cours. Un backtest aval ne doit pas exécuter rétrospectivement au cours de clôture déjà connu à 17 h. La reconstruction n’est pas une preuve de publication réelle à la date historique.

Chargement joblib uniquement depuis une source de confiance : `trusted=True` est obligatoire avant désérialisation ; les empreintes ne sont pas des signatures d’authenticité. Un changement de code, version, ressource ou artefact provoque un refus plutôt qu’un fit ou téléchargement automatique.

**Fichiers ignorés, limites et décisions collectives**

À reconstruire ou transmettre séparément par une source de confiance : `.venv/`, `data/processed/ml1/`, `data/processed/ml1_core_vix_experimental/`, `data/processed/ml1_training/`, `data/processed/ml1_inference_release/`. Le core, OOF et release ont été reconstruits en mode strict ; la variante VIX optionnelle n’a pas été régénérée à cette étape. Les sources brutes SPY/IEF nécessaires sont déjà suivies dans main et restent inchangées.

Les résultats PNG et deux Parquet déjà suivis sous `reports/ml1/` apparaissent dans le diff de branche. Ils n’ont pas été ajoutés de force pendant l’audit ni supprimés. Le membre 1 doit décider si ces neuf binaires doivent rester versionnés ou relever d’une autre politique d’artefacts. Aucune réécriture de commits ou suppression dans cette mission.

Revue collective requise : disponibilité historique des prix ajustés, risque de révisions et réception réelle avant 17 h, calendrier ARCX résolu vers XNYS, seuil de crise provisoire, cibles chevauchantes non indépendantes, comparaison annuelle/frozen, politique d’artefacts, garde de branche des CLI, environnement neuf/OpenMP et absence de validation financière réelle. Le VIX n’est pas certifié point-in-time. Aucune calibration sélectionnée, allocation, stratégie, performance 2023+, mise en production ou diffusion automatique.

La PR préparée est un support de revue, pas une demande de fusion automatique. Aucun commit, push, PR ou transmission au membre 1 n’est effectué par cet audit.
