**Rapport de validation du pipeline de données ML 1 — 14 novembre 2026**

Périmètre : préparation hors ligne des données du membre 3. Aucun modèle entraîné, aucun scaler ou imputeur ajusté, aucune optimisation, prédiction ou exécution de backtest. Aucun téléchargement de cours ou de séries économiques.

**1. Branche Git et état initial du dépôt**

Branche vérifiée avant intervention : `feature/ml1-crisis-detection`. Commit : `8d9ff5ff66c161181f55a1da56d59cae02394375`. Aucun travail effectué sur `main`.

Le seul élément préexistant non suivi était `reports/ml1/step1_remediation.md`. Il a été conservé tel quel, y compris les modifications faites depuis la mission précédente. Son empreinte et celles de tous les fichiers suivis ont été enregistrées avant intervention puis comparées après exécution : aucun changement.

**2. Sources réellement utilisées**

Véritable dossier brut : `src/data/financial_dataset/raw/`.

Entrées numériques du pipeline : `SPY_raw.csv` et `IEF_raw.csv`, colonnes `Date` et `Adj Close`. Les prix sont des clôtures ajustées en USD. Les dates horodatées sont converties en dates de séance America/New_York, sans utiliser la date UTC comme substitut.

| Source locale | Lignes | Valeurs valides | Valeurs manquantes | Période des valeurs valides | Unité |
|---|---:|---:|---:|---|---|
| SPY_raw.csv | 5 565 | 5 565 | 0 | 23/09/2004–04/11/2026 | USD ajustés |
| IEF_raw.csv | 5 565 | 5 565 | 0 | 23/09/2004–04/11/2026 | USD ajustés |
| LQD_raw.csv | 5 565 | 5 565 | 0 | 23/09/2004–04/11/2026 | USD ajustés |
| VNQ_raw.csv | 5 561 | 5 561 | 0 | 29/09/2004–04/11/2026 | USD ajustés |
| VIX_raw.csv, colonne VIXCLS | 5 770 | 5 598 | 172 | 23/09/2004–04/11/2026 | Points d’indice |
| BAA10Y_raw.csv | 5 770 | 5 528 | 242 | 23/09/2004–04/11/2026 | Points de pourcentage |
| DGS3MO_raw.csv | 5 770 | 5 533 | 237 | 23/09/2004–04/11/2026 | Points de pourcentage |
| DGS2_raw.csv | 5 770 | 5 533 | 237 | 23/09/2004–04/11/2026 | Points de pourcentage |
| DGS10_raw.csv | 5 770 | 5 533 | 237 | 23/09/2004–04/11/2026 | Points de pourcentage |
| T10Y2Y_raw.csv | 5 770 | 5 533 | 237 | 23/09/2004–04/11/2026 | Points de pourcentage |

Les quatre fichiers ETF contiennent aussi Open, High, Low, Close, Volume, Dividends, Stock Splits et Capital Gains ; ces colonnes ne servent pas à construire les cibles de cette version. Les fichiers FRED ont une colonne `observation_date` et une colonne de valeurs, sans preuve de publication historique.

Les fichiers `raw/final_dataset.csv` et `processed/final_dataset.csv` sont les versions filtrées de 5 517 lignes examinées lors de l’audit ; le premier est une copie du second effectuée par le notebook. Aucun des deux n’est une entrée de reconstruction du nouveau pipeline. Les anciens Parquet du membre 1 ne sont pas chargés pour produire les nouvelles tables. Toutes les sources inventoriées et leurs empreintes sont enregistrées dans les métadonnées.

**3. Rôle vérifié de app.py et des autres scripts**

`app.py` est absent de cette copie du dépôt, hors dépendances de `.venv`. Il est donc impossible de confirmer le rôle du fichier évoqué par le membre 1 ; aucun fichier portant ce nom n’a été inventé ou modifié.

La chaîne réellement présente est :

1. `src/data/download_datasets.py` télécharge les ETF via yfinance et les indicateurs via les points d’accès CSV FRED ; il écrit les fichiers bruts, fait une jointure par date, supprime les lignes incomplètes et écrit le CSV traité ainsi que son rapport qualité.
2. `notebooks/data_engineering_features.ipynb` lit ce CSV traité, en conserve une copie dans raw, calcule les variables et les anciennes cibles, aligne les tables puis écrit les quatre anciens Parquet ML 1/ML 2.
3. Le nouveau pipeline lit directement les prix bruts complets et effectue ses propres validations. Il n’importe pas le script de téléchargement et n’exécute aucune cellule du notebook.

Les deux anciens traitements sont conservés. Le nouveau code est indépendant de leurs effets de bord et de la cible `y_stress_regime`.

**4. Fichiers créés ou modifiés**

Nouveaux fichiers de code :

- `src/models/ml1/__init__.py` : version du pipeline, `0.1.0`.
- `src/models/ml1/core.py` : validation, décisions, cibles, features, garde de disponibilité et diagnostics.
- `src/models/ml1/pipeline.py` : lecture des sources, contrôle du calendrier, métadonnées, exports et CLI.

Nouveaux fichiers de tests : `tests/ml1/__init__.py`, `tests/ml1/test_core.py`, `tests/ml1/test_pipeline.py`.

Nouvelle documentation : `reports/ml1/step2_data_validation.md`, présent rapport.

Cinq sorties ont été créées dans `data/processed/ml1/`, ignoré par Git. Aucun script, notebook, dataset brut ou ancien Parquet n’a été modifié. Le rapport de l’étape 1 et les composants ML 2 sont inchangés.

L’environnement `.venv` existant a reçu uniquement la bibliothèque de calendrier nécessaire au contrôle hors ligne et ses nouvelles dépendances : `exchange-calendars 4.13.3`, `pyluach 2.3.0`, `toolz 1.2.0`, `korean_lunar_calendar 0.4.0`. Les bibliothèques déjà installées ont été conservées. Aucun changement de `requirements.txt` ou d’installation Python globale.

Environnement utilisé : Python 3.12.13, pandas 3.1.1, NumPy 2.5.4, pyarrow 26.0.1 et pytest 9.2.0. `pip check` ne signale aucun conflit. Les versions utiles à la construction figurent dans les métadonnées.

**5. Définition des cibles 7 % et 10 %**

Pour une séance t et sa clôture ajustée SPY P[t] :

```text
future_min_return_21(t) = min(P[t+1], ..., P[t+21]) / P[t] - 1
y_crisis_21d_07(t) = 1 si min(P[t+1], ..., P[t+21]) < 0,93 * P[t]
y_crisis_21d_10(t) = 1 si min(P[t+1], ..., P[t+21]) < 0,90 * P[t]
y_crisis_21d(t) = y_crisis_21d_07(t)
```

Sinon, le label vaut 0 si les 21 prix futurs et le prix de référence sont valides. Il reste manquant si un prix requis est invalide ou si l’horizon est incomplet. La fenêtre exclut t, inclut t+21 et exclut t+22. Un rebond terminal n’efface pas un franchissement antérieur.

Les comparaisons de prix utilisent des représentations décimales pour ne pas transformer une égalité exacte au seuil en franchissement à cause d’un arrondi de `P_min/P_t - 1`. Le rendement continu exporté reste un nombre flottant. Il s’agit d’événements sur clôtures ajustées, sans détection intrajournalière.

Le seuil opérationnel de 7 % est **provisoire, fixé par instruction de l’utilisateur**. Les deux candidats sont construits en mémoire et utilisés uniquement pour les diagnostics ; les colonnes candidates ne figurent pas dans les Parquet opérationnels. `y_stress_regime` n’est jamais utilisée.

Diagnostics descriptifs limités à 2005–2014, avec fin de cible au plus tard le 31/12/2014 :

| Seuil | Observations | Positifs | Négatifs | Fréquence |
|---|---:|---:|---:|---:|
| Plus de 7 % | 2 496 | 257 | 2 239 | 10,30 % |
| Plus de 10 % | 2 496 | 124 | 2 372 | 4,97 % |

Dates t admissibles : 03/01/2005–01/12/2014. Le CSV de diagnostic contient 22 lignes : deux totaux et deux lignes par année de 2005 à 2014. Ces observations partagent des fenêtres futures : elles ne comptent pas des crises indépendantes. Aucun résultat du test final 2023+ n’a servi à sélectionner le seuil, les variables ou les paramètres.

**6. Nombre d’observations et périodes couvertes**

| Table / étape | Observations | Colonnes | Période |
|---|---:|---:|---|
| Calendrier SPY brut validé | 5 565 | Prix SPY | 23/09/2004–04/11/2026 |
| features_ml1.parquet | 5 293 | 10 | 21/09/2005–06/10/2026 |
| targets_ml1.parquet | 5 293 | 7 | 21/09/2005–06/10/2026 |
| features_ml1_inference.parquet | 5 314 | 10 | 21/09/2005–04/11/2026 |

Les 251 premières séances n’ont pas toutes les features à cause des fenêtres de chauffe, notamment 252 séances pour le drawdown historique. Les 21 dernières séances ayant des features valides restent dans la table d’inférence malgré leur cible inconnue. Aucune séance SPY n’a été supprimée du calendrier de calcul.

Les tables supervisées contiennent toutes les dates étiquetables, y compris celles de la période de test final. Elles ne doivent donc jamais être passées intégralement à un entraînement. Le contrat enregistré réserve l’entraînement à 2005–2014 avec `target_end_session <= 2014-12-31` ; les périodes ultérieures exigent leurs propres masques chronologiques et purges. Aucun entraînement n’est réalisé ici.

**7. Features retenues et exclues**

Dix features sont retenues, dans un ordre stable :

- `SPY_ret_1d`, `SPY_ret_5d`, `SPY_ret_21d`, `SPY_ret_63d` : `P[t]/P[t-w] - 1`.
- `SPY_vol_5d`, `SPY_vol_21d`, `SPY_vol_63d` : écart-type échantillonnal des rendements historiques, `ddof=1`, annualisé par √252.
- `SPY_drawdown_252d` : prix courant divisé par le maximum des 252 dernières séances, moins 1.
- `SPY_vs_ma200` : prix courant divisé par la moyenne des 200 dernières séances, moins 1.
- `SPY_IEF_corr_63d` : corrélation historique des rendements SPY et IEF sur 63 séances.

Ces formules reprennent celles documentées par le membre 1, mais s’appliquent au calendrier brut conservé. Les rendements utilisent explicitement `pct_change(fill_method=None)`. Aucun prix n’est imputé.

Exclusions provisoires : VIX, variation VIX à cinq séances, BAA10Y, T10Y2Y, DGS3MO, DGS2 et DGS10. Les données historiques de publication/révision ne sont pas présentes dans les CSV. LQD et VNQ ne sont pas nécessaires au sous-ensemble ML 1 retenu ; leurs valeurs ne participent pas aux features.

Si IEF est absent d’un autre jeu d’entrée, le pipeline peut produire les neuf features SPY et documente l’exclusion de la corrélation. Si IEF a des trous, les séances SPY restent présentes ; les fenêtres de corrélation concernées restent manquantes. Cette solution ne fabrique aucune observation.

**8. Disponibilité temporelle des features**

Convention : décision à **17 h America/New_York**, après réception des clôtures, y compris lors des clôtures anticipées. Les changements d’heure sont pris en compte. Le contrôle vérifie que les clôtures planifiées précèdent la coupure sur toutes les séances.

Les features de prix utilisent uniquement des observations de date inférieure ou égale à t. Leur usage suppose que les clôtures ETF ont effectivement été reçues avant la coupure ; les CSV ne comportent pas de journal historique de réception. Les prix ajustés sont des instantanés et ne certifient pas un historique intégralement point-in-time. Cette hypothèse demeure une réserve explicite.

VIX provient ici de VIXCLS, série CBOE distribuée par FRED. Ni son heure historique d’arrivée ni celle des indicateurs macro n’est établie dans les fichiers. La version fiable de ce pipeline les exclut donc entièrement : aucun `shift(1)` n’est présenté comme une preuve de disponibilité.

Une fonction pure `align_known_observations` est testée pour une intégration future de sources vérifiées : elle exige `observation_date`, `known_at` avec fuseau et `value`, puis retient uniquement une version connue avant la décision. Une révision tardive d’une vieille observation ne remplace pas une observation plus récente. Sans observation admissible, elle laisse NA. Cette fonction n’est pas utilisée pour réintroduire les CSV FRED actuels et ne prouve pas l’authenticité d’horodatages fournis.

**9. Traitement des séances manquantes**

Calendrier de référence : `ARCX` dans exchange-calendars 4.13.3 ; cette bibliothèque le résout vers `XNYSExchangeCalendar`. C’est une référence logicielle, pas un audit officiel de la place. Son fonctionnement par séances est décrit dans la [documentation du projet](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md).

Sur la plage brute observée : **5 565 séances attendues et 5 565 observées**, aucune séance attendue manquante et aucune date inattendue. Les 205 jours de semaine sans séance sont identifiés comme fermetures par les règles du calendrier, et non comme trous de prix. Les clôtures anticipées comptent chacune pour une séance.

Le pipeline échoue explicitement si une séance attendue manque, si une date inattendue apparaît, si des dates sont dupliquées/non croissantes ou si un prix SPY est invalide. Il ne remplit ni ne supprime silencieusement ces observations. Les fermetures exceptionnelles des 29 et 30 octobre 2012 sont couvertes par un test distinct d’une séance attendue artificiellement supprimée.

Aucun manque FRED ne peut modifier le calendrier ou les cibles, puisque les séries concernées sont exclues des calculs de cette version. Les vérifications portent sur la plage effectivement couverte par les fichiers ; elles ne prétendent pas que les sources sont actualisées au jour de ce rapport.

**10. Résultats des tests pytest**

Commande exécutée :

```bash
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Résultat final : **58 tests réussis**, en 2,95 secondes dans cette exécution. La suite émet 64 avertissements de dépréciation `Pandas4Warning` provenant de l’usage interne de `Series.values` dans exchange-calendars avec pandas 3.1.1. Aucun test n’échoue dans la configuration normale. Une sonde avec cet avertissement converti en erreur confirme que cette combinaison n’est pas compatible avec un mode global « warnings as errors » sans traitement de la dépendance ; les avertissements n’ont pas été masqués dans les tests.

Les tests couvrent : validation des prix/dates ; seuils exacts et stricts ; t+1/t+21/t+22 ; rebond ; cibles incomplètes ou prix invalides ; correspondance du label opérationnel avec 7 % ; inclusion des positifs 10 % dans les positifs 7 % ; formules historiques ; stabilité par troncature ; IEF absent/manquant ; exclusion des publications futures et révisions tardives ; passage été/hiver ; diagnostics insensibles aux résultats postérieurs à 2014 ; calendrier ; alignement et cohérence inférence ; absence de colonnes cibles dans X ; empreintes ; non-écrasement ; reproductibilité octet par octet sur données synthétiques ; construction sans connexion réseau.

**11. Vérification de l’absence de fuite du futur**

- Les features historiques sont identiques lorsqu’on retire le futur : contrôles synthétiques et contrôle supplémentaire sur 15 coupures réparties dans les données locales réelles.
- Les bornes des 5 293 cibles supervisées ont été revérifiées par positions dans le calendrier SPY brut : écart exactement égal à 21, départ à t+1 et fin à t+21.
- Aucun nom de colonne n’est commun aux features et aux targets ; aucun label de stress n’est lu.
- Les manques FRED ou un faux CSV final filtré ne changent pas X/y, vérification par test de contamination synthétique.
- Les diagnostics et le choix provisoire du seuil n’utilisent pas les résultats 2023+. Un test modifie les prix postérieurs à 2014 et vérifie que les diagnostics d’entraînement restent identiques.
- Aucun ajustement statistique de prétraitement ou de sélection n’a lieu.

Ces contrôles établissent la causalité des transformations et l’exclusion des sources non vérifiables. Ils ne certifient pas les heures historiques de réception ou l’absence de révisions dans les instantanés ETF, réserve conservée dans les métadonnées.

**12. Tables produites et reconstruction**

Les cinq sorties sont :

```text
data/processed/ml1/features_ml1.parquet
data/processed/ml1/targets_ml1.parquet
data/processed/ml1/features_ml1_inference.parquet
data/processed/ml1/target_diagnostics.csv
data/processed/ml1/dataset_metadata.json
```

Les targets exportent sept colonnes : prix ajusté à t, prix futur minimal, rendement futur minimal, première et dernière séances de cible, nombre de prix futurs valides et `y_crisis_21d` nullable. Les features d’inférence ont exactement le même schéma que les features supervisées et contiennent toutes les dates admissibles, pas seulement les 21 dernières.

Les métadonnées enregistrent version, dépendances, sources/colonnes/unités, empreintes du code et des entrées, seuil/horizon, convention temporelle, exclusions, calendrier, périodes, règles de séparation temporelle, contrôles et empreintes des quatre fichiers de données. Les fichiers Parquet ont été relus et leurs schémas, alignements et empreintes vérifiés.

Depuis la racine du dépôt, sur un dossier de sortie encore absent :

```bash
.venv/bin/python -B -m src.models.ml1.pipeline
```

Le dossier principal existe désormais. Pour reconstruire sans l’écraser, choisir un nouveau dossier :

```bash
.venv/bin/python -B -m src.models.ml1.pipeline --output-dir data/processed/ml1/rebuild_001
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Chaque nouvelle exécution doit désigner un dossier inexistant. Les exports sont préparés dans un dossier temporaire puis publiés ensemble ; aucun ancien dossier de sortie n’est écrasé. Aucun accès réseau n’est nécessaire au pipeline. Pour recréer l’environnement, conserver les versions indiquées en partie 4, notamment `exchange-calendars==4.13.3` ; le fonctionnement natif éventuel de XGBoost n’intervient pas dans cette préparation.

**13. Limites et problèmes non résolus**

VIX, macro et crédit restent exclus jusqu’à production de preuves de disponibilité historique. Leur réintégration exigera un contrat et des tests supplémentaires ; aucune collecte de nouvelles séries ou de vintages n’a été autorisée ni effectuée.

Le seuil de 7 % reste provisoire. Les features de prix reposent sur des instantanés ajustés, sous convention de réception avant 17 h, sans journal historique de réception. Le calendrier ARCX/XNYS reste une référence logicielle à versionner. La combinaison actuelle émet des avertissements de compatibilité future pandas/exchange-calendars ; il faudra revalider toute mise à jour. `app.py` reste absent de cette copie locale.

Les exports supervisés contiennent la période réservée au test final : un futur entraîneur devra appliquer les masques chronologiques et purger les fenêtres qui franchissent les bornes. Les labels voisins ne sont pas des épisodes indépendants. Aucune performance prédictive n’est évaluée ici.

Les fichiers partagés ML 2 restent sur leur ancien calendrier et n’ont pas été migrés. Le membre 2 devra être averti avant toute éventuelle harmonisation ; cette mission n’en réalise aucune.

**14. État Git final**

Branche inchangée : `feature/ml1-crisis-detection`, commit `8d9ff5f`. Les seuls nouveaux éléments visibles sont le code ML 1, ses tests et ce rapport ; `step1_remediation.md` demeure non suivi et inchangé. Les fichiers de données générés et `.venv` sont ignorés par les règles existantes.

Aucun fichier préexistant suivi modifié, aucun ajout à l’index, aucun commit, push ou merge. Aucun fichier du membre 1 ou d’un autre membre modifié. Une première exécution de vérification a été conservée hors dépôt dans un répertoire temporaire ; les sorties actuelles correspondent aux empreintes de la version finale du code.

**15. Verdict : PRÊT AVEC RÉSERVES**

Le pipeline indépendant de données ML 1 est construit, exécuté et testé. Les sorties sont exploitables pour la revue et pour une future phase utilisant les dix features historiques retenues, sous réserve de respecter les hypothèses de disponibilité des prix et les séparations temporelles.

L’intégration des variables macro/VIX n’est pas validée. Aucun entraînement de baseline, Logistic Regression, Random Forest ou XGBoost n’a commencé. Toute étape suivante reste soumise à l’accord de l’utilisateur.
