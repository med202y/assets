**Rapport de validation chronologique ML 1 — étape 3 — 14 novembre 2026**

Mission limitée au découpage temporel, à la purge et à l’embargo. Aucun modèle créé ou entraîné, aucune métrique prédictive calculée, aucune donnée financière téléchargée.

**1. État Git initial**

Branche : `feature/ml1-crisis-detection`. Commit : `8d9ff5ff66c161181f55a1da56d59cae02394375`.

Le dépôt contenait déjà huit fichiers non suivis : les rapports des étapes 1 et 2, les trois fichiers Python de `src/models/ml1/` et les trois fichiers Python de `tests/ml1/`. Tous ont été préservés. Les empreintes SHA-256 des fichiers préexistants, suivis ou non, et des cinq sorties de l’étape 2 ont été enregistrées puis vérifiées après intervention.

**2. Architecture de validation mise en place**

Nouveau module : `src/models/ml1/time_validation.py`.

Interface réutilisable, importable par de futurs scripts sans les créer :

- `load_development_data()` vérifie le contrat et les empreintes, charge X, les contrôles temporels séparés et uniquement les labels admissibles avant 2023.
- `prepare_development_data()` applique la séparation stricte du développement et du test final, avec validation des horizons.
- `WalkForwardConfig` définit les années, l’embargo et les tailles minimales.
- `build_walk_forward_folds()` renvoie les huit folds déterministes à fenêtre d’entraînement croissante.
- `assert_fold_is_safe()` vérifie explicitement chronologie, purge, disponibilité des labels et protection du test final, y compris avec Python lancé sous `-O`.
- `get_fold_xy()` renvoie seulement `X_train`, `y_train`, `X_validation`, `y_validation`, sous forme de copies.
- `apply_post_validation_embargo()` fournit le composant d’embargo pour les configurations dont l’entraînement comporte des observations postérieures à la validation.

Les `train_indices`, `validation_indices` et `purged_train_indices` sont des positions entières triées, non modifiables, relatives à `DevelopmentData.X` et `DevelopmentData.y`, et non aux 5 293 lignes du Parquet initial. Les folds fournissent dates, effectifs, dernière fin de cible d’entraînement, horodatages de disponibilité et informations d’embargo. Une empreinte de l’index empêche de réutiliser les positions sur un autre découpage.

Contrat inspecté : index `Date` de type `DatetimeIndex`, `datetime64[us]`, normalisé, sans fuseau, unique et croissant. Les dates complètes des features et des contrôles de targets sont identiques sur 5 293 observations.

X contient exactement les dix features historiques SPY/IEF de l’étape 2. y est exclusivement la Series `y_crisis_21d`, en `Int8` nullable dans les fichiers, complète et binaire sur le développement retenu. Les champs `target_start_session`, `target_end_session` et `future_session_count` restent dans `target_info`, hors de l’interface destinée aux modèles. Les prix futurs et `future_min_return_21` ne sont pas chargés comme features ni comme y.

Le contrat des métadonnées est contrôlé : baisse strictement supérieure à 7 %, horizon de 21 séances SPY, décision à 17 h America/New_York, frontière du test au 01/01/2023, absence d’utilisation du label de stress. Aucun changement de seuil ou de variables.

**3. Définition des folds walk-forward**

Entraînement initial : observations disponibles de 2005–2014 dont la cible se termine au plus tard le 31/12/2014. Les features existantes commencent le 21/09/2005 à cause de leur chauffe.

Validation : huit fenêtres annuelles, de 2015 à 2022. Pour chaque année, toutes les observations passées admissibles sont candidates à l’entraînement, puis purgées selon leur fin de cible. Les observations de validation des années précédentes peuvent rejoindre l’entraînement lorsqu’elles sont devenues admissibles.

Tailles minimales déclarées : 252 observations d’entraînement et 100 observations de validation. Ces gardes sont fixes, sans réglage sur les résultats des labels. Une période vide ou insuffisante provoque une erreur explicite ; aucun fold n’est ignoré silencieusement. De petits jeux synthétiques peuvent utiliser des minima explicitement réduits.

Le développement retenu comporte **4 329 observations**, du 21/09/2005 au 30/11/2022. Les 21 dates de décembre 2022 dont les cibles se terminent en 2023 sont exclues du développement. Aucun mélange aléatoire ni `shuffle=True`.

**4. Tableau réel des huit folds**

Toutes les fenêtres d’entraînement commencent le **21/09/2005**. Les effectifs suivants ont été obtenus par exécution sur les Parquet existants.

| Validation | Dernière date train | Dernière fin de cible train | N train | Purge | Début validation | Fin validation | N validation |
|---|---|---|---:|---:|---|---|---:|
| 2015 | 01/12/2014 | 31/12/2014 | 2 315 | 21 | 02/01/2015 | 31/12/2015 | 252 |
| 2016 | 01/12/2015 | 31/12/2015 | 2 567 | 21 | 04/01/2016 | 30/12/2016 | 252 |
| 2017 | 30/11/2016 | 30/12/2016 | 2 819 | 21 | 03/01/2017 | 29/12/2017 | 251 |
| 2018 | 29/11/2017 | 29/12/2017 | 3 070 | 21 | 02/01/2018 | 31/12/2018 | 251 |
| 2019 | 28/11/2018 | 31/12/2018 | 3 321 | 21 | 02/01/2019 | 31/12/2019 | 252 |
| 2020 | 29/11/2019 | 31/12/2019 | 3 573 | 21 | 02/01/2020 | 31/12/2020 | 253 |
| 2021 | 01/12/2020 | 31/12/2020 | 3 826 | 21 | 04/01/2021 | 31/12/2021 | 252 |
| 2022 | 01/12/2021 | 31/12/2021 | 4 078 | 21 | 03/01/2022 | 30/11/2022 | 230 |

L’embargo ne retire aucune observation d’entraînement dans ces huit folds strictement croissants. La réduction de la validation 2022 protège la frontière du test final ; elle n’est pas un second embargo.

**5. Explication précise de la purge**

Soit V la première séance de validation. Une observation d’entraînement t doit vérifier simultanément :

```text
t < V
target_end_session(t) < V
position_SPY(t) + 21 < position_SPY(V)
```

Une cible qui se termine exactement à V est exclue. La comparaison des dates réelles de fin de cible constitue le contrôle principal ; le contrôle indépendant par positions dans le calendrier SPY confirme l’horizon et la purge. Le système ne dépend pas d’un simple écart de 21 lignes du dataset, ni de 21 jours calendaires.

Sur les données actuelles, les candidats sont continus en séances : les 21 observations immédiatement antérieures à V sont purgées. La dernière observation conservée est à la position V−22 ; sa dernière séance cible est V−1. Il n’y a pas d’observation supprimée en plus au titre d’un deuxième gap.

Exemple de la frontière initiale : l’observation du 01/12/2014 a une cible terminée le 31/12/2014 et reste admissible ; celle du 02/12/2014 atteint le 02/01/2015, première validation, et est exclue.

La disponibilité du label est représentée par 17 h America/New_York à sa dernière séance cible, conformément au contrat des clôtures. Elle doit être strictement antérieure à la première décision de validation. Pour le premier fold : 31/12/2014 à 22 h UTC, avant le 02/01/2015 à 22 h UTC. Cette vérification réussit dans les huit folds.

**6. Embargo et application**

La purge élimine les cibles d’entraînement qui atteignent la validation. L’embargo élimine des dates d’entraînement situées **après** la fenêtre de validation, lorsqu’une configuration autorise de telles dates.

Pour une dernière séance de validation E et un embargo de 21 séances :

```text
Dates d’entraînement interdites par l’embargo :
les séances SPY E+1, ..., E+21, bornes incluses.
Première séance postérieure autorisée par ce seul composant : E+22.
```

La durée est paramétrable par un entier positif ou nul, 21 par défaut. Le comptage utilise le calendrier SPY complet, même si certaines lignes de features sont absentes. Le helper expose les indices conservés/exclus et les bornes temporelles ; il est testé sur un entraînement situé des deux côtés de la validation. Il ne constitue pas, à lui seul, une validation croisée complète autorisant le futur.

Dans le walk-forward implémenté, tout l’entraînement est antérieur à la validation. L’intersection avec les dates d’embargo est donc vide : `applied=False`, zéro observation retirée. La configuration de 21 séances est néanmoins explicitement enregistrée dans chaque fold. Elle ne crée pas un second gap avant validation.

Aucun embargo persistant ne bannit des observations d’une année passée dans tous les folds ultérieurs. Lorsqu’elles rejoignent un entraînement ultérieur, leur fin de cible et la disponibilité de leur label sont revérifiées face à la nouvelle validation. Les dernières observations de validation précédente encore indisponibles sont purgées ; les autres peuvent être réutilisées.

**7. Vérification des frontières chronologiques**

Les dates sont uniques et strictement croissantes. Les dates train/validation ne se recoupent jamais. Chaque ensemble d’entraînement est entièrement antérieur à sa validation et croît par inclusion d’un fold au suivant.

Le calendrier est reconstruit à partir de la colonne Date du fichier SPY brut dont l’empreinte est vérifiée, puis rapproché du calendrier de référence existant. Aucun prix ou Parquet n’est modifié. Les métadonnées de chaque horizon sont vérifiées : début à t+1, fin à t+21 et `future_session_count == 21`.

Les labels de validation 2015–2021 peuvent se terminer dans l’année suivante, toujours avant 2023. Ils servent à une future évaluation rétrospective, pas à une décision antérieure à leur disponibilité. Une future sélection adaptative en début d’année ne devra pas utiliser les résultats de fin d’année précédente encore inconnus. Les contrôles de disponibilité de l’entraînement ne dispensent pas de cette règle de sélection.

**8. Protection du test final 2023+**

La lecture des labels applique les filtres au point d’entrée Parquet :

```text
Date >= 2005-01-01
Date < 2023-01-01
target_end_session < 2023-01-01
columns = [y_crisis_21d]
```

Aucune lecture non filtrée de la colonne y n’est effectuée. Les contrôles temporels et les index peuvent être lus sur l’ensemble du dataset, mais aucune valeur de label 2023+ n’est retournée au module de développement, contrôlée, comptée ou utilisée pour régler la validation.

Les 943 dates supervisées du 03/01/2023 au 06/10/2026 sont repérées uniquement par leur index. Le test final reste fermé.

Une protection supplémentaire exclut les **21 dates du 01/12/2022 au 30/12/2022** dont les cibles utiliseraient des prix de 2023. La validation 2022 s’arrête donc au 30/11/2022. Les dernières cibles de développement sont connues au plus tard le 30/12/2022.

Des tests placent des objets qui déclenchent une erreur à la moindre inspection dans les labels finaux et les labels de décembre 2022 franchissant la frontière : la préparation et les folds fonctionnent sans les examiner. Modifier les labels 2023+ synthétiques ne change aucun fold. Les années 2023+ sont refusées comme périodes de validation dans la configuration du projet.

**9. Résultats pytest**

Les 58 tests préexistants ont été exécutés sans modifier leurs fichiers : **58 réussis**. La nouvelle suite comporte **40 tests**.

Exécution finale de l’ensemble ML 1 :

```text
98 passed, 80 warnings in 3.81s
```

Les 80 avertissements sont les dépréciations pandas/exchange-calendars déjà identifiées, dont 64 dans les tests du pipeline et 16 dans les tests temporels. Ils n’ont pas été masqués ; aucun test n’échoue.

Couverture nouvelle : huit folds, ordre et absence de dates communes, purge de 21 séances, cible touchant la validation, contrôle calendaire indépendant, frontière 2014/2015, embargo postérieur paramétrable, absence de second gap, réemploi de validations passées, isolation du test final, déterminisme, indices non modifiables, petit jeu synthétique, folds vides/insuffisants, calendrier non continu en jours civils, séparation X/y/métadonnées, copies indépendantes et lecture réelle des labels avec filtres obligatoires.

Les effectifs du tableau et les contrôles d’horodatage ont aussi été vérifiés sur les données locales réelles. Aucun score de classification ou autre résultat prédictif n’a été calculé.

**10. Fichiers créés ou modifiés**

Trois fichiers nouveaux uniquement :

- `src/models/ml1/time_validation.py`
- `tests/ml1/test_time_validation.py`
- `reports/ml1/step3_walk_forward_validation.md`

Aucun fichier préexistant modifié. En particulier : scripts et tests de l’étape 2, rapports des étapes 1 et 2, fichiers bruts, anciens traitements des autres membres et cinq sorties de données ML 1 inchangés. Aucune nouvelle dépendance installée. Aucun fichier `train_ml1.py` ou `evaluate_ml1.py` créé.

**11. Limites et risques restants**

Le contrat de données conserve ses réserves : seuil de 7 % provisoire, prix ajustés issus d’instantanés, réception supposée avant 17 h sans journal historique, macro/VIX exclus, calendrier logiciel de référence et avertissements de dépendances.

Les minima d’effectifs ne garantissent pas la présence des deux classes dans chaque période. Cette condition et le comportement des futurs estimateurs devront être traités lors de l’étape autorisée d’entraînement, sans redécoupage opportuniste du test final.

Les fenêtres de cible se chevauchent entre observations d’une même période ; les observations et folds ne constituent pas des échantillons statistiquement indépendants. La purge règle la frontière train/validation, pas cette dépendance interne.

Les futurs scalers, imputeurs, sélecteurs, calibrations et choix de seuil de décision devront être ajustés exclusivement sur les sous-ensembles autorisés, avec une disponibilité temporelle cohérente. Le test final ne devra être ouvert qu’après gel du protocole et des choix. Cette mission n’implémente aucun de ces ajustements.

Les positions renvoyées ciblent le dataset de développement filtré : ne pas les appliquer directement aux Parquet complets. Utiliser `get_fold_xy()` ou les objets X/y associés au même index ; une empreinte protège contre un changement de dates.

**12. Commandes de relance**

Depuis la racine du dépôt, avec `.venv` existant :

```bash
# Inspecter les huit folds et leurs effectifs, sans écrire de données
.venv/bin/python -B -m src.models.ml1.time_validation

# Configuration explicite de l’embargo ; aucun second gap en expanding-window
.venv/bin/python -B -m src.models.ml1.time_validation --embargo-sessions 21

# Tests de l’étape 2 conservés
.venv/bin/python -B -m pytest tests/ml1/test_core.py tests/ml1/test_pipeline.py -q -p no:cacheprovider

# Tests de validation temporelle
.venv/bin/python -B -m pytest tests/ml1/test_time_validation.py -q -p no:cacheprovider

# Ensemble ML 1
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

Exemple d’interface pour de futurs scripts, sans entraîner de modèle :

```python
from src.models.ml1.time_validation import (
    load_development_data, build_walk_forward_folds, get_fold_xy,
)

data = load_development_data()
folds = build_walk_forward_folds(data)
X_train, y_train, X_validation, y_validation = get_fold_xy(data, folds[0])
```

Ces commandes ne reconstruisent pas les Parquet et ne téléchargent aucune donnée. Le test d’intégration réel est exécuté lorsque les sorties locales existent ; il est explicitement ignoré dans une copie du dépôt qui ne distribue pas les fichiers de données ignorés par Git. Les autres tests utilisent des données synthétiques.

**13. État Git final**

Branche toujours `feature/ml1-crisis-detection`, même commit `8d9ff5f`. Les huit fichiers non suivis préexistants restent présents et inchangés ; les trois fichiers de cette étape s’y ajoutent. Les données et `.venv` restent ignorés par Git.

Aucun changement de fichier suivi, aucun ajout à l’index, aucun commit, push ou merge. Les comparaisons d’empreintes confirment la préservation des fichiers préexistants et de toutes les sorties de l’étape 2.

**14. Verdict : PRÊT AVEC RÉSERVES**

Le système de validation chronologique de l’étape 3 est implémenté, déterministe, exécuté sur les données réelles et couvert par les tests. La purge, l’embargo et la frontière du test final sont explicites. Les réserves de disponibilité et de qualité des données de l’étape 2 demeurent.

Aucune baseline, régression logistique, Random Forest ou XGBoost n’a été lancée. Aucune métrique de performance n’a été calculée. L’étape suivante attend la validation de l’utilisateur.
