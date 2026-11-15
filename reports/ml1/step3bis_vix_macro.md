**Rapport ML 1 — étape 3 bis : VIX et macro — 14 novembre 2026**

Résultat : le jeu `core` à dix features est conservé comme référence et reste le choix par défaut. Une variante `core_vix` à douze features est préparée, mais demeure **expérimentale, non certifiée point-in-time et activable uniquement explicitement**. Les cinq indicateurs macro restent exclus. Aucun entraînement ou calcul de performance prédictive.

**1. État Git initial**

Branche vérifiée en premier : `feature/ml1-crisis-detection`, commit `8d9ff5ff66c161181f55a1da56d59cae02394375`. Onze fichiers non suivis préexistaient : les trois rapports, quatre modules et quatre fichiers de tests des étapes précédentes. Aucun fichier suivi n’était modifié.

Les empreintes des fichiers préexistants et des cinq sorties de base ont été enregistrées puis comparées après les travaux : aucun changement. Aucun travail sur `main`, ajout à l’index, commit, push ou merge.

**2. Sources inspectées**

Sources locales : `src/data/financial_dataset/raw/VIX_raw.csv`, `BAA10Y_raw.csv`, `T10Y2Y_raw.csv`, `DGS3MO_raw.csv`, `DGS2_raw.csv`, `DGS10_raw.csv`, ainsi que le calendrier SPY et les sorties ML 1 existantes.

Les pages officielles demandées ont été consultées : [historique Cboe VIX](https://www.cboe.com/tradable-products/vix/vix-historical-data), [fiche de données FRED VIXCLS](https://fred.stlouisfed.org/data/VIXCLS), [périodes temps réel FRED](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [dates de vintage](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html). Les fiches FRED/ALFRED propres aux cinq séries et la page H.15 de la Fed ont également été examinées.

Il s’agit d’une consultation documentaire et des métadonnées affichées. Aucun nouvel historique CSV, aucune réponse API de vintages et aucune série distante n’ont été enregistrés ou intégrés au pipeline. Les sept sorties nouvelles proviennent exclusivement des fichiers locaux et des sorties de base.

**3. Résultats de l’audit VIX**

Le fichier contient `observation_date` et `VIXCLS`. La provenance déclarée par le téléchargement existant est FRED, source Cboe. La fiche officielle confirme une série quotidienne de clôture, non désaisonnalisée, en points d’indice. La page Cboe propose des valeurs quotidiennes de clôture ; elle ne fournit pas la preuve de réception historique de notre dispositif. [FRED VIXCLS](https://fred.stlouisfed.org/data/VIXCLS), [Cboe](https://www.cboe.com/tradable-products/vix/vix-historical-data).

Résultats locaux :

| Contrôle | Résultat |
|---|---|
| Période | 23/09/2004–04/11/2026 |
| Lignes | 5 770 |
| Valeurs valides | 5 598 |
| Valeurs manquantes | 172 |
| Valeurs non manquantes invalides | 0 |
| Minimum / maximum | 9,14 / 82,69 |
| Dates | Uniques, croissantes |
| Manques sur les 5 565 séances SPY | 0 |
| Valeurs valides hors séances SPY | 33 |

Les 33 dates supplémentaires sont hors du calendrier SPY, notamment certains jours fériés à partir de 2022. Leur origine précise et leur convention de publication ne sont pas certifiées ici. Elles sont ignorées pour le calcul par séances SPY, sans être supprimées du fichier brut. Aucun jour hors séance n’est transformé en séance SPY.

Le CSV est un instantané : aucune date de publication, heure de réception ou succession de révisions n’y figure. La présence de la [fiche ALFRED VIXCLS](https://alfred.stlouisfed.org/series?seid=VIXCLS) ouvre une piste de reconstruction, sans prouver la couverture et l’heure de disponibilité de chaque observation historique.

**4. Convention temporelle retenue et limites**

La décision reste fixée à 17 h America/New_York, après réception des clôtures, conformément au pipeline de base.

Convention A, clôture VIX datée t : **non retenue**. Les pages consultées établissent la nature de la série, mais pas son arrivée avant notre coupure pour chaque date historique. L’heure actuelle de mise à jour d’une page ne reconstitue pas cet historique.

Convention B : retard d’une séance SPY, retenu **uniquement pour l’expérimentation**. Hypothèse : la clôture attribuée à la séance SPY précédente aurait été accessible avant la décision du jour. Ce retard réduit le risque d’utiliser la publication du jour même ; il ne prouve ni cette disponibilité ni l’absence de révisions ultérieures.

Des délais supérieurs à une séance restent possibles. L’instantané actuel peut aussi différer d’une version réellement connue dans le passé. **Le décalage n’est jamais présenté comme une certification point-in-time.** Le niveau t non retardé et les décalages nuls sont refusés par le constructeur de cette variante non vérifiée.

Si des preuves `known_at` horodatées sont fournies à la fonction, une valeur connue après la décision, ou dépourvue de preuve, est masquée. Cette garde est testée sur données synthétiques ; aucune preuve de ce type n’existe dans la source locale utilisée. Le champ `known_at_evidence_supplied=false` et les horodatages NaT le signalent. Zéro rejet de publication dans l’exécution locale ne signifie donc pas que les publications ont été historiquement validées.

**5. Variables VIX intégrées**

Les deux variables sont ajoutées exclusivement à `core_vix` :

```text
v[s] = VIXCLS réindexé sur les séances SPY
L[t] = v[t−1 séance SPY]

VIX_level[t] = L[t]
VIX_change_5d[t] = L[t] / L[t−5 séances SPY] − 1
```

Le retard est appliqué après réindexation, pas sur les lignes du calendrier FRED. Par exemple, une éventuelle valeur fournisseur un lundi de fermeture SPY ne remplace pas la clôture de la séance SPY précédente dans le calcul du mardi.

La variation utilise les deux niveaux admissibles selon la convention aux extrémités des cinq séances. Un manque à l’une des deux extrémités laisse la variation manquante. Aucun backfill, forward-fill, interpolation, remplacement par zéro ou ajustement statistique.

Toutes les lignes du calendrier et des tables de base sont conservées dans les exports. Si un manque affecte les douze features, le chargeur de développement expose les dates non admissibles et sélectionne explicitement les lignes complètes pour un usage futur ; cela ne modifie ni le calendrier des cibles ni les fichiers exportés.

**6. Analyse des indicateurs macro**

Les cinq fichiers couvrent le 23/09/2004–04/11/2026 et contiennent chacun 5 770 dates d’observation uniques et croissantes. Les dates sont des dates de référence, pas des preuves de publication.

| Variable | Utilité financière candidate | Valides | Manquantes | Manques sur SPY |
|---|---|---:|---:|---:|
| BAA10Y | Écart de crédit Baa–Treasury 10 ans, contexte de risque de crédit | 5 528 | 242 | 44 |
| T10Y2Y | Pente 10 ans–2 ans, contexte de courbe des taux | 5 533 | 237 | 42 |
| DGS3MO | Niveau des taux courts | 5 533 | 237 | 42 |
| DGS2 | Niveau des taux à deux ans | 5 533 | 237 | 42 |
| DGS10 | Niveau des taux longs et contexte obligataire | 5 533 | 237 | 42 |

Les DGS sont des taux annuels en pourcentage ; BAA10Y et T10Y2Y sont des écarts de taux, exprimés économiquement en points de pourcentage. Les descriptions sont confirmées par les fiches [BAA10Y](https://fred.stlouisfed.org/series/BAA10Y), [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y), [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO), [DGS2](https://fred.stlouisfed.org/series/DGS2), [DGS10](https://fred.stlouisfed.org/series/DGS10).

Chaque série possède une fiche ALFRED : [BAA10Y](https://alfred.stlouisfed.org/series?seid=BAA10Y), [T10Y2Y](https://alfred.stlouisfed.org/series?seid=T10Y2Y), [DGS3MO](https://alfred.stlouisfed.org/series?seid=DGS3MO), [DGS2](https://alfred.stlouisfed.org/series?seid=DGS2), [DGS10](https://alfred.stlouisfed.org/series?seid=DGS10). Leur existence ne certifie pas une archive exhaustive exploitable sur toute la période du projet. Aucune extraction de vintages n’a été effectuée.

Les cinq séries présentent donc la même limite opérationnelle : révisions de valeurs non mesurables depuis le CSV à une valeur par date, historique d’arrivée non enregistré et couverture réelle des vintages restant à vérifier. Aucune n’est supposée non révisable.

FRED décrit le mode courant comme l’information disponible aujourd’hui, tandis qu’ALFRED permet d’interroger ce qui était connu à une date passée. Les dates de vintage identifient des publications nouvelles ou révisées, au format journalier ; elles ne fournissent pas à elles seules une heure de publication intrajournalière. [Périodes temps réel](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [vintages](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).

La Fed annonce H.15 à 16 h 15 les jours de publication. Cela ne garantit ni la présence de la valeur datée t dans cette publication, ni sa réception chez FRED avant notre décision. [H.15](https://www.federalreserve.gov/releases/h15/).

Conditions d’intégration future : collecte préalablement autorisée, vérification de la couverture des vintages par série, reconstruction des versions connues à la décision, preuve ou borne conservatrice de publication/réception, gestion des révisions et des manques. Si seule une date de publication est vérifiée, une admissibilité à une séance strictement ultérieure peut être envisagée sous convention documentée ; elle ne remplace pas l’audit de couverture et de provenance. Un simple `shift(1)` des CSV courants ne suffit pas. Aucune variable macro n’est intégrée.

**7. Variables admises, expérimentales et exclues**

| Groupe | Statut dans cette version | Utilisation autorisée par l’interface |
|---|---|---|
| Dix features SPY/IEF existantes | Référence `core`, avec réserves de disponibilité des prix déjà documentées | Choix par défaut |
| VIX_level retardé d’une séance SPY | Expérimental, non certifié PIT | Opt-in explicite |
| VIX_change_5d sur niveaux retardés | Expérimental, même réserve | Opt-in explicite |
| VIX de clôture du jour sans retard | Exclu | Refusé |
| BAA10Y | Exclu du jeu opérationnel | Archive et disponibilité non validées |
| T10Y2Y | Exclu du jeu opérationnel | Archive et disponibilité non validées |
| DGS3MO | Exclu du jeu opérationnel | Archive et disponibilité non validées |
| DGS2 | Exclu du jeu opérationnel | Archive et disponibilité non validées |
| DGS10 | Exclu du jeu opérationnel | Archive et disponibilité non validées |

**8. Architecture des variantes**

`market_indicators.py` fournit les profils explicites, la lecture locale VIX, la construction causale, les traces de provenance, les exports et une CLI. `indicator_validation.py` adapte les variantes au moteur de l’étape 3 sans modifier ce dernier.

- `core` : dix features, référence et défaut. Les commandes des étapes 2 et 3 restent inchangées.
- `core_vix` : douze features, activation et chargement exigeant `allow_experimental=True` ou le drapeau CLI correspondant.
- `core_vix_macro` : variante future bloquée, même avec opt-in expérimental, tant que les sources macro ne sont pas validées.

Les sept sorties distinctes sont dans :

```text
data/processed/ml1_core_vix_experimental/
  features_ml1.parquet
  targets_ml1.parquet
  features_ml1_inference.parquet
  target_diagnostics.csv
  vix_session_features.parquet
  vix_lineage.parquet
  dataset_metadata.json
```

Les cibles et diagnostics d’entraînement sont copiés octet par octet depuis le jeu de base, sans inspection des valeurs finales ni recalcul de labels. Les dix colonnes de base sont vérifiées identiques. Les sorties existantes ne sont jamais écrasées.

Les métadonnées enregistrent le statut expérimental, la version `0.1.0-experimental`, les profils et variables, les sources et empreintes, le retard, les règles temporelles, les manques et dates admissibles, les limites, les références documentaires et les empreintes du code et des sorties. `vix_lineage.parquet` conserve les dates source/point de comparaison et laisse les heures réelles de disponibilité inconnues à NaT ; aucune heure de publication n’est inventée.

Reconstruction dans un nouveau dossier, depuis la racine :

```bash
# Consultation du profil de référence ; aucune sortie nouvelle
.venv/bin/python -B -m src.models.ml1.market_indicators

# Le dossier principal existe déjà : choisir un dossier de reconstruction absent
.venv/bin/python -B -m src.models.ml1.market_indicators --variant core_vix --allow-experimental --output-dir data/processed/ml1_core_vix_experimental_rebuild_001
```

**9. Dates, observations et valeurs manquantes**

| Table / contrôle | Résultat réel |
|---|---|
| Calendrier SPY conservé | 5 565 séances |
| Features VIX complètes sur calendrier entier | 5 559 lignes |
| Manques VIX_level sur calendrier entier | 1, à l’amorçage |
| Manques VIX_change_5d sur calendrier entier | 6, à l’amorçage |
| Features supervisées enrichies | 5 293 × 12, du 21/09/2005 au 06/10/2026 |
| Features d’inférence enrichies | 5 314 × 12, du 21/09/2005 au 04/11/2026 |
| Manques VIX dans ces deux tables après chauffe | 0 |
| Développement enrichi admissible avant 2023 | 4 329 lignes, jusqu’au 30/11/2022 |
| Dates de développement exclues pour manque VIX réel | 0 |

Les 21 dernières lignes d’inférence restent disponibles sans cible future connue. Aucune séance SPY n’a été retirée à cause du VIX. Les tests vérifient séparément les cas synthétiques où des manques existent.

**10. Tests pytest et résultats**

Les 98 tests existants sont conservés et continuent à réussir. Les 32 nouveaux tests couvrent la construction et les interfaces d’indicateurs.

Résultat final :

```text
130 passed, 92 warnings in 6.66s
```

Les avertissements sont les dépréciations pandas/exchange-calendars déjà connues : 64 dans les tests du pipeline, 16 dans ceux de validation temporelle et 12 dans les nouveaux tests. Aucun test ignoré ou en échec dans cette exécution.

Tests nouveaux : défaut `core`, opt-in et refus de promotion silencieuse, macro bloquée, retards sur séances SPY, variation à cinq séances, jours fournisseur hors SPY, manques et prix invalides, absence de remplissage, publication postérieure à la décision, absence de preuve, fuseaux, troncature du futur, préservation des entrées, sept exports distincts, cibles identiques, huit folds sûrs, protection du test final, refus d’écrasement, masques d’admissibilité, absence de connexion réseau et reproductibilité octet par octet.

Commande :

```bash
.venv/bin/python -B -m pytest tests/ml1 -q -p no:cacheprovider
```

**11. Vérification du data leakage**

Les transformations ne consultent pas de valeurs VIX futures et sont stables sur trois coupures synthétiques ainsi que quinze coupures réparties dans l’historique local réel. Le décalage porte sur les séances SPY, pas les jours calendaires ou les dates supplémentaires de FRED.

Les observations connues après la décision sont exclues lorsque des preuves de publication sont fournies. En revanche, leur absence dans la source réelle interdit de certifier l’absence de fuite liée aux publications ou aux révisions : cette limite justifie le statut expérimental.

Le chargeur de variante réutilise le chargeur protégé de l’étape 3 pour y : seules les dates et fins de cible antérieures à 2023 sont admises. Les labels de décembre 2022 débordant en 2023 restent exclus. Aucun label 2023+ n’est utilisé pour sélectionner variables, retard, seuil ou paramètres.

Seules les douze features autorisées sont transmises par l’interface de variante, avec une seule Series y séparée. Les métadonnées futures et la provenance ne deviennent jamais des features. Aucun scaler, imputeur statistique, sélecteur ou modèle n’est ajusté.

**12. Fichiers créés ou modifiés**

Quatre nouveaux fichiers visibles dans Git :

- `src/models/ml1/market_indicators.py`
- `src/models/ml1/indicator_validation.py`
- `tests/ml1/test_market_indicators.py`
- `reports/ml1/step3bis_vix_macro.md`

Sept nouvelles sorties dans le dossier expérimental, ignoré par les règles Git existantes. Aucun fichier préexistant modifié, notamment les trois rapports précédents, le pipeline `core`, le moteur walk-forward, les 98 tests existants, les CSV bruts et les Parquet de base. Aucune dépendance supplémentaire et aucune modification des composants d’autres membres.

Branche finale inchangée, même commit. Les onze fichiers non suivis préexistants et les quatre nouveaux restent non suivis. Aucun commit, push, merge ou ajout à l’index.

**13. Compatibilité avec l’étape 3**

Le moteur strict à dix features est conservé intégralement. L’adaptateur construit une projection des dix colonnes de base sur les mêmes dates admissibles, réutilise les folds, puis expose les douze features uniquement par son interface dédiée.

Sur les données réelles, les huit folds sont exactement identiques à ceux de `core` : entraînements de 2 315 à 4 078 observations, validations de 252, 252, 251, 251, 252, 253, 252 et 230 observations. Chaque purge reste de 21 séances, sans second gap d’embargo. Le test final reste fermé.

Exemple de préparation explicite, sans entraînement :

```python
from src.models.ml1.indicator_validation import (
    load_feature_variant, build_feature_variant_folds,
    get_feature_variant_fold_xy,
)

data = load_feature_variant("core_vix", allow_experimental=True)
folds = build_feature_variant_folds(data)
X_train, y_train, X_validation, y_validation = get_feature_variant_fold_xy(
    data, folds[0]
)
```

Les indices désignent le développement admissible de la variante, pas les lignes originales du Parquet complet. Sur des sources comportant des manques, les effectifs peuvent changer ; la purge reste fondée sur le calendrier SPY complet.

**14. Problèmes restant à résoudre**

Disponibilité historique du VIX et délais FRED, révisions éventuelles, couverture exacte des vintages ALFRED pour chacune des six séries, provenance des dates VIX hors SPY, autorisation d’une éventuelle collecte d’archives et contrôle d’horaires de publication/réception. La présence de fiches ALFRED et un décalage temporel ne résolvent pas ces points.

Les réserves de `core` demeurent : instantanés ETF ajustés, réception supposée avant 17 h, calendrier logiciel, seuil de crise provisoire et avertissements de dépendances. Aucun résultat de modèle n’est disponible pour comparer les variantes, et aucune comparaison de performance n’est entreprise ici.

**15. Verdict : PRÊT AVEC RÉSERVES**

L’audit documentaire et local, l’architecture explicite, les exports expérimentaux et les tests sont terminés. `core` reste la référence par défaut. `core_vix_macro` reste bloquée.

**Une future baseline VIX reste expérimentale.** Elle ne pourra être présentée comme une baseline historiquement certifiée point-in-time ; un éventuel usage exploratoire devra être explicitement validé. La prochaine baseline de référence peut conserver les dix features `core`, sous leurs réserves existantes.

Aucune baseline ou autre modèle n’a été lancé et aucune métrique prédictive calculée. Toute étape suivante attend la validation de l’utilisateur.
