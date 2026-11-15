**Rapport complet de remédiation ML 1 — membre 3**

Périmètre : préparation de l’environnement, comparaison descriptive des seuils et spécification des corrections. Aucun entraînement, optimisation, backtest ou calcul de prédiction. Les recommandations de modification des données restent à appliquer par le membre 1, après validation.

**1. État Git avant et après**

Avant : branche `feature/ml1-crisis-detection`, arbre propre, commit `8d9ff5ff66c161181f55a1da56d59cae02394375`.

Après : même branche et même commit ; seul nouveau fichier non suivi attendu : `reports/ml1/step1_remediation.md`. `.venv/` est ignoré par `.gitignore`, ligne 2. Aucun fichier préexistant suivi n’a été modifié, vérification par empreintes SHA-256 et par Git. Aucun commit, ajout à l’index, push ou modification des références Git pendant cette mission.

**2. Environnement Python créé et versions**

Interpréteur trouvé : `/opt/homebrew/bin/python3.12`, version **3.12.13**, architecture ARM64. Environnement créé à la racine : `.venv/`. `include-system-site-packages = false`, paquets utilisateur désactivés ; les imports vérifiés proviennent tous de `.venv/lib/python3.12/site-packages/`.

Les dépendances de `requirements.txt` ont été installées en premier, puis `scikit-learn`, `xgboost`, `shap`, `matplotlib` et `joblib`. `requirements.txt` et les installations Python globales sont inchangés. Le réseau a nécessité une autorisation technique.

Pour utiliser cet environnement : `source .venv/bin/activate`, ou appeler directement `.venv/bin/python`. Dans VS Code, sélectionner `.venv/bin/python` ; ce choix dans l’interface n’a pas été effectué pendant cette mission.

**3. Résultats des tests d’importation**

| Distribution | Version installée | Module importé | Résultat final |
|---|---|---|---|
| pandas | 3.1.1 | pandas | OK |
| numpy | 2.5.4 | numpy | OK |
| yfinance | 1.8.0 | yfinance | OK |
| fredapi | 0.5.2 | fredapi | OK |
| pyarrow | 26.0.1 | pyarrow | OK |
| pytest | 9.2.0 | pytest | OK |
| python-dotenv | 1.2.4 | dotenv | OK |
| scikit-learn | 1.9.1 | sklearn | OK |
| xgboost | 3.4.1 | xgboost | OK |
| shap | 0.54.1 | shap | OK |
| matplotlib | 3.11.2 | matplotlib | OK |
| joblib | 1.6.0 | joblib | OK |

`pip check` : **No broken requirements found**. Les 12 imports ont été vérifiés, puis réunis dans un même processus après correction du problème natif décrit ci-dessous. Les caches graphiques utilisés par les contrôles ont été dirigés vers un répertoire temporaire ; aucune configuration graphique globale n’a été changée.

XGBoost échouait initialement à charger `@rpath/libomp.dylib`, absente des emplacements Homebrew attendus. Le runtime fourni par la roue scikit-learn existe déjà dans `.venv`. Le chemin relatif suivant a été ajouté à la bibliothèque XGBoost locale, puis sa signature locale a été renouvelée :

```bash
install_name_tool -add_rpath '@loader_path/../../sklearn/.dylibs' .venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
codesign --force --sign - .venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
```

L’import XGBoost seul fonctionne, ainsi que les imports combinés. `threadpoolctl` constate un seul runtime OpenMP chargé, celui de scikit-learn. La signature locale a été vérifiée. Aucune installation Homebrew ou modification de Python global n’a été réalisée. Le mécanisme `@rpath` employé par XGBoost est documenté dans son [code de liaison macOS](https://github.com/dmlc/xgboost/blob/master/cmake/FindOpenMPMacOS.cmake).

Cette adaptation appartient uniquement à `.venv`, dépend de la présence du runtime scikit-learn et doit être revalidée après réinstallation ou mise à jour. Un simple gel des versions ne reproduit pas cette modification native ; les deux commandes ci-dessus font partie de la procédure locale. Empreintes SHA-256 :

- XGBoost avant adaptation : `5aef338ede7b84ae7d3405a306614de1d326ccaad033a2f14242aedd421359bc`.
- XGBoost après adaptation : `652d0e2c5c6891c1887b913450f93274d50baaca81e71a9ff96ab2200bbac8c1`.
- Runtime OpenMP réutilisé : `7fe04d06d824817d39e4c99ec44e837fd6b5f0f75f2cc3e77a384c298f3f1709`.

Ces tests valident les imports et la cohérence des dépendances, sans certifier l’exécution complète du notebook existant ou d’un futur entraînement.

**4. Définition recommandée de la cible**

Soit `s_i` la i-ème séance du calendrier SPY complet et `P_i` sa **clôture ajustée**, strictement positive et valide. Pour `x = 0,07` ou `0,10` :

```text
future_min_21(i) = min(P[i+1], ..., P[i+21])
future_min_return_21(i) = future_min_21(i) / P[i] - 1
y_crisis_21d_x(i) = 1 si future_min_21(i) < (1 - x) * P[i]
                   0 sinon, si les 21 prix futurs sont tous valides
                   NA si l’horizon est incomplet ou un prix requis invalide
```

Le comparateur est strict : une baisse exactement égale à 7 % ou 10 % ne déclenche pas la classe positive correspondante. La fenêtre exclut t et inclut t+21. Un rebond avant t+21 n’annule pas un franchissement antérieur. Les données disponibles permettent d’étudier les clôtures quotidiennes ajustées ; cette définition ne détecte pas un passage intrajournalier sous le seuil. Ne pas utiliser les `Low` non ajustés comme substitut.

Pendant l’étude des seuils, les noms conceptuels `y_crisis_21d_07` et `y_crisis_21d_10` évitent toute ambiguïté. Le nom final `y_crisis_21d` devra être accompagné des métadonnées du seuil retenu, de l’horizon, du calendrier et de la version des prix. Aucune nouvelle table cible n’a été exportée.

**`y_stress_regime` ne doit pas être utilisée comme cible de crise.** Elle mesure une volatilité future annualisée supérieure à 20 %. Elle peut être conservée séparément comme diagnostic de volatilité. La colonne actuelle `y_SPY_maxdd_21d_fwd` correspond au minimum futur rapporté au prix à t ; ce n’est pas un maximum drawdown crête–creux. Sa formule est proche de la quantité continue utile ci-dessus, mais son calendrier doit être corrigé et sa sémantique documentée.

La décision à t intervient après réception des données de clôture. Proposition opérationnelle à valider : coupure à **17 h America/New_York**, y compris lors des clôtures anticipées, sous réserve de réception effective du prix. Toute donnée reçue après cette coupure est exclue de cette décision. Cette convention ne suppose aucune exécution rétroactive au cours de clôture de t.

**5. Comparaison descriptive des seuils sur 2005–2014**

Source : `src/data/financial_dataset/raw/SPY_raw.csv`, colonne `Adj Close`, avant toute jointure macroéconomique. Le lecteur s’arrête à la fin de 2014 ; aucune valeur de 2015–2022 ni de la période de test final à partir de 2023 n’entre dans la comparaison.

Le fichier fournit 2 517 séances du 03/01/2005 au 31/12/2014. Les 21 dernières dates de décision sont exclues pour ne jamais emprunter de prix à 2015. Échantillon commun : **2 496 dates t, du 03/01/2005 au 01/12/2014** ; fins de fenêtre du 02/02/2005 au 31/12/2014. Les horizons contiennent exactement les 21 observations SPY suivantes disponibles dans cette source.

| Seuil strict de baisse | Positifs | Négatifs | Fréquence positive |
|---|---:|---:|---:|
| Plus de 7 % | 257 | 2 239 | 10,30 % |
| Plus de 10 % | 124 | 2 372 | 4,97 % |

| Année de t | Dates admissibles | Positifs à 7 % | Positifs à 10 % |
|---|---:|---:|---:|
| 2005 | 252 | 0 | 0 |
| 2006 | 251 | 0 | 0 |
| 2007 | 251 | 28 | 7 |
| 2008 | 253 | 108 | 61 |
| 2009 | 252 | 36 | 27 |
| 2010 | 252 | 36 | 8 |
| 2011 | 252 | 40 | 21 |
| 2012 | 250 | 7 | 0 |
| 2013 | 252 | 0 | 0 |
| 2014 | 231 | 2 | 0 |
| **Total** | **2 496** | **257** | **124** |

Le seuil de 7 % fournit 133 dates positives supplémentaires et environ 2,07 fois plus de positifs. Le seuil de 10 % caractérise des baisses plus sévères, avec une classe positive plus rare. Les labels se chevauchent temporellement : ces nombres ne comptent pas des crises indépendantes et ne constituent pas une mesure de performance prédictive.

**Aucun seuil définitif n’est sélectionné par cette seule fréquence.** Garder les deux candidats pour validation du besoin métier, puis une éventuelle validation temporelle avant 2023, après autorisation. Le test final ne doit intervenir ni dans ce choix ni dans les imputations ou réglages. Les fréquences sont calculées sur le calendrier SPY brut ; elles pourront différer sur le sous-échantillon effectivement utilisable après la chauffe des variables et la validation de disponibilité macro.

Vérifications réalisées : parcours explicite en arithmétique décimale, puis recalcul indépendant vectorisé NumPy, donnant les mêmes effectifs ; inclusion des positifs à 10 % dans ceux à 7 %. Neuf cas synthétiques passent : égalité à chacun des deux seuils exclue, dépassement strict de chacun inclus, t+21 inclus, horizon incomplet non étiqueté, prix futur manquant non étiqueté, prix constant donnant la classe 0 et rebond terminal après franchissement donnant la classe 1. Ce sont des contrôles de la définition proposée, pas des tests d’un pipeline corrigé.

Empreinte de l’échantillon d’entrée normalisé `date,Adj Close` 2005–2014 : `6a79f71078411d7da319fcdb79249f2d22a8e0d1075ff28c4fab136ab7b8412f`.

**6. Problème de calendrier et solution proposée**

L’audit initial a constaté 44 séances SPY supprimées après le début de VNQ, ainsi que quatre dates antérieures à sa disponibilité. La jointure par date suivie de `dropna()` retire des séances de marché lorsqu’une série macro manque. Sur les tables ML 1 auditées, 855 fenêtres couvrent ainsi 22 ou 23 séances SPY au lieu de 21. Les fenêtres de rendements, volatilités et corrélations sont également affectées. Le remplissage ultérieur du notebook ne restaure pas les lignes déjà supprimées.

Le calendrier maître doit être celui des séances SPY, rapproché d’un calendrier historique NYSE Arca et des fermetures exceptionnelles. Une clôture anticipée compte pour une séance. Un jour de fermeture de la bourse n’ajoute pas de ligne ni de rendement nul. Une séance attendue sans prix SPY est un défaut de couverture à investiguer ; elle ne doit pas être assimilée automatiquement à un jour férié. Les calendriers et horaires d’[NYSE, dont NYSE Arca](https://www.nyse.com/trade/hours-calendars) distinguent fermetures et clôtures anticipées.

Construire les rendements et cibles à partir de ce calendrier avant filtrage des variables. Joindre les informations macro par disponibilité, en préservant chaque séance SPY. Un trou FRED laisse une donnée manquante ou une ancienne observation admissible, sans supprimer le prix SPY. Ne jamais imputer les prix utilisés pour étiqueter une crise.

Politique macro proposée, à faire valider et implémenter par le membre 1 :

1. Conserver pour chaque observation : série, date d’observation, valeur, vintage, date/heure de publication, date/heure de disponibilité chez le fournisseur et source de cette preuve. Définir `known_at` comme le moment où cette version était effectivement accessible au dispositif de décision.
2. Pour une décision à t, sélectionner uniquement des versions telles que `known_at <= decision_time(t)`, avec une jointure temporelle vers le passé. Une correction publiée après t ne doit pas remplacer rétroactivement l’ancienne valeur. FRED distingue les observations connues aujourd’hui des données connues dans le passé ; les vintages identifient les publications nouvelles ou révisées. [Périodes temps réel FRED](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [dates de vintage](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).
3. Une simple date d’observation n’est pas une preuve de publication. Si seule la date de publication/vintage est vérifiable, rendre la valeur admissible à la décision de la première séance SPY strictement postérieure à cette date. Si la publication historique n’est pas vérifiable, conserver NA ou retirer explicitement la variable du contrat, sans prétendre qu’un décalage arbitraire d’une séance élimine la fuite.
4. Appliquer cette vérification séparément à VIXCLS, BAA10Y, DGS3MO, DGS2, DGS10 et T10Y2Y. La Fed annonce H.15 à 16 h 15 les jours de publication ; cela ne prouve ni la disponibilité de la valeur datée t à t, ni son arrivée immédiate chez FRED. Les délais exacts doivent être établis historiquement par série. [Publication H.15](https://www.federalreserve.gov/releases/h15/).
5. Après sélection des seules versions admissibles, autoriser un report vers le futur limité : proposition initiale non optimisée de **3 séances SPY et 7 jours calendaires au maximum depuis la date d’observation**, les deux bornes devant être respectées. Au-delà, laisser NA. Ce plafond est une proposition de gouvernance à valider sur les données d’entraînement ; il ne constitue pas une règle déjà implémentée. Interdire backfill, interpolation utilisant une borne future et remplacement par zéro.
6. Ajouter des indicateurs de manque et de report, l’âge en séances et en jours, la date de l’observation réutilisée et `known_at`. Produire un bilan par série des valeurs disponibles, reportées, périmées et non vérifiables. Recalculer les transformations VIX/macro uniquement à partir de valeurs admissibles ; documenter l’effet des répétitions sur leurs variations.
7. Conserver le calendrier complet même lorsqu’une variable reste NA. Définir séparément un masque d’admissibilité des variables. Un éventuel imputeur statistique ultérieur devra être ajusté uniquement sur l’entraînement, et jamais sur le test final. Aucune imputation statistique n’est effectuée ici.

Les CSV actuels ne portent pas les vintages ou horodatages historiques nécessaires. La politique est spécifiée, mais la disponibilité réelle des variables n’est donc pas encore sécurisée.

Contrôles à imposer pour chaque cible : index unique et trié, prix ajustés valides ; exactement 21 séances attendues entre t exclu et `target_end_session` inclus ; aucune séance attendue absente ; aucune date t+22 dans le minimum ; dernière fenêtre incomplète laissée NA ; étiquette indépendante des manques FRED. Aux frontières temporelles, purger les dates d’entraînement dont la fin de cible empiète sur la période suivante. L’échantillon descriptif ci-dessus applique déjà la borne de fin 2014, sans certifier à lui seul l’exhaustivité du calendrier historique de la source.

**7. Corrections à demander au membre 1**

Les numéros de cellules ci-dessous sont les index JSON du notebook, à partir de zéro. Aucun des fichiers cités n’a été modifié.

| Anomalie / emplacement | Modification recommandée | Validation attendue |
|---|---|---|
| `src/data/download_datasets.py`, lignes 241–250 : jointure et `dropna()` global | Préserver le calendrier SPY ; séparer prix, données macro disponibles et masque de variables ; journaliser les défauts de couverture | Une lacune FRED ne change ni l’index SPY ni ses cibles ; fermeture réelle distincte d’un prix absent |
| Notebook `data_engineering_features.ipynb`, cellule 5 : suppression dès qu’un ETF manque et `ffill(limit=5)` | Un manque d’un autre ETF ne doit pas effacer une séance SPY ; reporter uniquement des valeurs macro connues selon les bornes explicites | Tests de jours fériés bancaires, publication tardive, report à la limite et dépassement ; absence de remplissage de prix |
| Cellule 9, `build_features`, `select_ml1` et `USE_BAA10Y` | Calculer sur calendrier préservé ; utiliser `pct_change(fill_method=None)` ; vérifier disponibilité de chaque variable ; figer la présence de BAA10Y sans examiner le test final | Troncature du futur, absence d’infini, noms/ordre/types stables ; cohérence des rendements autour des séances rétablies |
| Cellule 11, `build_targets`, `fwd_maxdd`, `fwd_ret`, `fwd_vol` | Ajouter la cible de baisse strictement définie ; conserver le stress de volatilité séparé ; calculer toutes les fenêtres sur les séances complètes ; laisser les horizons incomplets NA | Tests des deux seuils, égalité, t+1, t+21, exclusion t+22, rebond, manque de prix et 21 séances exactes |
| Cellules 13 et 24, `align` et exports | Séparer les variables sans contrainte de cible, le jeu supervisé et les métadonnées d’horizon/disponibilité ; proposer des sorties versionnées sans écraser les livrables existants | Alignement X/y supervisé, dates de fin d’horizon et conservation des lignes récentes sans étiquette |
| Cellules 16 et 20, contrôles qualité | Ajouter tests de disponibilité historique, limites d’imputation, calendrier externe et frontières temporelles, au-delà des tests numériques existants | Toute donnée connue après la décision est rejetée ; aucun horizon d’entraînement ne franchit sa borne |
| Cellule 1 et `src/data/dataset_info.txt` | Documenter interpréteur, versions, contrat cible, calendrier, unités et disponibilité ; sortir l’installation active du notebook | Réexécution contrôlée et portable par le membre 1, sans installation implicite ni modification silencieuse de données |

Les formules ML 2 attendues restent `return_21 = P[i+21]/P[i] - 1` et `vol_21 = std(r[i+1], ..., r[i+21], ddof=1) * sqrt(252)`, avec rendements calculés entre séances consécutives et prix valides. Le rétablissement du calendrier changera potentiellement leurs valeurs, les volatilités historiques, corrélations, ratios, dates et nombres de lignes. Le membre 2 doit être informé du contrat et vérifier les nouvelles données avant toute migration. Aucun Parquet ML 2 n’a été régénéré ou écrasé.

Besoin distinct pour les prédictions récentes : produire des **features ML 1 sans cible future**, avec le même schéma et les mêmes règles de disponibilité que pour l’entraînement. Les tables supervisées actuelles s’arrêtent au 05/10/2026 alors que les prix locaux audités atteignent le 04/11/2026. Une date récente admissible doit rester disponible même si ses 21 prix futurs n’existent pas. Si ses données requises sont trop anciennes ou non vérifiables, signaler l’indisponibilité ; ne pas fabriquer une prédiction. Un test devra garantir la présence de la dernière date de prix lorsque toutes les conditions d’admissibilité sont réunies. Aucune prédiction n’est calculée ici.

**8. Fichiers créés ou modifiés**

- Créé : `.venv/`, environnement ignoré par Git, avec ses paquets et l’adaptation native XGBoost décrite plus haut.
- Créé : `reports/ml1/step1_remediation.md`, présent rapport, seul nouveau fichier visible dans l’état Git normal.
- Créés hors dépôt : scripts descriptifs, résultats et sauvegarde de la bibliothèque XGBoost initiale dans `/private/tmp/ml1-remediation-teti_y34/`. Ces éléments sont temporaires ; les résultats et la procédure utiles sont consignés ici.
- Aucun fichier préexistant suivi modifié ; `requirements.txt`, notebook, scripts, CSV et Parquet partagés inchangés. Aucune nouvelle table Parquet, aucun commit et aucun push.

**9. Problèmes non résolus**

Le seuil métier reste à approuver entre 7 % et 10 %. Le pipeline partagé conserve ses anomalies de calendrier et sa cible de volatilité. Les historiques de disponibilité/vintages macro restent à établir ; les plafonds d’imputation et l’heure de décision sont des propositions. Le calendrier historique complet de SPY reste à rapprocher d’une référence boursière indépendante. Les variables récentes pour inférence ne sont pas encore livrées séparément. La sélection de l’interpréteur dans VS Code reste à faire.

L’environnement est fonctionnel pour les imports, mais sa pérennité nécessite de conserver les versions et de revalider l’adaptation OpenMP après toute mise à jour. Aucun entraînement n’a servi à vérifier une compatibilité opérationnelle plus large. La clé FRED n’a pas été créée ; une collecte ultérieure de vintages nécessitera les accès appropriés. Ces limites n’empêchent pas la lecture des données locales.

**10. Verdict pour l’étape 2 : BLOQUÉ**

La préparation de l’environnement et le document de corrections sont terminés. L’étape 2 reste bloquée par la définition finale du seuil et la validation des corrections de données, du calendrier et de la disponibilité macro. Aucune étape suivante ne sera engagée sans accord de l’utilisateur.

**Inventaire exact des distributions installées, dépendances transitives comprises**

Cet inventaire constate l’état de `.venv` ; il ne modifie pas `requirements.txt`. L’adaptation native XGBoost décrite en partie 3 doit être ajoutée à toute procédure de reproduction sur ce Mac.

```text
Pygments==2.21.0
beautifulsoup4==4.15.0
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
contourpy==1.4.0
curl_cffi==0.17.0
cycler==0.12.1
fonttools==4.68.0
fredapi==0.5.2
idna==3.20
iniconfig==2.3.1
joblib==1.6.0
kiwisolver==1.5.1
lxml==6.1.3
matplotlib==3.11.2
narwhals==2.28.0
numpy==2.5.4
packaging==26.3
pandas==3.1.1
peewee==4.5.3
pillow==13.0.0
pip==26.0
platformdirs==4.14.5
pluggy==1.7.0
protobuf==7.36.2
pyarrow==26.0.1
pycparser==3.12
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==4.0.0
python-dotenv==1.2.4
pytz==2026.6
requests==2.34.2
scikit-learn==1.9.1
scipy==1.18.1
shap==0.54.1
slicer==0.0.8
soupsieve==4.1
threadpoolctl==3.8.0
tqdm==4.70.1
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
websockets==17.4
xgboost==3.4.1
yfinance==1.8.0
```
