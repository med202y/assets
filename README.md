# Asset Allocation ML

Plateforme d'aide à la décision pour l'allocation d'actifs entre quatre ETF : SPY (actions américaines), IEF (obligations d'État), LQD (obligations d'entreprise) et VNQ (immobilier coté).

## Principe

- **ML 1** estime la probabilité d'une forte baisse de SPY dans les 21 prochaines séances.
- **ML 2** prévoit la volatilité des quatre actifs.
- **L'optimiseur (CVXPY)** transforme ces estimations en pourcentages d'allocation selon le profil de risque.
- **Le backtest** vérifie la méthode sur le passé, avec frais de transaction et sans fuite d'information.
- **L'application Streamlit** présente le tout en quatre pages.

## Avertissement

Outil d'aide à la décision et projet d'apprentissage. Ce n'est pas un conseil en investissement. Aucune performance future n'est garantie.

## Structure

- `data/` : cache des données (non versionné)
- `src/data/` : téléchargement, nettoyage, variables, cibles
- `src/models/` : ML 1, ML 2, validation temporelle
- `src/optimizer/` : optimisation CVXPY, profils, contraintes
- `src/backtest/` : simulation, métriques, stratégies de référence
- `app/` : application Streamlit
- `notebooks/` : exploration
- `tests/` : tests automatiques
- `reports/` : rapport et figures

## Installation

```bash
git clone https://github.com/TON-PSEUDO/asset-allocation-ml.git
cd asset-allocation-ml
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Clé API FRED

Créer une clé gratuite sur https://fredaccount.stlouisfed.org/apikeys, puis la définir dans la variable d'environnement `FRED_API_KEY`. Ne jamais l'écrire dans le code ni la committer. Le frontend ne lit jamais cette clé et ne l'affiche jamais.

## Application web (Streamlit)

L'interface web est entièrement séparée du backend dans le dossier `app/` :

```
app/
├── Home.py                     # point d'entrée
├── pages/
│   ├── 1_Market_Overview.py    # prix normalisés, corrélations, drawdowns
│   ├── 2_Risk_Signals.py       # ML 1 (probabilité de krach) et ML 2 (volatilité)
│   ├── 3_Recommended_Allocation.py  # allocation sous contraintes (page centrale)
│   └── 4_Backtest_and_Method.py     # courbe de capital, métriques, méthode
├── services.py                 # SEUL pont vers src/
├── components.py               # graphiques Plotly et widgets réutilisables
└── config.py                   # libellés EN/FR, couleurs et valeurs par défaut
```

Pour lancer l'application :

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

L'application s'ouvre sur `http://localhost:8501`. Le profil de risque, la langue et les contraintes sont partagés entre les pages via `st.session_state`. Les appels coûteux sont mis en cache (`st.cache_data`) : aucun modèle n'est réentraîné et aucune optimisation n'est relancée à chaque interaction.

### Où brancher le vrai backend

Tous les appels à `src/` passent par `app/services.py`. Les pages ne contiennent aucune logique métier : elles affichent uniquement ce que `services` renvoie. Pour migrer l'interface (FastAPI/Flask + frontend custom), seul `app/` est à remplacer.

Fonctions de services utilisées par les pages :

| Fonction | Backend réel appelé |
| --- | --- |
| `get_prices`, `get_market_overview` | cache `src/data/financial_dataset/processed/final_dataset.csv` |
| `get_ml1_signal` | probabilités OOF/blind de `data/processed/ml1_training` et `ml1_inference_release`, métriques de `reports/ml1/evaluation` |
| `get_ml2_vol` | `src.models.ml2.predict_volatility.predict_volatility(date, use_oos=True)` |
| `get_allocation` | `src.optimizer.recommender.recommend(...)` avec covariance ML 2 et probabilité ML 1 précalculées |
| `get_backtest_results` | **stub** : `src/backtest` ne contient pas encore d'implémentation |

Quand un artefact ou une dépendance manque, `services.py` renvoie des valeurs de remplacement déterministes et l'interface l'indique explicitement (`demo data`). Le backtest affiché est un placeholder basé sur les vrais cours historiques, avec coûts de transaction de 10 bp par unité de rotation ; il ne s'agit pas encore d'un backtest walk-forward du strategy réelle.

### Avertissement

Outil d'aide à la décision et projet d'apprentissage. Ce n'est **pas** un conseil en investissement. Aucune performance future n'est garantie. Cet avertissement est affiché sur toutes les pages.

