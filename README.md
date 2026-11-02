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

Créer une clé gratuite sur https://fredaccount.stlouisfed.org/apikeys, puis la définir dans la variable d'environnement `FRED_API_KEY`. Ne jamais l'écrire dans le code ni la committer.


