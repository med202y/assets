"""UI configuration: labels, defaults and display constants.

This module contains NO business logic and NO calls to ``src``.  All user
facing strings live here so the interface can be translated (EN/FR) by adding
a dictionary, without touching pages, components or services.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #

DEFAULT_LANGUAGE = "EN"
SUPPORTED_LANGUAGES = ("EN", "FR")

LABELS: dict[str, dict[str, str]] = {
    "EN": {
        # Global
        "app_title": "Asset Allocation ML",
        "app_subtitle": "Model-based portfolio allocation across four ETFs",
        "disclaimer_title": "Important information",
        "disclaimer": (
            "This is a decision-support and learning project, **not investment "
            "advice**. No future performance is guaranteed. Model outputs are "
            "estimates with uncertainty."
        ),
        "language": "Language",
        "risk_profile": "Risk profile",
        "risk_profile_help": (
            "Sets the optimizer's risk aversion and default constraints."
        ),
        "data_as_of": "Data as of",
        "data_as_of_unknown": "not available",
        "loading": "Loading…",
        "no_data": "No data available for this selection.",
        "no_data_hint": (
            "Try widening the date range or check that the data cache has been "
            "built (see README)."
        ),
        "invalid_range": "The start date must be before the end date.",
        "error_title": "Something went wrong while loading this view.",
        "error_hint": "The backend did not return a usable result. No model was retrained.",
        "computed_at": "Computed at",
        "source": "Source",

        # Risk profiles
        "profile_conservative": "Conservative",
        "profile_balanced": "Balanced",
        "profile_aggressive": "Aggressive",
        "profile_conservative_desc": "Favours bonds and low volatility.",
        "profile_balanced_desc": "Balances growth and stability.",
        "profile_aggressive_desc": "Tolerates more SPY / VNQ risk.",

        # Assets
        "asset_SPY": "SPY — US equities",
        "asset_IEF": "IEF — US Treasuries",
        "asset_LQD": "LQD — Corporate bonds",
        "asset_VNQ": "VNQ — Listed real estate",
        "asset_bonds": "Bonds (IEF + LQD)",

        # Shared widgets
        "time_window": "Time window",
        "start_date": "Start",
        "end_date": "End",
        "full_range": "Full history",

        # Navigation / home
        "home_welcome": "Welcome",
        "home_intro": (
            "A decision-support tool that combines risk signals, constrained "
            "optimization and an honest backtest to propose an allocation across "
            "four ETFs."
        ),
        "home_journey": "Suggested journey",
        "home_step1": "Pick a risk profile in the sidebar.",
        "home_step2": "Check the market on **Market overview**.",
        "home_step3": "Read the signals on **Risk signals**.",
        "home_step4": "Get a suggestion on **Recommended allocation**.",
        "home_step5": "Verify it historically on **Backtest and method**.",
        "home_pages": "Pages",
        "home_p1": "Market overview — prices, correlations, drawdowns.",
        "home_p2": "Risk signals — ML 1 crash probability, ML 2 volatility.",
        "home_p3": "Recommended allocation — constrained optimizer output.",
        "home_p4": "Backtest and method — historical check and limitations.",

        # Page 1
        "p1_title": "Market overview",
        "p1_caption": (
            "Normalised prices (base 100) put the four ETFs on the same footing. "
            "They do not always move together, which is why diversification can "
            "smooth a portfolio's path — but it never removes risk."
        ),
        "p1_normalized": "Normalised prices (base 100)",
        "p1_correlation": "Correlation of daily returns",
        "p1_correlation_note": (
            "Lower or negative correlations are the ones that help diversification."
        ),
        "p1_drawdown": "Drawdown from previous peak",
        "p1_drawdown_note": (
            "Drawdowns show the pain of holding an asset through a falling market."
        ),

        # Page 2
        "p2_title": "Risk signals",
        "p2_intro": (
            "Two models feed the optimizer: a crash-probability signal for SPY and "
            "a volatility forecast for each ETF. Both are estimates, not facts."
        ),
        "ml1_title": "ML 1 — Probability of a large SPY drop within 21 days",
        "ml1_current": "Current probability",
        "ml1_history": "Historical signal with large SPY drawdowns marked",
        "ml1_marked": "Observed large drawdown",
        "ml1_weak": (
            "**Model quality warning:** out-of-sample discrimination is weak. "
            "Treat this signal as indicative only."
        ),
        "ml2_title": "ML 2 — Forecast volatility by asset",
        "ml2_bar": "Current forecast vs long-run average (annualised)",
        "ml2_forecast": "Forecast",
        "ml2_longrun": "Long-run average",
        "drivers_title": "What drives this",
        "drivers_ml1": "Top features — ML 1",
        "drivers_ml2": "Top features — ML 2",
        "quality_title": "Model quality (out-of-sample, temporal validation)",
        "quality_metric": "Metric",
        "quality_value": "Value",
        "quality_auc": "AUC (ML 1)",
        "quality_brier": "Brier score / calibration (ML 1)",
        "quality_mae": "Volatility forecast MAE (ML 2)",
        "quality_note": (
            "Temporal (walk-forward) validation trains on the past and tests on "
            "later data, so no future information leaks into these scores."
        ),

        # Page 3
        "p3_title": "Recommended allocation",
        "p3_intro": (
            "A model-based suggestion from the constrained optimizer. It is an "
            "estimate, not a prediction or investment advice."
        ),
        "constraints_title": "Constraints",
        "max_weight": "Maximum weight per asset (%)",
        "min_bonds": "Minimum bond share, IEF + LQD (%)",
        "compute": "Compute allocation",
        "computing": "Solving…",
        "allocation_title": "Suggested weights",
        "allocation_table_title": "Exact weights",
        "weight": "Weight (%)",
        "comparison_title": "Comparison",
        "current": "Current suggestion",
        "previous": "Previous month",
        "benchmark_weights": "Selected benchmark",
        "explanation_title": "Why this allocation?",
        "infeasible_title": "These constraints cannot be satisfied together.",
        "infeasible_hint": (
            "Try raising the maximum weight per asset or lowering the minimum "
            "bond share, then compute again."
        ),
        "uncertainty": (
            "Small changes in the signals can change these weights. Treat the "
            "result as one scenario, not a forecast."
        ),
        "not_computed": "Choose your constraints and press **Compute allocation**.",
        "profile_effect": (
            "Risk profile: **{profile}** — {description} The optimizer uses this "
            "profile's risk aversion and default limits."
        ),
        "explain_spy_reduced": (
            "**SPY reduced to {weight:.0f}%**: the crash signal is {p:.0%} and "
            "forecast volatility is {vol:.0%}."
        ),
        "explain_spy_kept": (
            "**SPY kept at {weight:.0f}%**: the crash signal is low ({p:.0%})."
        ),
        "explain_bonds": (
            "**Bonds (IEF + LQD): {bonds:.0f}%**, respecting your {floor:.0f}% minimum."
        ),
        "explain_profile": (
            "The **{profile}** profile sets the risk limits; the optimizer worked "
            "within them."
        ),
        "explain_relaxed": (
            "A constraint had to be relaxed by the optimizer: {reason}."
        ),
        "placeholder_notice": (
            "Demo data: a model artifact is unavailable in this environment. "
            "The bridge falls back to deterministic placeholder values."
        ),
        "placeholder_backtest_notice": (
            "Placeholder backtest: `src/backtest` is not implemented yet. The curves "
            "use real historical ETF prices with a fixed target and the stated costs."
        ),

        # Page 4
        "p4_title": "Backtest and method",
        "p4_intro": (
            "Historical simulation of the strategy and benchmarks, net of "
            "assumed transaction costs."
        ),
        "benchmarks": "Benchmarks",
        "benchmark_SPYSINGLE": "SPY only",
        "benchmark_6040": "60/40 (SPY / IEF)",
        "benchmark_EQW": "Equal weight (4 ETFs)",
        "transaction_costs": "Transaction costs (backtest assumption)",
        "transaction_costs_readonly": "Read-only: set in the backtest configuration.",
        "equity_curve": "Equity curve (net of costs, base 100)",
        "metrics_title": "Performance metrics (net of costs)",
        "metric": "Metric",
        "strategy": "Strategy",
        "annual_return": "Annualised return",
        "volatility": "Annualised volatility",
        "sharpe": "Sharpe ratio",
        "max_drawdown": "Maximum drawdown",
        "turnover": "Annual turnover",
        "weights_over_time": "Portfolio weights over time",
        "underperform": (
            "The strategy does **not** beat all selected benchmarks after costs. "
            "This is shown as-is."
        ),
        "method_title": "Method and limits",
        "method_walk_forward": (
            "**Walk-forward validation.** Models are trained on past data and "
            "evaluated on later data, mirroring how the tool would have been used."
        ),
        "method_no_leak": (
            "**No information leakage.** Signals at each date use only data "
            "available up to that date; weights are applied with a lag."
        ),
        "method_costs": (
            "**Costs.** Returns are net of assumed transaction costs of {cost}. "
            "Slippage and taxes are not modelled."
        ),
        "method_data": (
            "**Data.** ETF prices from the market data cache; macro series from "
            "FRED (API key read from the environment, never displayed)."
        ),
        "method_limits": (
            "**Limits.** Historical results do not guarantee future outcomes. "
            "This project is for learning and decision support, not investment advice."
        ),
    },
    "FR": {
        "app_title": "Asset Allocation ML",
        "app_subtitle": "Allocation de portefeuille fondée sur des modèles, sur quatre ETF",
        "disclaimer_title": "Information importante",
        "disclaimer": (
            "Ce projet est un outil d'aide à la décision et d'apprentissage, **et non "
            "un conseil en investissement**. Aucune performance future n'est garantie. "
            "Les sorties des modèles sont des estimations incertaines."
        ),
        "language": "Langue",
        "risk_profile": "Profil de risque",
        "risk_profile_help": (
            "Définit l'aversion au risque et les limites par défaut de l'optimiseur."
        ),
        "data_as_of": "Données au",
        "data_as_of_unknown": "non disponible",
        "loading": "Chargement…",
        "no_data": "Aucune donnée disponible pour cette sélection.",
        "no_data_hint": (
            "Élargissez la période ou vérifiez que le cache de données est construit "
            "(voir le README)."
        ),
        "invalid_range": "La date de début doit précéder la date de fin.",
        "error_title": "Une erreur est survenue lors du chargement de cette vue.",
        "error_hint": "Le backend n'a pas renvoyé de résultat exploitable. Aucun modèle n'a été réentraîné.",
        "computed_at": "Calculé le",
        "source": "Source",

        "profile_conservative": "Prudent",
        "profile_balanced": "Équilibré",
        "profile_aggressive": "Dynamique",
        "profile_conservative_desc": "Privilégie les obligations et la faible volatilité.",
        "profile_balanced_desc": "Équilibre croissance et stabilité.",
        "profile_aggressive_desc": "Tolère davantage de risque SPY / VNQ.",

        "asset_SPY": "SPY — Actions américaines",
        "asset_IEF": "IEF — Obligations du Trésor américain",
        "asset_LQD": "LQD — Obligations d'entreprises",
        "asset_VNQ": "VNQ — Immobilier coté",
        "asset_bonds": "Obligations (IEF + LQD)",

        "time_window": "Période",
        "start_date": "Début",
        "end_date": "Fin",
        "full_range": "Historique complet",

        "home_welcome": "Bienvenue",
        "home_intro": (
            "Un outil d'aide à la décision qui combine signaux de risque, "
            "optimisation sous contraintes et backtest honnête pour proposer une "
            "allocation sur quatre ETF."
        ),
        "home_journey": "Parcours suggéré",
        "home_step1": "Choisissez un profil de risque dans la barre latérale.",
        "home_step2": "Observez le marché dans **Vue de marché**.",
        "home_step3": "Lisez les signaux dans **Signaux de risque**.",
        "home_step4": "Obtenez une suggestion dans **Allocation recommandée**.",
        "home_step5": "Vérifiez l'historique dans **Backtest et méthode**.",
        "home_pages": "Pages",
        "home_p1": "Vue de marché — prix, corrélations, drawdowns.",
        "home_p2": "Signaux de risque — probabilité de krach ML 1, volatilité ML 2.",
        "home_p3": "Allocation recommandée — sortie de l'optimiseur sous contraintes.",
        "home_p4": "Backtest et méthode — vérification historique et limites.",

        "p1_title": "Vue de marché",
        "p1_caption": (
            "Les prix normalisés (base 100) mettent les quatre ETF sur un pied "
            "d'égalité. Ils ne évoluent pas toujours ensemble, d'où l'intérêt de la "
            "diversification — qui ne supprime toutefois aucun risque."
        ),
        "p1_normalized": "Prix normalisés (base 100)",
        "p1_correlation": "Corrélation des rendements journaliers",
        "p1_correlation_note": (
            "Les corrélations faibles ou négatives sont celles qui aident la diversification."
        ),
        "p1_drawdown": "Drawdown depuis le précédent sommet",
        "p1_drawdown_note": (
            "Les drawdowns montrent la difficulté de détenir un actif en baisse."
        ),

        "p2_title": "Signaux de risque",
        "p2_intro": (
            "Deux modèles alimentent l'optimiseur : une probabilité de krach pour SPY "
            "et une prévision de volatilité pour chaque ETF. Ce sont des estimations."
        ),
        "ml1_title": "ML 1 — Probabilité d'une forte baisse de SPY sous 21 jours",
        "ml1_current": "Probabilité actuelle",
        "ml1_history": "Signal historique et importants drawdowns de SPY",
        "ml1_marked": "Drawdown observé",
        "ml1_weak": (
            "**Avertissement qualité :** la discrimination hors échantillon est faible. "
            "Ce signal n'est qu'indicatif."
        ),
        "ml2_title": "ML 2 — Volatilité prévue par actif",
        "ml2_bar": "Prévision actuelle vs moyenne de long terme (annualisée)",
        "ml2_forecast": "Prévision",
        "ml2_longrun": "Moyenne long terme",
        "drivers_title": "Ce qui explique ce résultat",
        "drivers_ml1": "Principales variables — ML 1",
        "drivers_ml2": "Principales variables — ML 2",
        "quality_title": "Qualité des modèles (hors échantillon, validation temporelle)",
        "quality_metric": "Métrique",
        "quality_value": "Valeur",
        "quality_auc": "AUC (ML 1)",
        "quality_brier": "Score de Brier / calibration (ML 1)",
        "quality_mae": "Erreur MAE de volatilité (ML 2)",
        "quality_note": (
            "La validation temporelle (walk-forward) entraîne sur le passé et teste sur "
            "des données ultérieures : aucune information future ne fuit dans les scores."
        ),

        "p3_title": "Allocation recommandée",
        "p3_intro": (
            "Une suggestion fondée sur les modèles et l'optimiseur sous contraintes. "
            "Ce n'est ni une prévision ni un conseil en investissement."
        ),
        "constraints_title": "Contraintes",
        "max_weight": "Poids maximal par actif (%)",
        "min_bonds": "Part minimale d'obligations, IEF + LQD (%)",
        "compute": "Calculer l'allocation",
        "computing": "Optimisation…",
        "allocation_title": "Poids suggérés",
        "allocation_table_title": "Poids exacts",
        "weight": "Poids (%)",
        "comparison_title": "Comparaison",
        "current": "Suggestion actuelle",
        "previous": "Mois précédent",
        "benchmark_weights": "Benchmark sélectionné",
        "explanation_title": "Pourquoi cette allocation ?",
        "infeasible_title": "Ces contraintes ne peuvent pas être satisfaites ensemble.",
        "infeasible_hint": (
            "Augmentez le poids maximal par actif ou réduisez la part minimale "
            "d'obligations, puis recalculez."
        ),
        "uncertainty": (
            "De petites variations des signaux peuvent modifier ces poids. "
            "Considérez le résultat comme un scénario, pas une prévision."
        ),
        "not_computed": "Choisissez vos contraintes puis appuyez sur **Calculer l'allocation**.",
        "profile_effect": (
            "Profil de risque : **{profile}** — {description} L'optimiseur utilise "
            "l'aversion au risque et les limites par défaut de ce profil."
        ),
        "explain_spy_reduced": (
            "**SPY réduit à {weight:.0f} %** : le signal de krach est de {p:.0%} et "
            "la volatilité prévue de {vol:.0%}."
        ),
        "explain_spy_kept": (
            "**SPY maintenu à {weight:.0f} %** : le signal de krach est faible ({p:.0%})."
        ),
        "explain_bonds": (
            "**Obligations (IEF + LQD) : {bonds:.0f} %**, dans le respect de votre "
            "minimum de {floor:.0f} %."
        ),
        "explain_profile": (
            "Le profil **{profile}** fixe les limites de risque ; l'optimiseur "
            "les a respectées."
        ),
        "explain_relaxed": (
            "Une contrainte a dû être assouplie par l'optimiseur : {reason}."
        ),
        "placeholder_notice": (
            "Données de démonstration : un artefact de modèle est indisponible dans "
            "cet environnement. Le pont utilise des valeurs de remplacement déterministes."
        ),
        "placeholder_backtest_notice": (
            "Backtest de démonstration : `src/backtest` n'est pas encore implémenté. "
            "Les courbes utilisent de vrais cours historiques, une cible fixe et les coûts indiqués."
        ),

        "p4_title": "Backtest et méthode",
        "p4_intro": (
            "Simulation historique de la stratégie et des benchmarks, "
            "coûts de transaction déduits."
        ),
        "benchmarks": "Benchmarks",
        "benchmark_SPYSINGLE": "SPY seul",
        "benchmark_6040": "60/40 (SPY / IEF)",
        "benchmark_EQW": "Poids égaux (4 ETF)",
        "transaction_costs": "Coûts de transaction (hypothèse du backtest)",
        "transaction_costs_readonly": "Lecture seule : défini dans la configuration du backtest.",
        "equity_curve": "Courbe de capital (coûts déduits, base 100)",
        "metrics_title": "Métriques de performance (coûts déduits)",
        "metric": "Métrique",
        "strategy": "Stratégie",
        "annual_return": "Rendement annualisé",
        "volatility": "Volatilité annualisée",
        "sharpe": "Ratio de Sharpe",
        "max_drawdown": "Drawdown maximal",
        "turnover": "Rotation annuelle",
        "weights_over_time": "Poids du portefeuille dans le temps",
        "underperform": (
            "La stratégie **ne surperforme pas** tous les benchmarks sélectionnés "
            "après coûts. Ce résultat est affiché tel quel."
        ),
        "method_title": "Méthode et limites",
        "method_walk_forward": (
            "**Validation walk-forward.** Les modèles sont entraînés sur le passé "
            "et évalués sur des données ultérieures."
        ),
        "method_no_leak": (
            "**Aucune fuite d'information.** À chaque date, les signaux n'utilisent "
            "que des données connues à cette date ; les poids sont appliqués avec décalage."
        ),
        "method_costs": (
            "**Coûts.** Les rendements sont nets de coûts de transaction supposés de "
            "{cost}. Le slippage et les taxes ne sont pas modélisés."
        ),
        "method_data": (
            "**Données.** Cours des ETF depuis le cache de marché ; séries macro "
            "depuis FRED (clé API lue dans l'environnement, jamais affichée)."
        ),
        "method_limits": (
            "**Limites.** Les résultats historiques ne garantissent pas les résultats "
            "futurs. Ce projet est un outil d'apprentissage et d'aide à la décision, "
            "pas un conseil en investissement."
        ),
    },
}


def label(key: str, language: str = DEFAULT_LANGUAGE, **format_values: Any) -> str:
    """Return a UI string from the label catalogue.

    Unknown keys return the key itself so missing translations are visible in
    review instead of failing silently.
    """
    catalog = LABELS.get(language, LABELS[DEFAULT_LANGUAGE])
    text = catalog.get(key, LABELS[DEFAULT_LANGUAGE].get(key, key))
    if format_values:
        try:
            return text.format(**format_values)
        except (KeyError, ValueError):
            return text
    return text


# --------------------------------------------------------------------------- #
# Display / default constants
# --------------------------------------------------------------------------- #

ASSETS = ("SPY", "IEF", "LQD", "VNQ")
BOND_ASSETS = ("IEF", "LQD")

RISK_PROFILES = ("conservative", "balanced", "aggressive")
DEFAULT_RISK_PROFILE = "balanced"

DEFAULT_MAX_WEIGHT = 60.0
DEFAULT_MIN_BONDS = 20.0
MIN_WEIGHT_SLIDER = 10.0
MAX_WEIGHT_SLIDER = 100.0
WEIGHT_SLIDER_STEP = 5.0

BENCHMARKS = ("SPY_ONLY", "6040", "EQW")
DEFAULT_BENCHMARKS = ("6040",)

# Plotly palette, shared by every chart.
COLORS = {
    "SPY": "#2563EB",
    "IEF": "#16A34A",
    "LQD": "#F59E0B",
    "VNQ": "#DC2626",
    "strategy": "#4F46E5",
    "benchmark_1": "#64748B",
    "benchmark_2": "#94A3B8",
    "benchmark_3": "#CBD5E1",
    "warning": "#B45309",
    "accent": "#0F766E",
}

PLOTLY_TEMPLATE = "plotly_white"
CHART_HEIGHT = 360
