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
        "plain_words_title": "In simple words",
        "glossary_page_title": "Words used on this page",
        "glossary_intro": "Click to see what each term means.",

        # Ticker symbols and how to read the charts.
        "ticker_legend_title": "What do SPY, IEF, LQD and VNQ mean?",
        "ticker_legend_intro": (
            "These are ticker symbols: short names for the four funds used in "
            "this tool."
        ),
        "chart_help_prefix": "How to read this:",
        "chart_normalized": (
            "All four funds start at 100. A rising line means that fund grew "
            "more than the others in this period; a falling line means it lost "
            "value. This is a look back, not a forecast."
        ),
        "chart_correlation": (
            "Each square shows how two funds moved together. Red means they "
            "often moved in opposite directions, which helps diversification; "
            "blue means they moved together. Pale squares mean little link."
        ),
        "chart_drawdown": (
            "This shows how far each fund fell below its previous high. A "
            "deeper dip means a more painful period. It does not predict future "
            "losses."
        ),
        "chart_gauge": (
            "The needle is the model's estimate of the chance that US shares "
            "fall sharply in the next 21 trading days. It is an educated guess, "
            "not a promise."
        ),
        "chart_quality": (
            "These scores test the models on data they did not learn from. For "
            "AUC, higher is better. For Brier and MAE, lower is better. They are "
            "checks on the past, not grades for the future."
        ),
        "chart_ml1_history": (
            "The line is the model's crash probability over time. Dotted lines "
            "mark periods when SPY really did fall a lot, so you can see whether "
            "the model warned early."
        ),
        "chart_volatility": (
            "A blue bar is the model's guess of how jumpy a fund may be; the "
            "grey bar is its long-run average. Taller bars mean more expected "
            "ups and downs."
        ),
        "chart_feature_importance": (
            "Longer bars are the inputs the model looked at most. They explain "
            "what influenced it, not whether it was right."
        ),
        "chart_weights_donut": (
            "Each slice is the share of your money the optimizer suggests for "
            "one fund. The whole circle is 100%."
        ),
        "chart_weights_table": (
            "These are the exact percentages behind the chart. Together they "
            "should add up to 100%."
        ),
        "chart_comparison": (
            "This compares the suggestion with last month and with a simple "
            "benchmark. It shows how different the new mix is."
        ),
        "chart_equity": (
            "The lines show what 100 would have become if you had followed each "
            "mix. Higher past lines are better past results, but they do not "
            "guarantee future ones."
        ),
        "chart_metrics": (
            "Annual return and Sharpe: higher is better. Volatility, maximum "
            "drawdown and turnover: lower is usually better."
        ),
        "chart_weights_area": (
            "The stacked bands show how the mix changed over time. The total "
            "height is always 100%."
        ),

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

        # One-paragraph explanation in plain language for each page.
        "p1_plain": (
            "This page answers one question: which of the four funds went up or "
            "down the most, and do they move together? We start every fund at 100 "
            "so they can be compared fairly. The last chart shows how far each fund "
            "fell from its best point — a reminder of what a bad period can feel "
            "like. No model is used here, only past prices."
        ),
        "p2_plain": (
            "Two computer models give us clues about risk. The first estimates the "
            "chance of a big drop in US shares over the next month of trading. The "
            "second estimates how jumpy each fund is likely to be. Both are "
            "educated guesses, not facts. We also show how well each model did on "
            "data it had not seen before, because a model that only remembers the "
            "past is not useful."
        ),
        "p3_plain": (
            "This is the main page. You say how much risk you want and set two "
            "simple rules: no fund larger than X%, and at least Y% in bonds. The "
            "optimiser then searches for a mix that follows your rules and keeps "
            "risk in check. You get exact percentages, a comparison with a simple "
            "alternative, and a plain-language reason for the result. Nothing is "
            "learned again here — the saved signals are simply rearranged."
        ),
        "p4_plain": (
            "This page asks: if we had followed a simple mix in the past, what "
            "would have happened? We start with 100 and show how that mix and a "
            "few simple alternatives grew after trading costs. Important: this is "
            "a demonstration using a fixed mix, not a full historical test of the "
            "real model-driven strategy. The real backtest code does not exist "
            "yet, so read these curves as an illustration."
        ),

        # Navigation / home
        "home_welcome": "Welcome",
        "home_intro": (
            "A simple way to decide how to share your money between four funds — "
            "with a clear explanation of why."
        ),
        "home_solves_title": "What does this platform do?",
        "home_solves": (
            "Deciding how much to put in each investment is confusing: prices move "
            "up and down, and there is no obvious 'right' mix. This tool does that "
            "thinking with you. It looks at how the market behaved in the past, uses "
            "two computer models to judge risk, and suggests a mix of four funds "
            "that fits how much risk you are comfortable with. It is a starting "
            "point for a decision, not a set of instructions."
        ),
        "home_regular_title": "For regular people",
        "home_regular": (
            "If you are saving for a home, retirement, or a rainy day, think of this "
            "as a simple recipe for spreading your money. You choose calm, middle, "
            "or adventurous, set two easy limits, and the tool shows a possible mix "
            "of four funds. It then explains, in everyday language, why it chose "
            "that mix. Useful when you want a second opinion you can actually "
            "understand."
        ),
        "home_finance_title": "For finance people",
        "home_finance": (
            "You get a transparent pipeline instead of a black box: saved ML 1 crash "
            "probabilities and ML 2 volatility forecasts feed a CVXPY optimiser with "
            "profile and constraint limits, then a comparison against simple "
            "benchmarks. Every figure is shown with its source and its limitations, "
            "so you can explore scenarios and challenge the assumptions."
        ),
        "home_journey": "A simple path through the site",
        "home_step1": "Choose your comfort with risk in the sidebar.",
        "home_step2": "See how the four funds behaved in the past.",
        "home_step3": "See what the two models say about risk now.",
        "home_step4": "Get a suggested mix and the reason behind it.",
        "home_step5": "Look at an illustrative past check and the limits.",
        "home_pages": "The four pages",
        "home_p1": "Market overview — how the funds moved.",
        "home_p2": "Risk signals — what the models think.",
        "home_p3": "Recommended allocation — a suggested mix.",
        "home_p4": "Backtest and method — a past check and its limits.",

        # Page 1
        "p1_title": "Market overview",
        "p1_caption": (
            "All four funds start at 100, so you can compare their moves. They do "
            "not always move together, which can make the ride smoother — but "
            "never removes risk."
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
            "Two models give the optimizer clues about risk: the chance of a big "
            "fall in US shares, and how jumpy each fund may be. Both are guesses, "
            "not facts."
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
            "A suggested mix based on the models and your rules. It is one possible "
            "scenario, not a prediction or investment advice."
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
            "An illustrative look at how a simple mix would have done in the past, "
            "after assumed trading costs. This is not a full test of the real strategy."
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
        "plain_words_title": "En mots simples",
        "glossary_page_title": "Mots utilisés sur cette page",
        "glossary_intro": "Cliquez pour voir la signification de chaque terme.",

        # Symboles boursiers et lecture des graphiques.
        "ticker_legend_title": "Que signifient SPY, IEF, LQD et VNQ ?",
        "ticker_legend_intro": (
            "Ce sont des symboles boursiers : les noms courts des quatre fonds "
            "utilisés dans cet outil."
        ),
        "chart_help_prefix": "Comment lire ceci :",
        "chart_normalized": (
            "Les quatre fonds commencent tous à 100. Une ligne qui monte signifie "
            "que ce fonds a plus progressé que les autres sur la période ; une "
            "ligne qui baisse signifie qu'il a perdu de la valeur. C'est un regard "
            "sur le passé, pas une prévision."
        ),
        "chart_correlation": (
            "Chaque case montre si deux fonds ont bougé ensemble. Le rouge signifie "
            "qu'ils ont souvent bougé en sens opposé, ce qui aide la diversification ; "
            "le bleu signifie qu'ils ont bougé ensemble. Les cases pâles indiquent "
            "peu de lien."
        ),
        "chart_drawdown": (
            "Ce graphique montre de combien chaque fonds est tombé sous son plus "
            "haut précédent. Une baisse plus profonde signifie une période plus "
            "difficile. Il ne prédit pas les pertes futures."
        ),
        "chart_gauge": (
            "L'aiguille montre l'estimation du modèle sur la chance d'une forte "
            "baisse des actions américaines sous 21 séances. C'est une supposition "
            "éclairée, pas une promesse."
        ),
        "chart_quality": (
            "Ces scores testent les modèles sur des données qu'ils n'ont pas vues. "
            "Pour l'AUC, plus c'est élevé, mieux c'est. Pour le score de Brier et "
            "le MAE, plus c'est bas, mieux c'est. Ce sont des contrôles du passé, "
            "pas des notes pour le futur."
        ),
        "chart_ml1_history": (
            "La ligne montre la probabilité de krach estimée par le modèle au fil "
            "du temps. Les lignes pointillées marquent les périodes où SPY a "
            "réellement beaucoup baissé : vous voyez si le modèle avait prévenu."
        ),
        "chart_volatility": (
            "Une barre bleue est l'estimation de l'agitation possible d'un fonds ; "
            "la barre grise est sa moyenne de long terme. Des barres plus hautes "
            "signifient davantage de hauts et de bas attendus."
        ),
        "chart_feature_importance": (
            "Les barres les plus longues sont les données que le modèle a le plus "
            "regardées. Elles expliquent ce qui l'a influencé, pas s'il avait raison."
        ),
        "chart_weights_donut": (
            "Chaque part représente la portion de votre argent que l'optimiseur "
            "suggère pour un fonds. Le cercle entier représente 100 %."
        ),
        "chart_weights_table": (
            "Voici les pourcentages exacts derrière le graphique. Ensemble, ils "
            "doivent totaliser 100 %."
        ),
        "chart_comparison": (
            "Cela compare la suggestion avec le mois précédent et avec un mélange "
            "simple de référence. Vous voyez à quel point le nouveau mélange est "
            "différent."
        ),
        "chart_equity": (
            "Les lignes montrent ce que 100 seraient devenus si vous aviez suivi "
            "chaque mélange. Des lignes plus hautes indiquent de meilleurs résultats "
            "passés, sans garantir les résultats futurs."
        ),
        "chart_metrics": (
            "Rendement annualisé et ratio de Sharpe : plus c'est élevé, mieux c'est. "
            "Volatilité, drawdown maximal et rotation : plus c'est bas, mieux c'est "
            "en général."
        ),
        "chart_weights_area": (
            "Les bandes empilées montrent comment le mélange a changé dans le temps. "
            "La hauteur totale reste toujours à 100 %."
        ),

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

        # Explication en langage courant pour chaque page.
        "p1_plain": (
            "Cette page répond à une seule question : lequel des quatre fonds a le "
            "plus monté ou baissé, et évoluent-ils ensemble ? Tous les fonds "
            "commencent à 100 pour être comparés équitablement. Le dernier "
            "graphique montre de combien chacun est tombé depuis son plus haut : "
            "un rappel de ce que peut ressembler une mauvaise période. Aucun modèle "
            "ici, seulement des prix passés."
        ),
        "p2_plain": (
            "Deux programmes donnent des indices sur le risque. Le premier estime "
            "la chance d'une forte baisse des actions américaines au cours du "
            "prochain mois de bourse. Le second estime à quel point chaque fonds "
            "devrait bouger. Ce sont des suppositions éclairées, pas des faits. "
            "Nous montrons aussi comment chaque modèle s'est comporté sur des "
            "données qu'il n'avait jamais vues : un modèle qui ne sait que "
            "mémoriser le passé n'est pas utile."
        ),
        "p3_plain": (
            "C'est la page principale. Vous indiquez le risque que vous acceptez et "
            "fixez deux règles simples : aucun fonds au-dessus de X %, et au moins "
            "Y % en obligations. L'optimiseur cherche alors un mélange qui respecte "
            "vos règles tout en maîtrisant le risque. Vous obtenez des pourcentages "
            "précis, une comparaison avec une solution simple et une explication en "
            "langage courant. Rien n'est réentraîné ici : les signaux enregistrés "
            "sont simplement réorganisés."
        ),
        "p4_plain": (
            "Cette page demande : si nous avions suivi un mélange simple dans le "
            "passé, que se serait-il passé ? Nous partons de 100 et montrons comment "
            "ce mélange et quelques solutions simples ont évolué après frais. "
            "Attention : c'est une démonstration avec un mélange fixe, pas un vrai "
            "test historique de la stratégie pilotée par les modèles. Le vrai code "
            "de backtest n'existe pas encore ; lisez ces courbes comme une "
            "illustration."
        ),

        "home_welcome": "Bienvenue",
        "home_intro": (
            "Une façon simple de décider comment répartir votre argent entre "
            "quatre fonds — avec une explication claire du pourquoi."
        ),
        "home_solves_title": "Que fait cette plateforme ?",
        "home_solves": (
            "Décider combien placer dans chaque investissement est déroutant : les "
            "prix montent et descendent, et il n'existe pas de « bon » mélange "
            "évident. Cet outil réfléchit avec vous. Il regarde ce qui s'est passé "
            "par le passé, utilise deux programmes pour juger le risque, puis propose "
            "un mélange de quatre fonds adapté au niveau de risque que vous acceptez. "
            "C'est un point de départ pour une décision, pas des consignes."
        ),
        "home_regular_title": "Pour le grand public",
        "home_regular": (
            "Si vous épargnez pour un logement, la retraite ou un imprévu, voyez "
            "cet outil comme une recette simple pour répartir votre argent. Vous "
            "choisissez calme, intermédiaire ou aventureux, vous fixez deux limites "
            "faciles, et l'outil montre un mélange possible de quatre fonds. Il "
            "explique ensuite en langage courant pourquoi il l'a choisi. Utile "
            "quand vous voulez un second avis que vous pouvez vraiment comprendre."
        ),
        "home_finance_title": "Pour les professionnels",
        "home_finance": (
            "Vous obtenez une chaîne transparente plutôt qu'une boîte noire : les "
            "probabilités de krach ML 1 et les prévisions de volatilité ML 2 "
            "enregistrées alimentent un optimiseur CVXPY avec profils et "
            "contraintes, puis une comparaison à des mélanges simples. Chaque "
            "chiffre est présenté avec sa source et ses limites, afin d'explorer "
            "des scénarios et de contester les hypothèses."
        ),
        "home_journey": "Un parcours simple",
        "home_step1": "Choisissez votre tolérance au risque dans la barre latérale.",
        "home_step2": "Voyez comment les quatre fonds ont évolué par le passé.",
        "home_step3": "Découvrez ce que les deux modèles disent du risque aujourd'hui.",
        "home_step4": "Obtenez un mélange suggéré et la raison derrière ce choix.",
        "home_step5": "Regardez une vérification illustrative et ses limites.",
        "home_pages": "Les quatre pages",
        "home_p1": "Vue de marché — comment les fonds ont bougé.",
        "home_p2": "Signaux de risque — ce que pensent les modèles.",
        "home_p3": "Allocation recommandée — un mélange suggéré.",
        "home_p4": "Backtest et méthode — une vérification et ses limites.",

        "p1_title": "Vue de marché",
        "p1_caption": (
            "Les quatre fonds commencent tous à 100 pour que vous puissiez "
            "comparer leurs évolutions. Ils ne bougent pas toujours ensemble, ce "
            "qui peut adoucir le parcours — sans jamais supprimer le risque."
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
            "Deux modèles donnent des indices à l'optimiseur : la chance d'une "
            "forte baisse des actions américaines et l'agitation probable de chaque "
            "fonds. Ce sont des suppositions, pas des faits."
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
            "Un mélange suggéré à partir des modèles et de vos règles. C'est un "
            "scénario possible, ni une prévision ni un conseil en investissement."
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
            "Un aperçu illustratif de ce qu'aurait donné un mélange simple par le "
            "passé, frais déduits. Ce n'est pas un vrai test de la stratégie réelle."
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


# Plain-language explanations for technical words. Each entry is
# (short name, simple definition). Pages only pass keys to the renderer.
GLOSSARY: dict[str, dict[str, tuple[str, str]]] = {
    "EN": {
        "etf": ("ETF (fund)", "A fund that trades like a stock and holds many assets at once. SPY, IEF, LQD and VNQ are ETFs."),
        "spy": ("SPY", "A fund that follows the S&P 500, the 500 largest US companies. It is the 'US shares' part of the portfolio."),
        "ief": ("IEF", "A fund of US Treasury bonds with medium to long maturity. It is usually the calmest of the four funds."),
        "lqd": ("LQD", "A fund of bonds issued by strong companies (investment grade). Its risk sits between government bonds and shares."),
        "vnq": ("VNQ", "A fund of listed property companies (real estate). It can be more volatile than the bond funds."),
        "allocation": ("Allocation", "How the money is split between the four funds, shown as percentages."),
        "weight": ("Weight", "The percentage of the portfolio in one fund. 25% means a quarter of the money."),
        "risk_profile": ("Risk profile", "How much ups and downs you are willing to accept. Conservative means calmer; aggressive accepts more risk."),
        "signal": ("Signal", "A clue from a model about what might happen. It is not a certainty."),
        "model": ("Model", "A computer program that learns patterns from past data."),
        "forecast": ("Forecast", "An estimate of what may happen next, based on past data."),
        "volatility": ("Volatility", "How much a fund's price jumps around. Higher means a bumpier ride."),
        "correlation": ("Correlation", "How much two funds move together. A low number means they often move differently."),
        "drawdown": ("Drawdown", "How far a fund has fallen from its highest point so far."),
        "base100": ("Base 100", "All funds are shown starting at 100 so their moves can be compared fairly."),
        "diversification": ("Diversification", "Owning different kinds of funds so they do not all fall at the same time."),
        "crash_probability": ("Crash probability", "The model's estimate of the chance of a big fall in US shares in the next 21 trading days."),
        "out_of_sample": ("Out of sample", "Testing a model on data it was not built with. This is a fairer test."),
        "walk_forward": ("Walk-forward validation", "Testing by pretending you are in the past: learn from older data, then check the next period."),
        "auc": ("AUC", "A score from 0.5 to 1. 0.5 is guessing; closer to 1 is better at spotting risky times."),
        "brier": ("Brier score", "A score for probability estimates. Lower is better."),
        "mae": ("MAE", "Average forecasting error. Lower is better."),
        "feature": ("Feature", "An input the model looks at, such as past returns or volatility."),
        "optimizer": ("Optimizer", "A tool that searches for the mix of funds that best fits the limits you set."),
        "constraint": ("Constraint", "A rule you set, such as 'no fund above 60%'."),
        "infeasible": ("Infeasible", "The rules cannot all be followed at the same time."),
        "benchmark": ("Benchmark", "A simple comparison mix, such as 60% shares and 40% bonds."),
        "covariance": ("Covariance", "How the funds' price moves relate to each other. The optimizer uses this to judge risk."),
        "backtest": ("Backtest", "Checking how a method would have behaved using past data."),
        "equity_curve": ("Equity curve", "A line showing how 100 would have grown, or shrunk, over time."),
        "annualised_return": ("Annualised return", "The average yearly gain if the observed growth were spread evenly over a year."),
        "sharpe": ("Sharpe ratio", "Return compared with risk. Higher is better."),
        "max_drawdown": ("Maximum drawdown", "The worst fall from a high point in the period shown."),
        "turnover": ("Turnover", "How much of the portfolio was traded. More trading usually means more costs."),
        "transaction_cost": ("Transaction cost", "The fee assumed each time the mix is changed."),
        "no_lookahead": ("No look-ahead", "The method only uses information known at the time, never future data."),
        "rebalancing": ("Rebalancing", "Adjusting the mix back to its target from time to time."),
        "uncertainty": ("Uncertainty", "The result is an estimate. Small changes in the inputs can change it."),
    },
    "FR": {
        "etf": ("ETF (fonds)", "Un fonds qui se négocie comme une action et contient plusieurs actifs. SPY, IEF, LQD et VNQ sont des ETF."),
        "spy": ("SPY", "Un fonds qui suit le S&P 500, les 500 plus grandes entreprises américaines. C'est la part « actions américaines » du portefeuille."),
        "ief": ("IEF", "Un fonds d'obligations du Trésor américain à moyen et long terme. C'est généralement le plus calme des quatre fonds."),
        "lqd": ("LQD", "Un fonds d'obligations émises par des entreprises solides (catégorie investment grade). Son risque se situe entre les obligations d'État et les actions."),
        "vnq": ("VNQ", "Un fonds de sociétés immobilières cotées (REIT). Il peut être plus volatil que les fonds obligataires."),
        "allocation": ("Allocation", "La répartition de l'argent entre les quatre fonds, en pourcentages."),
        "weight": ("Poids", "Le pourcentage du portefeuille dans un fonds. 25 % signifie un quart de l'argent."),
        "risk_profile": ("Profil de risque", "La part des hauts et des bas que vous acceptez. Prudent = plus calme ; dynamique = plus de risque."),
        "signal": ("Signal", "Un indice fourni par un modèle sur ce qui pourrait arriver. Ce n'est pas une certitude."),
        "model": ("Modèle", "Un programme informatique qui apprend des régularités dans des données passées."),
        "forecast": ("Prévision", "Une estimation de ce qui pourrait arriver, fondée sur des données passées."),
        "volatility": ("Volatilité", "L'importance des variations de prix. Plus elle est élevée, plus le parcours est cahoteux."),
        "correlation": ("Corrélation", "La mesure dans laquelle deux fonds évoluent ensemble. Un chiffre faible signifie qu'ils bougent souvent différemment."),
        "drawdown": ("Drawdown", "L'ampleur de la baisse depuis le point le plus haut atteint."),
        "base100": ("Base 100", "Tous les fonds commencent à 100 pour que leurs évolutions soient comparables."),
        "diversification": ("Diversification", "Détenir différents types de fonds pour qu'ils ne baissent pas tous en même temps."),
        "crash_probability": ("Probabilité de krach", "L'estimation du modèle sur la chance d'une forte baisse des actions américaines sous 21 séances."),
        "out_of_sample": ("Hors échantillon", "Tester un modèle sur des données qui n'ont pas servi à le construire. C'est un test plus juste."),
        "walk_forward": ("Validation walk-forward", "Tester comme si l'on était dans le passé : apprendre sur des données anciennes, puis vérifier la période suivante."),
        "auc": ("AUC", "Un score de 0,5 à 1. 0,5 équivaut à deviner ; plus on approche de 1, mieux le modèle repère les moments risqués."),
        "brier": ("Score de Brier", "Une note pour les probabilités estimées. Plus elle est basse, mieux c'est."),
        "mae": ("MAE", "L'erreur moyenne de prévision. Plus elle est basse, mieux c'est."),
        "feature": ("Variable", "Une donnée regardée par le modèle, comme les rendements ou la volatilité passés."),
        "optimizer": ("Optimiseur", "Un outil qui cherche le mélange de fonds qui respecte le mieux les limites fixées."),
        "constraint": ("Contrainte", "Une règle que vous fixez, par exemple « aucun fonds au-dessus de 60 % »."),
        "infeasible": ("Infaisable", "Les règles ne peuvent pas toutes être respectées en même temps."),
        "benchmark": ("Benchmark", "Un mélange simple de comparaison, par exemple 60 % d'actions et 40 % d'obligations."),
        "covariance": ("Covariance", "La façon dont les mouvements des fonds sont liés. L'optimiseur s'en sert pour juger le risque."),
        "backtest": ("Backtest", "Vérifier comment une méthode aurait fonctionné avec des données passées."),
        "equity_curve": ("Courbe de capital", "Une ligne qui montre comment 100 aurait grandi, ou diminué, dans le temps."),
        "annualised_return": ("Rendement annualisé", "Le gain moyen annuel si la croissance observée était étalée sur une année."),
        "sharpe": ("Ratio de Sharpe", "Le rendement comparé au risque. Plus il est élevé, mieux c'est."),
        "max_drawdown": ("Drawdown maximal", "La plus forte baisse depuis un sommet sur la période affichée."),
        "turnover": ("Rotation", "La part du portefeuille qui a été échangée. Plus il y a d'échanges, plus les coûts sont élevés."),
        "transaction_cost": ("Coût de transaction", "Les frais supposés à chaque changement de répartition."),
        "no_lookahead": ("Sans fuite d'information", "La méthode n'utilise que des informations connues à l'époque, jamais des données futures."),
        "rebalancing": ("Rééquilibrage", "Rajuster périodiquement le mélange vers sa cible."),
        "uncertainty": ("Incertitude", "Le résultat est une estimation. De petites modifications des données peuvent le changer."),
    },
}


def glossary(key: str, language: str = DEFAULT_LANGUAGE) -> tuple[str, str]:
    """Return (name, definition) for a technical term."""
    catalog = GLOSSARY.get(language, GLOSSARY[DEFAULT_LANGUAGE])
    if key in catalog:
        return catalog[key]
    return GLOSSARY[DEFAULT_LANGUAGE].get(key, (key, ""))


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
