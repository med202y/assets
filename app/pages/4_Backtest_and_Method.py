"""Page 4 — Backtest and method."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import components, config, services

components.setup_page()
components.init_session_state()

try:
    as_of = services.get_data_as_of()
    sidebar_error = None
except Exception as exc:  # noqa: BLE001
    as_of, sidebar_error = None, exc

components.render_sidebar(as_of)
components.render_header("p4_title")
st.caption(components.t("p4_intro"))
components.plain_words(components.t("p4_plain"))
components.ticker_legend()

if sidebar_error is not None:
    components.render_error(sidebar_error, "data as of")

try:
    benchmarks = st.multiselect(
        components.t("benchmarks"),
        options=list(config.BENCHMARKS),
        format_func=lambda value: components.t("benchmark_" + value),
        key="selected_benchmarks",
    )

    cost = services.get_transaction_cost()
    components.info_block(
        f"**{components.t('transaction_costs')}:** {components.fmt_pct(cost, 2)}  ·  "
        f"{components.t('transaction_costs_readonly')}"
    )

    with st.spinner(components.t("loading")):
        backtest = services.get_backtest_results(tuple(benchmarks))

    if backtest.get("source") == services.SOURCE_PLACEHOLDER:
        components.info_block(components.t("placeholder_backtest_notice"))

    components.equity_curve_chart(backtest["equity_curve"])
    components.chart_help("equity")
    if backtest.get("underperforms"):
        components.warning_block(components.t("underperform"))

    metrics_col, weights_col = st.columns([1, 2])
    with metrics_col:
        components.section(components.t("metrics_title"))
        components.metrics_table(backtest["metrics"])
        components.chart_help("metrics")
    with weights_col:
        components.section(components.t("weights_over_time"))
        components.weights_area_chart(backtest["weights_history"])
        components.chart_help("weights_area")

    st.divider()
    components.section(components.t("method_title"))
    for key in (
        "method_walk_forward",
        "method_no_leak",
        "method_costs",
        "method_data",
        "method_limits",
    ):
        message = components.t(key, cost=components.fmt_pct(cost, 2))
        st.markdown(f"- {message}")
except Exception as exc:  # noqa: BLE001
    components.render_error(exc, "backtest and method")

components.glossary(
    [
        "backtest",
        "benchmark",
        "spy",
        "ief",
        "lqd",
        "vnq",
        "equity_curve",
        "annualised_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "transaction_cost",
        "walk_forward",
        "no_lookahead",
        "rebalancing",
    ]
)
