"""Page 1 — Market overview."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import components, services

components.setup_page()
components.init_session_state()

try:
    as_of = services.get_data_as_of()
    sidebar_error = None
except Exception as exc:  # noqa: BLE001
    as_of, sidebar_error = None, exc

components.render_sidebar(as_of)
components.render_header("p1_title")
st.caption(components.t("p1_caption"))
components.plain_words(components.t("p1_plain"))
components.ticker_legend()

if sidebar_error is not None:
    components.render_error(sidebar_error, "data as of")

try:
    bounds = services.get_date_bounds()
    col_start, col_end = st.columns(2)
    start = col_start.date_input(
        components.t("start_date"),
        value=bounds["default_start"],
        min_value=bounds["min_date"],
        max_value=bounds["max_date"],
    )
    end = col_end.date_input(
        components.t("end_date"),
        value=bounds["max_date"],
        min_value=bounds["min_date"],
        max_value=bounds["max_date"],
    )

    if start > end:
        st.warning(components.t("invalid_range"))
    else:
        with st.spinner(components.t("loading")):
            overview = services.get_market_overview(start, end)

        components.normalized_price_chart(overview["prices_normalized"])
        components.chart_help("normalized")
        corr_col, dd_col = st.columns([1, 2])
        with corr_col:
            components.correlation_heatmap(overview["correlation"])
            components.chart_help("correlation")
        with dd_col:
            components.drawdown_chart(overview["drawdowns"])
            components.chart_help("drawdown")
except Exception as exc:  # noqa: BLE001
    components.render_error(exc, "market overview")

components.glossary(
    [
        "etf",
        "spy",
        "ief",
        "lqd",
        "vnq",
        "base100",
        "correlation",
        "drawdown",
        "diversification",
    ]
)
