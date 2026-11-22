"""Page 2 — Risk signals (ML 1 and ML 2)."""

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
components.render_header("p2_title")
st.caption(components.t("p2_intro"))

if sidebar_error is not None:
    components.render_error(sidebar_error, "data as of")

try:
    with st.spinner(components.t("loading")):
        ml1 = services.get_ml1_signal()
        ml2 = services.get_ml2_vol()

    if services.SOURCE_PLACEHOLDER in (ml1.get("source"), ml2.get("source")):
        components.info_block(components.t("placeholder_notice"))

    # --- ML 1 ---------------------------------------------------------------
    components.section(components.t("ml1_title"))
    gauge_col, quality_col = st.columns([1, 2])
    with gauge_col:
        components.ml1_gauge(ml1["current_probability"])
    with quality_col:
        components.section(components.t("quality_title"))
        components.quality_table(ml1["quality_table"])
        st.caption(components.t("quality_note"))
        if ml1.get("weak"):
            components.warning_block(components.t("ml1_weak"))

    components.ml1_history_chart(ml1["history"], ml1["large_drawdown_dates"])

    st.divider()

    # --- ML 2 ---------------------------------------------------------------
    components.section(components.t("ml2_title"))
    vol_col, quality_col2 = st.columns([2, 1])
    with vol_col:
        components.volatility_forecast_chart(ml2["forecast_table"])
    with quality_col2:
        components.quality_table(ml2["quality_table"])

    st.divider()

    # --- Drivers ------------------------------------------------------------
    components.section(components.t("drivers_title"))
    drivers_ml1, drivers_ml2 = st.columns(2)
    with drivers_ml1:
        components.feature_importance_chart(ml1["top_features"], components.t("drivers_ml1"))
    with drivers_ml2:
        components.feature_importance_chart(ml2["top_features"], components.t("drivers_ml2"))
except Exception as exc:  # noqa: BLE001
    components.render_error(exc, "risk signals")
