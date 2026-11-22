"""Streamlit entry point for Asset Allocation ML.

Run with:  streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``streamlit run app/Home.py`` adds ``app/`` to sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import components, services

components.setup_page()
components.init_session_state()

try:
    as_of = services.get_data_as_of()
except Exception as exc:  # noqa: BLE001 - never show a raw traceback in the UI
    as_of = None
    components.render_error(exc, "data as of")

components.render_sidebar(as_of)
components.render_header("home_welcome")
st.markdown(components.t("home_intro"))

# Explain the problem and who the platform helps, in everyday language.
components.section(components.t("home_solves_title"))
st.markdown(components.t("home_solves"))

regular_col, finance_col = st.columns([1, 1], gap="large")
with regular_col:
    components.section(components.t("home_regular_title"))
    st.markdown(components.t("home_regular"))
with finance_col:
    components.section(components.t("home_finance_title"))
    st.markdown(components.t("home_finance"))

st.divider()

intro_col, pages_col = st.columns([1, 1], gap="large")

with intro_col:
    components.section(components.t("home_journey"))
    for key in ("home_step1", "home_step2", "home_step3", "home_step4", "home_step5"):
        st.markdown(f"- {components.t(key)}")

with pages_col:
    components.section(components.t("home_pages"))
    for key in ("home_p1", "home_p2", "home_p3", "home_p4"):
        st.markdown(f"- {components.t(key)}")

components.glossary(
    ["etf", "allocation", "risk_profile", "model", "signal", "optimizer", "backtest"]
)

st.divider()
profile = st.session_state["risk_profile"]
components.info_block(
    f"**{components.t('risk_profile')}:** {components.t('profile_' + profile)}  ·  "
    f"**{components.t('data_as_of')}:** {as_of or components.t('data_as_of_unknown')}"
)
components.render_disclaimer()
