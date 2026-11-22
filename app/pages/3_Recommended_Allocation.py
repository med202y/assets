"""Page 3 — Recommended allocation (core page)."""

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
components.render_header("p3_title")
st.caption(components.t("p3_intro"))
components.plain_words(components.t("p3_plain"))

if sidebar_error is not None:
    components.render_error(sidebar_error, "data as of")

profile = st.session_state["risk_profile"]
st.markdown(
    components.t(
        "profile_effect",
        profile=components.t("profile_" + profile),
        description=components.t("profile_" + profile + "_desc"),
    )
)

try:
    components.section(components.t("constraints_title"))
    col_max, col_min, col_bench = st.columns([1, 1, 1])
    max_weight = col_max.slider(
        components.t("max_weight"),
        min_value=config.MIN_WEIGHT_SLIDER,
        max_value=config.MAX_WEIGHT_SLIDER,
        step=config.WEIGHT_SLIDER_STEP,
        key="max_weight",
    )
    min_bonds = col_min.slider(
        components.t("min_bonds"),
        min_value=0.0,
        max_value=config.MAX_WEIGHT_SLIDER,
        step=config.WEIGHT_SLIDER_STEP,
        key="min_bonds",
    )
    benchmark = col_bench.selectbox(
        components.t("benchmark_weights"),
        options=list(config.BENCHMARKS),
        format_func=lambda value: components.t("benchmark_" + value),
        key="allocation_benchmark",
    )

    signature = (profile, float(max_weight), float(min_bonds), benchmark)

    # Re-solve automatically when shared profile/constraints change, so the
    # recommendation stays consistent across pages. The optimizer is cached in
    # services, so this is cheap and never retrains a model.
    if st.session_state["allocation_signature"] != signature:
        st.session_state["allocation_signature"] = signature
        if st.session_state["allocation_result"] is not None:
            with st.spinner(components.t("computing")):
                st.session_state["allocation_result"] = services.get_allocation(*signature)

    if st.button(components.t("compute"), type="primary"):
        with st.spinner(components.t("computing")):
            st.session_state["allocation_result"] = services.get_allocation(*signature)

    result = st.session_state["allocation_result"]

    if result is None:
        components.info_block(components.t("not_computed"))
    elif result.get("status") == "infeasible":
        st.error(components.t("infeasible_title"))
        components.warning_block(result.get("message") or components.t("infeasible_hint"))
    elif result.get("status") == "error":
        st.error(result.get("message") or components.t("error_title"))
    else:
        weights_col, table_col = st.columns([1, 1])
        with weights_col:
            components.weights_donut(result["weights"])
        with table_col:
            components.section(components.t("allocation_table_title"))
            components.weights_table(result["weights"])
            components.weights_comparison_chart(result["comparison"])

        if result.get("source") == services.SOURCE_PLACEHOLDER:
            components.info_block(components.t("placeholder_notice"))

        st.divider()
        components.section(components.t("explanation_title"))
        st.markdown(result["explanation"])
        components.info_block(components.t("uncertainty"))
        components.render_disclaimer()
except Exception as exc:  # noqa: BLE001
    components.render_error(exc, "recommended allocation")

components.glossary(
    [
        "allocation",
        "weight",
        "risk_profile",
        "constraint",
        "optimizer",
        "benchmark",
        "infeasible",
        "uncertainty",
    ]
)
