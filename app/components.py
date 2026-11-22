"""Reusable, presentation-only Streamlit widgets and Plotly charts.

This module contains NO business logic and never imports from ``src``.  Every
function takes ready-to-render data from :mod:`app.services`, so swapping the
frontend later only requires replacing this file and the pages.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import config

logger = logging.getLogger(__name__)

# Streamlit is responsive by default, but the fixed column ratios and chart
# heights used here need a little help on narrow screens. This CSS only applies
# below the mobile breakpoint and changes presentation, never data or logic.
MOBILE_CSS = """
<style>
@media (max-width: 768px) {
  /* Stack every multi-column row on phones. */
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 0.6rem !important;
  }
  [data-testid="stHorizontalBlock"] > div {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }

  /* Keep charts inside the viewport. */
  .js-plotly-plot,
  .plot-container,
  .svg-container {
    max-width: 100% !important;
    height: 300px !important;
  }
  .modebar { display: none !important; }

  /* Tables scroll sideways instead of stretching the page. */
  [data-testid="stDataFrame"] {
    width: 100% !important;
    overflow-x: auto !important;
  }

  /* Slightly smaller headings and safer text wrapping. */
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.3rem !important; }
  h3 { font-size: 1.1rem !important; }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] p {
    overflow-wrap: anywhere;
  }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Session state and text helpers
# --------------------------------------------------------------------------- #

def setup_page() -> None:
    """Call before any other Streamlit command on every page."""
    st.set_page_config(
        page_title=config.label("app_title", config.DEFAULT_LANGUAGE),
        page_icon="📊",
        layout="wide",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialise shared UI state. Safe to call on every page."""
    defaults: dict[str, Any] = {
        "language": config.DEFAULT_LANGUAGE,
        "risk_profile": config.DEFAULT_RISK_PROFILE,
        "max_weight": config.DEFAULT_MAX_WEIGHT,
        "min_bonds": config.DEFAULT_MIN_BONDS,
        "selected_benchmarks": list(config.DEFAULT_BENCHMARKS),
        "allocation_result": None,
        "allocation_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def t(key: str, **format_values: Any) -> str:
    """Translate a label using the language stored in session state."""
    language = st.session_state.get("language", config.DEFAULT_LANGUAGE)
    return config.label(key, language, **format_values)


def _l(key: str) -> str:
    """Translate a key when it exists, otherwise render it unchanged."""
    language = st.session_state.get("language", config.DEFAULT_LANGUAGE)
    return t(key) if key in config.LABELS.get(language, {}) else key


def plain_words(message: str) -> None:
    """Show a short, jargon-free explanation of the page."""
    st.info(f"**{t('plain_words_title')}**\n\n{message}")


def glossary(terms: Sequence[str]) -> None:
    """Show an expander defining the technical words used on this page.

    Definitions live in ``config.GLOSSARY``; pages only pass keys, so adding a
    language or editing wording never requires touching a page file.
    """
    if not terms:
        return
    language = st.session_state.get("language", config.DEFAULT_LANGUAGE)
    with st.expander(t("glossary_page_title")):
        st.caption(t("glossary_intro"))
        for term in terms:
            name, definition = config.glossary(term, language)
            st.markdown(f"**{name}** — {definition}")


def ticker_legend() -> None:
    """Explain the four ETF ticker symbols in plain language."""
    language = st.session_state.get("language", config.DEFAULT_LANGUAGE)
    with st.expander(t("ticker_legend_title")):
        st.caption(t("ticker_legend_intro"))
        for asset in config.ASSETS:
            name, definition = config.glossary(asset.lower(), language)
            st.markdown(f"**{name}** — {definition}")


def chart_help(key: str) -> None:
    """Show a one-line, plain-language guide to the chart just rendered."""
    st.caption(f"**{t('chart_help_prefix')}** {t('chart_' + key)}")


# --------------------------------------------------------------------------- #
# Shared sidebar / common widgets
# --------------------------------------------------------------------------- #

def persist_session_kwargs() -> dict[str, Any]:
    """Keep keyed widget values alive across page switches.

    Streamlit >= 1.67 drops a keyed widget's value on page switch unless
    ``persist_state="session"`` is set. Older versions do not know the argument,
    so it is only passed when supported.
    """
    version = tuple(
        int(part) for part in st.__version__.split(".")[:2] if part.isdigit()
    )
    return {"persist_state": "session"} if version >= (1, 67) else {}


def render_sidebar(as_of: Any = None) -> None:
    """Render the shared sidebar. Call once from Home and every page.

    Widget values are bound to :func:`init_session_state` keys, so changing the
    risk profile or language immediately affects all pages.
    """
    init_session_state()

    with st.sidebar:
        st.title(config.label("app_title", st.session_state["language"]))
        st.caption(config.label("app_subtitle", st.session_state["language"]))
        st.divider()

        st.selectbox(
            t("language"),
            options=list(config.SUPPORTED_LANGUAGES),
            key="language",
            **persist_session_kwargs(),
        )

        st.radio(
            t("risk_profile"),
            options=list(config.RISK_PROFILES),
            format_func=lambda profile: t(f"profile_{profile}"),
            help=t("risk_profile_help"),
            key="risk_profile",
            **persist_session_kwargs(),
        )

        as_of_text = str(as_of) if as_of not in (None, "") else t("data_as_of_unknown")
        st.markdown(f"**{t('data_as_of')}:** {as_of_text}")

        st.divider()
        st.info(f"**{t('disclaimer_title')}**\n\n{t('disclaimer')}")
        st.caption(t("uncertainty"))


def render_header(title_key: str, intro_key: str | None = None) -> None:
    """Standard page header."""
    st.title(t(title_key))
    if intro_key:
        st.caption(t(intro_key))


def render_disclaimer() -> None:
    """Extra in-page reminder of the non-advice disclaimer."""
    st.info(f"**{t('disclaimer_title')}** — {t('disclaimer')}")


def section(title: str) -> None:
    st.subheader(title)


def render_empty_state() -> None:
    st.info(f"{t('no_data')}\n\n{t('no_data_hint')}")


def render_error(error: Exception | None = None, context: str | None = None) -> None:
    """Show a friendly error; log details to the terminal, never to the UI."""
    if error is not None:
        logger.exception("View failed to load%s", f" ({context})" if context else "")
    st.error(f"{t('error_title')}\n\n{t('error_hint')}")


def fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #

def _new_figure(title: str | None = None, height: int = config.CHART_HEIGHT) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=55, r=25, t=55 if title else 25, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(text=title, x=0.01, xanchor="left") if title else None,
    )
    return fig


def _draw(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # Streamlit < 1.50
        st.plotly_chart(fig, use_container_width=True)


def _dataframe(frame: pd.DataFrame, **kwargs: Any) -> None:
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:  # Streamlit < 1.50
        st.dataframe(frame, use_container_width=True, **kwargs)


def _is_empty(data: Any) -> bool:
    if data is None:
        return True
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.empty
    if isinstance(data, (dict, list, tuple, set)):
        return len(data) == 0
    return False


def _as_frame(data: Any) -> pd.DataFrame:
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def normalized_price_chart(data: pd.DataFrame) -> None:
    """Line chart of prices already rebased to 100 by the service layer."""
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(t("p1_normalized"))
    for column in frame.columns:
        color = config.COLORS.get(str(column), "#475569")
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=str(column),
                line=dict(color=color, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>" + str(column) + ": %{y:.1f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Base 100", gridcolor="#E2E8F0")
    _draw(fig)


def correlation_heatmap(correlation: pd.DataFrame) -> None:
    if _is_empty(correlation):
        render_empty_state()
        return
    frame = _as_frame(correlation)
    fig = go.Figure(
        data=go.Heatmap(
            z=frame.to_numpy(),
            x=[str(c) for c in frame.columns],
            y=[str(i) for i in frame.index],
            colorscale=[(0, "#DC2626"), (0.5, "#F8FAFC"), (1, "#2563EB")],
            zmin=-1,
            zmax=1,
            text=frame.round(2).to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        height=config.CHART_HEIGHT,
        margin=dict(l=60, r=25, t=45, b=45),
        title=dict(text=t("p1_correlation"), x=0.01, xanchor="left"),
    )
    _draw(fig)


def drawdown_chart(data: pd.DataFrame) -> None:
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(t("p1_drawdown"))
    for column in frame.columns:
        color = config.COLORS.get(str(column), "#475569")
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=str(column),
                line=dict(color=color, width=1.6),
                fill="tozeroy",
                fillcolor=_with_alpha(color, 0.08),
                hovertemplate="%{x|%Y-%m-%d}<br>" + str(column) + ": %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="%", gridcolor="#E2E8F0")
    _draw(fig)


def _with_alpha(hex_color: str, alpha: float) -> str:
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    except (ValueError, IndexError):
        return "rgba(100,116,139,0.1)"


def ml1_gauge(probability: Any) -> None:
    try:
        value = float(probability)
    except (TypeError, ValueError):
        render_empty_state()
        return
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number=dict(suffix="%", valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#64748B"),
                bar=dict(color=config.COLORS["accent"]),
                bgcolor="#F8FAFC",
                borderwidth=1,
                bordercolor="#CBD5E1",
                steps=[
                    {"range": [0, 15], "color": "#DCFCE7"},
                    {"range": [15, 35], "color": "#FEF9C3"},
                    {"range": [35, 100], "color": "#FEE2E2"},
                ],
                threshold=dict(line=dict(color="#B45309", width=3), value=35),
            ),
        )
    )
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        height=240,
        margin=dict(l=30, r=30, t=30, b=20),
    )
    _draw(fig)


def ml1_history_chart(history: pd.DataFrame, event_dates: Sequence[Any] = ()) -> None:
    """Plot the ML 1 probability time series and mark observed large drawdowns."""
    if _is_empty(history):
        render_empty_state()
        return
    frame = _as_frame(history)
    fig = _new_figure(t("ml1_history"))
    x = frame.index
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame.iloc[:, 0],
            mode="lines",
            name=t("ml1_current"),
            line=dict(color=config.COLORS["strategy"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
        )
    )
    marked = False
    for event in event_dates:
        if event not in frame.index:
            continue
        fig.add_vline(
            x=event,
            line=dict(color=config.COLORS["warning"], width=1, dash="dot"),
            annotation_text=t("ml1_marked") if not marked else None,
            annotation_position="top",
        )
        marked = True
    fig.update_yaxes(title_text="%", tickformat=".0%", range=[0, 1], gridcolor="#E2E8F0")
    _draw(fig)


def volatility_forecast_chart(data: pd.DataFrame) -> None:
    """Grouped bar chart: current forecast vs long-run average per asset."""
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(t("ml2_bar"))
    fig.add_trace(
        go.Bar(
            x=frame.index.astype(str),
            y=frame["forecast"],
            name=t("ml2_forecast"),
            marker_color=config.COLORS["strategy"],
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame.index.astype(str),
            y=frame["long_run"],
            name=t("ml2_longrun"),
            marker_color="#CBD5E1",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%", tickformat=".0%", gridcolor="#E2E8F0")
    _draw(fig)


def feature_importance_chart(data: pd.DataFrame, title: str) -> None:
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(title)
    fig.add_trace(
        go.Bar(
            x=frame.iloc[:, 1],
            y=frame.iloc[:, 0].astype(str),
            orientation="h",
            marker_color=config.COLORS["accent"],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="relative importance", gridcolor="#E2E8F0")
    fig.update_layout(height=max(220, 45 * len(frame) + 100))
    _draw(fig)


def weights_donut(weights: Mapping[str, float]) -> None:
    if _is_empty(weights):
        render_empty_state()
        return
    labels = [str(k) for k in weights]
    values = [float(v) for v in weights.values()]
    colors = [config.COLORS.get(k, "#64748B") for k in weights]
    fig = _new_figure(t("allocation_title"))
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.05))
    _draw(fig)


def weights_comparison_chart(comparison: pd.DataFrame) -> None:
    """Grouped bar chart of weights by scenario; services supplies the frame."""
    if _is_empty(comparison):
        render_empty_state()
        return
    frame = _as_frame(comparison)
    fig = _new_figure(t("comparison_title"))
    for i, scenario in enumerate(frame.columns):
        label = _l(str(scenario))
        fig.add_trace(
            go.Bar(
                x=[_l(str(a)) for a in frame.index],
                y=frame[scenario],
                name=label,
                marker_color=list(config.COLORS.values())[i % len(config.COLORS)],
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%", range=[0, 100], gridcolor="#E2E8F0")
    _draw(fig)


def equity_curve_chart(data: pd.DataFrame) -> None:
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(t("equity_curve"))
    for i, column in enumerate(frame.columns):
        canonical = str(column)
        label = _l(canonical)
        color = (
            config.COLORS["strategy"]
            if canonical == "strategy"
            else config.COLORS.get(f"benchmark_{i}", "#94A3B8")
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.2 if i == 0 else 1.5, dash="solid" if i == 0 else "dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>" + label + ": %{y:.1f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Base 100", gridcolor="#E2E8F0")
    _draw(fig)


def weights_area_chart(data: pd.DataFrame) -> None:
    if _is_empty(data):
        render_empty_state()
        return
    frame = _as_frame(data)
    fig = _new_figure(t("weights_over_time"))
    for column in frame.columns:
        color = config.COLORS.get(str(column), "#94A3B8")
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                stackgroup="one",
                name=str(column),
                line=dict(width=0.5, color=color),
                fillcolor=_with_alpha(color, 0.65),
                hovertemplate="%{x|%Y-%m-%d}<br>" + str(column) + ": %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="%", range=[0, 100], gridcolor="#E2E8F0")
    _draw(fig)


# --------------------------------------------------------------------------- #
# Tables and info blocks
# --------------------------------------------------------------------------- #

def weights_table(weights: Mapping[str, float]) -> None:
    if _is_empty(weights):
        render_empty_state()
        return
    frame = pd.DataFrame(
        {t("asset_" + str(k)): [float(v)] for k, v in weights.items()},
        index=[t("weight")],
    ).T
    _dataframe(frame.style.format({t("weight"): "{:.2f}%"}))


def metrics_table(metrics: pd.DataFrame) -> None:
    if _is_empty(metrics):
        render_empty_state()
        return
    frame = _as_frame(metrics).copy()
    if "metric" in frame.columns:
        frame["metric"] = frame["metric"].map(lambda value: _l(str(value)))
        frame = frame.set_index("metric")
    frame = frame.rename(columns=lambda column: _l(str(column)))
    _dataframe(frame.style.format("{:.2f}"))


def quality_table(metrics: pd.DataFrame) -> None:
    if _is_empty(metrics):
        render_empty_state()
        return
    frame = _as_frame(metrics).copy()
    if "metric" in frame.columns:
        frame["metric"] = frame["metric"].map(
            lambda value: _l("quality_" + str(value))
        )
    _dataframe(frame, hide_index=True)


def feature_table(features: pd.DataFrame, title: str) -> None:
    if _is_empty(features):
        render_empty_state()
        return
    st.markdown(f"**{title}**")
    _dataframe(_as_frame(features), hide_index=True)


def info_block(message: str) -> None:
    st.info(message)


def warning_block(message: str) -> None:
    st.warning(message)


def chip_list(values: Iterable[str]) -> str:
    return " · ".join(f"`{v}`" for v in values)
