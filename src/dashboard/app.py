"""
src/dashboard/app.py — LLM Cost Autopilot Dashboard

Run with:
    streamlit run src/dashboard/app.py

Features:
  - Headline metric cards (requests, cost, baseline, savings $, savings %)
  - Cost per day bar chart: actual vs baseline
  - Routing distribution pie chart
  - Quality score histogram with min-threshold line
  - Escalation rate line chart over time
  - Request audit table (last 50, colour-coded)
  - Live routing config panel (change tier→model without restart)
  - Retrain button + classifier metadata
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.database import (
    get_summary_stats,
    get_recent_requests,
    get_cost_timeseries,
    get_connection,
    init_db,
)
from src.config import load_registry

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOUR_ACTUAL   = "#4F8EF7"   # blue  — actual cost
COLOUR_BASELINE = "#F7824F"   # orange — what GPT-4o / highest-quality would cost
COLOUR_SAVINGS  = "#2ECC71"   # green  — savings
COLOUR_WARN     = "#F4A62A"   # amber  — escalated rows
COLOUR_BAD      = "#E74C3C"   # red    — low quality rows

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Metric card overrides */
[data-testid="stMetricValue"]          { font-size: 2rem !important; }
[data-testid="stMetricLabel"]          { font-size: 0.85rem !important; color: #888; }

/* Savings card — green tint */
.savings-card [data-testid="stMetricValue"] { color: #2ECC71 !important; }

/* Sidebar section headers */
.sidebar-section { font-weight: 700; font-size: 0.9rem;
                   text-transform: uppercase; letter-spacing: 0.05em;
                   color: #888; margin-top: 1.2rem; margin-bottom: 0.3rem; }

/* Audit table row colouring — applied via HTML */
.row-escalated { background-color: rgba(244, 166, 42, 0.15) !important; }
.row-bad       { background-color: rgba(231,  76, 60, 0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached DB queries — refresh every 60 s
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def _summary_stats() -> dict:
    try:
        init_db()
        return get_summary_stats()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def _recent_requests(limit: int = 50) -> list[dict]:
    try:
        return get_recent_requests(limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=60)
def _cost_timeseries(days: int = 7) -> list[dict]:
    try:
        return get_cost_timeseries(days=days)
    except Exception:
        return []


@st.cache_data(ttl=60)
def _escalation_timeseries(days: int = 7) -> pd.DataFrame:
    """Daily escalation rate (%) for the line chart."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    DATE(timestamp)                         AS day,
                    ROUND(AVG(escalated) * 100, 1)          AS escalation_rate,
                    COUNT(*)                                AS num_requests
                FROM requests
                WHERE timestamp > datetime('now', ? || ' days')
                GROUP BY day
                ORDER BY day
            """, (f"-{days}",)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _quality_distribution() -> list[float]:
    """Return list of quality_score values for histogram."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT quality_score FROM requests
                WHERE quality_score IS NOT NULL
            """).fetchall()
        return [r["quality_score"] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=60)
def _routing_distribution() -> dict[str, int]:
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT routed_model, COUNT(*) AS n
                FROM requests
                GROUP BY routed_model
                ORDER BY n DESC
            """).fetchall()
        return {r["routed_model"]: r["n"] for r in rows}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _pending_failures() -> int:
    try:
        with get_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) AS n FROM routing_failures WHERE used_in_retrain=0"
            ).fetchone()["n"]
    except Exception:
        return 0


@st.cache_data(ttl=60)
def _classifier_info() -> dict:
    try:
        from src.classifier.predict import get_model_info
        return get_model_info()
    except Exception:
        return {"model_name": "Not trained", "test_accuracy": None,
                "loaded": False, "model_path": ""}


def _clear_cache():
    _summary_stats.clear()
    _recent_requests.clear()
    _cost_timeseries.clear()
    _escalation_timeseries.clear()
    _quality_distribution.clear()
    _routing_distribution.clear()
    _pending_failures.clear()
    _classifier_info.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚀 Cost Autopilot")

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)
    days_back = st.slider("Days to show", min_value=1, max_value=30, value=7)
    tier_filter = st.multiselect(
        "Tier filter",
        options=[1, 2, 3],
        default=[1, 2, 3],
        format_func=lambda t: {1: "Tier 1 — Simple", 2: "Tier 2 — Moderate",
                                3: "Tier 3 — Complex"}[t],
    )

    # ── Live routing config ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Live Routing Config</div>',
                unsafe_allow_html=True)

    try:
        from src.router.router import get_router
        router  = get_router()
        current = router.get_routing_config()
        registry = load_registry()
        model_keys = list(registry.keys())
        model_labels = {k: registry[k].display_name for k in model_keys}

        def _model_index(key: str) -> int:
            return model_keys.index(key) if key in model_keys else 0

        new_t1 = st.selectbox(
            "Tier 1 model",
            options=model_keys,
            index=_model_index(current.get("tier_1_model", model_keys[0])),
            format_func=lambda k: model_labels[k],
            key="sidebar_t1",
        )
        new_t2 = st.selectbox(
            "Tier 2 model",
            options=model_keys,
            index=_model_index(current.get("tier_2_model", model_keys[0])),
            format_func=lambda k: model_labels[k],
            key="sidebar_t2",
        )
        new_t3 = st.selectbox(
            "Tier 3 model",
            options=model_keys,
            index=_model_index(current.get("tier_3_model", model_keys[0])),
            format_func=lambda k: model_labels[k],
            key="sidebar_t3",
        )

        if st.button("💾 Save routing config", use_container_width=True):
            try:
                router.update_routing({
                    "tier_1_model": new_t1,
                    "tier_2_model": new_t2,
                    "tier_3_model": new_t3,
                })
                st.success("✅ Routing updated — takes effect immediately")
                _clear_cache()
                st.rerun()
            except ValueError as e:
                st.error(f"Invalid config: {e}")

    except Exception as e:
        st.warning(f"Router unavailable: {e}")

    # ── Classifier info ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Classifier</div>',
                unsafe_allow_html=True)

    clf_info = _classifier_info()
    pending  = _pending_failures()

    if clf_info.get("loaded"):
        acc = clf_info.get("test_accuracy") or 0.0
        st.metric("Accuracy", f"{acc:.1%}")
        st.metric("Model", clf_info.get("model_name", "—"))
    else:
        st.warning("Classifier not loaded. Run `python -m src.classifier.train` first.")

    st.metric("Pending failures", pending,
              help="Routing failures not yet absorbed by retrain")

    if st.button("🔁 Retrain now", use_container_width=True,
                 disabled=pending == 0):
        with st.spinner("Retraining classifier…"):
            try:
                from scripts.retrain import retrain
                result = retrain()
                if result["model_replaced"]:
                    st.success(
                        f"✅ Model updated: {result['old_accuracy']:.1%} → "
                        f"{result['new_accuracy']:.1%} "
                        f"({result['n_failures']} new examples)"
                    )
                else:
                    st.info(f"ℹ️ Status: {result['status']}")
                _clear_cache()
                st.rerun()
            except Exception as e:
                st.error(f"Retrain failed: {e}")

    # ── Refresh ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Auto-refresh</div>',
                unsafe_allow_html=True)
    if st.button("🔄 Refresh now", use_container_width=True):
        _clear_cache()
        st.rerun()
    st.caption("Data auto-refreshes every 60 s via cache TTL.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main content
# ═══════════════════════════════════════════════════════════════════════════════

st.title("LLM Cost Autopilot")
st.caption(f"Last loaded: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

stats = _summary_stats()
if "error" in stats:
    st.error(f"Database error: {stats['error']}")
    st.stop()


# ── Section 1: Headline metric cards ──────────────────────────────────────────

st.subheader("Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total requests",
    f"{stats['total_requests']:,}",
)
col2.metric(
    "Actual cost",
    f"${stats['total_cost_usd']:.4f}",
)
col3.metric(
    "Baseline cost",
    f"${stats['total_baseline_cost']:.4f}",
    help="What you would have paid routing everything to the highest-quality model",
)

# Savings card — big green number, the project's headline
savings_usd = stats.get("savings_usd", 0.0)
savings_pct = stats.get("savings_pct", 0.0)
col4.metric(
    "💰 Saved",
    f"${savings_usd:.4f}",
    delta=f"{savings_pct:.1f}% cheaper",
    delta_color="normal",
)

col5.metric(
    "Avg quality score",
    f"{stats['avg_quality_score']:.1f}/5" if stats.get("avg_quality_score") else "—",
    help="Average LLM-as-judge score (1–5). Only populated after async verification runs.",
)

# Second row
col6, col7, col8 = st.columns(3)
col6.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
col7.metric(
    "Escalation rate (7d)",
    f"{stats['escalation_rate_pct']:.1f}%",
    help="% of requests where auto-escalation fired. >20% signals classifier needs retraining.",
)
col8.metric(
    "Requests by tier",
    " / ".join(
        f"T{t}:{n}" for t, n in sorted(stats.get("requests_by_tier", {}).items())
    ) or "—",
)

st.divider()


# ── Section 2: Cost per day bar chart ─────────────────────────────────────────

st.subheader("Daily cost: actual vs baseline")

ts_rows = _cost_timeseries(days=days_back)

if ts_rows:
    df_ts = pd.DataFrame(ts_rows)
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=df_ts["day"], y=df_ts["cost_actual"],
        name="Actual cost",
        marker_color=COLOUR_ACTUAL,
        hovertemplate="$%{y:.6f}<extra>Actual</extra>",
    ))
    fig_cost.add_trace(go.Bar(
        x=df_ts["day"], y=df_ts["cost_baseline"],
        name="Baseline (highest-quality model)",
        marker_color=COLOUR_BASELINE,
        opacity=0.6,
        hovertemplate="$%{y:.6f}<extra>Baseline</extra>",
    ))
    fig_cost.update_layout(
        barmode="group",
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_cost, use_container_width=True)
else:
    st.info("No requests logged yet — send some prompts to see cost trends.")

st.divider()


# ── Section 3: Routing distribution + Quality histogram (side by side) ────────

col_pie, col_hist = st.columns(2)

with col_pie:
    st.subheader("Routing distribution")
    routing_dist = _routing_distribution()

    if routing_dist:
        # Use display names where available
        try:
            reg = load_registry()
            labels = [reg[k].display_name if k in reg else k for k in routing_dist]
        except Exception:
            labels = list(routing_dist.keys())

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=list(routing_dist.values()),
            hole=0.4,
            marker_colors=[COLOUR_ACTUAL, COLOUR_BASELINE, COLOUR_SAVINGS],
            hovertemplate="%{label}: %{value} requests (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No routing data yet.")

with col_hist:
    st.subheader("Quality score distribution")
    quality_scores = _quality_distribution()
    min_cheap_score = 3.0  # from routing.yaml default

    try:
        from src.config import load_routing
        min_cheap_score = load_routing()["quality"]["min_cheap_score"]
    except Exception:
        pass

    if quality_scores:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=quality_scores,
            nbinsx=20,
            marker_color=COLOUR_ACTUAL,
            name="Quality scores",
            hovertemplate="Score %{x}: %{y} requests<extra></extra>",
        ))
        fig_hist.add_vline(
            x=min_cheap_score,
            line_dash="dash",
            line_color=COLOUR_BAD,
            annotation_text=f"Min threshold ({min_cheap_score})",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            xaxis_title="Score (1–5)",
            yaxis_title="Count",
            xaxis=dict(range=[0.5, 5.5]),
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info(
            "No quality scores yet. Scores appear after the async verifier runs "
            "(a few seconds after each request)."
        )

st.divider()


# ── Section 4: Escalation rate over time ──────────────────────────────────────

st.subheader("Escalation rate over time")

df_esc = _escalation_timeseries(days=days_back)

if not df_esc.empty and "escalation_rate" in df_esc.columns:
    fig_esc = go.Figure()
    fig_esc.add_trace(go.Scatter(
        x=df_esc["day"],
        y=df_esc["escalation_rate"],
        mode="lines+markers",
        line=dict(color=COLOUR_WARN, width=2),
        marker=dict(size=6),
        name="Escalation rate %",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    # 20% alert threshold line
    fig_esc.add_hline(
        y=20,
        line_dash="dash",
        line_color=COLOUR_BAD,
        annotation_text="20% alert threshold",
        annotation_position="bottom right",
    )
    fig_esc.update_layout(
        xaxis_title="Date",
        yaxis_title="Escalation rate (%)",
        yaxis=dict(range=[0, max(25, df_esc["escalation_rate"].max() + 5)]),
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_esc, use_container_width=True)
    if df_esc["escalation_rate"].max() > 20:
        st.warning(
            "⚠️ Escalation rate exceeded 20% — consider triggering a retrain "
            "using the sidebar button."
        )
else:
    st.info("No escalation data yet.")

st.divider()


# ── Section 5: Request audit table ────────────────────────────────────────────

st.subheader("Recent requests (last 50)")

rows = _recent_requests(limit=50)

if rows:
    df = pd.DataFrame(rows)

    # Apply tier filter from sidebar
    if tier_filter and "complexity_tier" in df.columns:
        df = df[df["complexity_tier"].isin(tier_filter)]

    # Format columns for display
    display_cols = {
        "timestamp":              "Time",
        "prompt_preview":         "Prompt",
        "complexity_tier":        "Tier",
        "routed_model":           "Model",
        "cost_usd":               "Cost ($)",
        "cost_if_highest_quality":"Baseline ($)",
        "latency_ms":             "Latency (ms)",
        "quality_score":          "Quality",
        "escalated":              "Escalated",
    }
    df_display = df[[c for c in display_cols if c in df.columns]].copy()
    df_display = df_display.rename(columns=display_cols)

    # Format numeric columns
    for col in ["Cost ($)", "Baseline ($)"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x:.6f}" if pd.notna(x) else "—"
            )
    if "Latency (ms)" in df_display.columns:
        df_display["Latency (ms)"] = df_display["Latency (ms)"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "—"
        )
    if "Quality" in df_display.columns:
        df_display["Quality"] = df_display["Quality"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
    if "Escalated" in df_display.columns:
        df_display["Escalated"] = df_display["Escalated"].apply(
            lambda x: "⚠️ Yes" if x else ""
        )
    if "Time" in df_display.columns:
        df_display["Time"] = pd.to_datetime(
            df_display["Time"], errors="coerce"
        ).dt.strftime("%H:%M:%S")
    if "Prompt" in df_display.columns:
        df_display["Prompt"] = df_display["Prompt"].apply(
            lambda x: (x[:60] + "…") if x and len(str(x)) > 60 else x
        )

    # Colour-code rows
    def _row_style(row):
        if row.get("Escalated") == "⚠️ Yes":
            return [f"background-color: rgba(244,166,42,0.15)"] * len(row)
        quality_raw = row.get("Quality", "—")
        try:
            if float(quality_raw) < 3.0:
                return [f"background-color: rgba(231,76,60,0.15)"] * len(row)
        except (ValueError, TypeError):
            pass
        return [""] * len(row)

    styled = df_display.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, height=420)

    # Legend
    st.caption(
        "🟡 Amber = escalated row &nbsp;&nbsp; 🔴 Red = quality score < 3.0"
    )
else:
    st.info("No requests logged yet.")

# ── Auto-rerun every 60 s ──────────────────────────────────────────────────────
time.sleep(60)
st.rerun()
