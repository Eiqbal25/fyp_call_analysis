"""
dashboard/app.py
=================
Phase 5: Visualization & QA Dashboard

Streamlit web-based QA dashboard that renders:
  - Executive Summary KPIs (calls, avg quality score, avg sentiment, alerts)
  - Overall Quality Distribution chart
  - Agent vs. Customer Talk Ratio chart
  - Sentiment Volatility by Role
  - Compliance Component Adherence
  - Top 5 Calls by Quality Score
  - Per-call granular analysis: diarized transcript, sentiment trajectory,
    compliance flags, method comparison

Launch with:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DASHBOARD_TITLE, OUTPUTS_DIR,
    AGENT_COLOR, CUSTOMER_COLOR,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #7dd3fc; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .alert-high   { background: #3f0d0d; border-left: 4px solid #ef4444; padding: 0.5rem; }
    .alert-medium { background: #3f2d0d; border-left: 4px solid #f59e0b; padding: 0.5rem; }
    .transcript-agent    { color: #60a5fa; }
    .transcript-customer { color: #f87171; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────

def load_results_json(results_path: str) -> dict:
    """
    Load pipeline output JSON from disk.
    Not cached — always reads fresh so dashboard updates after each main.py run.
    Use st.button('Refresh') or re-select the call to trigger a reload.
    """
    if not os.path.isfile(results_path):
        return {}
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Could not parse results JSON: {e}")
        return {}


def _get_all_calls(data: dict) -> list[str]:
    return list(data.get("calls", {}).keys())


def _get_call_data(data: dict, call_id: str) -> dict:
    return data.get("calls", {}).get(call_id, {})


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(data: dict) -> str:
    st.sidebar.title("📞 Call QA Navigator")
    st.sidebar.markdown("---")

    results_path = st.sidebar.text_input(
        "Results JSON path",
        value=os.path.join(OUTPUTS_DIR, "pipeline_results.json"),
    )

    call_ids = _get_all_calls(data)
    if not call_ids:
        st.sidebar.warning("No calls loaded. Run main.py first.")
        return None

    # Filter by risk
    risk_filter = st.sidebar.multiselect(
        "Filter by Risk Severity",
        options=["All", "High", "Medium", "Low", "Clean"],
        default=["All"],
    )

    # Rating filter
    rating_filter = st.sidebar.multiselect(
        "Filter by QA Rating",
        options=["All", "Good", "Fair", "Needs Improvement"],
        default=["All"],
    )

    # Apply filters
    filtered = []
    for cid in call_ids:
        cd = _get_call_data(data, cid)
        risk   = cd.get("compliance", {}).get("risk_severity",  "Clean")
        rating = cd.get("qa_result",  {}).get("rating",         "Fair")
        if ("All" in risk_filter   or risk   in risk_filter) and \
           ("All" in rating_filter or rating in rating_filter):
            filtered.append(cid)

    selected_call = st.sidebar.selectbox("Select Call to Audit", filtered or call_ids)
    st.sidebar.markdown("---")
    st.sidebar.caption("FYP1 | Speaker Role Detection & Segmented Analysis")
    return selected_call


# ─────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY KPIs
# ─────────────────────────────────────────────────────────────

def render_kpis(data: dict):
    calls = data.get("calls", {})
    if not calls:
        return

    n_calls      = len(calls)
    qa_scores    = [c.get("qa_result",   {}).get("qa_score",    50) for c in calls.values()]
    cust_sents   = [c.get("sentiment",   {}).get("customer_avg_compound", 0) for c in calls.values()]
    high_risks   = sum(1 for c in calls.values()
                       if c.get("compliance", {}).get("risk_severity") == "High")

    avg_qa   = np.mean(qa_scores)   if qa_scores  else 0
    avg_sent = np.mean(cust_sents) if cust_sents else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_calls}</div>
            <div class="metric-label">Calls Analysed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_qa:.1f}%</div>
            <div class="metric-label">Avg Quality Score</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        sent_str = f"{avg_sent:+.3f}"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{sent_str}</div>
            <div class="metric-label">Avg Cust Sentiment</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ef4444">{high_risks}</div>
            <div class="metric-label">Critical Alerts</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")


# ─────────────────────────────────────────────────────────────
# OVERVIEW CHARTS
# ─────────────────────────────────────────────────────────────

def render_overview_charts(data: dict):
    calls = data.get("calls", {})
    if not calls:
        return

    call_ids  = list(calls.keys())
    qa_scores = [calls[c].get("qa_result", {}).get("qa_score",    50) for c in call_ids]
    ratings   = [calls[c].get("qa_result", {}).get("rating",    "Fair") for c in call_ids]
    agent_pcts  = [calls[c].get("talk_ratio", {}).get("agent_talk_pct",    50) for c in call_ids]
    cust_pcts   = [calls[c].get("talk_ratio", {}).get("customer_talk_pct", 50) for c in call_ids]

    col1, col2 = st.columns(2)

    with col1:
        rating_counts = pd.Series(ratings).value_counts()
        fig = px.pie(
            names=rating_counts.index,
            values=rating_counts.values,
            title="Overall Quality Distribution",
            color=rating_counts.index,
            color_discrete_map={
                "Good": "#2ca02c",
                "Fair": "#ff7f0e",
                "Needs Improvement": "#d62728",
            },
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=agent_pcts, y=qa_scores,
            mode="markers+text",
            text=call_ids,
            textposition="top center",
            marker=dict(size=12, color=qa_scores, colorscale="RdYlGn",
                        showscale=True, colorbar=dict(title="QA Score")),
        ))
        fig2.update_layout(
            title="Agent vs Customer Talk Ratio",
            xaxis_title="Agent Talk % ",
            yaxis_title="QA Score",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Compliance adherence chart
    compliance_keys = {
        "Greeting":   "greeting_passed",
        "Closing":    "closing_passed",
        "Recorded":   "recorded_passed",
        "Identity":   "identity_passed",
    }
    n_calls = len(calls)
    comp_rates = {
        label: sum(1 for c in calls.values()
                   if c.get("compliance", {}).get(key, False)) / n_calls * 100
        for label, key in compliance_keys.items()
    }

    fig3 = go.Figure(go.Bar(
        x=list(comp_rates.keys()),
        y=list(comp_rates.values()),
        marker_color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"],
        text=[f"{v:.1f}%" for v in comp_rates.values()],
        textposition="outside",
    ))
    fig3.update_layout(
        title="Compliance Component Adherence (%)",
        yaxis=dict(range=[0, 115]),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PER-CALL GRANULAR VIEW
# ─────────────────────────────────────────────────────────────

def render_call_detail(call_data: dict, call_id: str):
    st.markdown(f"## 🔍 Granular Analysis: `{call_id}`")

    qa     = call_data.get("qa_result",  {})
    talk   = call_data.get("talk_ratio", {})
    sent   = call_data.get("sentiment",  {})
    comp   = call_data.get("compliance", {})
    hybrid = call_data.get("transcript_hybrid", [])

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QA Score",       f"{qa.get('qa_score', 0):.1f} / 100",
              delta=qa.get("rating", ""))
    c2.metric("Agent Talk",     f"{talk.get('agent_talk_pct', 0):.1f}%")
    c3.metric("Cust Sentiment", f"{sent.get('customer_avg_compound', 0):+.3f}",
              delta=sent.get("sentiment_trend", ""))
    c4.metric("Compliance",     f"{comp.get('compliance_score', 0)*100:.0f}%",
              delta=comp.get("compliance_label", ""))

    st.markdown("---")

    # ── Alerts
    if comp.get("risk_severity") in ("High", "Medium"):
        sev = comp.get("risk_severity")
        cls = "alert-high" if sev == "High" else "alert-medium"
        kws = ", ".join(comp.get("risk_keywords_found", []))
        st.markdown(
            f'<div class="{cls}">🚨 <b>{sev} Risk</b> — Keywords detected: {kws}</div>',
            unsafe_allow_html=True)
        st.markdown("")

    for note in comp.get("notes", []):
        st.info(note)

    # ── Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Transcript", "📈 Sentiment", "✅ Compliance", "📊 Method Comparison"])

    with tab1:
        render_transcript(hybrid)

    with tab2:
        render_sentiment_chart(sent, call_id)

    with tab3:
        render_compliance_detail(comp)

    with tab4:
        render_method_comparison(call_data)


def render_transcript(segments: list[dict]):
    if not segments:
        st.warning("No transcript available.")
        return

    st.markdown("**Diarized Transcript** (🔵 Agent | 🔴 Customer)")
    for seg in segments:
        role  = seg.get("predicted_role", "Unknown")
        text  = seg.get("text", "")
        ts    = seg.get("start", 0)
        conf  = seg.get("final_confidence", seg.get("confidence", 0))
        sent_score = seg.get("sentiment_compound", 0)

        if role == "Agent":
            icon  = "🔵"
            color = AGENT_COLOR
        elif role == "Customer":
            icon  = "🔴"
            color = CUSTOMER_COLOR
        else:
            icon  = "⚪"
            color = "#888"

        st.markdown(
            f"{icon} **{role}** `[{ts:.1f}s]` _(conf: {conf:.2f} | "
            f"sent: {sent_score:+.2f})_  \n{text}",
        )


def render_sentiment_chart(sent: dict, call_id: str):
    traj_cust  = sent.get("trajectory_customer", [])
    traj_agent = sent.get("trajectory_agent",    [])

    if not traj_cust and not traj_agent:
        st.warning("No sentiment trajectory data available.")
        return

    fig = go.Figure()

    if traj_cust:
        ts_c, sc_c = zip(*traj_cust)
        fig.add_trace(go.Scatter(
            x=ts_c, y=sc_c, mode="lines+markers",
            name="Customer", line=dict(color=CUSTOMER_COLOR, width=2),
            marker=dict(size=6),
        ))

    if traj_agent:
        ts_a, sc_a = zip(*traj_agent)
        fig.add_trace(go.Scatter(
            x=ts_a, y=sc_a, mode="lines+markers",
            name="Agent", line=dict(color=AGENT_COLOR, width=2, dash="dash"),
            marker=dict(size=6, symbol="square"),
        ))

    fig.add_hline(y=0,     line_dash="dot",  line_color="gray")
    fig.add_hline(y=0.05,  line_dash="dash", line_color="green",  opacity=0.3)
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red",    opacity=0.3)

    fig.update_layout(
        title=f"Sentiment Trajectory — {call_id} ({sent.get('sentiment_trend', 'Stable')})",
        xaxis_title="Time (seconds)",
        yaxis_title="VADER Compound Score",
        yaxis=dict(range=[-1.1, 1.1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Agent Avg",    f"{sent.get('agent_avg_compound',    0):+.3f}",
                delta=sent.get("agent_polarity_label", ""))
    col2.metric("Customer Avg", f"{sent.get('customer_avg_compound', 0):+.3f}",
                delta=sent.get("customer_polarity_label", ""))
    col3.metric("Interpretation", sent.get("interpretation", "—"))


def render_compliance_detail(comp: dict):
    checks = {
        "Greeting":                 comp.get("greeting_passed",  False),
        "Closing":                  comp.get("closing_passed",   False),
        "Recorded Disclaimer":      comp.get("recorded_passed",  False),
        "Identity Verification":    comp.get("identity_passed",  False),
    }
    col1, col2 = st.columns(2)
    for i, (label, passed) in enumerate(checks.items()):
        col = col1 if i % 2 == 0 else col2
        icon = "✅" if passed else "❌"
        col.markdown(f"{icon} **{label}**")

    st.markdown("---")
    st.markdown(f"**Compliance Score:** `{comp.get('compliance_score', 0)*100:.0f}%` "
                f"— {comp.get('compliance_label', '')}")

    if comp.get("risk_keywords_found"):
        st.markdown(f"**Risk Keywords Found:** {', '.join(comp['risk_keywords_found'])}")

    if comp.get("non_compliant_items"):
        st.markdown(f"**Non-Compliant Items:** {', '.join(comp['non_compliant_items'])}")


def render_method_comparison(call_data: dict):
    metrics = call_data.get("method_metrics", {})
    if not metrics:
        st.info("Method comparison metrics not available for this call.")
        return

    rows = []
    for method, m in metrics.items():
        rows.append({
            "Method":    method,
            "Accuracy":  f"{m.get('accuracy', 0):.1f}%",
            "F1-Score":  f"{m.get('f1', 0):.1f}%",
            "Precision": f"{m.get('precision', 0):.1f}%",
            "Recall":    f"{m.get('recall', 0):.1f}%",
        })
    st.table(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.title(f"📞 {DASHBOARD_TITLE}")

    # Load results — with manual refresh button
    results_path = os.path.join(OUTPUTS_DIR, "pipeline_results.json")

    col_refresh, col_info = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Results"):
            st.rerun()
    with col_info:
        if os.path.isfile(results_path):
            mtime = os.path.getmtime(results_path)
            import datetime
            st.caption(
                f"Results last updated: "
                f"{datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            )

    data = load_results_json(results_path)

    if not data:
        st.warning(
            "⚠️ No results found. Please run **`python main.py`** first to process calls.\n\n"
            f"Results are saved to: `{results_path}`"
        )
        st.stop()

    selected_call = render_sidebar(data)
    if selected_call is None:
        st.stop()

    # Executive Summary
    st.subheader("📊 Executive Summary Dashboard")
    render_kpis(data)

    st.markdown("---")
    st.subheader("📈 Overview Analytics")
    render_overview_charts(data)

    st.markdown("---")
    call_data = _get_call_data(data, selected_call)
    if call_data:
        render_call_detail(call_data, selected_call)
    else:
        st.warning(f"No data found for call: {selected_call}")

    # Validation summary if available
    validation = data.get("validation_results", {})
    if validation:
        st.markdown("---")
        st.subheader("🔬 Validation Results")

        agg = validation.get("aggregated", {})
        cols = st.columns(3)
        for i, (mname, metrics) in enumerate(agg.items()):
            with cols[i]:
                label = {"m1": "Method 1 (Keyword)",
                         "m2": "Method 2 (Acoustic)",
                         "m3": "Method 3 (Hybrid)"}.get(mname, mname)
                st.metric(label, f"Acc: {metrics.get('accuracy', 0):.1f}%",
                          delta=f"F1: {metrics.get('f1', 0):.1f}%")

        ttest = validation.get("ttest", {})
        if ttest.get("conclusion"):
            st.info(f"**Statistical Test:** {ttest['conclusion']}")


if __name__ == "__main__":
    main()
