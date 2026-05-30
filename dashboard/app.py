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
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Could not parse results JSON: {e}")
        return {}


def _normalise_calls(data: dict) -> dict:
    """Convert calls list → dict if needed."""
    calls = data.get("calls", {})
    if isinstance(calls, list):
        return {c["call_id"]: c for c in calls if isinstance(c, dict)}
    return calls


def _get_all_calls(data: dict) -> list:
    return sorted(_normalise_calls(data).keys())


def _get_call_data(data: dict, call_id: str) -> dict:
    return _normalise_calls(data).get(call_id, {})


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(data: dict) -> str:
    st.sidebar.title("📞 Call QA Navigator")
    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Dashboard mode",
        ["🔧 System (Production)", "🎙️ Live Analysis", "🎓 FYP Evaluation"],
        horizontal=False,
    )
    st.session_state["dashboard_mode"] = mode
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
    calls = _normalise_calls(data)
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
    calls = _normalise_calls(data)
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
    hybrid = call_data.get("transcript_m3", [])

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QA Score",       f"{qa.get('qa_score', 0):.1f} / 100",
              delta=qa.get("rating", ""))
    c2.metric("Agent Talk",     f"{talk.get('agent_talk_pct', 0):.1f}%")
    c3.metric("Cust Sentiment", f"{sent.get('customer_avg_compound', 0):+.3f}",
              delta=sent.get("sentiment_trend", ""))
    c4.metric("Compliance",     f"{comp.get('compliance_score', 0)*100:.0f}%",
              delta=comp.get("compliance_label", ""))

    # ── Outcome + Rude Behavior row ───────────────────────────
    outcome = call_data.get("call_outcome", {})
    rude    = call_data.get("rude_behavior", {})
    if outcome or rude:
        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        if outcome:
            emoji  = outcome.get("emoji", "✅")
            o_text = outcome.get("outcome", "Unknown")
            r1.metric("Call Outcome", f"{emoji} {o_text}")
        if rude:
            a_level = rude.get("agent_rudeness_level", "NONE")
            c_level = rude.get("customer_rudeness_level", "NONE")
            a_color = "🔴" if a_level=="HIGH" else "🟡" if a_level=="MEDIUM" else "🟢"
            c_color = "🔴" if c_level=="HIGH" else "🟡" if c_level=="MEDIUM" else "🟢"
            r2.metric("Agent Behavior", f"{a_color} {a_level}")
            r3.metric("Customer Behavior", f"{c_color} {c_level}")
            a_count = rude.get("total_agent_incidents", 0)
            c_count = rude.get("total_customer_incidents", 0)
            r4.metric("Incidents", f"A:{a_count} C:{c_count}")
        # Show warnings
        if rude and rude.get("warnings"):
            for w in rude["warnings"]:
                if "CRITICAL" in w:
                    st.error(w)
                elif "WARNING" in w:
                    st.warning(w)
                elif "NOTE" in w:
                    st.info(w)

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
    # ── Call Summary ──────────────────────────────────────────
    summary = call_data.get("call_summary", {})
    if summary and any(summary.values()):
        st.markdown("---")
        st.markdown("### 📋 Call Summary")
        col1, col2, col3 = st.columns(3)
        col1.info(f"**📌 Topic**\n\n{summary.get('topic', 'N/A')}")
        col2.info(f"**📝 Summary**\n\n{summary.get('summary', 'N/A')}")
        col3.info(f"**✅ Outcome**\n\n{summary.get('outcome', 'N/A')}")

    # ── Advanced Analytics Row ───────────────────────────────
    rt   = call_data.get("response_time", {})
    intr = call_data.get("interruptions", {})
    lang = call_data.get("language", {})
    wer  = call_data.get("wer", {})

    if rt or intr or lang:
        st.markdown("---")
        a1, a2, a3, a4 = st.columns(4)
        if lang:
            a1.metric("🌐 Language", lang.get("detected_language", "Unknown"))
        if rt and rt.get("avg_response_time_sec") is not None:
            a2.metric("⏱ Avg Response Time",
                      f"{rt['avg_response_time_sec']:.2f}s",
                      delta=rt.get("rating", ""))
        if intr:
            a3.metric("🗣 Interruptions",
                      f"{intr.get('total_interruptions', 0)} total",
                      delta=f"Agent: {intr.get('agent_interruptions',0)} | Cust: {intr.get('customer_interruptions',0)}")
        if wer and wer.get("wer") is not None:
            wer_val = wer["wer"]
            wer_quality = "Excellent" if wer_val < 10 else "Good" if wer_val < 25 else "Fair" if wer_val < 40 else "Poor"
            a4.metric("📝 WER", f"{wer_val:.1f}%", delta=wer_quality)

    render_speaker_timeline(hybrid, call_id)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💬 Transcript", "📈 Sentiment", "✅ Compliance",
         "📊 Method Comparison", "⚠️ Rude Behavior"])

    with tab1:
        view_mode = st.radio(
            "View", ["Method 3 — Final", "M1 vs M2 vs M3 Comparison"],
            horizontal=True, key=f"view_{call_id}"
        )
        if view_mode == "Method 3 — Final":
            render_transcript(hybrid)
        else:
            render_method_transcript_comparison(call_data)

    with tab2:
        render_sentiment_chart(sent, call_id)

    with tab3:
        render_compliance_detail(comp)

    with tab4:
        render_method_comparison(call_data)

    with tab5:
        render_rude_timeline(hybrid, rude, call_id)


def render_speaker_timeline(segments: list, call_id: str):
    """Horizontal bar chart showing who spoke when."""
    if not segments:
        return
    agent_segs    = [(s["start"], s.get("end", s["start"] + s.get("duration", 1)))
                     for s in segments if s.get("predicted_role") == "Agent"]
    customer_segs = [(s["start"], s.get("end", s["start"] + s.get("duration", 1)))
                     for s in segments if s.get("predicted_role") == "Customer"]
    total = max((s.get("end", s["start"]) for s in segments), default=1)
    fig = go.Figure()
    for start, end in agent_segs:
        fig.add_shape(type="rect", x0=start, x1=end, y0=0.55, y1=0.95,
                      fillcolor=AGENT_COLOR, opacity=0.8, line_width=0)
    for start, end in customer_segs:
        fig.add_shape(type="rect", x0=start, x1=end, y0=0.05, y1=0.45,
                      fillcolor=CUSTOMER_COLOR, opacity=0.8, line_width=0)
    fig.update_layout(
        title=f"Speaker Timeline — {call_id}",
        xaxis=dict(title="Time (seconds)", range=[0, total]),
        yaxis=dict(tickvals=[0.25, 0.75], ticktext=["Customer", "Agent"], range=[0, 1]),
        height=180, margin=dict(l=100, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_accuracy_overview(data: dict):
    """Bar chart comparing accuracy across all calls."""
    calls_raw = data.get("calls", [])
    # Handle both list and dict formats
    if isinstance(calls_raw, dict):
        calls = list(calls_raw.values())
    else:
        calls = calls_raw
    if not calls:
        return
    call_ids, accuracies = [], []
    for call in sorted(calls, key=lambda x: x.get("call_id", "")):
        cid = call.get("call_id", "")
        m3  = call.get("method_metrics", {}).get("Method3-LLM", {})
        acc = m3.get("accuracy", 0)
        call_ids.append(cid.replace("_", " "))
        accuracies.append(acc)
    if not any(accuracies):
        return
    fig = px.bar(
        x=accuracies, y=call_ids, orientation="h",
        title="Classification Accuracy per Call (Method 3 LLM)",
        labels={"x": "Accuracy (%)", "y": ""},
        color=accuracies,
        color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
        range_color=[50, 100],
    )
    fig.add_vline(x=90, line_dash="dash", line_color="green",
                  annotation_text="90% target")
    fig.update_layout(height=520, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_rude_timeline(segments: list, rude_result: dict, call_id: str):
    """Show rude behavior incidents on the call timeline."""
    if not rude_result:
        return
    incidents = (rude_result.get("agent_incidents", []) +
                 rude_result.get("customer_incidents", []))
    if not incidents:
        st.success("No rude behavior incidents detected in this call.")
        return
    total = max((s.get("end", s["start"]) for s in segments), default=1)
    fig = go.Figure()
    for inc in incidents:
        ts    = inc.get("timestamp", 0)
        role  = inc.get("role", "Agent")
        sev   = inc.get("severity", "MEDIUM")
        color = "#c0392b" if sev == "HIGH" else "#e67e22"
        y_pos = 0.7 if role == "Agent" else 0.3
        txt   = inc.get("text", "")[:50]
        fig.add_trace(go.Scatter(
            x=[ts], y=[y_pos], mode="markers",
            marker=dict(size=16 if sev == "HIGH" else 10, color=color, symbol="x"),
            hovertemplate=f"[{ts}s] {role} {sev}<br>{txt}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        title=f"Rude Behavior Incidents — {call_id}",
        xaxis=dict(title="Time (seconds)", range=[0, total]),
        yaxis=dict(tickvals=[0.3, 0.7], ticktext=["Customer", "Agent"], range=[0, 1]),
        height=200, margin=dict(l=100, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Incident Details:**")
    for inc in incidents:
        icon = "🔴" if inc["severity"] == "HIGH" else "🟡"
        ts_v = inc["timestamp"]
        ro_v = inc["role"]
        se_v = inc["severity"]
        tx_v = inc["text"][:60]
        st.markdown(f"{icon} `[{ts_v}s]` **{ro_v}** — {se_v} | {tx_v}")



def render_method_transcript_comparison(call_data: dict):
    """Show M1 vs M2 vs M3 labels side by side for every segment."""
    import pandas as pd

    m1_segs = call_data.get("transcript_m1") or call_data.get("method1", {}).get("classified", [])
    m2_segs = call_data.get("transcript_m2") or call_data.get("method2", {}).get("classified", [])
    m3_segs = call_data.get("transcript_m3") or call_data.get("method3", {}).get("classified", [])

    if not m3_segs:
        st.warning("No segments found.")
        return

    m1_lookup = {s.get("text","").strip().lower(): s.get("predicted_role","?") for s in m1_segs}
    m2_lookup = {s.get("text","").strip().lower(): s.get("predicted_role","?") for s in m2_segs}

    rows = []
    for seg in m3_segs:
        text    = seg.get("text", "").strip()
        key     = text.lower()
        m1_role = m1_lookup.get(key, "?")
        m2_role = m2_lookup.get(key, "?")
        m3_role = seg.get("predicted_role", "?")

        def icon(r):
            return "🔵 Agent" if r == "Agent" else "🔴 Customer" if r == "Customer" else "❓"

        agree = "✅" if m1_role == m3_role and m2_role == m3_role else (
                "⚠️" if m1_role != m3_role or m2_role != m3_role else "✅")

        rows.append({
            "Time":      f"{seg.get('start', 0):.1f}s",
            "Text":      text[:60] + "..." if len(text) > 60 else text,
            "Method 1":  icon(m1_role),
            "Method 2":  icon(m2_role),
            "Method 3":  icon(m3_role),
            "Agreement": agree,
        })

    df = pd.DataFrame(rows)

    # Summary
    if rows:
        agree_count = sum(1 for r in rows if r["Agreement"] == "✅")
        disagree    = len(rows) - agree_count
        c1, c2, c3 = st.columns(3)
        c1.metric("Total segments", len(rows))
        c2.metric("All methods agree", agree_count)
        c3.metric("Methods disagree", disagree)

    st.dataframe(df, use_container_width=True, height=500)
    st.caption("🔵 Agent | 🔴 Customer | ✅ All agree | ⚠️ Methods disagree")


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
    import pandas as pd

    m1_stats = call_data.get("method1", {}).get("stats", {})
    m2_stats = call_data.get("method2", {}).get("stats", {})
    m3_stats = call_data.get("method3", {}).get("stats", {})

    if not any([m1_stats, m2_stats, m3_stats]):
        st.info("Method comparison metrics not available for this call.")
        return

    def fmt(stats):
        return {
            "Segments":        stats.get("n_segments", 0),
            "Avg Confidence":  f"{stats.get('mean_confidence', 0):.3f}",
            "Min Confidence":  f"{stats.get('min_confidence', 0):.3f}",
            "Max Confidence":  f"{stats.get('max_confidence', 0):.3f}",
            "Std Confidence":  f"{stats.get('std_confidence', 0):.3f}",
        }

    rows = []
    if m1_stats:
        rows.append({"Method": "Method 1 — Keyword/Lexical", **fmt(m1_stats)})
    if m2_stats:
        rows.append({"Method": "Method 2 — Acoustic DNN",    **fmt(m2_stats)})
    if m3_stats:
        rows.append({"Method": "Method 3 — LLM (Llama 3.3 70B)", **fmt(m3_stats)})

    df = pd.DataFrame(rows).set_index("Method")

    # Show agent/customer distribution per method
    st.subheader("📊 Confidence Statistics per Method")
    st.dataframe(df, use_container_width=True)

    # Show label distribution
    st.subheader("🏷️ Label Distribution")
    col1, col2, col3 = st.columns(3)
    for col, key, name in [
        (col1, "method1", "Method 1"),
        (col2, "method2", "Method 2"),
        (col3, "method3", "Method 3 (LLM)"),
    ]:
        segs = call_data.get(key, {}).get("classified", [])
        if segs:
            agents    = sum(1 for s in segs if s.get("predicted_role") == "Agent")
            customers = sum(1 for s in segs if s.get("predicted_role") == "Customer")
            with col:
                st.metric(f"{name} — Agent",    agents)
                st.metric(f"{name} — Customer", customers)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# LIVE ANALYSIS — Upload WAV and run full pipeline
# ─────────────────────────────────────────────────────────────

def run_live_analysis(wav_bytes: bytes, filename: str) -> dict:
    """Run full pipeline on uploaded WAV file."""
    import tempfile, sys, importlib
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Save uploaded file to temp
    call_id = filename.replace(".wav", "").replace(".mp3", "")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    progress = st.progress(0, text="Starting analysis...")
    status   = st.empty()

    try:
        # ── Phase 1: Preprocessing ────────────────────────────
        status.info("⚙️ Phase 1: Audio preprocessing...")
        progress.progress(10, text="Preprocessing audio...")
        from preprocessing.audio_processor import preprocess_audio
        prep = preprocess_audio(tmp_path, call_id)
        progress.progress(20, text="Transcribing with Whisper...")

        # ── Phase 1B: Whisper transcription ──────────────────
        status.info("🎙️ Phase 1B: Transcribing with Whisper (1-3 min)...")
        progress.progress(25, text="Loading Whisper model...")
        import torch, whisper, soundfile as sf
        import librosa
        from collections import Counter

        device = "cuda" if torch.cuda.is_available() else "cpu"

        @st.cache_resource
        def load_whisper():
            return whisper.load_model("medium", device=device)

        wmodel = load_whisper()
        y, sr  = librosa.load(tmp_path, sr=16000, mono=True)
        progress.progress(30, text="Transcribing audio...")
        result = wmodel.transcribe(y, language=None, word_timestamps=True,
                                   beam_size=5, no_speech_threshold=0.6)

        # Split Whisper segments on silence gaps
        def split_segments(raw_segs, silence_thresh=0.35):
            out = []
            for seg in raw_segs:
                words = seg.get("words", [])
                if not words:
                    out.append({"text": seg["text"].strip(),
                                "start": seg["start"], "end": seg["end"]})
                    continue
                split_pts = [i for i in range(len(words)-1)
                             if float(words[i+1]["start"]) - float(words[i]["end"]) >= silence_thresh]
                if not split_pts:
                    out.append({"text": seg["text"].strip(),
                                "start": seg["start"], "end": seg["end"]})
                    continue
                bounds = [-1] + split_pts + [len(words)-1]
                for j in range(len(bounds)-1):
                    sw = words[bounds[j]+1:bounds[j+1]+1]
                    if not sw: continue
                    txt = " ".join(w.get("word","").strip() for w in sw).strip()
                    if not txt: continue
                    out.append({"text": txt,
                                "start": float(sw[0]["start"]),
                                "end":   float(sw[-1]["end"])})
            return out

        whisper_segs = split_segments(result["segments"])
        progress.progress(40, text="Diarizing speakers with pyannote...")

        # ── Phase 1C: pyannote speaker diarization ────────────
        status.info("🔊 Phase 1C: Speaker diarization with pyannote...")
        try:
            from pyannote.audio import Pipeline
            from huggingface_hub import login as hf_login
            import os as _os

            hf_token = _os.environ.get("HF_TOKEN", "")

            @st.cache_resource
            def load_diarizer(token):
                if token:
                    hf_login(token=token, add_to_git_credential=False)
                pipe = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=token or True)
                pipe = pipe.to(torch.device(device))
                return pipe

            diarizer  = load_diarizer(hf_token)
            import tempfile as _tmp
            with _tmp.NamedTemporaryFile(suffix=".wav", delete=False) as _t:
                sf.write(_t.name, y, sr)
                _tpath = _t.name
            diarization = diarizer(_tpath, num_speakers=2)
            _os.unlink(_tpath)

            # Parse diarization
            label_map, spk_segs = {}, []
            for turn, _, label in diarization.itertracks(yield_label=True):
                if label not in label_map:
                    label_map[label] = len(label_map)
                spk_id = label_map[label]
                s, e   = round(float(turn.start), 3), round(float(turn.end), 3)
                if e - s < 0.3: continue
                if spk_segs and spk_segs[-1]["speaker_id"] == spk_id and s - spk_segs[-1]["end"] < 0.5:
                    spk_segs[-1]["end"] = e
                else:
                    spk_segs.append({"speaker_id": spk_id, "start": s, "end": e})

            # Align Whisper + pyannote
            diarized = []
            for ws in whisper_segs:
                best_spk, best_ov = 0, -1.0
                for ss in spk_segs:
                    ov = max(0.0, min(ws["end"], ss["end"]) - max(ws["start"], ss["start"]))
                    if ov > best_ov:
                        best_ov, best_spk = ov, ss["speaker_id"]
                diarized.append({
                    "speaker_id":  best_spk,
                    "speaker_raw": f"SPEAKER_{best_spk}",
                    "text":        ws["text"],
                    "start":       round(ws["start"], 3),
                    "end":         round(ws["end"], 3),
                    "duration":    round(ws["end"] - ws["start"], 3),
                })

            # fix_speakers voice embedding
            spk_counts = Counter(s["speaker_id"] for s in diarized)
            if len(spk_counts) < 2:
                status.warning("⚠️ Only one speaker detected — running voice clustering...")
                from fix_speakers import fix_call
                from resemblyzer import VoiceEncoder
                enc = VoiceEncoder(device=device)
                # Save diarized to temp JSON, fix, reload
                import json as _json, tempfile as _tmp2
                with _tmp2.NamedTemporaryFile(suffix="_diarized.json",
                                              delete=False, mode="w") as jf:
                    _json.dump(diarized, jf)
                    jpath = jf.name
                import shutil
                wav_dest = jpath.replace("_diarized.json", ".wav")
                shutil.copy(tmp_path, wav_dest)
                # Monkey-patch DATA_DIR and OUTPUTS_DIR for fix_speakers
                import fix_speakers as _fs
                _orig_data = _fs.DATA_DIR
                _orig_out  = _fs.OUTPUTS_DIR
                _fs.DATA_DIR    = _os.path.dirname(wav_dest)
                _fs.OUTPUTS_DIR = _os.path.dirname(jpath)
                _fs.fix_call(call_id, enc, force=True)
                _fs.DATA_DIR    = _orig_data
                _fs.OUTPUTS_DIR = _orig_out
                with open(jpath) as jf2:
                    diarized = _json.load(jf2)
                _os.unlink(jpath)
                _os.unlink(wav_dest)

        except Exception as pyannote_err:
            status.warning(f"⚠️ pyannote unavailable ({pyannote_err}) — using Whisper-only diarization")
            diarized = [{
                "speaker_id":  0,
                "speaker_raw": "SPEAKER_0",
                "text":        ws["text"],
                "start":       round(ws["start"], 3),
                "end":         round(ws["end"], 3),
                "duration":    round(ws["end"] - ws["start"], 3),
            } for ws in whisper_segs]

        progress.progress(50, text="Classifying speakers...")
        status.info("🔍 Phase 2: Classifying speakers with 3 methods...")

        # ── Phase 2A: Method 1 ────────────────────────────────
        from methods.method1_lexical import classify_transcript_lexical, analyze_keyword_frequency
        from methods.method3_llm import compute_confidence_statistics
        m1_classified = classify_transcript_lexical(diarized)
        m1_stats      = compute_confidence_statistics(m1_classified)
        progress.progress(60, text="Running acoustic method...")

        # ── Phase 2B: Method 2 — Acoustic DNN ───────────────────
        progress.progress(62, text="Running acoustic DNN...")
        try:
            from methods.method2_acoustic import classify_transcript_acoustic
            # Extract per-speaker audio for acoustic method
            speaker_audio = {}
            spk_ids = list(set(s["speaker_id"] for s in diarized))
            for spk_id in spk_ids:
                spk_segs_list = [s for s in diarized if s["speaker_id"] == spk_id]
                chunks = []
                for s in spk_segs_list:
                    st_s = int(s["start"] * sr)
                    en_s = int(s["end"]   * sr)
                    chunks.append(y[st_s:en_s])
                if chunks:
                    speaker_audio[spk_id] = np.concatenate(chunks)
            m2_classified = classify_transcript_acoustic(diarized, speaker_audio)
            m2_stats      = compute_confidence_statistics(m2_classified)
        except Exception as m2_err:
            status.warning(f"⚠️ Method 2 fallback: {m2_err}")
            m2_classified = m1_classified.copy()
            m2_stats      = m1_stats

        # ── Phase 2C: Method 3 LLM ────────────────────────────
        progress.progress(65, text="Calling LLM API (Groq)...")
        status.info("🤖 Phase 2C: LLM classification via Groq API...")
        from methods.method3_llm import classify_transcript_llm
        m3_classified = classify_transcript_llm(diarized)
        m3_stats      = compute_confidence_statistics(m3_classified)
        progress.progress(80, text="Running analytics...")

        # ── Phase 3: Analytics ────────────────────────────────
        status.info("📊 Phase 3: Analytics...")
        from analytics.sentiment import analyze_sentiment
        from analytics.talk_ratio import compute_talk_time_ratio, compute_turn_taking, compute_qa_score
        from analytics.compliance import check_compliance
        from analytics.rude_behavior import detect_rude_behavior
        from analytics.advanced import detect_call_outcome, detect_language, compute_agent_response_time, detect_interruptions
        from methods.method3_llm import generate_call_summary

        sentiment_result  = analyze_sentiment(m3_classified)
        m3_with_sentiment = sentiment_result["segments_with_sentiment"]
        duration          = prep.get("duration_sec", len(y)/sr)
        talk_ratio        = compute_talk_time_ratio(m3_with_sentiment, duration)
        turn_flow         = compute_turn_taking(m3_with_sentiment)
        compliance_result = check_compliance(m3_with_sentiment)
        rude_result       = detect_rude_behavior(m3_with_sentiment)
        outcome_result    = detect_call_outcome(m3_with_sentiment)
        response_time     = compute_agent_response_time(m3_with_sentiment)
        interruptions     = detect_interruptions(m3_with_sentiment)
        language_result   = detect_language(m3_with_sentiment)
        summary_result    = generate_call_summary(m3_with_sentiment, call_id)
        qa_result         = compute_qa_score(talk_ratio, turn_flow,
                                             sentiment_result, compliance_result,
                                             rude_result=rude_result)

        progress.progress(100, text="Done!")
        status.success("✅ Analysis complete!")

        return {
            "call_id":       call_id,
            "preprocessing": prep,
            "method1":       {"classified": m1_classified, "stats": m1_stats},
            "method2":       {"classified": m2_classified, "stats": m2_stats},
            "method3":       {"classified": m3_classified, "stats": m3_stats},
            "transcript_m1": m1_classified,
            "transcript_m2": m2_classified,
            "transcript_m3": m3_with_sentiment,
            "sentiment":     {k: v for k, v in sentiment_result.items()
                              if k != "segments_with_sentiment"},
            "talk_ratio":    talk_ratio,
            "turn_flow":     turn_flow,
            "compliance":    compliance_result,
            "rude_behavior": rude_result,
            "call_outcome":  outcome_result,
            "call_summary":  summary_result,
            "response_time": response_time,
            "interruptions": interruptions,
            "language":      language_result,
            "qa_result":     qa_result,
        }

    except Exception as e:
        status.error(f"❌ Error: {e}")
        progress.empty()
        raise
    finally:
        os.unlink(tmp_path)


def render_live_analysis():
    """Live analysis tab — upload WAV and get instant results."""
    st.header("🎙️ Live Call Analysis")
    st.caption("Upload a raw audio file to run the full pipeline instantly.")

    uploaded = st.file_uploader(
        "Drop your call recording here",
        type=["wav", "mp3"],
        help="Supports WAV and MP3. Max recommended: 10 minutes."
    )

    if not uploaded:
        st.info("👆 Upload a WAV or MP3 file to begin analysis.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.audio(uploaded)
    with col2:
        st.metric("File", uploaded.name)
        st.metric("Size", f"{uploaded.size / 1024:.0f} KB")

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running pipeline..."):
            try:
                call_data = run_live_analysis(uploaded.read(), uploaded.name)
                st.session_state["live_result"] = call_data
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

    # Show results if available
    if "live_result" in st.session_state:
        call_data = st.session_state["live_result"]
        st.markdown("---")
        render_call_detail(call_data, call_data.get("call_id", "uploaded_call"))


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
    # ── Mode switch ──────────────────────────────────────────
    mode = st.session_state.get("dashboard_mode", "🔧 System (Production)")
    if mode == "🎓 FYP Evaluation":
        render_fyp_evaluation(data)
        return

    if mode == "🎙️ Live Analysis":
        render_live_analysis()
        return

    st.subheader("📊 Executive Summary Dashboard")
    render_kpis(data)

    st.markdown("---")
    st.subheader("📈 Overview Analytics")
    render_overview_charts(data)

    st.markdown("---")
    st.subheader("🎯 Per-Call Accuracy Overview")
    render_accuracy_overview(data)

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

        # Try reading from evaluation_summary.json first (full 30-call evaluation)
        import json as _json
        eval_json = os.path.join(OUTPUTS_DIR, "evaluation_summary.json")
        if os.path.isfile(eval_json):
            with open(eval_json, encoding="utf-8") as _f:
                eval_data = _json.load(_f)
            agg    = eval_data.get("aggregated", {})
            source = eval_data.get("source", "evaluate.py")
            n_calls = eval_data.get("n_calls", 0)
            caption = f"✅ Full evaluation: {n_calls} calls — run `python evaluate.py` to refresh."
        else:
            agg     = validation.get("aggregated", {})
            caption = "⚠️ Stale single-call result. Run `python evaluate.py` for full accuracy."

        shown = set()
        cols  = st.columns(3)
        col_i = 0
        label_map = {
            "m1":     "Method 1 (Keyword)",
            "m2":     "Method 2 (Acoustic)",
            "m3_llm": "Method 3 (LLM)",
            "m3":     "Method 3 (LLM)",
        }
        for mname, metrics in agg.items():
            if col_i >= 3: break
            label = label_map.get(mname, mname)
            if label in shown: continue
            shown.add(label)
            with cols[col_i]:
                st.metric(label, f"Acc: {metrics.get('accuracy', 0):.1f}%",
                          delta=f"F1: {metrics.get('f1', 0):.1f}%")
            col_i += 1
        st.caption(caption)

        ttest = validation.get("ttest", {})
        if ttest.get("conclusion"):
            st.info(f"**Statistical Test:** {ttest['conclusion']}")


def render_fyp_evaluation(data: dict):
    """FYP research evaluation view — reads from label_accuracy_summary.json."""
    import os, json as _json, pandas as pd
    from config import HUMAN_TRANSCRIPTS_DIR, OUTPUTS_DIR

    st.header("🎓 FYP Evaluation — System vs Human Transcript")
    st.caption("Research mode. Run compare_labels.py first to generate accuracy data.")

    calls = _normalise_calls(data)
    if not calls:
        st.warning("No calls loaded. Run main.py first.")
        return

    # ── Load accuracy summary from compare_labels.py output ──
    json_path = os.path.join(OUTPUTS_DIR, "label_accuracy_summary.json")
    if not os.path.isfile(json_path):
        st.warning("No accuracy data found. Run: python compare_labels.py")
        return

    with open(json_path, encoding="utf-8") as f:
        acc_data = _json.load(f)

    per_call  = acc_data.get("per_call", {})
    overall   = acc_data.get("overall_accuracy", 0)
    overall_f1 = acc_data.get("overall_f1", 0)
    n_calls   = acc_data.get("total_calls", 0)

    # ── Top KPIs ──────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Average Accuracy", f"{overall:.1f}%")
    k2.metric("Average F1", f"{overall_f1:.1f}%")
    k3.metric("Calls Evaluated", str(n_calls))

    # ── Per-call accuracy bar chart ───────────────────────────
    st.subheader("📊 Per-Call Accuracy (Method 3 LLM)")
    rows = [{"Call": cid, "Accuracy": v["accuracy"], "F1": v["f1"],
             "Correct": v["correct"], "Wrong": v["wrong"]}
            for cid, v in per_call.items()]

    if rows:
        df_acc = pd.DataFrame(rows).sort_values("Accuracy")
        avg_acc = df_acc["Accuracy"].mean()
        st.metric("Average Accuracy", f"{avg_acc:.1f}%")

        import plotly.express as px
        fig = px.bar(df_acc, x="Accuracy", y="Call", orientation="h",
                     color="Accuracy",
                     color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
                     range_color=[50, 100],
                     title="Per-Call Accuracy (Method 3 LLM)")
        fig.add_vline(x=90, line_dash="dash", line_color="green",
                      annotation_text="90% target")
        fig.update_layout(height=600, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_acc.set_index("Call"), use_container_width=True)

    # ── Method-by-Method Label Comparison ───────────────────
    st.subheader("🔍 Method Label Comparison")
    st.caption("Select a call to see how Method 1, 2, and 4 label each segment vs human ground truth.")

    call_options = sorted(per_call.keys())
    selected_eval_call = st.selectbox("Select call", call_options)

    call_data = calls.get(selected_eval_call, {})
    ht_path   = os.path.join(HUMAN_TRANSCRIPTS_DIR, f"{selected_eval_call}.csv")

    if call_data and os.path.isfile(ht_path):
        gt_df = pd.read_csv(ht_path)
        if "role" in gt_df.columns:
            gt_df = gt_df.rename(columns={"role": "ground_truth_role"})

        gt_lookup = {r["text"].strip().lower(): r["ground_truth_role"]
                     for _, r in gt_df.iterrows()}

        m1_segs = call_data.get("method1", {}).get("classified", [])
        m2_segs = call_data.get("method2", {}).get("classified", [])
        m3_segs = call_data.get("method3", {}).get("classified", [])

        m1_lookup = {s.get("text","").strip().lower(): s.get("predicted_role","?") for s in m1_segs}
        m2_lookup = {s.get("text","").strip().lower(): s.get("predicted_role","?") for s in m2_segs}

        comp_rows = []
        for seg in m3_segs:
            text    = seg.get("text", "").strip()
            key     = text.lower()
            gt_role = gt_lookup.get(key, "")
            if not gt_role:
                continue  # skip segments not in human transcript
            m1_role = m1_lookup.get(key, "?")
            m2_role = m2_lookup.get(key, "?")
            m3_role = seg.get("predicted_role", "?")
            comp_rows.append({
                "Text":      text[:60] + "..." if len(text) > 60 else text,
                "Human ✓":   gt_role,
                "Method 1":  m1_role,
                "Method 2":  m2_role,
                "Method 3":  m3_role,
                "M1":        "✅" if m1_role == gt_role else "❌",
                "M2":        "✅" if m2_role == gt_role else "❌",
                "M3":        "✅" if m3_role == gt_role else "❌",
            })

        if comp_rows:
            df_comp = pd.DataFrame(comp_rows)
            m1_acc = (df_comp["M1"] == "✅").mean() * 100
            m2_acc = (df_comp["M2"] == "✅").mean() * 100
            m3_acc = (df_comp["M3"] == "✅").mean() * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Method 1", f"{m1_acc:.1f}%")
            c2.metric("Method 2", f"{m2_acc:.1f}%")
            c3.metric("Method 3", f"{m3_acc:.1f}%",
                      delta=f"+{m3_acc - m1_acc:.1f}% vs M1")

            st.dataframe(
                df_comp[["Text", "Human ✓", "Method 1", "Method 2", "Method 3", "M1", "M2", "M3"]],
                use_container_width=True, height=500,
            )
        else:
            st.info("Run compare_labels.py to see segment-level comparison.")

    # ── Validation report ─────────────────────────────────────
    st.subheader("📋 Evaluation Report")
    report_path = os.path.join(os.path.dirname(HUMAN_TRANSCRIPTS_DIR),
                               "outputs", "latest", "1_evaluation_report.txt")
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.code(f.read(), language=None)
    else:
        st.info("Run python evaluate.py to generate the evaluation report.")


if __name__ == "__main__":
    main()
