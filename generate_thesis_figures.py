"""
generate_thesis_figures.py
==========================
Generates all aggregate figures needed for thesis Chapter 4.
Reads from pipeline_results.json.

Outputs (saved to outputs/latest/):
  m1_confidence_distribution.png  — Method 1 confidence score distribution
  keyword_density_analysis.png    — Agent vs Customer keyword density
  talk_time_distribution.png      — Talk time ratio across all calls
  sentiment_summary.png           — Agent vs Customer sentiment across calls
  compliance_summary.png          — Compliance scores per call
  risk_severity_distribution.png  — Risk severity distribution

Usage:
    python generate_thesis_figures.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_JSON = os.path.join(BASE_DIR, "outputs", "latest", "pipeline_results.json")
OUT_DIR       = os.path.join(BASE_DIR, "outputs", "latest")

COLORS = {
    "agent":    "#2196F3",
    "customer": "#FF5722",
    "silence":  "#9E9E9E",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral":  "#9E9E9E",
    "bar_blue": "#1976D2",
    "bar_orange": "#E64A19",
}

CATEGORY_COLORS = {
    "eng_prof":    "#1976D2",
    "eng_rudeagt": "#D32F2F",
    "eng_rudecust":"#F57C00",
    "long":        "#7B1FA2",
    "manglish":    "#00796B",
    "my_prof":     "#388E3C",
    "my_rude":     "#C62828",
    "my_sales":    "#1565C0",
}

def get_category_color(call_id):
    for prefix, color in CATEGORY_COLORS.items():
        if call_id.startswith(prefix.replace("_", "")):
            return color
        if call_id.startswith(prefix):
            return color
    return "#607D8B"


def load_data():
    with open(PIPELINE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    calls = data.get("calls", {})
    if isinstance(calls, list):
        calls = {c["call_id"]: c for c in calls}
    return calls


# ── FIGURE 1: M1 Confidence Distribution ─────────────────────
def plot_m1_confidence_distribution(calls: dict):
    all_confidences = []
    for call_data in calls.values():
        segs = call_data.get("method1", {}).get("classified", [])
        for s in segs:
            conf = s.get("final_confidence", s.get("confidence", None))
            if conf is not None:
                all_confidences.append(float(conf))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Method 1 — Keyword/Lexical Confidence Score Distribution",
                 fontsize=13, fontweight="bold")

    # Histogram
    axes[0].hist(all_confidences, bins=20, color=COLORS["bar_blue"],
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(all_confidences), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {np.mean(all_confidences):.3f}")
    axes[0].set_title("Distribution of Confidence Scores")
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot per category
    categories = {
        "Eng Prof":     [],
        "Eng RudeAgt":  [],
        "Eng RudeCust": [],
        "Manglish":     [],
        "Malay Prof":   [],
        "Malay Rude":   [],
        "Sales":        [],
    }
    prefixes = {
        "Eng Prof":     ["eng_prof"],
        "Eng RudeAgt":  ["eng_rudeagt"],
        "Eng RudeCust": ["eng_rudecust"],
        "Manglish":     ["manglish"],
        "Malay Prof":   ["my_prof"],
        "Malay Rude":   ["my_rude"],
        "Sales":        ["my_sales"],
    }
    for cid, call_data in calls.items():
        segs = call_data.get("method1", {}).get("classified", [])
        for cat, plist in prefixes.items():
            if any(cid.startswith(p) for p in plist):
                for s in segs:
                    c = s.get("final_confidence", s.get("confidence"))
                    if c is not None:
                        categories[cat].append(float(c))

    data_to_plot = [v for v in categories.values() if v]
    labels = [k for k, v in categories.items() if v]
    bp = axes[1].boxplot(data_to_plot, patch_artist=True, labels=labels)
    box_colors = ["#1976D2","#D32F2F","#F57C00","#00796B","#388E3C","#C62828","#1565C0"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Confidence Score by Call Category")
    axes[1].set_ylabel("Confidence Score")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "m1_confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: m1_confidence_distribution.png")


# ── FIGURE 2: Keyword Density Analysis ───────────────────────
def plot_keyword_density_analysis(calls: dict):
    agent_densities    = []
    customer_densities = []
    call_ids           = []

    for cid in sorted(calls.keys()):
        segs = calls[cid].get("method1", {}).get("classified", [])
        if not segs:
            continue
        a_dens = np.mean([s.get("agent_density",    0) for s in segs])
        c_dens = np.mean([s.get("customer_density", 0) for s in segs])
        agent_densities.append(a_dens)
        customer_densities.append(c_dens)
        call_ids.append(cid)

    x = np.arange(len(call_ids))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Method 1 — Keyword Density Analysis Across 30 Calls",
                 fontsize=13, fontweight="bold")

    # Bar chart
    bars_a = axes[0].bar(x - width/2, agent_densities,    width,
                         label="Agent Keywords",    color=COLORS["agent"],    alpha=0.8)
    bars_c = axes[0].bar(x + width/2, customer_densities, width,
                         label="Customer Keywords", color=COLORS["customer"], alpha=0.8)
    axes[0].set_title("Average Keyword Density per Call (Agent vs Customer)")
    axes[0].set_ylabel("Keyword Density (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(call_ids, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Scatter: agent density vs customer density
    colors_scatter = [get_category_color(cid) for cid in call_ids]
    axes[1].scatter(agent_densities, customer_densities,
                    c=colors_scatter, s=80, alpha=0.8, edgecolors="white", linewidths=0.5)
    for i, cid in enumerate(call_ids):
        axes[1].annotate(cid, (agent_densities[i], customer_densities[i]),
                         fontsize=6, alpha=0.7, ha="left", va="bottom")
    axes[1].set_xlabel("Agent Keyword Density (%)")
    axes[1].set_ylabel("Customer Keyword Density (%)")
    axes[1].set_title("Agent vs Customer Keyword Density (per call)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "keyword_density_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: keyword_density_analysis.png")


# ── FIGURE 3: Talk Time Distribution ─────────────────────────
def plot_talk_time_distribution(calls: dict):
    call_ids    = []
    agent_pcts  = []
    cust_pcts   = []
    silence_pcts = []
    classifications = []

    for cid in sorted(calls.keys()):
        tr = calls[cid].get("talk_ratio", {})
        if not tr:
            continue
        call_ids.append(cid)
        agent_pcts.append(tr.get("agent_talk_pct",    0))
        cust_pcts.append(tr.get("customer_talk_pct",  0))
        silence_pcts.append(tr.get("silence_pct",     0))
        classifications.append(tr.get("interaction_classification", ""))

    x = np.arange(len(call_ids))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Talk Time Distribution Across 30 Calls", fontsize=13, fontweight="bold")

    # Stacked bar
    axes[0].bar(x, agent_pcts,   label="Agent",    color=COLORS["agent"],    alpha=0.85)
    axes[0].bar(x, cust_pcts,    bottom=agent_pcts,
                label="Customer", color=COLORS["customer"], alpha=0.85)
    bot2 = [a + c for a, c in zip(agent_pcts, cust_pcts)]
    axes[0].bar(x, silence_pcts, bottom=bot2,
                label="Silence",  color=COLORS["silence"],  alpha=0.65)
    axes[0].axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.5,
                    label="50% line")
    axes[0].set_title("Talk Time Ratio per Call (Stacked)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(call_ids, rotation=45, ha="right", fontsize=8)
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Box plot: agent vs customer talk %
    avg_agent = np.mean(agent_pcts)
    avg_cust  = np.mean(cust_pcts)
    bp = axes[1].boxplot([agent_pcts, cust_pcts, silence_pcts],
                         patch_artist=True,
                         labels=["Agent", "Customer", "Silence"])
    colors_bp = [COLORS["agent"], COLORS["customer"], COLORS["silence"]]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_title(
        f"Talk Time Distribution Summary  "
        f"(Avg Agent={avg_agent:.1f}%  Customer={avg_cust:.1f}%)"
    )
    axes[1].set_ylabel("Percentage (%)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "talk_time_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: talk_time_distribution.png")


# ── FIGURE 4: Sentiment Summary ──────────────────────────────
def plot_sentiment_summary(calls: dict):
    call_ids    = []
    agent_sent  = []
    cust_sent   = []
    mismatches  = []

    for cid in sorted(calls.keys()):
        s = calls[cid].get("sentiment", {})
        if not s:
            continue
        call_ids.append(cid)
        agent_sent.append(s.get("agent_avg_compound",    0))
        cust_sent.append(s.get("customer_avg_compound",  0))
        mismatches.append(s.get("emotional_mismatch",    False))

    x = np.arange(len(call_ids))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Sentiment Analysis Across 30 Calls", fontsize=13, fontweight="bold")

    # Bar chart
    agent_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                    for v in agent_sent]
    cust_colors  = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                    for v in cust_sent]
    axes[0].bar(x - width/2, agent_sent,   width, color=COLORS["agent"],
                alpha=0.8, label="Agent Sentiment")
    axes[0].bar(x + width/2, cust_sent,    width, color=COLORS["customer"],
                alpha=0.8, label="Customer Sentiment")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="-")
    for i, mm in enumerate(mismatches):
        if mm:
            axes[0].annotate("⚠", (x[i], max(agent_sent[i], cust_sent[i]) + 0.02),
                             ha="center", fontsize=10, color="red")
    axes[0].set_title("Average Sentiment Compound Score per Call (VADER)")
    axes[0].set_ylabel("Compound Score (-1 to +1)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(call_ids, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Scatter: agent vs customer sentiment
    colors_s = [get_category_color(cid) for cid in call_ids]
    axes[1].scatter(agent_sent, cust_sent, c=colors_s, s=80,
                    alpha=0.8, edgecolors="white", linewidths=0.5)
    for i, cid in enumerate(call_ids):
        axes[1].annotate(cid, (agent_sent[i], cust_sent[i]),
                         fontsize=6, alpha=0.7)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Agent Sentiment")
    axes[1].set_ylabel("Customer Sentiment")
    axes[1].set_title("Agent vs Customer Sentiment Correlation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sentiment_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: sentiment_summary.png")


# ── FIGURE 5: Compliance Summary ─────────────────────────────
def plot_compliance_summary(calls: dict):
    call_ids    = []
    scores      = []
    greeting    = []
    closing     = []
    recorded    = []
    identity    = []

    for cid in sorted(calls.keys()):
        c = calls[cid].get("compliance", {})
        if not c:
            continue
        call_ids.append(cid)
        scores.append(c.get("compliance_score", 0))
        greeting.append(1 if c.get("greeting_passed")  else 0)
        closing.append( 1 if c.get("closing_passed")   else 0)
        recorded.append(1 if c.get("recorded_passed")  else 0)
        identity.append(1 if c.get("identity_passed")  else 0)

    x = np.arange(len(call_ids))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Compliance Analysis Across 30 Calls", fontsize=13, fontweight="bold")

    # Compliance score bar
    bar_colors = ["#4CAF50" if s >= 75 else "#FF9800" if s >= 50
                  else "#F44336" for s in scores]
    axes[0].bar(x, scores, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(75, color="green",  linestyle="--", linewidth=1,
                    alpha=0.7, label="75% (Good)")
    axes[0].axhline(50, color="orange", linestyle="--", linewidth=1,
                    alpha=0.7, label="50% (Fair)")
    axes[0].set_title("Compliance Score per Call (0-100)")
    axes[0].set_ylabel("Compliance Score (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(call_ids, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].set_ylim(0, 110)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Checklist heatmap
    checklist = np.array([greeting, closing, recorded, identity])
    im = axes[1].imshow(checklist, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["Greeting", "Closing", "Recorded", "Identity"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(call_ids, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("SOP Checklist Pass/Fail per Call (Green=Pass, Red=Fail)")
    plt.colorbar(im, ax=axes[1], orientation="vertical",
                 label="Pass(1) / Fail(0)", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compliance_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: compliance_summary.png")


# ── FIGURE 6: Risk Severity Distribution ─────────────────────
def plot_risk_severity_distribution(calls: dict):
    risk_counts    = Counter()
    agent_rude     = Counter()
    cust_rude      = Counter()
    outcomes       = Counter()

    for call_data in calls.values():
        comp = call_data.get("compliance", {})
        rude = call_data.get("rude_behavior", {})
        out  = call_data.get("call_outcome", {})

        risk_counts[comp.get("risk_severity", "Clean")] += 1
        agent_rude[rude.get("agent_rudeness_level",    "NONE")] += 1
        cust_rude[rude.get("customer_rudeness_level",  "NONE")] += 1
        outcomes[out.get("outcome", "Unknown")] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Behavioral & Risk Analysis — All 30 Calls",
                 fontsize=13, fontweight="bold")

    # Risk severity pie
    risk_labels = list(risk_counts.keys())
    risk_vals   = list(risk_counts.values())
    risk_colors = {"Clean": "#4CAF50", "Low": "#FFEB3B",
                   "Medium": "#FF9800", "High": "#F44336"}
    axes[0][0].pie(risk_vals, labels=risk_labels, autopct="%1.0f%%",
                   colors=[risk_colors.get(l, "#9E9E9E") for l in risk_labels],
                   startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0][0].set_title("Risk Severity Distribution")

    # Agent rudeness bar
    rude_order  = ["NONE", "LOW", "MEDIUM", "HIGH"]
    rude_colors = {"NONE": "#4CAF50", "LOW": "#FFEB3B",
                   "MEDIUM": "#FF9800", "HIGH": "#F44336"}
    vals_a = [agent_rude.get(k, 0) for k in rude_order]
    axes[0][1].bar(rude_order, vals_a,
                   color=[rude_colors[k] for k in rude_order],
                   alpha=0.85, edgecolor="white")
    axes[0][1].set_title("Agent Rudeness Level")
    axes[0][1].set_ylabel("Number of Calls")
    for i, v in enumerate(vals_a):
        if v > 0:
            axes[0][1].text(i, v + 0.1, str(v), ha="center", fontsize=11,
                            fontweight="bold")
    axes[0][1].grid(True, alpha=0.3, axis="y")

    # Customer rudeness bar
    vals_c = [cust_rude.get(k, 0) for k in rude_order]
    axes[1][0].bar(rude_order, vals_c,
                   color=[rude_colors[k] for k in rude_order],
                   alpha=0.85, edgecolor="white")
    axes[1][0].set_title("Customer Rudeness Level")
    axes[1][0].set_ylabel("Number of Calls")
    for i, v in enumerate(vals_c):
        if v > 0:
            axes[1][0].text(i, v + 0.1, str(v), ha="center", fontsize=11,
                            fontweight="bold")
    axes[1][0].grid(True, alpha=0.3, axis="y")

    # Call outcomes pie
    out_labels = list(outcomes.keys())
    out_vals   = list(outcomes.values())
    out_colors = {"Resolved": "#4CAF50", "Unresolved": "#F44336",
                  "Escalated": "#FF9800", "Transferred": "#2196F3"}
    wedge_colors = [out_colors.get(l, "#9E9E9E") for l in out_labels]
    axes[1][1].pie(out_vals, labels=out_labels, autopct="%1.0f%%",
                   colors=wedge_colors, startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1][1].set_title("Call Outcome Distribution")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "risk_severity_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: risk_severity_distribution.png")


def main():
    if not os.path.exists(PIPELINE_JSON):
        print(f"❌ {PIPELINE_JSON} not found — run main.py first")
        return

    print(f"Loading pipeline results...")
    calls = load_data()
    print(f"Loaded {len(calls)} calls\n")

    plot_m1_confidence_distribution(calls)
    plot_keyword_density_analysis(calls)
    plot_talk_time_distribution(calls)
    plot_sentiment_summary(calls)
    plot_compliance_summary(calls)
    plot_risk_severity_distribution(calls)

    print(f"\n✅ All 6 figures saved to: {OUT_DIR}")
    print("Files:")
    for f in ["m1_confidence_distribution.png", "keyword_density_analysis.png",
              "talk_time_distribution.png", "sentiment_summary.png",
              "compliance_summary.png", "risk_severity_distribution.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
