"""
utils/file_utils.py
====================
Shared file I/O utilities used across the pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_json(data: dict, path: str, indent: int = 2) -> None:
    """Serialize dict to JSON, handling numpy types automatically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            # pandas types
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.tolist()
            # numpy types
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=_NpEncoder)
    logger.debug(f"Saved JSON → {path}")


def load_json(path: str) -> dict:
    """Load JSON from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_transcript_txt(classified: list[dict], path: str) -> None:
    """
    Save a classified transcript to a human-readable .txt file.
    Format: [HH:MM:SS] ROLE: text
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    for seg in classified:
        start = seg.get("start", 0)
        role  = seg.get("predicted_role", "Unknown")
        text  = seg.get("text", "").strip()
        m, s  = divmod(int(start), 60)
        h, m  = divmod(m, 60)
        timestamp = f"{h:02d}:{m:02d}:{s:02d}"
        lines.append(f"[{timestamp}] {role}: {text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.debug(f"Transcript saved → {path}")


def save_analytics_csv(all_results: dict, path: str) -> None:
    """
    Export per-call analytics summary to a flat CSV for reporting.

    Columns: call_id, duration_sec, snr_before, snr_after,
             agent_talk_pct, customer_talk_pct, silence_pct,
             customer_sentiment, agent_sentiment, sentiment_trend,
             compliance_score, risk_severity, qa_score, rating
    """
    rows = []
    # Handle both list and dict formats
    if isinstance(all_results, list):
        all_results = {c["call_id"]: c for c in all_results if isinstance(c, dict)}
    for cid, r in all_results.items():
        prep  = r.get("preprocessing", {})
        talk  = r.get("talk_ratio",    {})
        sent  = r.get("sentiment",     {})
        comp  = r.get("compliance",    {})
        qa    = r.get("qa_result",     {})
        rows.append({
            "call_id":              cid,
            "duration_sec":         prep.get("duration_sec",       0),
            "snr_before_db":        prep.get("snr_before_db",      0),
            "snr_after_db":         prep.get("snr_after_db",       0),
            "snr_improvement_db":   prep.get("snr_improvement_db", 0),
            "agent_talk_pct":       talk.get("agent_talk_pct",     0),
            "customer_talk_pct":    talk.get("customer_talk_pct",  0),
            "silence_pct":          talk.get("silence_pct",        0),
            "interaction_type":     talk.get("interaction_classification", ""),
            "customer_sentiment":   sent.get("customer_avg_compound",   0),
            "agent_sentiment":      sent.get("agent_avg_compound",      0),
            "sentiment_trend":      sent.get("sentiment_trend",         ""),
            "interpretation":       sent.get("interpretation",          ""),
            "compliance_score":     comp.get("compliance_score",    0),
            "compliance_label":     comp.get("compliance_label",    ""),
            "risk_severity":        comp.get("risk_severity",       ""),
            "qa_score":             qa.get("qa_score",   0),
            "rating":               qa.get("rating",     ""),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Analytics CSV saved → {path} | rows={len(df)}")


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
