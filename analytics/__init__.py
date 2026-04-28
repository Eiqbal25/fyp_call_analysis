"""
analytics/
==========
Phase 3: Automated Call Analytics

Modules:
    talk_ratio   — Talk-time ratio, silence analysis, QA score (composite)
    sentiment    — VADER sentiment trajectory + role-separated analysis
    compliance   — SOP checklist, behavioural risk flagging
"""

from analytics.talk_ratio import (
    compute_talk_time_ratio,
    compute_turn_taking,
    compute_qa_score,
)
from analytics.sentiment import (
    analyze_sentiment,
    score_sentiment,
)
from analytics.compliance import (
    check_compliance,
)

__all__ = [
    "compute_talk_time_ratio",
    "compute_turn_taking",
    "compute_qa_score",
    "analyze_sentiment",
    "score_sentiment",
    "check_compliance",
]
