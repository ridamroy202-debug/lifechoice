"""
gate_service.py — AIP gate logic, MCQ evaluation, and integrity controls.

Rules (all read from DB, never hardcoded):
- AIP-11 always fires live API regardless of aip_count_used.
- FA1/FA2/FA3: live API if aip_count_used < cap, question_bank cache if >= cap.
- evaluateMCQAnswer: pure DB lookup, zero API cost.
- Integrity probe: separate live Sonnet call, does NOT increment aip_count_used.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from typing import Any

from app.persistence import (
    cache_question,
    get_cached_questions,
    get_learner_cc_aip_count,
    get_platform_setting,
    get_question_by_id,
    get_reference_responses,
    get_unreviewed_integrity_events,
    increment_learner_aip11_attempts,
    increment_learner_cc_aip_count,
    increment_question_served,
    log_aip_usage,
    log_integrity_event,
    store_reference_response,
    update_learner_cc_status,
    utc_now_iso,
)
from app.remote_backend import RemoteBackendError, remote_backend_client

logger = logging.getLogger(__name__)

# ── Company/location pools for contextual injection (Phase 6.2) ───────────────
_COMPANY_NAMES = [
    "Kenvara Solutions", "Brightfield Analytics", "Nexora Consulting", "Talvex Group",
    "Orindal Systems", "Crestwave Digital", "Lumivex Partners", "Stratford Dynamics",
    "Pinnacle Edge", "Vortex Innovations", "Clearpath Advisory", "Meridian Works",
    "Solaris Ventures", "Irongate Technologies", "Cobalt Bridge", "Wavefront Labs",
    "Thornfield Associates", "Apex Meridian", "Cascade Intellect", "Redstone Collective",
]
_LOCATIONS = [
    "Nairobi, Kenya", "Lagos, Nigeria", "Accra, Ghana", "Johannesburg, South Africa",
    "London, UK", "Manchester, UK", "Dubai, UAE", "Singapore", "Kuala Lumpur, Malaysia",
    "Sydney, Australia", "Toronto, Canada", "New York, USA", "Berlin, Germany",
    "Amsterdam, Netherlands", "Stockholm, Sweden", "Bangalore, India", "Mumbai, India",
    "São Paulo, Brazil", "Mexico City, Mexico", "Cairo, Egypt",
]
_SITUATIONAL_DETAILS = [
    "Your team has just missed its second consecutive delivery deadline due to unclear role assignments.",
    "A key stakeholder has requested a complete pivot in project direction with two weeks remaining.",
    "Three team members have raised concerns about workload distribution in the last sprint review.",
    "The organisation is adopting a new AI tool and staff resistance is higher than anticipated.",
    "Budget has been cut by 30% mid-project and scope must be renegotiated.",
    "A critical dependency on an external vendor has been delayed by six weeks.",
    "Two senior team members have conflicting views on the technical approach.",
    "The client has expanded the project scope without adjusting the timeline.",
    "A compliance audit has flagged gaps in the current process documentation.",
    "Remote team members in different time zones are struggling with communication delays.",
]


def get_aip_cap() -> int:
    """Always reads from DB. Never hardcoded."""
    return int(get_platform_setting("aip_cap_per_cc", "14"))


def _get_session_fa_count(session_id: str, competency_title: str) -> int:
    """Fallback counter using session_id when learner_id/competency_id are unavailable."""
    from app.db import get_connection
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM aip_usage_log
            WHERE session_id = ? AND competency_name = ? AND question_source = 'live_api'
            """,
            (session_id, competency_title),
        ).fetchone()
    return int(row["cnt"]) if row else 0


def get_next_fa_question(
    session_id: str,
    learner_id: str | None,
    competency_title: str,
    core_competency_id: int | None,
    micro_credential_id: int | None,
    fa_phase: str,
    attempt_number: int = 1,
    auth_token: str | None = None,
    is_retry: bool = False,
) -> dict[str, Any]:
    """
    Gate logic for FA1/FA2/FA3 MCQ questions.
    - is_retry=False (first attempt for this slot): fetch from live API, cache locally.
    - is_retry=True (slot was already failed): serve from question bank cache, zero cost.
    - AIP-11 is never routed here — always live AI in orchestrator.
    """
    if is_retry:
        # Zero-cost path: serve from local cache with anti-repeat
        recent_ids = _get_recent_served_ids(learner_id, core_competency_id, fa_phase, limit=5)
        cached = get_cached_questions(competency_title, fa_phase, exclude_remote_ids=recent_ids, limit=5)
        if not cached:
            cached = get_cached_questions(competency_title, fa_phase, limit=5)
        if not cached:
            raise RuntimeError(
                f"No cached questions for {competency_title}/{fa_phase}. "
                "Admin must trigger question bank generation first."
            )
        q = random.choice(cached)
        increment_question_served(q["id"])
        log_aip_usage(
            learner_id=learner_id, core_competency_id=core_competency_id,
            micro_credential_id=micro_credential_id, question_source="question_bank",
            aip_phase=fa_phase, aip_count_at_time=0, attempt_number=attempt_number,
            question_bank_id=q["id"], api_cost_estimate_usd=0.00,
            session_id=session_id, competency_name=competency_title,
        )
        return {
            "question": q["question_text"], "option_a": q["option_a"],
            "option_b": q["option_b"], "option_c": q["option_c"], "option_d": q["option_d"],
            "correct_answer": str(q["correct_answer"]).upper(),
            "explanation": q.get("explanation", ""), "source": "question_bank",
            "question_bank_id": q["id"], "aip_phase": fa_phase,
            "remote_question_id": q.get("remote_question_id"),
        }

    # First attempt — fetch from live API
    try:
        payload = remote_backend_client.fetch_question_bank(
            competency_title, fa_phase, count=5, token=auth_token
        )
        questions_data = payload.get("questions", {})
        questions_list = questions_data.get("questions", []) if isinstance(questions_data, dict) else []
        if not questions_list:
            raise RemoteBackendError("Empty question list from remote API")

        q = random.choice(questions_list)
        qb_id = cache_question(
            remote_question_id=int(q.get("id", 0)), core_competency_id=core_competency_id,
            competency_title=competency_title, micro_credential=q.get("micro_credential"),
            question_text=q.get("question", ""), option_a=q.get("option_a", ""),
            option_b=q.get("option_b", ""), option_c=q.get("option_c", ""),
            option_d=q.get("option_d", ""), correct_answer=str(q.get("correct_answer", "")).upper(),
            explanation=q.get("explanation"), difficulty_level=q.get("difficulty"), aip_phase=fa_phase,
        )
        log_aip_usage(
            learner_id=learner_id, core_competency_id=core_competency_id,
            micro_credential_id=micro_credential_id, question_source="live_api",
            aip_phase=fa_phase, aip_count_at_time=1, attempt_number=attempt_number,
            question_bank_id=qb_id, api_cost_estimate_usd=0.002,
            session_id=session_id, competency_name=competency_title,
        )
        return {
            "question": q.get("question", ""), "option_a": q.get("option_a", ""),
            "option_b": q.get("option_b", ""), "option_c": q.get("option_c", ""),
            "option_d": q.get("option_d", ""), "correct_answer": str(q.get("correct_answer", "")).upper(),
            "explanation": q.get("explanation", ""), "source": "live_api",
            "question_bank_id": qb_id, "aip_phase": fa_phase,
            "remote_question_id": int(q.get("id", 0)),
        }
    except RemoteBackendError as exc:
        logger.warning("Remote question bank fetch failed, falling back to cache: %s", exc)
        cached = get_cached_questions(competency_title, fa_phase, limit=5)
        if not cached:
            raise RuntimeError(
                f"Remote API failed and no cached questions for {competency_title}/{fa_phase}."
            ) from exc
        q = random.choice(cached)
        increment_question_served(q["id"])
        log_aip_usage(
            learner_id=learner_id, core_competency_id=core_competency_id,
            micro_credential_id=micro_credential_id, question_source="question_bank",
            aip_phase=fa_phase, aip_count_at_time=0, attempt_number=attempt_number,
            question_bank_id=q["id"], api_cost_estimate_usd=0.00,
            session_id=session_id, competency_name=competency_title,
        )
        return {
            "question": q["question_text"], "option_a": q["option_a"],
            "option_b": q["option_b"], "option_c": q["option_c"], "option_d": q["option_d"],
            "correct_answer": str(q["correct_answer"]).upper(),
            "explanation": q.get("explanation", ""), "source": "question_bank",
            "question_bank_id": q["id"], "aip_phase": fa_phase,
            "remote_question_id": q.get("remote_question_id"),
        }
    return {
        "question": q["question_text"],
        "option_a": q["option_a"],
        "option_b": q["option_b"],
        "option_c": q["option_c"],
        "option_d": q["option_d"],
        "correct_answer": str(q["correct_answer"]).upper(),
        "explanation": q.get("explanation", ""),
        "source": "question_bank",
        "question_bank_id": q["id"],
        "aip_phase": fa_phase,
        "remote_question_id": q.get("remote_question_id"),
    }


def _get_recent_served_ids(
    learner_id: str | None,
    core_competency_id: int | None,
    fa_phase: str,
    limit: int = 5,
) -> list[int]:
    """Returns remote_question_ids of the last N questions served to this learner for this CC+phase."""
    if not learner_id or not core_competency_id:
        return []
    from app.db import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT qb.remote_question_id
            FROM aip_usage_log aul
            JOIN question_bank qb ON qb.id = aul.question_bank_id
            WHERE aul.learner_id = ? AND aul.core_competency_id = ? AND aul.aip_phase = ?
              AND aul.question_bank_id IS NOT NULL
            ORDER BY aul.id DESC
            LIMIT ?
            """,
            (learner_id, core_competency_id, fa_phase, limit),
        ).fetchall()
    return [int(row["remote_question_id"]) for row in rows if row["remote_question_id"]]


def evaluate_mcq_answer(question_bank_id: int, learner_answer: str) -> dict[str, Any]:
    """
    Pure DB lookup. No API call. Returns pass/fail with zero cost.
    learner_answer should be 'A', 'B', 'C', or 'D'.
    """
    q = get_question_by_id(question_bank_id)
    if not q:
        return {"passed": False, "correct_answer": None, "explanation": "", "api_cost": 0.00}

    correct = str(q["correct_answer"]).strip().upper()
    submitted = str(learner_answer).strip().upper()
    passed = submitted == correct
    return {
        "passed": passed,
        "correct_answer": correct,
        "submitted_answer": submitted,
        "explanation": q.get("explanation", ""),
        "api_cost": 0.00,
    }


def build_injected_context(competency_title: str) -> dict[str, Any]:
    """Generates randomized scenario context for AIP-11 (Phase 6.2)."""
    return {
        "company_name": random.choice(_COMPANY_NAMES),
        "location": random.choice(_LOCATIONS),
        "situational_detail": random.choice(_SITUATIONAL_DETAILS),
        "competency": competency_title,
    }


def inject_context_into_scenario(scenario_text: str, context: dict[str, Any]) -> str:
    """Prepends injected context to the AIP-11 scenario prompt."""
    company = context.get("company_name", "")
    location = context.get("location", "")
    detail = context.get("situational_detail", "")
    prefix = f"**Context:** You are working at {company} in {location}. {detail}\n\n"
    return prefix + scenario_text


def check_time_floor(displayed_at: str | None, floor_seconds: int) -> bool:
    """Returns True if enough time has passed since the question was displayed."""
    if not displayed_at:
        return True  # No timestamp recorded — allow submission
    try:
        displayed = datetime.fromisoformat(displayed_at)
        elapsed = (datetime.now(timezone.utc) - displayed).total_seconds()
        return elapsed >= floor_seconds
    except (ValueError, TypeError):
        return True


def check_integrity_hold(
    learner_id: str | None,
    core_competency_id: int | None,
    session_id: str | None,
    micro_credential_id: int | None,
) -> dict[str, Any]:
    """
    Checks if AIP-11 verdict should be held for admin review.
    Returns {hold: bool, reason: str, event_count: int}.
    """
    if not learner_id or not core_competency_id:
        return {"hold": False, "reason": "", "event_count": 0}

    events = get_unreviewed_integrity_events(learner_id, core_competency_id)
    if len(events) >= 3:
        return {
            "hold": True,
            "reason": f"{len(events)} unreviewed integrity events exist for this learner on this CC.",
            "event_count": len(events),
            "events": events,
        }

    # Auto-hold if similarity_flag + (paste_detected OR time_floor_breach)
    types = {e["event_type"] for e in events}
    if "similarity_flag" in types and ("paste_detected" in types or "time_floor_breach" in types):
        return {
            "hold": True,
            "reason": "Combined similarity flag and paste/time-floor breach detected.",
            "event_count": len(events),
            "events": events,
        }

    return {"hold": False, "reason": "", "event_count": len(events)}


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure Python cosine similarity — no external dependencies."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def run_similarity_check(
    learner_id: str | None,
    session_id: str | None,
    core_competency_id: int | None,
    micro_credential_id: int | None,
    submitted_answer: str,
    answer_embedding: list[float] | None,
) -> None:
    """
    Silent background similarity check (Phase 6.6).
    Logs similarity_flag to integrity_log if cosine similarity > 0.91.
    Does NOT delay the learner response.
    """
    if not answer_embedding or not core_competency_id:
        return

    references = get_reference_responses(core_competency_id)
    for ref in references:
        ref_embedding = ref.get("embedding")
        if not ref_embedding:
            continue
        similarity = compute_cosine_similarity(answer_embedding, ref_embedding)
        if similarity > 0.91:
            log_integrity_event(
                learner_id=learner_id,
                session_id=session_id,
                micro_credential_id=micro_credential_id,
                core_competency_id=core_competency_id,
                event_type="similarity_flag",
                event_detail={
                    "similarity_score": round(similarity, 4),
                    "reference_id": ref["id"],
                    "answer_preview": submitted_answer[:200],
                },
            )
            logger.info(
                "Similarity flag logged: learner=%s cc=%s score=%.4f",
                learner_id, core_competency_id, similarity,
            )
            break  # Log once per submission
