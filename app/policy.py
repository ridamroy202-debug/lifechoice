from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.persistence import add_anomaly_flag, get_unresolved_anomalies
from app.state import LearnerSession, STREAK_BONUS_POINTS


ENCOURAGEMENT_MESSAGES = [
    "Good. You applied the idea correctly and can build on it.",
    "Correct. The reasoning is strong enough to move forward.",
    "That works. You are showing reliable mastery of this step.",
]

SUPPORTIVE_MESSAGES = [
    "Not there yet. I am going to reteach this in a simpler way.",
    "That answer has gaps. We will slow down and repair the concept.",
    "You are close, but the reasoning is incomplete. I will reframe it differently.",
]


def encouragement_message(passed: bool, streak_bonus: bool = False) -> str:
    message = ENCOURAGEMENT_MESSAGES[0] if passed else SUPPORTIVE_MESSAGES[0]
    if passed and streak_bonus:
        message += " Streak bonus applied."
    return message


def detect_and_record_anomalies(session: LearnerSession, learner_input: str, route: str, *, is_assessment: bool = False) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    normalized = re.sub(r'\s+', ' ', learner_input.strip().lower())
    recent_user_messages = [msg for msg in session.messages if msg.role == 'user']
    previous_user_messages = recent_user_messages[:-1] if recent_user_messages else []

    if len(normalized) > 30:
        for prior in reversed(previous_user_messages[-3:]):
            prior_norm = re.sub(r'\s+', ' ', prior.content.strip().lower())
            if prior_norm == normalized:
                flags.append({'flag_type': 'near_duplicate_answer', 'severity': 'medium', 'details': {'route': route, 'content': learner_input[:250]}})
                break

    last_assistant = next((msg for msg in reversed(session.messages) if msg.role == 'assistant'), None)
    if last_assistant is not None:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_assistant.created_at)
            if delta.total_seconds() < 2 and len(normalized) > 80:
                flags.append({'flag_type': 'implausibly_fast_response', 'severity': 'low', 'details': {'route': route, 'seconds_since_prompt': delta.total_seconds()}})
        except ValueError:
            pass

    if is_assessment and session.current_assessment_attempts >= 2:
        flags.append({'flag_type': 'repeated_assessment_retry', 'severity': 'medium', 'details': {'route': route, 'attempts': session.current_assessment_attempts}})

    for flag in flags:
        add_anomaly_flag(session.session_id, session.learner_id, flag['flag_type'], flag['severity'], flag['details'])
    return get_unresolved_anomalies(session.session_id)


def build_gamification_payload(session: LearnerSession, *, competency_badge: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'points_total': session.points_total,
        'points_delta': session.last_points_delta,
        'streak_count': session.streak_count,
        'streak_bonus_awarded': session.streak_bonus_awarded,
        'streak_bonus_points': STREAK_BONUS_POINTS if session.streak_bonus_awarded else 0,
        'progress_percent': session.overall_progress_percent,
        'competency_progress_percent': session.competency_progress_percent,
        'competency_badge': competency_badge,
        'completion_badge': session.completion_badge,
        'earned_badges': session.earned_badges,
        'feedback_message': session.last_feedback_message,
        'session_summary': session.session_summary,
    }


def build_session_summary(session: LearnerSession) -> dict[str, Any]:
    return session.build_session_summary()


def build_session_runtime_payload(session: LearnerSession) -> dict[str, Any]:
    return {
        'remote_sync': session.remote_sync_status,
        'interaction_type': session.current_interaction_type,
        'delivery_format': session.current_delivery_format,
        'current_aip_code': session.current_aip_code,
        'current_aip_name': session.current_aip_name,
        'aip_budget_total': session.aip_budget_total,
        'competency_aip_count': session.competency_aip_count,
        'competency_aip_history': session.competency_aip_history,
        'live_aip_call_count': len(session.live_aip_call_history),
        'mc_completion_reflection_fired': session.mc_completion_reflection_fired,
        'latest_binary_outcome': session.latest_binary_outcome,
        'developing_competency_active': session.developing_competency_active,
        'developing_competency_reason': session.developing_competency_reason,
        'local_assessment_passed': session.local_assessment_passed,
        'remote_assessment_synced': session.remote_assessment_synced,
        'remote_assessment_passed': session.remote_assessment_passed,
        'current_assessment_sync_error': session.current_assessment_sync_error,
        'formative_slot': session.formative_slot_number,
        'required_next_action': session.required_next_action,
        'unlimited_attempts': True,
    }
