
import json
import os
import re
from typing import Any

import yaml

from app.crews.ai_tutor_agents_crew import TutorCrew
from app.crews.assessment_crew import AssessmentCrew
from app.crews.learning_path_planner import PathPlnner
from app.crews.level_classifier_crew import LevelClassifierCrew
from app.crews.pre_assessment_crew import PreAssessCrew
from app.crews.studey_materils_crew import StudyMeterial
from app.remote_backend import RemoteBackendError, remote_backend_client
from app.session_manager import save_session
from app.state import CompetencyResult, LearnerSession

PASS_THRESHOLD = 75.0
PRE_ASSESSMENT_QUESTION_COUNT = 2
BASE_LEARNING_INTERACTIONS = 6

_RUBRICS_CACHE: dict[str, Any] | None = None


def _load_all_rubrics() -> dict[str, Any]:
    global _RUBRICS_CACHE
    if _RUBRICS_CACHE is not None:
        return _RUBRICS_CACHE

    rubrics_path = os.path.join(os.path.dirname(__file__), "config", "rubrics.yaml")
    try:
        with open(rubrics_path, "r", encoding="utf-8") as file:
            _RUBRICS_CACHE = yaml.safe_load(file) or {}
    except (FileNotFoundError, yaml.YAMLError):
        _RUBRICS_CACHE = {}
    return _RUBRICS_CACHE


def _load_rubric(competency: str) -> dict[str, Any]:
    all_rubrics = _load_all_rubrics()
    key = competency.lower().replace(" ", "_")
    rubric = all_rubrics.get(key, all_rubrics.get("default", {}))
    if rubric and rubric.get("criteria"):
        return rubric
    return {
        "criteria": [
            {"name": "Accuracy", "weight": 0.34},
            {"name": "Application", "weight": 0.33},
            {"name": "Reasoning", "weight": 0.33},
        ],
        "pass_threshold": PASS_THRESHOLD,
    }


def _extract_subparts_from_plan(plan_text: str, fallback_turns: int = BASE_LEARNING_INTERACTIONS) -> list[str]:
    if not plan_text or not plan_text.strip():
        return [
            f"Concept {index}: Teach one practical idea and end with an applied check."
            for index in range(1, fallback_turns + 1)
        ]

    chunks = re.split(r"\n(?=\s*(?:\d+[\).:-]|[-*])\s+)", plan_text.strip())
    candidates = [chunk.strip() for chunk in chunks if chunk.strip()]
    cleaned: list[str] = []
    for chunk in candidates:
        item = re.sub(r"^\s*(?:\d+[\).:-]|[-*])\s+", "", chunk).strip()
        if item:
            cleaned.append(item)

    while len(cleaned) < fallback_turns:
        cleaned.append(
            f"Concept {len(cleaned) + 1}: Explain the next building block with a practical example."
        )
    return cleaned[:fallback_turns]


def _is_technical_competency(name: str) -> bool:
    tokens = (
        "regression",
        "statistics",
        "probability",
        "python",
        "sql",
        "analysis",
        "machine learning",
        "model",
        "algorithm",
        "data",
        "math",
        "programming",
        "code",
        "database",
        "api",
        "engineering",
        "architecture",
        "cloud",
    )
    lowered = name.lower()
    return any(token in lowered for token in tokens)


def _get_competency_details(session: LearnerSession, competency: str) -> dict[str, Any]:
    return session.competency_details.get(competency, {})


def _competency_prompt_label(session: LearnerSession, competency: str) -> str:
    details = _get_competency_details(session, competency)
    description = str(details.get("description") or "").strip()
    if description:
        return f"{competency} - {description}"
    return competency


def _three_word_competency_brief(competency: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", competency)
    if not words:
        return "Focused skill path"
    brief_words = words[:3]
    if len(brief_words) == 1:
        brief_words.extend(["skill", "focus"])
    elif len(brief_words) == 2:
        brief_words.append("focus")
    return " ".join(word.title() for word in brief_words[:3])


def build_competency_intro(session: LearnerSession) -> str:
    competency = session.current_competency
    details = _get_competency_details(session, competency)
    description = str(details.get("description") or "").strip()
    brief = _three_word_competency_brief(competency)

    lines = [
        f"**{brief}**",
        "",
        f"We are starting the competency **{competency}**.",
    ]
    if description:
        lines.append(f"This competency focuses on: {description}")
    lines.extend(
        [
            "In this competency session, I will teach one concept at a time, check mastery in short formative moments, and unlock the final applied assessment only after you show readiness.",
            "Next step: call `POST /pre-assessment/start` for the competency pre-assessment.",
        ]
    )
    intro_message = "\n".join(lines)
    session.intro_delivered = True
    session.competency_interaction = 1
    session.add_message("assistant", intro_message)
    save_session(session)
    return intro_message


def _normalize_remote_rubric(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None

    raw_criteria = payload.get("rubric_criteria") or {}
    iterable: list[dict[str, Any]]
    if isinstance(raw_criteria, dict):
        iterable = raw_criteria.get("criteria", [])
    elif isinstance(raw_criteria, list):
        iterable = raw_criteria
    else:
        iterable = []

    criteria = []
    total = max(1, len(iterable))
    weight = round(1 / total, 3)
    for index, item in enumerate(iterable, start=1):
        criteria.append(
            {
                "name": str(item.get("criterion") or item.get("name") or f"Criterion {index}"),
                "description": str(item.get("descriptor") or item.get("met_indicator") or ""),
                "weight": weight,
                "criterion_id": str(item.get("id") or f"c{index}"),
            }
        )
    if not criteria:
        return None

    return {
        "criteria": criteria,
        "pass_threshold": float(payload.get("pass_threshold") or PASS_THRESHOLD),
        "scenario_template": payload.get("scenario_template"),
        "difficulty_level": payload.get("difficulty_level"),
        "source": "remote",
    }


def _load_assessment_context(session: LearnerSession, competency: str) -> tuple[dict[str, Any], str, str]:
    details = _get_competency_details(session, competency)
    remote_competency_id = details.get("id")

    if competency in session.rubric_cache:
        cached = session.rubric_cache[competency]
        scenario = cached.get("scenario_template") or details.get("description") or session.study_materials.get(competency, "")
        return cached, str(scenario), cached.get("source", "cache")

    if isinstance(remote_competency_id, int):
        remote_payload = remote_backend_client.fetch_rubric(remote_competency_id)
        normalized = _normalize_remote_rubric(remote_payload)
        if normalized:
            session.rubric_cache[competency] = normalized
            scenario = normalized.get("scenario_template") or details.get("description") or session.study_materials.get(competency, "")
            return normalized, str(scenario), "remote"

    fallback = _load_rubric(competency)
    fallback["source"] = "local_fallback"
    session.rubric_cache[competency] = fallback
    warning = f"Remote rubric missing for competency '{competency}'. Using local fallback rubric."
    if warning not in session.backend_warnings:
        session.backend_warnings.append(warning)
    scenario = details.get("description") or session.study_materials.get(competency, "")
    return fallback, str(scenario), "local_fallback"


def _ensure_remote_learning_session(session: LearnerSession, competency: str) -> int | None:
    existing = session.remote_learning_sessions.get(competency)
    if existing:
        return existing

    details = _get_competency_details(session, competency)
    remote_competency_id = details.get("id")
    if session.source != "remote" or not session.remote_access_id or not isinstance(remote_competency_id, int):
        return None

    try:
        payload = remote_backend_client.start_learning_session(
            mc_access_id=session.remote_access_id,
            competency_id=remote_competency_id,
            token=session.remote_auth_token,
        )
    except RemoteBackendError as exc:
        warning = f"Could not start remote learning session for '{competency}': {exc}"
        if warning not in session.backend_warnings:
            session.backend_warnings.append(warning)
        return None

    remote_session_id = payload.get("id")
    if isinstance(remote_session_id, int):
        session.remote_learning_sessions[competency] = remote_session_id
        return remote_session_id
    return None


def _safe_json_loads(raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _difficulty_from_level(level: str) -> str:
    return {
        "beginner": "support",
        "intermediate": "standard",
        "advanced": "stretch",
    }.get(level, "standard")


def _raise_difficulty(current: str) -> str:
    order = ["support", "standard", "stretch"]
    return order[min(order.index(current), len(order) - 2) + 1]


def _lower_difficulty(current: str) -> str:
    order = ["support", "standard", "stretch"]
    return order[max(order.index(current) - 1, 0)]


def _delivery_mode(session: LearnerSession) -> str:
    modes = [
        "analogy",
        "step_by_step",
        "comparison_table",
        "worked_example",
        "mini_challenge",
        "error_clinic",
    ]
    index = len(session.delivery_history) % len(modes)
    return modes[index]


def _build_personalization_state(session: LearnerSession) -> dict[str, str]:
    delivery_mode = _delivery_mode(session)
    state = {
        "difficulty_tier": session.current_difficulty,
        "delivery_mode": delivery_mode,
        "support_style": "high-guidance" if session.current_difficulty == "support" else "coaching",
        "feedback_style": "encouraging" if session.consecutive_formative_fails == 0 else "supportive-recovery",
        "mastery_status": "ready" if session.final_assessment_unlocked else "in_progress",
        "revision_mode": "active" if session.revision_required else "inactive",
        "spaced_learning_rule": "one concept per interaction",
    }
    session.personalization_state = state
    session.delivery_history.append(delivery_mode)
    return state


async def _setup_competency(session: LearnerSession):
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    details = _get_competency_details(session, competency)
    context_description = str(details.get("description") or "").strip()

    if competency not in session.study_materials:
        material = StudyMeterial().crew().kickoff(
            inputs={
                "topic": session.topic,
                "competency": competency_label,
                "user_level": session.user_level,
                "context_description": context_description,
            }
        )
        session.study_materials[competency] = material.raw

    if competency not in session.learning_plans:
        plan = PathPlnner().crew().kickoff(
            inputs={
                "topic": session.topic,
                "competency": competency_label,
                "user_level": session.user_level,
                "weak_areas": ", ".join(session.weak_areas) or "none",
                "context_description": context_description,
                "interaction_budget": BASE_LEARNING_INTERACTIONS,
            }
        )
        session.learning_plans[competency] = plan.raw

    if competency not in session.competency_subparts:
        session.competency_subparts[competency] = _extract_subparts_from_plan(
            session.learning_plans.get(competency, ""),
            fallback_turns=BASE_LEARNING_INTERACTIONS,
        )


def _generate_preassessment_prompt(session: LearnerSession) -> str:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    result = PreAssessCrew().crew().kickoff(
        inputs={
            "topic": session.topic,
            "competencies": competency_label,
            "chat_history": session.format_recent_history(),
            "user_message": f"Generate {PRE_ASSESSMENT_QUESTION_COUNT} applied pre-assessment questions for this competency.",
            "turn_number": 1,
        }
    )
    return result.raw.strip()


def _classify_competency_readiness(session: LearnerSession) -> dict[str, Any]:
    result = LevelClassifierCrew().crew().kickoff(
        inputs={
            "topic": session.current_competency,
            "chat_history": session.format_recent_history(12),
        }
    )
    return _safe_json_loads(
        result.raw,
        {"level": "beginner", "weak_areas": [session.current_competency.lower()]},
    )


def _format_formative_feedback(passed: bool, percent: float, summary: str) -> str:
    prefix = (
        "You got that formative check right enough to move forward."
        if passed
        else "That formative check is not strong enough yet, so I am going to reteach the concept in a different way."
    )
    return f"{prefix} Score: {percent:.1f}%. {summary}".strip()


def _update_formative_slot(session: LearnerSession, passed: bool):
    slot = session.current_formative_slot
    if slot < 0:
        return
    while len(session.formative_slots) <= slot:
        session.formative_slots.append(None)
    session.formative_slots[slot] = passed
    session.formative_check_results = [item for item in session.formative_slots if item is not None]


def _all_formative_slots_passed(session: LearnerSession) -> bool:
    return bool(session.formative_slots) and all(item is True for item in session.formative_slots)


def _apply_formative_outcome(session: LearnerSession, passed: bool, percent: float, summary: str):
    _update_formative_slot(session, passed)
    session.formative_feedback_log.append({"passed": passed, "score": percent, "summary": summary})
    session.awaiting_formative_response = False
    session.current_formative_prompt = None

    if passed:
        session.consecutive_formative_passes += 1
        session.consecutive_formative_fails = 0
        if session.consecutive_formative_passes >= 2:
            session.current_difficulty = _raise_difficulty(session.current_difficulty)
        if session.current_subpart_index < len(session.competency_subparts.get(session.current_competency, [])) - 1:
            session.current_subpart_index += 1
    else:
        session.consecutive_formative_passes = 0
        session.consecutive_formative_fails += 1
        session.current_difficulty = _lower_difficulty(session.current_difficulty)
        if session.consecutive_formative_fails >= 2 or sum(item is False for item in session.formative_slots) >= 2:
            session.revision_required = True

    if session.learning_turn >= BASE_LEARNING_INTERACTIONS and _all_formative_slots_passed(session):
        session.final_assessment_unlocked = True


def _build_formative_rubric(session: LearnerSession) -> dict[str, Any]:
    concept = session.current_subpart or session.current_competency
    return {
        "criteria": [
            {
                "name": "Concept accuracy",
                "description": f"Understands the current concept: {concept}",
                "weight": 0.34,
            },
            {
                "name": "Applied reasoning",
                "description": "Uses the concept in the scenario rather than only defining it",
                "weight": 0.33,
            },
            {
                "name": "Clear explanation",
                "description": "Explains why the chosen action makes sense",
                "weight": 0.33,
            },
        ],
        "pass_threshold": PASS_THRESHOLD,
    }


def _evaluate_formative_response(session: LearnerSession, learner_answer: str) -> tuple[bool, float, str]:
    rubric = _build_formative_rubric(session)
    result = AssessmentCrew().crew().kickoff(
        inputs={
            "competency": f"Formative check for {session.current_competency}",
            "scenario": session.current_formative_prompt or session.current_subpart or session.current_competency,
            "user_response": learner_answer,
            "rubric_json": json.dumps(rubric),
        }
    )
    payload = _safe_json_loads(result.raw, {"overall_percent": 0.0, "pass": False, "summary": result.raw})
    overall = float(payload.get("overall_percent", 0.0) or 0.0)
    passed = bool(payload.get("pass", overall >= PASS_THRESHOLD))
    summary = str(payload.get("summary") or "").strip() or "No detailed formative feedback was returned."
    return passed, overall, summary


def _should_ask_formative_check(session: LearnerSession) -> bool:
    if session.learning_turn <= 0:
        return False
    if session.learning_turn % 2 == 0:
        return True
    return session.revision_required and session.learning_turn > BASE_LEARNING_INTERACTIONS


def _interaction_goal(session: LearnerSession) -> str:
    if session.learning_turn <= 2:
        return "Teach the next core concept clearly and simply."
    if session.learning_turn <= 4:
        return "Show how to apply the concept with reasoning and examples."
    if session.learning_turn <= 6:
        return "Prepare the learner for the mastery gate with practice and precise feedback."
    return "Run focused revision on the weakest concept and repair misconceptions."


def _parse_formative_prompt(ai_response: str) -> str:
    parts = ai_response.rsplit("**Formative Check**", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return ai_response.strip()


def _generate_assessment_prompt(session: LearnerSession) -> str:
    competency = session.current_competency
    details = _get_competency_details(session, competency)
    description = str(details.get("description") or "").strip()
    weakest = ", ".join(session.weak_areas[:2]) or "the main steps and reasoning"
    return (
        f"**Final Competency Assessment - {competency}**\n\n"
        f"Scenario: You are working on a real project that depends on **{competency}**. "
        f"{description or 'Use the competency in a realistic workplace situation.'}\n\n"
        "Your task:\n"
        "1. Explain the situation and objective.\n"
        "2. Describe the exact steps you would take.\n"
        "3. Justify why this approach is appropriate.\n"
        "4. Mention at least one risk, mistake, or tradeoff you would watch for.\n\n"
        f"Focus especially on: {weakest}.\n"
        f"Passing threshold is **{PASS_THRESHOLD:.0f}%**."
    )


def _record_remote_teaching_interaction(
    session: LearnerSession,
    competency: str,
    ai_prompt: str,
    ai_response: str,
    learner_input: str | None,
    formative_passed: bool | None,
):
    remote_learning_session_id = _ensure_remote_learning_session(session, competency)
    if not remote_learning_session_id:
        return

    try:
        remote_backend_client.record_interaction(
            session_id=remote_learning_session_id,
            interaction_type="teaching",
            ai_prompt=ai_prompt,
            ai_response=ai_response,
            learner_input=learner_input,
            formative_passed=formative_passed,
            token=session.remote_auth_token,
        )
    except RemoteBackendError as exc:
        warning = f"Remote interaction sync failed for '{competency}': {exc}"
        if warning not in session.backend_warnings:
            session.backend_warnings.append(warning)


def _generate_learning_response(session: LearnerSession, user_message: str, formative_feedback: str = "") -> str:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    current_subpart = session.current_subpart or competency
    personalization = _build_personalization_state(session)
    include_formative = _should_ask_formative_check(session)

    result = TutorCrew().crew().kickoff(
        inputs={
            "topic": session.topic,
            "competency": competency_label,
            "user_level": session.user_level,
            "weak_areas": ", ".join(session.weak_areas) or "none",
            "chat_history": session.format_recent_history(),
            "user_message": user_message,
            "turn_number": session.learning_turn,
            "interaction_number": session.competency_interaction,
            "current_subpart": current_subpart,
            "subpart_index": session.current_subpart_index + 1,
            "total_subparts": len(session.competency_subparts.get(competency, [])),
            "study_material": session.study_materials.get(competency, ""),
            "chat_stage": session.chat_stage,
            "bloom_level": session.bloom_level,
            "competency_is_technical": "yes" if _is_technical_competency(competency) else "no",
            "formative_feedback": formative_feedback or "No formative feedback yet for this turn.",
            "delivery_mode": personalization["delivery_mode"],
            "difficulty_tier": personalization["difficulty_tier"],
            "support_style": personalization["support_style"],
            "feedback_style": personalization["feedback_style"],
            "interaction_goal": _interaction_goal(session),
            "include_formative_check": "yes" if include_formative else "no",
            "revision_required": "yes" if session.revision_required else "no",
        }
    )
    ai_response = result.raw.strip()
    session.add_message("assistant", ai_response)

    if include_formative:
        session.awaiting_formative_response = True
        needs_new_slot = (
            session.current_formative_slot < 0
            or session.current_formative_slot >= len(session.formative_slots)
            or session.formative_slots[session.current_formative_slot] is not None
        )
        if needs_new_slot:
            session.current_formative_slot = len(session.formative_slots)
            session.formative_slots.append(None)
        session.current_formative_prompt = _parse_formative_prompt(ai_response)

    return ai_response


def _reset_learning_after_assessment_fail(session: LearnerSession):
    session.phase = "learning"
    session.learning_turn = 0
    session.competency_interaction = 2
    session.current_subpart_index = 0
    session.formative_check_results = []
    session.formative_slots = []
    session.awaiting_formative_response = False
    session.current_formative_slot = -1
    session.current_formative_prompt = None
    session.revision_required = False
    session.revision_turns_used = 0
    session.consecutive_formative_passes = 0
    session.consecutive_formative_fails = 0
    session.current_difficulty = _difficulty_from_level(session.user_level)
    session.final_assessment_unlocked = False
    session.current_assessment_prompt = None


async def handle_pre_assessment_start(session: LearnerSession) -> dict[str, Any]:
    if not session.intro_delivered:
        build_competency_intro(session)

    if session.pre_assessment_prompt:
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "interaction_number": 2,
            "message": session.pre_assessment_prompt,
            "already_started": True,
        }

    prompt = _generate_preassessment_prompt(session)
    session.pre_assessment_prompt = prompt
    session.pre_assessment_turn = 1
    session.competency_interaction = max(session.competency_interaction, 2)
    session.add_message("assistant", prompt)
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "interaction_number": 2,
        "competency": session.current_competency,
        "message": prompt,
        "question_count": PRE_ASSESSMENT_QUESTION_COUNT,
    }


async def handle_pre_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    session.add_message("user", user_answer)
    classifier_payload = _classify_competency_readiness(session)
    session.user_level = classifier_payload.get("level", session.user_level)
    session.weak_areas = classifier_payload.get("weak_areas", session.weak_areas)
    session.current_difficulty = _difficulty_from_level(session.user_level)
    session.pre_assessment_completed = True
    session.phase = "learning"

    await _setup_competency(session)

    session.learning_turn = 1
    session.competency_interaction = 3
    teaching_response = _generate_learning_response(
        session,
        user_message=f"Learner pre-assessment answer: {user_answer}",
        formative_feedback="Use the pre-assessment answer to personalize the first teaching interaction.",
    )
    _record_remote_teaching_interaction(
        session,
        session.current_competency,
        ai_prompt=session.current_subpart or session.current_competency,
        ai_response=teaching_response,
        learner_input=user_answer,
        formative_passed=None,
    )

    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "done": True,
        "level": session.user_level,
        "weak_areas": session.weak_areas,
        "interaction_number": session.competency_interaction,
        "competency": session.current_competency,
        "current_subpart_index": session.current_subpart_index + 1,
        "current_subpart": session.current_subpart,
        "total_subparts": len(session.competency_subparts.get(session.current_competency, [])),
        "chat_stage": session.chat_stage,
        "bloom_level": session.bloom_level,
        "message": teaching_response,
        "ready_for_assessment": False,
        "personalization_state": session.personalization_state,
    }


def _learning_window_exhausted(session: LearnerSession) -> bool:
    return session.learning_turn >= session.max_learning_window


async def handle_learning(session: LearnerSession, user_message: str) -> dict[str, Any]:
    session.add_message("user", user_message)
    competency = session.current_competency
    formative_feedback = ""
    formative_passed: bool | None = None

    if session.awaiting_formative_response:
        formative_passed, formative_percent, formative_summary = _evaluate_formative_response(session, user_message)
        formative_feedback = _format_formative_feedback(formative_passed, formative_percent, formative_summary)
        _apply_formative_outcome(session, formative_passed, formative_percent, formative_summary)

    if session.final_assessment_unlocked:
        session.phase = "competency_assessment"
        session.competency_interaction += 1
        session.current_assessment_prompt = _generate_assessment_prompt(session)
        session.add_message("assistant", session.current_assessment_prompt)
        save_session(session)
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "competency": competency,
            "interaction_number": session.competency_interaction,
            "message": session.current_assessment_prompt,
            "assessment_prompt": session.current_assessment_prompt,
            "ready_for_assessment": True,
            "backend_warnings": session.backend_warnings,
        }

    if _learning_window_exhausted(session) and not session.final_assessment_unlocked:
        session.revision_required = False
        _reset_learning_after_assessment_fail(session)
        session.learning_turn = 1
        session.competency_interaction = 3
        relearn_feedback = "Mastery gate not met. Restart from interaction 3 and reteach the weakest concepts with simpler explanations."
        ai_response = _generate_learning_response(session, user_message, relearn_feedback)
        _record_remote_teaching_interaction(
            session,
            competency,
            session.current_subpart or competency,
            ai_response,
            user_message,
            formative_passed,
        )
        save_session(session)
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "competency": competency,
            "interaction_number": session.competency_interaction,
            "learning_turn": session.learning_turn,
            "message": ai_response,
            "ready_for_assessment": False,
            "mastery_reset": True,
            "backend_warnings": session.backend_warnings,
        }

    session.learning_turn += 1
    session.competency_interaction = 2 + session.learning_turn
    if session.learning_turn > session.max_learning_turns:
        session.revision_required = True
        session.revision_turns_used = session.learning_turn - session.max_learning_turns

    ai_response = _generate_learning_response(session, user_message, formative_feedback)
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=session.current_subpart or competency,
        ai_response=ai_response,
        learner_input=user_message,
        formative_passed=formative_passed,
    )
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "competency": competency,
        "interaction_number": session.competency_interaction,
        "learning_turn": session.learning_turn,
        "max_learning_turns": session.max_learning_turns,
        "revision_turns_used": session.revision_turns_used,
        "current_subpart_index": session.current_subpart_index + 1,
        "total_subparts": len(session.competency_subparts.get(competency, [])),
        "current_subpart": session.current_subpart,
        "chat_stage": session.chat_stage,
        "bloom_level": session.bloom_level,
        "is_doubt_phase": session.is_doubt_phase,
        "awaiting_formative_response": session.awaiting_formative_response,
        "formative_check_results": session.formative_check_results,
        "message": ai_response,
        "ready_for_assessment": False,
        "personalization_state": session.personalization_state,
        "backend_warnings": session.backend_warnings,
    }


async def handle_competency_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    rubric, scenario, rubric_source = _load_assessment_context(session, competency)
    prompt = session.current_assessment_prompt or _generate_assessment_prompt(session)
    session.current_assessment_attempts += 1
    session.add_message("user", user_answer)

    result = AssessmentCrew().crew().kickoff(
        inputs={
            "competency": competency_label,
            "scenario": prompt or scenario,
            "user_response": user_answer,
            "rubric_json": json.dumps(rubric),
        }
    )
    evaluation = _safe_json_loads(result.raw, {"overall_percent": 0.0, "pass": False, "summary": result.raw})
    overall = float(evaluation.get("overall_percent", 0.0) or 0.0)
    passed = bool(evaluation.get("pass", overall >= PASS_THRESHOLD))
    summary = str(evaluation.get("summary") or "").strip() or "No assessment summary returned."

    remote_learning_session_id = _ensure_remote_learning_session(session, competency)
    if remote_learning_session_id:
        try:
            remote_backend_client.submit_assessment(
                session_id=remote_learning_session_id,
                scenario_question=prompt,
                learner_response=user_answer,
                rubric_score=overall,
                ai_feedback=summary,
                token=session.remote_auth_token,
            )
        except RemoteBackendError as exc:
            warning = f"Remote assessment sync failed for '{competency}': {exc}"
            if warning not in session.backend_warnings:
                session.backend_warnings.append(warning)

    if passed:
        session.completed_competencies.append(
            CompetencyResult(
                competency=competency,
                score=overall,
                passed=True,
                feedback=summary,
            )
        )
        session.add_message("assistant", summary)
        if session.is_last_competency:
            session.phase = "completed"
            save_session(session)
            return {
                "session_id": session.session_id,
                "phase": "completed",
                "assessed_competency": competency,
                "score": overall,
                "passed": True,
                "message": (
                    f"Assessment passed for **{competency}** with **{overall:.1f}%**. "
                    "All competencies are complete. Credential can now be issued."
                ),
                "assessment_detail": evaluation,
                "rubric_source": rubric_source,
                "backend_warnings": session.backend_warnings,
            }

        session.advance_to_next_competency()
        next_intro = build_competency_intro(session)
        save_session(session)
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "assessed_competency": competency,
            "score": overall,
            "passed": True,
            "message": (
                f"Assessment passed for **{competency}** with **{overall:.1f}%**. "
                f"Next competency: **{session.current_competency}**.\n\n{next_intro}"
            ),
            "assessment_detail": evaluation,
            "rubric_source": rubric_source,
            "next_competency": session.current_competency,
            "backend_warnings": session.backend_warnings,
        }

    _reset_learning_after_assessment_fail(session)
    session.learning_turn = 1
    session.competency_interaction = 3
    relearn_feedback = (
        f"Assessment score was {overall:.1f}%. Restart from interaction 3. "
        "Use the assessment feedback to reteach the weakest concepts."
    )
    learning_message = _generate_learning_response(session, user_answer, relearn_feedback)
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=session.current_subpart or competency,
        ai_response=learning_message,
        learner_input=user_answer,
        formative_passed=False,
    )
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "assessed_competency": competency,
        "score": overall,
        "passed": False,
        "message": learning_message,
        "assessment_feedback": summary,
        "assessment_detail": evaluation,
        "rubric_source": rubric_source,
        "interaction_number": session.competency_interaction,
        "backend_warnings": session.backend_warnings,
    }


async def handle_final_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    session.add_message("user", user_answer)
    summary = {
        "competencies_completed": len(session.completed_competencies),
        "total_competencies": len(session.competencies),
        "final_answer_received": True,
    }
    session.phase = "completed"
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": "completed",
        "message": "This engine now completes the credential after the final competency assessment. No separate final assessment is required.",
        "summary": summary,
        "backend_warnings": session.backend_warnings,
    }
