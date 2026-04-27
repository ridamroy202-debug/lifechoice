import hashlib
import logging
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

import yaml

from app.crews.ai_tutor_agents_crew import TutorCrew
from app.crews.assessment_crew import AssessmentCrew
from app.crews.pre_assessment_crew import PreAssessCrew
from app.persistence import (
    append_event_log,
    create_badge,
    get_locked_rubric,
    get_rubric_source_hash,
    get_rubric_version,
    normalize_rubric_key,
    upsert_learner_competency_progress,
    record_competency_attempt,
    record_final_assessment,
    record_formative_check,
    utc_now_iso,
)
from app.policy import (
    build_gamification_payload,
    build_session_runtime_payload,
    build_session_summary,
    detect_and_record_anomalies,
    encouragement_message,
)
from app.remote_backend import RemoteBackendError, remote_backend_client
from app.session_manager import save_session
from app.state import CompetencyResult, LearnerSession

PASS_THRESHOLD = 75.0
PRE_ASSESSMENT_QUESTION_COUNT = 2
BASE_LEARNING_INTERACTIONS = 6
REVISION_INTERACTIONS = 2
EASY_PASS_THRESHOLD = 90.0
INTRO_SENTENCE_LIMIT = 3
TEACHING_WORD_TARGETS = {
    "foundation": {"floor": 420, "target": 560},
    "guided_application": {"floor": 460, "target": 620},
    "mastery_gate": {"floor": 500, "target": 680},
    "revision": {"floor": 440, "target": 580},
}
ACADEMIC_STAGE_WORD_BONUS = {
    "bachelor": 0,
    "masters": 60,
    "phd": 120,
    "professional": 40,
}
REQUIRED_TUTOR_SECTIONS = (
    "## Title",
    "## Learner Feedback",
    "## What This Concept Means",
    "## How It Works",
    "## Visual Aid",
    "## Example",
    "## Key Takeaway",
    "## Next Learner Action",
)

_RUBRICS_CACHE: dict[str, Any] | None = None
logger = logging.getLogger(__name__)
_FORMATIVE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from", "how",
    "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "so", "that",
    "the", "their", "them", "they", "this", "to", "what", "when", "which", "why", "with",
    "would", "you", "your", "startup", "mission", "statement", "focus", "align", "aligned",
    "develop", "developing", "key", "elements", "visual", "identity", "brand", "question",
    "through",
}
_FORMATIVE_REASONING_MARKERS = (
    "because",
    "so that",
    "which helps",
    "this helps",
    "this supports",
    "this aligns",
    "therefore",
    "so the",
    "ensures",
    "to show",
    "to signal",
)
_FORMATIVE_VISUAL_ELEMENT_KEYWORDS = {
    "logo", "logos", "typography", "typeface", "font", "fonts", "color", "colors",
    "palette", "palettes", "imagery", "image", "images", "icon", "icons", "iconography",
    "layout", "layouts", "grid", "grids", "illustration", "illustrations", "photography",
    "shape", "shapes", "mark", "wordmark", "visual", "voice", "messaging",
}
COMPETENT_LABEL = "COMPETENT"
NOT_YET_COMPETENT_LABEL = "NOT YET COMPETENT"
LIVE_AI_AIP_CODES = frozenset(
    {
        "AIP-02",
        "AIP-03",
        "AIP-04",
        "AIP-05",
        "AIP-06",
        "AIP-07",
        "AIP-08",
        "AIP-09",
        "AIP-10",
        "AIP-11",
        "AIP-12",
        "AIP-14",
    }
)


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


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", _normalize_whitespace(text)) if part.strip()]


def _truncate_sentences(text: str, max_sentences: int) -> str:
    sentences = _sentence_split(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences])


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _teaching_length_policy(session: LearnerSession) -> dict[str, int]:
    stage = session.chat_stage
    base = TEACHING_WORD_TARGETS.get(stage, TEACHING_WORD_TARGETS["guided_application"]).copy()
    bonus = ACADEMIC_STAGE_WORD_BONUS.get(session.academic_stage, 0)
    base["floor"] += bonus
    base["target"] += bonus
    base["ceiling"] = base["target"] + 180
    return base


def _stage_teaching_instruction(session: LearnerSession) -> str:
    mapping = {
        "foundation": (
            "Introduce one foundational concept with intuition, mechanism, and a concrete workplace anchor. "
            "Keep the tone approachable and scaffold the explanation for a learner who may still be building confidence."
        ),
        "guided_application": (
            "Teach one concept through application. Show how the concept is used, what decision points matter, and why the sequence works."
        ),
        "mastery_gate": (
            "Deepen one concept to university-grade clarity. Include decision criteria, failure patterns, and retrieval practice that forces applied reasoning."
        ),
        "revision": (
            "Reteach the weakest concept using a meaningfully different format, simpler language where needed, and a new example that directly addresses the misconception."
        ),
    }
    return mapping.get(session.chat_stage, mapping["guided_application"])


def _missing_tutor_sections(ai_response: str, include_formative: bool) -> list[str]:
    lowered = ai_response.lower()
    missing = [section for section in REQUIRED_TUTOR_SECTIONS if section.lower() not in lowered]
    if include_formative and "formative check" not in lowered:
        missing.append("## Formative Check")
    return missing


def _log_session_event(session: LearnerSession, route: str, event_type: str, payload: dict[str, Any]) -> None:
    append_event_log(session.session_id, session.learner_id, route, event_type, payload)


def _record_session_interaction(
    session: LearnerSession,
    *,
    interaction_type: str,
    interaction_number: int,
    delivery_format: str | None = None,
    concept: str | None = None,
) -> None:
    concept_name = concept or session.current_subpart or session.current_competency
    session.record_interaction_event(
        interaction_type=interaction_type,  # type: ignore[arg-type]
        concept=concept_name,
        delivery_format=delivery_format,
        interaction_number=interaction_number,
        phase=session.phase,
    )
    _log_session_event(
        session,
        "/interaction",
        "interaction_recorded",
        {
            "interaction_number": interaction_number,
            "interaction_type": interaction_type,
            "delivery_format": delivery_format,
            "concept": concept_name,
            "phase": session.phase,
        },
    )


def _binary_outcome_label(passed: bool) -> str:
    return COMPETENT_LABEL if passed else NOT_YET_COMPETENT_LABEL


def _run_mapped_ai_call(
    session: LearnerSession,
    aip_code: str,
    *,
    purpose: str,
    crew_factory,
    inputs: dict[str, Any],
) -> Any:
    if aip_code not in LIVE_AI_AIP_CODES:
        raise RuntimeError(f"Unmapped AI call attempted for {aip_code}.")
    session.record_live_aip_call(aip_code=aip_code, purpose=purpose, metadata={"input_keys": sorted(inputs.keys())})
    return crew_factory().crew().kickoff(inputs=inputs)


def _record_aip(
    session: LearnerSession,
    aip_code: str,
    *,
    trigger: str,
    scope: str = "cc",
    outcome: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    session.record_aip_event(
        aip_code=aip_code,
        trigger=trigger,
        scope=scope,  # type: ignore[arg-type]
        outcome=outcome,
        metadata=metadata,
    )


def _build_static_study_material(session: LearnerSession, competency: str, context_description: str) -> str:
    description = context_description or f"Use {competency} to deliver measurable professional outcomes."
    weak_focus = ", ".join(session.weak_areas[:3]) or "scope, stakeholders, and measurable outcomes"
    return (
        f"# {competency}\n\n"
        f"## Core Definition\n{description}\n\n"
        f"## Why It Matters\n"
        f"This competency matters in {session.topic} because it turns planning into repeatable delivery decisions.\n\n"
        f"## Core Focus Areas\n"
        f"- Objective and scope clarity\n"
        f"- Stakeholder alignment\n"
        f"- Measurable outcomes and review points\n"
        f"- Common failure patterns: {weak_focus}\n"
    )


def _build_static_learning_plan(session: LearnerSession, competency: str, context_description: str) -> str:
    description = context_description or f"Apply {competency.lower()} in a professional setting."
    return "\n".join(
        [
            f"1. First concept and intuition - explain the core purpose of {competency} in practical terms.",
            f"2. Mechanism and workflow - show the repeatable steps used to perform {competency.lower()} well.",
            f"3. Worked example - walk through a realistic scenario using {competency.lower()} and measurable outcomes.",
            f"4. Decision quality - compare strong versus weak choices, tradeoffs, and stakeholder implications in {competency.lower()}.",
            f"5. Common mistakes and recovery - identify predictable errors and how to correct them in {competency.lower()}.",
            f"6. Assessment readiness - consolidate the key ideas from: {description}",
        ]
    )


def _derive_default_weak_areas(session: LearnerSession) -> list[str]:
    competency = session.current_competency.lower()
    return [
        f"understanding of {competency} principles",
        "application of measurable outcomes in decisions",
        "clear justification of chosen actions",
    ]


def _classify_answer_depth(answer: str) -> str:
    lowered = answer.lower()
    advanced_markers = ("tradeoff", "constraint", "risk", "dependency", "mitigate", "measure", "metric")
    intermediate_markers = ("because", "stakeholder", "scope", "priority", "outcome", "plan")
    if sum(marker in lowered for marker in advanced_markers) >= 2:
        return "advanced"
    if sum(marker in lowered for marker in intermediate_markers) >= 2:
        return "intermediate"
    return "beginner"

    _log_session_event(
        session,
        "/aip",
        "aip_recorded",
        {
            "aip_code": aip_code,
            "trigger": trigger,
            "scope": scope,
            "outcome": outcome,
            "metadata": metadata or {},
        },
    )


def _build_static_remediation_message(
    session: LearnerSession,
    *,
    title: str,
    summary: str,
    weakest_focus: str,
) -> str:
    competency = session.current_competency
    return (
        f"## {title}\n\n"
        f"### Developing Competency\n"
        f"You are not yet competent in **{competency}**. Review this targeted guidance before reattempting.\n\n"
        f"### What needs attention\n"
        f"{summary}\n\n"
        f"### Remedial focus\n"
        f"- Re-state the goal of **{competency}** in your own words.\n"
        f"- Work through one concrete example focused on: {weakest_focus}.\n"
        f"- Explain why your chosen action fits the scenario, not just what you would do.\n"
        f"- Prepare a tighter, evidence-based answer before the next gate.\n\n"
        "### Next step\n"
        "Use the next interaction to show a clearer and more applied response."
    )


def _sync_prompt_with_metadata(session: LearnerSession, ai_prompt: str, interaction_type: str) -> str:
    payload = {
        "interaction_type": interaction_type,
        "gamification": build_gamification_payload(session),
        "required_next_action": session.required_next_action,
    }
    return f"{ai_prompt}\n\n[AI_ENGINE_SYNC]{json.dumps(payload, sort_keys=True)}"


def _set_remote_sync_success(session: LearnerSession, remote_session_id: int | None) -> None:
    session.set_remote_sync(outcome="synced", backend_session_id=remote_session_id)
    _log_session_event(
        session,
        "/remote-sync",
        "remote_sync_succeeded",
        {
            "backend_session_id": remote_session_id,
            "last_sync_at": session.remote_sync_status.get("last_sync_at"),
        },
    )


def _set_remote_sync_failure(session: LearnerSession, warning: str, remote_session_id: int | None = None) -> None:
    if warning not in session.backend_warnings:
        session.backend_warnings.append(warning)
    session.set_remote_sync(outcome="warning", backend_session_id=remote_session_id, warning=warning)
    _log_session_event(
        session,
        "/remote-sync",
        "remote_sync_failed",
        {
            "backend_session_id": remote_session_id,
            "warning": warning,
        },
    )


def _extract_remote_learning_session(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    session_payload = payload.get("session") if isinstance(payload.get("session"), dict) else payload
    return session_payload if isinstance(session_payload, dict) else {}


def _remote_competency_pass_confirmed(remote_session: dict[str, Any]) -> bool | None:
    if not remote_session:
        return None
    mastery = remote_session.get("mastery_achieved")
    status = str(remote_session.get("status") or "").strip().lower()
    if mastery is True or status == "completed":
        return True
    if mastery is False or status in {"active", "in_progress", "failed", "pending"}:
        return False
    return None


def _sync_remote_competency_assessment(
    session: LearnerSession,
    *,
    competency: str,
    remote_required: bool,
    remote_learning_session_id: int | None,
    prompt: str,
    user_answer: str,
    overall: float,
    summary: str,
    local_passed: bool,
) -> dict[str, Any]:
    session.local_assessment_passed = local_passed
    session.remote_assessment_synced = False
    session.remote_assessment_passed = None
    session.current_assessment_sync_error = None

    if not remote_learning_session_id:
        if remote_required:
            warning = f"Remote assessment sync failed for '{competency}': no remote learning session is available."
            session.current_assessment_sync_error = warning
            _set_remote_sync_failure(session, warning, None)
            _log_session_event(
                session,
                "/assessment/competency",
                "remote_assessment_sync_failed",
                {
                    "competency": competency,
                    "remote_session_id": None,
                    "local_passed": local_passed,
                    "local_score": overall,
                    "warning": warning,
                },
            )
            return {
                "remote_required": True,
                "confirmed": None,
                "remote_passed": None,
                "remote_session": None,
                "submit_response": None,
                "fetch_response": None,
                "warning": warning,
            }
        return {
            "remote_required": False,
            "confirmed": local_passed,
            "remote_passed": None,
            "remote_session": None,
            "submit_response": None,
            "fetch_response": None,
            "warning": None,
        }

    submit_payload = {
        "session_id": remote_learning_session_id,
        "scenario_question": prompt,
        "learner_response": user_answer,
        "rubric_score": overall,
        "ai_feedback": summary,
    }
    _log_session_event(
        session,
        "/assessment/competency",
        "remote_assessment_sync_started",
        {
            "competency": competency,
            "remote_session_id": remote_learning_session_id,
            "local_passed": local_passed,
            "local_score": overall,
            "payload": submit_payload,
        },
    )

    try:
        submit_response = remote_backend_client.submit_assessment(
            session_id=remote_learning_session_id,
            scenario_question=prompt,
            learner_response=user_answer,
            rubric_score=overall,
            ai_feedback=summary,
            token=session.remote_auth_token,
        )
        fetch_response = remote_backend_client.fetch_learning_session(
            remote_learning_session_id,
            token=session.remote_auth_token,
        )
    except RemoteBackendError as exc:
        warning = f"Remote assessment sync failed for '{competency}': {exc}"
        session.current_assessment_sync_error = warning
        _set_remote_sync_failure(session, warning, remote_learning_session_id)
        _log_session_event(
            session,
            "/assessment/competency",
            "remote_assessment_sync_failed",
            {
                "competency": competency,
                "remote_session_id": remote_learning_session_id,
                "local_passed": local_passed,
                "local_score": overall,
                "payload": submit_payload,
                "warning": warning,
            },
        )
        return {
            "remote_required": True,
            "confirmed": None,
            "remote_passed": None,
            "remote_session": None,
            "submit_response": None,
            "fetch_response": None,
            "warning": warning,
        }

    remote_session = _extract_remote_learning_session(fetch_response)
    remote_passed = _remote_competency_pass_confirmed(remote_session)
    session.remote_assessment_synced = True
    session.remote_assessment_passed = remote_passed
    if remote_passed is None:
        warning = f"Remote assessment confirmation for '{competency}' was inconclusive."
        session.current_assessment_sync_error = warning
        _set_remote_sync_failure(session, warning, remote_learning_session_id)
    else:
        session.current_assessment_sync_error = None
        _set_remote_sync_success(session, remote_learning_session_id)

    _log_session_event(
        session,
        "/assessment/competency",
        "remote_assessment_sync_completed",
        {
            "competency": competency,
            "remote_session_id": remote_learning_session_id,
            "local_passed": local_passed,
            "local_score": overall,
            "payload": submit_payload,
            "submit_response": submit_response,
            "fetch_response": fetch_response,
            "remote_passed": remote_passed,
        },
    )
    return {
        "remote_required": True,
        "confirmed": remote_passed,
        "remote_passed": remote_passed,
        "remote_session": remote_session,
        "submit_response": submit_response,
        "fetch_response": fetch_response,
        "warning": session.current_assessment_sync_error,
    }


def _enforce_question_count(text: str, max_questions: int = PRE_ASSESSMENT_QUESTION_COUNT) -> str:
    questions = [part.strip() for part in re.findall(r"[^?]*\?", text, flags=re.MULTILINE) if part.strip()]
    selected = questions[:max_questions]
    if not selected:
        selected = [
            "In a real scenario, what would you do first and why?",
            "What result would tell you the approach is working?",
        ][:max_questions]
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(selected, start=1))


def _runtime_fields(session: LearnerSession) -> dict[str, Any]:
    return build_session_runtime_payload(session)


def build_competency_intro(session: LearnerSession) -> str:
    competency = session.current_competency
    details = _get_competency_details(session, competency)
    description = _truncate_sentences(str(details.get("description") or "").strip(), 1)
    brief = _three_word_competency_brief(competency)
    sentence_one = f"We are starting **{competency}**, and this competency matters because it helps you make sound decisions in real work situations."
    sentence_two = (
        f"By the end of this session, you should be able to use {competency.lower()} with clear reasoning, practical structure, and confident judgment."
    )
    sentence_three = (
        f"We will move one concept at a time through **{brief}**, starting with a short diagnostic so the teaching depth matches your current level."
    )
    if description:
        sentence_one = f"We are starting **{competency}**: {description}"
    intro_message = f"{sentence_one} {sentence_two} {sentence_three}"
    session.intro_delivered = True
    session.competency_interaction = 1
    session.add_message("assistant", intro_message)
    _record_session_interaction(session, interaction_type="intro", interaction_number=1, concept=competency)
    _record_aip(session, "AIP-01", trigger="intro_delivered", metadata={"competency": competency})
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=f"Introduce the competency {competency} in warm, simple language.",
        ai_response=intro_message,
        learner_input=None,
        formative_passed=None,
        interaction_type="intro",
        allow_session_creation=True,
    )
    save_session(session)
    return intro_message


def _normalize_remote_rubric(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None

    rubric_root = payload.get("rubric_rules") if isinstance(payload.get("rubric_rules"), dict) else payload
    raw_criteria = rubric_root.get("rubric_rules") or rubric_root.get("rubric_criteria") or {}
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
                "name": str(item.get("criterion_name") or item.get("criterion") or item.get("name") or f"Criterion {index}"),
                "description": str(item.get("criterion_descriptor") or item.get("descriptor") or item.get("met_indicator") or ""),
                "weight": float(item.get("weight") or weight),
                "criterion_id": str(item.get("criterion_id") or item.get("id") or f"c{index}"),
            }
        )
    if not criteria:
        return None

    payload_json = json.dumps({"criteria": criteria, "pass_threshold": rubric_root.get("pass_threshold") or PASS_THRESHOLD}, sort_keys=True)
    return {
        "rubric_key": normalize_rubric_key(str(rubric_root.get("competency_title") or "")) or None,
        "display_name": rubric_root.get("competency_title"),
        "criteria": criteria,
        "pass_threshold": float(rubric_root.get("pass_threshold") or PASS_THRESHOLD),
        "scenario_template": rubric_root.get("scenario_template"),
        "difficulty_level": rubric_root.get("difficulty_level"),
        "version": int(rubric_root.get("version") or 1),
        "source_hash": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "source": "remote_db",
    }


def _load_assessment_context(session: LearnerSession, competency: str) -> tuple[dict[str, Any], str, str]:
    details = _get_competency_details(session, competency)
    if competency in session.rubric_cache:
        cached = session.rubric_cache[competency]
        scenario = cached.get("scenario_template") or details.get("description") or session.study_materials.get(competency, "")
        return cached, str(scenario), str(cached.get("source") or "db_locked")

    remote_competency_id = session.current_remote_competency_id or details.get("id")
    if remote_competency_id:
        try:
            remote_payload = remote_backend_client.fetch_competency_rubric(int(remote_competency_id), token=session.remote_auth_token)
        except RemoteBackendError:
            remote_payload = None
        remote_rubric = _normalize_remote_rubric(remote_payload)
        if remote_rubric:
            if not remote_rubric.get("rubric_key"):
                remote_rubric["rubric_key"] = normalize_rubric_key(competency)
            if not remote_rubric.get("display_name"):
                remote_rubric["display_name"] = competency
            session.rubric_cache[competency] = remote_rubric
            scenario = remote_rubric.get("scenario_template") or details.get("description") or session.study_materials.get(competency, "")
            return remote_rubric, str(scenario), "remote_db"

    locked = get_locked_rubric(competency)
    if not locked:
        raise RuntimeError(f"Locked rubric missing for competency '{competency}'. Provision it in the database before assessment.")

    locked["source_hash"] = get_rubric_source_hash(competency)
    session.rubric_cache[competency] = locked
    scenario = locked.get("scenario_template") or details.get("description") or session.study_materials.get(competency, "")
    return locked, str(scenario), "db_locked"


def _ensure_remote_learning_session(session: LearnerSession, competency: str) -> int | None:
    existing = session.remote_learning_sessions.get(competency)
    if existing:
        _set_remote_sync_success(session, existing)
        return existing

    details = _get_competency_details(session, competency)
    remote_competency_id = details.get("id")
    if session.source != "remote" or not isinstance(remote_competency_id, int):
        return None

    try:
        payload = remote_backend_client.start_learning_session(
            competency_id=remote_competency_id,
            token=session.remote_auth_token,
        )
    except RemoteBackendError as exc:
        warning = f"Could not start remote learning session for '{competency}': {exc}"
        _set_remote_sync_failure(session, warning)
        return None

    session_payload = payload.get("session") if isinstance(payload.get("session"), dict) else payload
    remote_session_id = session_payload.get("id")
    if isinstance(remote_session_id, int):
        session.remote_learning_sessions[competency] = remote_session_id
        _set_remote_sync_success(session, remote_session_id)
        return remote_session_id
    _set_remote_sync_failure(session, f"Remote learning session for '{competency}' did not return an integer id.")
    return None


def _safe_json_loads(raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_text, str):
        stripped = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            raw_text = fenced_match.group(1)
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _normalize_eval_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _coerce_boolish(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "y", "1", "met", "pass", "passed", "proficient"}:
        return True
    if lowered in {"false", "no", "n", "0", "not_met", "not met", "fail", "failed"}:
        return False
    return None


def _normalize_formative_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize_formative_text(text: str) -> list[str]:
    return [_normalize_formative_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _extract_significant_prompt_terms(prompt: str) -> set[str]:
    tokens = set()
    for token in _tokenize_formative_text(prompt):
        if len(token) < 4 or token in _FORMATIVE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _build_formative_heuristics(prompt: str, learner_answer: str, competency: str) -> dict[str, Any]:
    prompt_terms = _extract_significant_prompt_terms(prompt)
    answer_terms = set(_tokenize_formative_text(learner_answer))
    shared_terms = sorted(prompt_terms & answer_terms)
    mission_match = re.search(r'"([^"]+)"', prompt or "")
    mission_terms = set(_tokenize_formative_text(mission_match.group(1))) if mission_match else set()
    mission_overlap = sorted(term for term in mission_terms if term in answer_terms and len(term) >= 4)
    has_reasoning = any(marker in learner_answer.lower() for marker in _FORMATIVE_REASONING_MARKERS)
    requires_visual_elements = any(
        clue in f"{competency} {prompt}".lower()
        for clue in ("brand", "visual", "identity", "logo", "typography", "color", "imagery")
    )
    visual_elements = sorted(term for term in answer_terms if term in _FORMATIVE_VISUAL_ELEMENT_KEYWORDS)
    mentions_specific_elements = len(visual_elements) >= (2 if requires_visual_elements else 1)

    scenario_relevance = bool(mission_overlap) or len(shared_terms) >= 3
    explanation_quality = has_reasoning
    concrete_application = mentions_specific_elements if requires_visual_elements else bool(shared_terms) and has_reasoning
    if not scenario_relevance and requires_visual_elements and concrete_application and explanation_quality:
        scenario_relevance = True

    met_count = sum([scenario_relevance, concrete_application, explanation_quality])
    overall_percent = round((met_count / 3.0) * 100.0, 2)
    return {
        "scenario_relevance": scenario_relevance,
        "concrete_application": concrete_application,
        "explanation_quality": explanation_quality,
        "mission_overlap": mission_overlap,
        "shared_terms": shared_terms[:10],
        "visual_elements": visual_elements,
        "requires_visual_elements": requires_visual_elements,
        "overall_percent": overall_percent,
        "pass": met_count >= 2 and scenario_relevance and explanation_quality,
    }


def _normalize_binary_evaluation(evaluation: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    criteria = rubric.get("criteria", [])
    raw_scores = evaluation.get("criteria_scores") or []
    by_key: dict[str, dict[str, Any]] = {}
    for item in raw_scores:
        for candidate_key in (
            item.get("criterion_id"),
            item.get("name"),
            item.get("criterion"),
            item.get("criterion_name"),
        ):
            key = _normalize_eval_key(candidate_key)
            if key:
                by_key[key] = item

    normalized_scores: list[dict[str, Any]] = []
    total = 0.0
    matched_count = 0
    for index, criterion in enumerate(criteria, start=1):
        criterion_id = str(criterion.get("criterion_id") or f"c{index}")
        weight = float(criterion.get("weight", 0.0) or 0.0)
        candidate = (
            by_key.get(_normalize_eval_key(criterion_id))
            or by_key.get(_normalize_eval_key(criterion.get("name")))
            or by_key.get(_normalize_eval_key(criterion.get("description")))
        )
        if candidate is None and len(raw_scores) == len(criteria):
            candidate = raw_scores[index - 1]
        candidate = candidate or {}
        if candidate:
            matched_count += 1
        met_value = _coerce_boolish(candidate.get("met"))
        if met_value is None:
            raw_score = candidate.get("score") or candidate.get("value")
            if isinstance(raw_score, (int, float)):
                met = float(raw_score) >= 75.0
            else:
                label = str(candidate.get("rating") or candidate.get("assessment") or "").lower()
                met = label in {"met", "pass", "passed", "proficient", "yes", "true"}
        else:
            met = met_value
        evidence = str(candidate.get("evidence") or candidate.get("reason") or candidate.get("summary") or "").strip()
        normalized_scores.append({
            "criterion_id": criterion_id,
            "met": met,
            "evidence": evidence,
        })
        if met:
            total += weight * 100.0

    overall = round(total, 2)
    raw_overall = evaluation.get("overall_percent")
    try:
        raw_overall_value = float(raw_overall)
    except (TypeError, ValueError):
        raw_overall_value = None
    raw_pass_value = _coerce_boolish(evaluation.get("pass"))

    if raw_overall_value is not None:
        needs_fallback = matched_count == 0 or (
            matched_count < max(1, len(criteria) // 2) and raw_overall_value >= PASS_THRESHOLD > overall
        )
        if needs_fallback:
            overall = round(raw_overall_value, 2)

    passed = overall >= PASS_THRESHOLD
    if raw_pass_value is not None and (matched_count == 0 or raw_pass_value):
        passed = raw_pass_value if raw_overall_value is None else bool(raw_pass_value or overall >= PASS_THRESHOLD)
    return {
        "criteria_scores": normalized_scores,
        "overall_percent": overall,
        "pass": passed,
        "summary": str(evaluation.get("summary") or "").strip(),
    }


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
        "guided_explanation",
        "worked_example",
        "comparison_table",
        "scenario_walkthrough",
        "mini_challenge",
        "decision_framework",
    ]
    index = len(session.delivery_format_history) % len(modes)
    candidate = modes[index]
    if session.delivery_format_history and candidate == session.delivery_format_history[-1]:
        candidate = modes[(index + 1) % len(modes)]
    return candidate


def _build_personalization_state(session: LearnerSession) -> dict[str, str]:
    delivery_mode = _delivery_mode(session)
    state = {
        "difficulty_tier": session.current_difficulty,
        "delivery_mode": delivery_mode,
        "support_style": "high-guidance" if session.current_difficulty == "support" else ("challenging-coach" if session.current_difficulty == "stretch" else "coaching"),
        "feedback_style": "encouraging" if session.consecutive_formative_fails == 0 else "empathetic-recovery",
        "mastery_status": "ready" if session.final_assessment_unlocked else "in_progress",
        "revision_mode": "active" if session.revision_required else "inactive",
        "spaced_learning_rule": "one concept per interaction",
        "academic_stage": session.academic_stage,
        "academic_guidance": session.academic_guidance,
    }
    session.personalization_state = state
    return state


async def _setup_competency(session: LearnerSession):
    competency = session.current_competency
    details = _get_competency_details(session, competency)
    context_description = str(details.get("description") or "").strip()

    if competency not in session.study_materials:
        session.study_materials[competency] = _build_static_study_material(session, competency, context_description)

    if competency not in session.learning_plans:
        session.learning_plans[competency] = _build_static_learning_plan(session, competency, context_description)

    if competency not in session.competency_subparts:
        session.competency_subparts[competency] = _extract_subparts_from_plan(
            session.learning_plans.get(competency, ""),
            fallback_turns=BASE_LEARNING_INTERACTIONS,
        )


def _generate_preassessment_prompt(session: LearnerSession) -> str:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    result = _run_mapped_ai_call(
        session,
        "AIP-02",
        purpose="diagnostic_prompt_generation",
        crew_factory=PreAssessCrew,
        inputs={
            "topic": session.topic,
            "competencies": competency_label,
            "chat_history": session.format_recent_history(),
            "user_message": f"Generate {PRE_ASSESSMENT_QUESTION_COUNT} applied pre-assessment questions for this competency.",
            "turn_number": 1,
        },
    )
    return _enforce_question_count(result.raw.strip())


def _classify_competency_readiness(session: LearnerSession) -> dict[str, Any]:
    learner_answer = next((msg.content for msg in reversed(session.messages) if msg.role == "user"), "")
    if not learner_answer or len(_normalize_whitespace(learner_answer)) < 12:
        return {"level": "beginner", "weak_areas": _derive_default_weak_areas(session)}

    level = _classify_answer_depth(learner_answer)
    weak_areas = _derive_default_weak_areas(session)
    lowered = learner_answer.lower()
    if "stakeholder" in lowered:
        weak_areas = [item for item in weak_areas if "stakeholder" not in item.lower()] or weak_areas
    if "measure" in lowered or "metric" in lowered or "outcome" in lowered:
        weak_areas = [item for item in weak_areas if "measurable outcomes" not in item.lower()] or weak_areas
    if "because" in lowered or "therefore" in lowered or "tradeoff" in lowered:
        weak_areas = [item for item in weak_areas if "justification" not in item.lower()] or weak_areas
    return {"level": level, "weak_areas": weak_areas[:3]}


def _diagnostic_answer_is_meaningful(session: LearnerSession, learner_answer: str) -> bool:
    text = _normalize_whitespace(learner_answer)
    if not text:
        return False

    lowered = text.lower()
    if lowered in {
        "string",
        "test",
        "testing",
        "hello",
        "hi",
        "ok",
        "okay",
        "yes",
        "no",
        "n/a",
        "na",
        "none",
    }:
        return False

    tokens = _tokenize_formative_text(text)
    if len(tokens) < 4:
        return False

    answer_terms = set(tokens)
    prompt_terms = _extract_significant_prompt_terms(
        f"{session.current_competency} {session.pre_assessment_prompt or ''}"
    )
    shared_terms = prompt_terms & answer_terms
    has_reasoning = any(marker in lowered for marker in _FORMATIVE_REASONING_MARKERS) or any(
        marker in lowered for marker in ("i would", "i will", "first", "then", "so that")
    )
    has_structured_action = any(
        term in answer_terms
        for term in {
            "stakeholder",
            "stakeholders",
            "scope",
            "goal",
            "goals",
            "outcome",
            "outcomes",
            "measure",
            "metrics",
            "priority",
            "priorities",
            "team",
            "process",
            "criteria",
            "risk",
            "analysis",
            "plan",
        }
    )

    if len(shared_terms) >= 2 and (has_reasoning or len(tokens) >= 8):
        return True
    if len(tokens) >= 10 and has_reasoning and has_structured_action:
        return True
    return False


def _format_formative_feedback(passed: bool, percent: float, summary: str, *, streak_bonus: bool = False) -> str:
    prefix = encouragement_message(passed, streak_bonus)
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


def _apply_formative_outcome(session: LearnerSession, passed: bool, percent: float, summary: str, *, easy_pass: bool) -> str:
    _update_formative_slot(session, passed)
    current_slot = max(session.formative_slot_number or 1, 1)
    feedback_aip = {
        1: "AIP-06",
        2: "AIP-08",
        3: "AIP-10",
    }.get(current_slot)
    if feedback_aip:
        _record_aip(
            session,
            feedback_aip,
            trigger=f"fa{current_slot}_feedback",
            outcome=_binary_outcome_label(passed),
            metadata={"score": percent},
        )
    points_delta = session.award_points_for_formative(passed)
    session.formative_feedback_log.append(
        {
            "passed": passed,
            "score": percent,
            "summary": summary,
            "points_delta": points_delta,
            "easy_pass": easy_pass,
        }
    )
    session.awaiting_formative_response = False
    session.current_formative_prompt = None

    if passed:
        session.developing_competency_active = False
        session.developing_competency_reason = None
        session.consecutive_formative_passes += 1
        session.consecutive_formative_fails = 0
        session.consecutive_easy_passes = session.consecutive_easy_passes + 1 if easy_pass else 0
        if session.consecutive_easy_passes >= 2:
            session.current_difficulty = _raise_difficulty(session.current_difficulty)
        if session.current_subpart_index < len(session.competency_subparts.get(session.current_competency, [])) - 1:
            session.current_subpart_index += 1
    else:
        session.developing_competency_active = True
        session.developing_competency_reason = "Formative gate not yet cleared."
        session.consecutive_formative_passes = 0
        session.consecutive_easy_passes = 0
        session.consecutive_formative_fails += 1
        session.current_difficulty = _lower_difficulty(session.current_difficulty)
        if session.consecutive_formative_fails >= 2 or sum(item is False for item in session.formative_slots) >= 2:
            session.revision_required = True

    if session.learning_turn >= BASE_LEARNING_INTERACTIONS and _all_formative_slots_passed(session):
        session.final_assessment_unlocked = True

    return _format_formative_feedback(passed, percent, summary, streak_bonus=session.streak_bonus_awarded)


def _build_formative_rubric(session: LearnerSession) -> dict[str, Any]:
    concept = session.current_subpart or session.current_competency
    prompt = session.current_formative_prompt or concept
    requires_visual_elements = any(
        clue in f"{session.current_competency} {prompt}".lower()
        for clue in ("brand", "visual", "identity", "logo", "typography", "color", "imagery")
    )
    application_description = "Uses the concept in the scenario rather than only defining it"
    if requires_visual_elements:
        application_description = (
            "Names concrete visual identity elements or brand-system decisions that fit the scenario, "
            "such as logo logic, color palette, typography, imagery, layout, or brand voice."
        )
    return {
        "criteria": [
            {
                "criterion_id": "formative_accuracy",
                "name": "Scenario relevance",
                "description": f"Addresses the specific scenario and mission in the prompt while using the current concept: {prompt}",
                "weight": 0.34,
            },
            {
                "criterion_id": "formative_application",
                "name": "Concrete application",
                "description": application_description,
                "weight": 0.33,
            },
            {
                "criterion_id": "formative_explanation",
                "name": "Clear explanation",
                "description": "Explains why the chosen action or element fits the scenario instead of giving only generic statements",
                "weight": 0.33,
            },
        ],
        "pass_threshold": PASS_THRESHOLD,
        "binary_scoring": True,
    }


def _evaluate_formative_response(session: LearnerSession, learner_answer: str) -> tuple[bool, float, str, bool]:
    rubric = _build_formative_rubric(session)
    prompt = session.current_formative_prompt or session.current_subpart or session.current_competency
    feedback_aip = {
        1: "AIP-06",
        2: "AIP-08",
        3: "AIP-10",
    }.get(max(session.formative_slot_number or 1, 1), "AIP-06")
    result = _run_mapped_ai_call(
        session,
        feedback_aip,
        purpose="formative_evaluation",
        crew_factory=AssessmentCrew,
        inputs={
            "competency": f"Formative check for {session.current_competency}",
            "scenario": prompt,
            "user_response": learner_answer,
            "rubric_json": json.dumps(rubric),
        },
    )
    payload = _safe_json_loads(result.raw, {"criteria_scores": [], "overall_percent": 0.0, "pass": False, "summary": result.raw})
    if not payload.get("criteria_scores") and isinstance(payload.get("summary"), str):
        embedded = _safe_json_loads(payload["summary"], {})
        if embedded.get("criteria_scores"):
            payload = embedded
    normalized = _normalize_binary_evaluation(payload, rubric)
    heuristics = _build_formative_heuristics(prompt, learner_answer, session.current_competency)
    overall = float(normalized.get("overall_percent", 0.0) or 0.0)
    passed = bool(normalized.get("pass", overall >= PASS_THRESHOLD))
    summary = str(normalized.get("summary") or "").strip() or "No detailed formative feedback was returned."

    if not passed and heuristics["pass"] and heuristics["overall_percent"] >= PASS_THRESHOLD:
        overall = max(overall, float(heuristics["overall_percent"]))
        passed = True
        summary = (
            f"{summary} Heuristic validation confirmed the answer was scenario-specific, concrete, "
            "and justified, so the formative check was accepted."
        ).strip()
    elif passed and not heuristics["scenario_relevance"]:
        summary = (
            f"{summary} Warning: the answer passed overall, but scenario-specific alignment looked weak in heuristic validation."
        ).strip()

    logger.info(
        "Formative evaluation competency=%s prompt=%r learner_answer=%r raw_payload=%s normalized=%s heuristics=%s final_passed=%s final_percent=%.2f",
        session.current_competency,
        prompt[:300],
        learner_answer[:500],
        json.dumps(payload, ensure_ascii=True),
        json.dumps(normalized, ensure_ascii=True),
        json.dumps(heuristics, ensure_ascii=True),
        passed,
        overall,
    )
    return passed, overall, summary, overall >= EASY_PASS_THRESHOLD


def _should_ask_formative_check(session: LearnerSession) -> bool:
    if session.learning_turn <= 0:
        return False
    if session.learning_turn % 2 == 0:
        return True
    return session.revision_required and session.learning_turn > BASE_LEARNING_INTERACTIONS


def _target_formative_slot(session: LearnerSession) -> int:
    if 0 <= session.current_formative_slot < len(session.formative_slots):
        current_value = session.formative_slots[session.current_formative_slot]
        if current_value is False:
            return session.current_formative_slot

    for idx, value in enumerate(session.formative_slots):
        if value is not True:
            return idx

    return len(session.formative_slots)


def _interaction_goal(session: LearnerSession) -> str:
    if session.learning_turn <= 2:
        return "Teach one foundational concept with intuitive explanation, mechanism detail, and a clear workplace example."
    if session.learning_turn <= 4:
        return "Teach one applied concept with decision reasoning, tradeoffs, and a concrete scenario."
    if session.learning_turn <= 6:
        return "Prepare the learner for the mastery gate with university-grade explanation, retrieval practice, and precise feedback."
    return "Run focused revision on the weakest concept and repair misconceptions."


def _parse_formative_prompt(ai_response: str) -> str:
    match = re.split(r"(?i)(?:\*\*|##\s*)formative check(?:\*\*)?", ai_response, maxsplit=1)
    if len(match) == 2:
        return match[1].strip()
    return ai_response.strip()


def _alternate_delivery_mode(current_mode: str | None) -> str:
    modes = [
        "guided_explanation",
        "worked_example",
        "comparison_table",
        "scenario_walkthrough",
        "mini_challenge",
        "decision_framework",
    ]
    if current_mode not in modes:
        return modes[0]
    index = modes.index(current_mode)
    return modes[(index + 1) % len(modes)]


def _is_repeated_explanation(session: LearnerSession, ai_response: str) -> bool:
    last_assistant = next((msg for msg in reversed(session.messages[:-1]) if msg.role == "assistant"), None)
    if last_assistant is None:
        return False
    current = re.sub(r"\s+", " ", ai_response.strip().lower())
    previous = re.sub(r"\s+", " ", last_assistant.content.strip().lower())
    if not current or not previous:
        return False
    if current == previous:
        return True
    return SequenceMatcher(a=current[:1200], b=previous[:1200]).ratio() >= 0.9


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


def _build_final_assessment_rubric(session: LearnerSession) -> dict[str, Any]:
    return {
        "criteria": [
            {
                "name": "Integrated application",
                "description": "Combines multiple completed competencies into one coherent solution",
                "weight": 0.30,
            },
            {
                "name": "Scenario reasoning",
                "description": "Explains why the proposed plan fits the scenario and constraints",
                "weight": 0.25,
            },
            {
                "name": "Execution detail",
                "description": "Describes concrete steps, checks, and decisions rather than staying generic",
                "weight": 0.25,
            },
            {
                "name": "Risk awareness",
                "description": "Identifies tradeoffs, failure modes, or quality controls",
                "weight": 0.20,
            },
        ],
        "pass_threshold": PASS_THRESHOLD,
    }


def _generate_final_assessment_prompt(session: LearnerSession) -> str:
    competency_names = [item.competency for item in session.completed_competencies] or session.competencies
    focus_list = ", ".join(competency_names)
    weakest = ", ".join(session.weak_areas[:3]) or "clarity, application, and reasoning"
    return (
        f"**Final Micro-Credential Assessment - {session.topic}**\n\n"
        f"You must now solve one integrated scenario that combines these competencies: **{focus_list}**.\n\n"
        "Scenario: You are responsible for delivering a real project outcome for a stakeholder. "
        "Build one response that shows you can combine the full micro-credential skill set in practice.\n\n"
        "Your answer must include:\n"
        "1. The project goal and stakeholder need.\n"
        "2. The step-by-step plan using the relevant competencies together.\n"
        "3. Why this plan is the right one for the scenario.\n"
        "4. At least one quality check, risk, or limitation you would manage.\n"
        "5. The final outcome you expect to deliver.\n\n"
        f"Pay extra attention to these weaker areas: {weakest}.\n"
        f"Pass threshold: **{PASS_THRESHOLD:.0f}%**."
    )


def _record_remote_teaching_interaction(
    session: LearnerSession,
    competency: str,
    ai_prompt: str,
    ai_response: str,
    learner_input: str | None,
    formative_passed: bool | None,
    *,
    interaction_type: str,
    allow_session_creation: bool = True,
):
    remote_learning_session_id = session.current_remote_learning_session_id
    if not remote_learning_session_id and allow_session_creation:
        remote_learning_session_id = _ensure_remote_learning_session(session, competency)
    if not remote_learning_session_id:
        return

    try:
        remote_backend_client.record_interaction(
            session_id=remote_learning_session_id,
            interaction_type=interaction_type,
            ai_prompt=_sync_prompt_with_metadata(session, ai_prompt, interaction_type),
            ai_response=ai_response,
            learner_input=learner_input,
            formative_passed=formative_passed,
            token=session.remote_auth_token,
        )
    except RemoteBackendError as exc:
        warning = f"Remote interaction sync failed for '{competency}': {exc}"
        _set_remote_sync_failure(session, warning, remote_learning_session_id)
        return
    _set_remote_sync_success(session, remote_learning_session_id)


def _generate_learning_response(session: LearnerSession, user_message: str, formative_feedback: str = "") -> tuple[str, str]:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    competency_description = str(_get_competency_details(session, competency).get("description") or "").strip()
    current_subpart = session.current_subpart or competency
    personalization = _build_personalization_state(session)
    include_formative = _should_ask_formative_check(session)
    length_policy = _teaching_length_policy(session)
    previous_formats = ", ".join(session.delivery_format_history[-3:]) or "none yet"
    live_aip_code = {
        1: "AIP-03",
        2: "AIP-04",
        3: "AIP-05",
        4: "AIP-07",
        5: "AIP-09",
        6: "AIP-11",
    }.get(session.learning_turn, "AIP-04")

    def _kickoff(delivery_mode: str, anti_repeat_instruction: str = "") -> str:
        result = _run_mapped_ai_call(
            session,
            live_aip_code,
            purpose="teaching_generation",
            crew_factory=TutorCrew,
            inputs={
                "topic": session.topic,
                "competency": competency_label,
                "competency_description": competency_description or "No backend competency description available.",
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
                "remote_micro_credential_level": session.remote_micro_credential_level or "not_specified",
                "academic_stage": session.academic_stage,
                "academic_guidance": personalization["academic_guidance"],
                "formative_feedback": formative_feedback or "No formative feedback yet for this turn.",
                "delivery_mode": delivery_mode,
                "difficulty_tier": personalization["difficulty_tier"],
                "support_style": personalization["support_style"],
                "feedback_style": personalization["feedback_style"],
                "interaction_goal": _interaction_goal(session),
                "include_formative_check": "yes" if include_formative else "no",
                "revision_required": "yes" if session.revision_required else "no",
                "stage_instruction": _stage_teaching_instruction(session),
                "response_word_floor": length_policy["floor"],
                "response_word_target": length_policy["target"],
                "response_word_ceiling": length_policy["ceiling"],
                "previous_delivery_modes": previous_formats,
                "anti_repeat_instruction": anti_repeat_instruction or "Do not repeat the previous explanation verbatim.",
            },
        )
        return result.raw.strip()

    ai_response = _kickoff(personalization["delivery_mode"])
    missing_sections = _missing_tutor_sections(ai_response, include_formative)
    if _is_repeated_explanation(session, ai_response) or _word_count(ai_response) < length_policy["floor"] or missing_sections:
        alternate_mode = _alternate_delivery_mode(personalization["delivery_mode"])
        personalization["delivery_mode"] = alternate_mode
        session.personalization_state["delivery_mode"] = alternate_mode
        ai_response = _kickoff(
            alternate_mode,
            anti_repeat_instruction=(
                "Use a different explanatory approach, different structure, and different example from the previous assistant message. "
                f"Expand the response to at least {length_policy['floor']} words and include these missing sections if absent: "
                f"{', '.join(missing_sections) if missing_sections else 'all required sections'}."
            ),
        )
    session.add_message("assistant", ai_response)
    interaction_type = "revision" if session.revision_required and session.learning_turn > BASE_LEARNING_INTERACTIONS else "teach"

    if include_formative:
        session.awaiting_formative_response = True
        target_slot = _target_formative_slot(session)
        while len(session.formative_slots) <= target_slot:
            session.formative_slots.append(None)
        session.current_formative_slot = target_slot
        session.current_formative_prompt = _parse_formative_prompt(ai_response)
        formative_aip = {
            1: "AIP-05",
            2: "AIP-07",
            3: "AIP-09",
        }.get(session.formative_slot_number or 1)
        if formative_aip:
            _record_aip(
                session,
                formative_aip,
                trigger=f"fa{session.formative_slot_number}_prompt_generated",
                metadata={"delivery_mode": personalization["delivery_mode"]},
            )
        if interaction_type != "revision":
            interaction_type = "formative"

    _record_session_interaction(
        session,
        interaction_type=interaction_type,
        interaction_number=session.competency_interaction,
        delivery_format=personalization["delivery_mode"],
        concept=current_subpart,
    )
    return ai_response, interaction_type


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
    session.consecutive_easy_passes = 0
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
            **_runtime_fields(session),
        }

    prompt = _generate_preassessment_prompt(session)
    session.pre_assessment_prompt = prompt
    session.pre_assessment_turn = 1
    session.competency_interaction = max(session.competency_interaction, 2)
    session.add_message("assistant", prompt)
    _record_session_interaction(session, interaction_type="diagnostic", interaction_number=2, concept=session.current_competency)
    _record_aip(session, "AIP-02", trigger="diagnostic_prompt_generated", metadata={"question_count": PRE_ASSESSMENT_QUESTION_COUNT})
    _record_remote_teaching_interaction(
        session,
        session.current_competency,
        ai_prompt=f"Run a short diagnostic for {session.current_competency}.",
        ai_response=prompt,
        learner_input=None,
        formative_passed=None,
        interaction_type="diagnostic",
        allow_session_creation=True,
    )
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "interaction_number": 2,
        "competency": session.current_competency,
        "message": prompt,
        "question_count": PRE_ASSESSMENT_QUESTION_COUNT,
        **_runtime_fields(session),
    }


async def handle_pre_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    if not _diagnostic_answer_is_meaningful(session, user_answer):
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "message": (
                "Please answer the diagnostic with a short, specific response that explains what you would do and why. "
                "The previous input was too limited to place you at the right starting point."
            ),
            "diagnostic_validation_failed": True,
            "counted_as_interaction": False,
            **_runtime_fields(session),
        }

    session.add_message("user", user_answer)
    anomalies = detect_and_record_anomalies(session, user_answer, "/pre-assessment/chat")
    classifier_payload = _classify_competency_readiness(session)
    session.user_level = classifier_payload.get("level", session.user_level)
    session.weak_areas = classifier_payload.get("weak_areas", session.weak_areas)
    session.current_difficulty = _difficulty_from_level(session.user_level)
    session.pre_assessment_completed = True
    session.phase = "learning"

    session.competency_attempts[session.current_competency] = session.competency_attempts.get(session.current_competency, 1)
    record_competency_attempt(
        session.session_id,
        session.current_competency,
        session.competency_attempt_number,
        "in_progress",
    )

    await _setup_competency(session)

    session.learning_turn = 1
    session.competency_interaction = 3
    teaching_response, interaction_type = _generate_learning_response(
        session,
        user_message=f"Learner pre-assessment answer: {user_answer}",
        formative_feedback="Use the pre-assessment answer to personalize the first teaching interaction.",
    )
    _record_aip(session, "AIP-03", trigger="first_teaching_turn_generated", metadata={"user_level": session.user_level})
    _record_remote_teaching_interaction(
        session,
        session.current_competency,
        ai_prompt=session.current_subpart or session.current_competency,
        ai_response=teaching_response,
        learner_input=user_answer,
        formative_passed=None,
        interaction_type="teaching" if interaction_type == "teach" else "formative_check",
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
        "gamification": build_gamification_payload(session),
        "anomaly_flags": anomalies,
        **_runtime_fields(session),
    }


def _learning_window_exhausted(session: LearnerSession) -> bool:
    limit = session.max_learning_turns + REVISION_INTERACTIONS if session.revision_required else session.max_learning_turns
    return session.learning_turn >= limit


async def handle_learning(session: LearnerSession, user_message: str) -> dict[str, Any]:
    session.add_message("user", user_message)
    anomalies = detect_and_record_anomalies(session, user_message, "/learn/chat")
    competency = session.current_competency
    formative_feedback = ""
    formative_passed: bool | None = None

    if session.awaiting_formative_response:
        formative_passed, formative_percent, formative_summary, easy_pass = _evaluate_formative_response(session, user_message)
        formative_feedback = _apply_formative_outcome(
            session,
            formative_passed,
            formative_percent,
            formative_summary,
            easy_pass=easy_pass,
        )
        record_formative_check(
            session.session_id,
            competency,
            session.competency_attempt_number,
            session.current_formative_slot,
            passed=formative_passed,
            score=formative_percent,
            learner_response=user_message,
            feedback=formative_feedback,
            difficulty_tier=session.current_difficulty,
            delivery_format=session.personalization_state.get("delivery_mode") or session.last_delivery_format or "explanation",
        )
        session.last_feedback_message = formative_feedback
        _log_session_event(
            session,
            "/learn/chat",
            "formative_evaluated",
            {
                "competency": competency,
                "passed": formative_passed,
                "score": formative_percent,
                "easy_pass": easy_pass,
                "formative_slot": session.formative_slot_number,
            },
        )

    if session.final_assessment_unlocked:
        session.phase = "competency_assessment"
        session.competency_interaction += 1
        session.current_assessment_prompt = _generate_assessment_prompt(session)
        session.add_message("assistant", session.current_assessment_prompt)
        _record_aip(session, "AIP-11", trigger="competency_assessment_prompt_generated")
        _record_session_interaction(
            session,
            interaction_type="final_assessment",
            interaction_number=session.competency_interaction,
            concept=session.current_competency,
        )
        save_session(session)
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "competency": competency,
            "interaction_number": session.competency_interaction,
            "message": session.current_assessment_prompt,
            "assessment_prompt": session.current_assessment_prompt,
            "ready_for_assessment": True,
            "gamification": build_gamification_payload(session),
            "anomaly_flags": anomalies,
            "backend_warnings": session.backend_warnings,
            **_runtime_fields(session),
        }

    if _learning_window_exhausted(session) and not session.final_assessment_unlocked:
        session.revision_required = False
        _reset_learning_after_assessment_fail(session)
        session.learning_turn = 1
        session.competency_interaction = 3
        relearn_feedback = "Mastery gate not met. Restart from interaction 3 and reteach the weakest concepts with simpler explanations and a different format."
        ai_response, interaction_type = _generate_learning_response(session, user_message, relearn_feedback)
        _record_remote_teaching_interaction(
            session,
            competency,
            session.current_subpart or competency,
            ai_response,
            user_message,
            formative_passed,
            interaction_type="teaching" if interaction_type == "teach" else "formative_check",
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
            "revision_required": session.revision_required,
            "gamification": build_gamification_payload(session),
            "anomaly_flags": anomalies,
            "backend_warnings": session.backend_warnings,
            **_runtime_fields(session),
        }

    session.learning_turn += 1
    session.competency_interaction = 2 + session.learning_turn
    if session.revision_required and session.learning_turn > session.max_learning_turns:
        session.revision_turns_used = session.learning_turn - session.max_learning_turns

    ai_response, interaction_type = _generate_learning_response(session, user_message, formative_feedback)
    if session.learning_turn == 2:
        _record_aip(
            session,
            "AIP-04",
            trigger="worked_example_delivered",
            metadata={"delivery_mode": session.personalization_state.get("delivery_mode")},
        )
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=session.current_subpart or competency,
        ai_response=ai_response,
        learner_input=user_message,
        formative_passed=formative_passed,
        interaction_type="spaced_review" if interaction_type == "revision" else ("formative_check" if interaction_type == "formative" else "teaching"),
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
        "current_formative_prompt": session.current_formative_prompt,
        "revision_required": session.revision_required,
        "formative_check_results": session.formative_check_results,
        "message": ai_response,
        "ready_for_assessment": False,
        "personalization_state": session.personalization_state,
        "gamification": build_gamification_payload(session),
        "anomaly_flags": anomalies,
        "backend_warnings": session.backend_warnings,
        **_runtime_fields(session),
    }


async def handle_competency_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    competency = session.current_competency
    competency_label = _competency_prompt_label(session, competency)
    rubric, scenario, rubric_source = _load_assessment_context(session, competency)
    prompt = session.current_assessment_prompt or _generate_assessment_prompt(session)
    session.current_assessment_attempts += 1
    session.add_message("user", user_answer)
    anomalies = detect_and_record_anomalies(session, user_answer, "/assessment/competency", is_assessment=True)

    result = _run_mapped_ai_call(
        session,
        "AIP-12",
        purpose="competency_assessment_scoring",
        crew_factory=AssessmentCrew,
        inputs={
            "competency": competency_label,
            "scenario": prompt or scenario,
            "user_response": user_answer,
            "rubric_json": json.dumps(rubric),
        },
    )
    evaluation = _safe_json_loads(result.raw, {"criteria_scores": [], "overall_percent": 0.0, "pass": False, "summary": result.raw})
    normalized = _normalize_binary_evaluation(evaluation, rubric)
    overall = float(normalized.get("overall_percent", 0.0) or 0.0)
    passed = bool(normalized.get("pass", overall >= PASS_THRESHOLD))
    summary = str(normalized.get("summary") or "").strip() or "No assessment summary returned."
    provisional_binary_outcome = _binary_outcome_label(passed)
    session.local_assessment_passed = passed
    session.remote_assessment_synced = False
    session.remote_assessment_passed = None
    session.current_assessment_sync_error = None
    rubric_key = rubric.get("rubric_key") or normalize_rubric_key(competency)
    rubric_version = int(rubric.get("version") or get_rubric_version(competency) or 1)
    rubric_hash = rubric.get("source_hash") or get_rubric_source_hash(competency)
    remote_competency_id = session.current_remote_competency_id
    remote_micro_credential_id = session.remote_micro_credential_id
    _record_session_interaction(
        session,
        interaction_type="competency_assessment",
        interaction_number=max(session.competency_interaction, 9),
        concept=competency,
    )
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=prompt,
        ai_response=summary,
        learner_input=user_answer,
        formative_passed=passed,
        interaction_type="competency_assessment",
        allow_session_creation=False,
    )

    remote_learning_session_id = _ensure_remote_learning_session(session, competency)
    remote_confirmation = _sync_remote_competency_assessment(
        session,
        competency=competency,
        remote_required=session.source == "remote",
        remote_learning_session_id=remote_learning_session_id,
        prompt=prompt,
        user_answer=user_answer,
        overall=overall,
        summary=summary,
        local_passed=passed,
    )

    if remote_confirmation["remote_required"] and remote_confirmation["confirmed"] is None:
        session.latest_binary_outcome = None
        pending_message = (
            f"{summary}\n\n"
            "The assessment was scored locally, but backend confirmation did not complete. "
            "The competency remains locked until the remote assessment state is confirmed."
        )
        session.add_message("assistant", pending_message)
        save_session(session)
        _log_session_event(
            session,
            "/assessment/competency",
            "assessment_confirmation_pending",
            {
                "competency": competency,
                "local_passed": passed,
                "score": overall,
                "warning": session.current_assessment_sync_error,
            },
        )
        return {
            "session_id": session.session_id,
            "phase": session.phase,
            "assessed_competency": competency,
            "score": overall,
            "passed": False,
            "message": pending_message,
            "assessment_feedback": summary,
            "assessment_detail": normalized,
            "binary_outcome": None,
            "rubric_source": rubric_source,
            "rubric_version": rubric_version,
            "rubric_source_hash": rubric_hash,
            "awaiting_backend_confirmation": True,
            "backend_warnings": session.backend_warnings,
            "anomaly_flags": anomalies,
            "gamification": build_gamification_payload(session),
            **_runtime_fields(session),
        }

    authoritative_passed = passed if not remote_confirmation["remote_required"] else bool(remote_confirmation["confirmed"])
    binary_outcome = _binary_outcome_label(authoritative_passed)
    session.latest_binary_outcome = binary_outcome
    _record_aip(
        session,
        "AIP-12",
        trigger="competency_assessment_scored",
        outcome=binary_outcome,
        metadata={
            "score": overall,
            "local_passed": passed,
            "remote_passed": remote_confirmation["confirmed"],
        },
    )

    if authoritative_passed:
        session.developing_competency_active = False
        session.developing_competency_reason = None
        record_competency_attempt(
            session.session_id,
            competency,
            session.competency_attempt_number,
            "passed",
            score=overall,
            rubric_key=rubric_key,
            evaluation={**normalized, "rubric_version": rubric_version, "rubric_source_hash": rubric_hash},
        )
        if session.learner_id and remote_micro_credential_id and remote_competency_id:
            upsert_learner_competency_progress(
                session.learner_id,
                int(remote_micro_credential_id),
                int(remote_competency_id),
                competency,
                passed=True,
                latest_session_id=session.session_id,
                latest_score=overall,
            )
        session.completed_competencies.append(
            CompetencyResult(
                competency=competency,
                score=overall,
                passed=True,
                feedback=summary,
            )
        )
        badge = create_badge(
            session.session_id,
            session.learner_id,
            competency,
            f"{competency} Badge",
            {"score": overall, "awarded_date": utc_now_iso(), "badge_type": "competency"},
        )
        session.earned_badges.append(badge)
        session.add_message("assistant", summary)
        _record_aip(session, "AIP-13", trigger="competency_summary_delivered", outcome=binary_outcome, metadata={"score": overall})
        _log_session_event(session, "/assessment/competency", "badge_issued", badge)
        if session.is_last_competency:
            session.phase = "final_assessment"
            session.final_assessment_prompt = _generate_final_assessment_prompt(session)
            session.add_message("assistant", session.final_assessment_prompt)
            _record_session_interaction(
                session,
                interaction_type="final_assessment",
                interaction_number=session.competency_interaction + 1,
                concept=session.topic,
            )
            _record_remote_teaching_interaction(
                session,
                competency,
                ai_prompt=f"Present the final integrated assessment for {session.topic}.",
                ai_response=session.final_assessment_prompt,
                learner_input=None,
                formative_passed=None,
                interaction_type="final_assessment",
                allow_session_creation=False,
            )
            save_session(session)
            _log_session_event(
                session,
                "/assessment/competency",
                "assessment_state_committed",
                {
                    "competency": competency,
                    "authoritative_passed": True,
                    "local_passed": passed,
                    "remote_passed": remote_confirmation["confirmed"],
                    "phase": session.phase,
                    "next_action": session.required_next_action,
                },
            )
            return {
                "session_id": session.session_id,
                "phase": "final_assessment",
                "assessed_competency": competency,
                "score": overall,
                "passed": True,
                "message": (
                    f"Assessment passed for **{competency}** with **{overall:.1f}%**. "
                    "All competency assessments are complete. Continue with the final micro-credential assessment."
                ),
                "assessment_detail": normalized,
                "binary_outcome": binary_outcome,
                "rubric_source": rubric_source,
                "rubric_version": rubric_version,
                "rubric_source_hash": rubric_hash,
                "final_assessment_prompt": session.final_assessment_prompt,
                "ready_for_final_assessment": True,
                "gamification": build_gamification_payload(session, competency_badge=badge),
                "anomaly_flags": anomalies,
                "backend_warnings": session.backend_warnings,
                **_runtime_fields(session),
            }

        session.advance_to_next_competency()
        session.competency_attempts[session.current_competency] = 1
        next_intro = build_competency_intro(session)
        record_competency_attempt(session.session_id, session.current_competency, session.competency_attempt_number, "in_progress")
        save_session(session)
        _log_session_event(
            session,
            "/assessment/competency",
            "assessment_state_committed",
            {
                "assessed_competency": competency,
                "authoritative_passed": True,
                "local_passed": passed,
                "remote_passed": remote_confirmation["confirmed"],
                "phase": session.phase,
                "next_competency": session.current_competency,
            },
        )
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
            "assessment_detail": normalized,
            "binary_outcome": binary_outcome,
            "rubric_source": rubric_source,
            "rubric_version": rubric_version,
            "rubric_source_hash": rubric_hash,
            "next_competency": session.current_competency,
            "gamification": build_gamification_payload(session, competency_badge=badge),
            "anomaly_flags": anomalies,
            "backend_warnings": session.backend_warnings,
            **_runtime_fields(session),
        }

    record_competency_attempt(
        session.session_id,
        competency,
        session.competency_attempt_number,
        "failed",
        score=overall,
        rubric_key=rubric_key,
        evaluation={**normalized, "rubric_version": rubric_version, "rubric_source_hash": rubric_hash},
    )
    if session.learner_id and remote_micro_credential_id and remote_competency_id:
        upsert_learner_competency_progress(
            session.learner_id,
            int(remote_micro_credential_id),
            int(remote_competency_id),
            competency,
            passed=False,
            latest_session_id=session.session_id,
            latest_score=overall,
        )
    session.competency_attempts[competency] = session.competency_attempt_number + 1
    _reset_learning_after_assessment_fail(session)
    session.learning_turn = 1
    session.competency_interaction = 3
    session.developing_competency_active = True
    session.developing_competency_reason = "Competency assessment not yet competent."
    learning_message = _build_static_remediation_message(
        session,
        title=f"{binary_outcome} - Competency Assessment",
        summary=summary,
        weakest_focus=", ".join(session.weak_areas[:2]) or competency.lower(),
    )
    interaction_type = "revision"
    session.add_message("assistant", learning_message)
    _record_session_interaction(
        session,
        interaction_type="revision",
        interaction_number=session.competency_interaction,
        concept=session.current_competency,
    )
    _record_remote_teaching_interaction(
        session,
        competency,
        ai_prompt=session.current_subpart or competency,
        ai_response=learning_message,
        learner_input=user_answer,
        formative_passed=False,
        interaction_type="spaced_review" if interaction_type == "revision" else ("formative_check" if interaction_type == "formative" else "teaching"),
    )
    save_session(session)
    _log_session_event(
        session,
        "/assessment/competency",
        "assessment_state_committed",
        {
            "competency": competency,
            "authoritative_passed": False,
            "local_passed": passed,
            "remote_passed": remote_confirmation["confirmed"],
            "phase": session.phase,
            "interaction_number": session.competency_interaction,
        },
    )
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "assessed_competency": competency,
        "score": overall,
        "passed": False,
        "message": learning_message,
        "assessment_feedback": summary,
        "assessment_detail": normalized,
        "binary_outcome": binary_outcome,
        "rubric_source": rubric_source,
        "rubric_version": rubric_version,
        "rubric_source_hash": rubric_hash,
        "interaction_number": session.competency_interaction,
        "gamification": build_gamification_payload(session),
        "anomaly_flags": anomalies,
        "backend_warnings": session.backend_warnings,
        **_runtime_fields(session),
    }


async def handle_final_assessment(session: LearnerSession, user_answer: str) -> dict[str, Any]:
    prompt = session.final_assessment_prompt or _generate_final_assessment_prompt(session)
    rubric = _build_final_assessment_rubric(session)
    session.final_assessment_attempts += 1
    session.add_message("user", user_answer)
    anomalies = detect_and_record_anomalies(session, user_answer, "/assessment/final", is_assessment=True)

    result = _run_mapped_ai_call(
        session,
        "AIP-14" if session.is_last_competency else "AIP-12",
        purpose="final_assessment_scoring",
        crew_factory=AssessmentCrew,
        inputs={
            "competency": f"Final micro-credential assessment for {session.topic}",
            "scenario": prompt,
            "user_response": user_answer,
            "rubric_json": json.dumps(rubric),
        },
    )
    evaluation = _safe_json_loads(result.raw, {"criteria_scores": [], "overall_percent": 0.0, "pass": False, "summary": result.raw})
    normalized = _normalize_binary_evaluation(evaluation, rubric)
    overall = float(normalized.get("overall_percent", 0.0) or 0.0)
    passed = bool(normalized.get("pass", overall >= PASS_THRESHOLD))
    summary = str(normalized.get("summary") or "").strip() or "No final assessment summary returned."
    binary_outcome = _binary_outcome_label(passed)
    session.latest_binary_outcome = binary_outcome
    record_final_assessment(session.session_id, session.final_assessment_attempts, prompt, user_answer, normalized, overall, passed)
    _record_remote_teaching_interaction(
        session,
        session.current_competency,
        ai_prompt=prompt,
        ai_response=summary,
        learner_input=user_answer,
        formative_passed=passed,
        interaction_type="final_assessment",
        allow_session_creation=False,
    )
    _record_aip(
        session,
        "AIP-14" if passed else "AIP-12",
        trigger="mc_completion_reflection_delivered" if passed else "final_assessment_scored",
        scope="mc" if passed else "cc",
        outcome=binary_outcome,
        metadata={"score": overall},
    )

    if passed:
        session.phase = "completed"
        session.completed_at = utc_now_iso()
        session.developing_competency_active = False
        session.developing_competency_reason = None
        session.add_message("assistant", summary)
        completion_badge = create_badge(
            session.session_id,
            session.learner_id,
            session.topic,
            f"{session.topic} Completion Badge",
            {"awarded_date": utc_now_iso(), "badge_type": "completion", "score": overall},
        )
        session.completion_badge = completion_badge
        session.earned_badges.append(completion_badge)
        _log_session_event(session, "/assessment/final", "badge_issued", completion_badge)
        session_summary = build_session_summary(session)
        save_session(session)
        return {
            "session_id": session.session_id,
            "phase": "completed",
            "passed": True,
            "score": overall,
            "message": (
                f"Final assessment passed with **{overall:.1f}%**. "
                "The full micro-credential is complete. Certificate generation is now available."
            ),
            "assessment_detail": normalized,
            "binary_outcome": binary_outcome,
            "final_assessment_prompt": prompt,
            "gamification": build_gamification_payload(session),
            "session_summary": session_summary,
            "anomaly_flags": anomalies,
            "backend_warnings": session.backend_warnings,
            **_runtime_fields(session),
        }

    _reset_learning_after_assessment_fail(session)
    session.learning_turn = 1
    session.competency_interaction = 3
    session.add_message("assistant", summary)
    session.developing_competency_active = True
    session.developing_competency_reason = "Final assessment not yet competent."
    learning_message = _build_static_remediation_message(
        session,
        title=f"{binary_outcome} - Final Assessment",
        summary=summary,
        weakest_focus=", ".join(session.weak_areas[:3]) or session.current_competency.lower(),
    )
    interaction_type = "revision"
    session.add_message("assistant", learning_message)
    _record_session_interaction(
        session,
        interaction_type="revision",
        interaction_number=session.competency_interaction,
        concept=session.current_competency,
    )
    _record_remote_teaching_interaction(
        session,
        session.current_competency,
        ai_prompt=session.current_subpart or session.current_competency,
        ai_response=learning_message,
        learner_input=user_answer,
        formative_passed=False,
        interaction_type="spaced_review" if interaction_type == "revision" else ("formative_check" if interaction_type == "formative" else "teaching"),
    )
    save_session(session)
    return {
        "session_id": session.session_id,
        "phase": "learning",
        "passed": False,
        "score": overall,
        "message": learning_message,
        "assessment_feedback": summary,
        "assessment_detail": normalized,
        "binary_outcome": binary_outcome,
        "interaction_number": session.competency_interaction,
        "mastery_reset": True,
        "gamification": build_gamification_payload(session),
        "anomaly_flags": anomalies,
        "backend_warnings": session.backend_warnings,
        **_runtime_fields(session),
    }
