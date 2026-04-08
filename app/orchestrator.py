import json
import os
import re
import yaml
from app.state import LearnerSession, CompetencyResult
from app.session_manager import get_session, save_session
from app.remote_backend import RemoteBackendError, remote_backend_client
from app.crews.pre_assessment_crew import PreAssessCrew
from app.crews.level_classifier_crew import LevelClassifierCrew
from app.crews.learning_path_planner import PathPlnner
from app.crews.studey_materils_crew import StudyMeterial
from app.crews.ai_tutor_agents_crew import TutorCrew
from app.crews.assessment_crew import AssessmentCrew

PRE_ASSESSMENT_TURNS = 4
PASS_THRESHOLD = 75.0

# ── Rubric Loading ──────────────────────────────────────────────────────────

_RUBRICS_CACHE: dict | None = None

def _load_all_rubrics() -> dict:
    """Load rubrics from YAML file (cached after first call)."""
    global _RUBRICS_CACHE
    if _RUBRICS_CACHE is not None:
        return _RUBRICS_CACHE

    rubrics_path = os.path.join(os.path.dirname(__file__), 'config', 'rubrics.yaml')
    try:
        with open(rubrics_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        data = {}

    _RUBRICS_CACHE = data
    return data


def _load_rubric(competency: str, user_level: str = 'intermediate') -> dict:
    """Return a rubric dict for the given competency with level-adjusted weights."""
    all_rubrics = _load_all_rubrics()

    # Try to find a competency-specific rubric, fall back to default
    key = competency.lower().replace(' ', '_')
    rubric = all_rubrics.get(key, all_rubrics.get('default', {}))

    if not rubric or 'criteria' not in rubric:
        rubric = {
            'criteria': [
                {'name': 'Accuracy', 'weight': 0.30},
                {'name': 'Depth of Understanding', 'weight': 0.25},
                {'name': 'Application', 'weight': 0.25},
                {'name': 'Critical Thinking', 'weight': 0.10},
                {'name': 'Clarity of Expression', 'weight': 0.10},
            ]
        }

    # Apply level-adjusted weights if available
    level_adjustments = all_rubrics.get('level_adjustments', {}).get(user_level, {})
    if level_adjustments:
        adjusted_criteria = []
        total_weight = 0.0
        for c in rubric['criteria']:
            modifier = level_adjustments.get(c['name'], 1.0)
            new_weight = c['weight'] * modifier
            adjusted = {**c, 'weight': new_weight}
            adjusted_criteria.append(adjusted)
            total_weight += new_weight
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for c in adjusted_criteria:
                c['weight'] = round(c['weight'] / total_weight, 3)
        rubric = {'criteria': adjusted_criteria}

    return rubric


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_subparts_from_plan(plan_text: str, fallback_turns: int = 22) -> list[str]:
    """Parse planner output into sequential sub-parts."""
    if not plan_text or not plan_text.strip():
        return [f"Part {i}: Applied learning step." for i in range(1, fallback_turns + 1)]

    text = plan_text.strip()
    chunks = re.split(r"\n(?=\s*(?:\d+[\).:\-]|[-*])\s+)", text)
    parts = [c.strip() for c in chunks if c.strip()]
    cleaned: list[str] = []

    if len(parts) <= 1:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        parts = [ln for ln in lines if re.match(r"^(?:\d+[\).:\-]|[-*])\s+", ln)] or lines

    for part in parts:
        item = re.sub(r"^\s*(?:\d+[\).:\-]|[-*])\s+", "", part).strip()
        if item:
            cleaned.append(item)

    if not cleaned:
        cleaned = [f"Part {i}: Applied learning step." for i in range(1, fallback_turns + 1)]

    # Pad to fallback_turns if we got fewer items
    while len(cleaned) < fallback_turns:
        cleaned.append(f"Part {len(cleaned) + 1}: Continued practice and exploration.")

    return cleaned[:fallback_turns]


def _is_technical_competency(name: str) -> bool:
    tokens = (
        "regression", "statistics", "probability", "python", "sql",
        "analysis", "machine learning", "model", "algorithm", "data",
        "math", "programming", "code", "database", "api", "engineering",
        "architecture", "devops", "cloud", "network",
    )
    lowered = name.lower()
    return any(tok in lowered for tok in tokens)


def _get_competency_details(session: LearnerSession, competency: str) -> dict:
    return session.competency_details.get(competency, {})


def _competency_prompt_label(session: LearnerSession, competency: str) -> str:
    details = _get_competency_details(session, competency)
    description = str(details.get('description') or '').strip()
    if description:
        return f"{competency} — {description}"
    return competency


def _normalize_remote_rubric(payload: dict | None) -> dict | None:
    if not payload:
        return None
    raw_criteria = payload.get('rubric_criteria') or {}
    if isinstance(raw_criteria, dict):
        iterable = raw_criteria.get('criteria', [])
    elif isinstance(raw_criteria, list):
        iterable = raw_criteria
    else:
        iterable = []

    criteria = []
    total = max(1, len(iterable))
    weight = round(1 / total, 3)
    for idx, item in enumerate(iterable, start=1):
        criteria.append(
            {
                'name': str(item.get('criterion') or item.get('name') or f'Criterion {idx}'),
                'description': str(item.get('descriptor') or item.get('met_indicator') or ''),
                'weight': weight,
                'criterion_id': str(item.get('id') or f'c{idx}'),
                'met_indicator': str(item.get('met_indicator') or ''),
            }
        )
    if not criteria:
        return None
    return {
        'criteria': criteria,
        'pass_threshold': float(payload.get('pass_threshold') or PASS_THRESHOLD),
        'scenario_template': payload.get('scenario_template'),
        'difficulty_level': payload.get('difficulty_level'),
        'source': 'remote',
    }


def _load_assessment_context(session: LearnerSession, competency: str) -> tuple[dict, str, str]:
    details = _get_competency_details(session, competency)
    remote_competency_id = details.get('id')
    if competency in session.rubric_cache:
        cached = session.rubric_cache[competency]
        scenario = cached.get('scenario_template') or details.get('description') or session.study_materials.get(competency, 'N/A')
        return cached, str(scenario), cached.get('source', 'cache')

    if isinstance(remote_competency_id, int):
        remote_payload = remote_backend_client.fetch_rubric(remote_competency_id)
        normalized = _normalize_remote_rubric(remote_payload)
        if normalized:
            session.rubric_cache[competency] = normalized
            scenario = normalized.get('scenario_template') or details.get('description') or session.study_materials.get(competency, 'N/A')
            return normalized, str(scenario), 'remote'

    fallback = _load_rubric(competency, session.user_level)
    fallback['source'] = 'local_fallback'
    session.rubric_cache[competency] = fallback
    warning = f"Remote rubric missing for competency '{competency}'. Using local fallback rubric."
    if warning not in session.backend_warnings:
        session.backend_warnings.append(warning)
    scenario = details.get('description') or session.study_materials.get(competency, 'N/A')
    return fallback, str(scenario), 'local_fallback'


def _ensure_remote_learning_session(session: LearnerSession, competency: str) -> int | None:
    existing = session.remote_learning_sessions.get(competency)
    if existing:
        return existing
    details = _get_competency_details(session, competency)
    remote_competency_id = details.get('id')
    if session.source != 'remote' or not session.remote_access_id or not isinstance(remote_competency_id, int):
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
    remote_session_id = payload.get('id')
    if isinstance(remote_session_id, int):
        session.remote_learning_sessions[competency] = remote_session_id
        return remote_session_id
    return None


async def _setup_competency(session: LearnerSession):
    """Generate study material + learning plan for current competency (if not cached)."""
    comp = session.current_competency
    competency_label = _competency_prompt_label(session, comp)
    if comp not in session.study_materials:
        material = StudyMeterial().crew().kickoff(inputs={
            'topic': session.topic,
            'competency': competency_label,
            'USER_LEVEL': session.user_level,
            'user_level': session.user_level,
        })
        session.study_materials[comp] = material.raw

    if comp not in session.learning_plans:
        plan = PathPlnner().crew().kickoff(inputs={
            'topic': session.topic,
            'competency': competency_label,
            'USER_LEVEL': session.user_level,
            'user_level': session.user_level,
            'weak_areas': ', '.join(session.weak_areas),
        })
        session.learning_plans[comp] = plan.raw

    if comp not in session.competency_subparts:
        session.competency_subparts[comp] = _extract_subparts_from_plan(
            session.learning_plans.get(comp, ''),
            fallback_turns=session.max_learning_turns,
        )


# ── Phase Handlers ────────────────────────────────────────────────────────

async def handle_pre_assessment_start(session: LearnerSession) -> dict:
    """
    Generates the FIRST pre-assessment question based on topic + competencies.
    Call this once after /session/start to get the opening question to show the learner.
    No user input needed — this is a pure generation step.
    """
    if session.pre_assessment_turn > 0:
        return {
            'session_id': session.session_id,
            'phase': session.phase,
            'message': "Pre-assessment already started. Use POST /pre-assessment/chat to continue.",
            'already_started': True,
        }

    all_competencies = ', '.join(session.competencies)

    result = PreAssessCrew().crew().kickoff(inputs={
        'topic': session.topic,
        'competencies': all_competencies,
        'chat_history': 'No conversation yet.',
        'user_message': f'Please start the pre-assessment for: {all_competencies}',
        'turn_number': 1,
    })
    first_question = result.raw
    session.add_message('assistant', first_question)

    save_session(session)
    return {
        'session_id': session.session_id,
        'phase': 'pre_assessment',
        'turn': 0,
        'max_turns': PRE_ASSESSMENT_TURNS,
        'message': first_question,
        'instruction': (
            f"Answer this question by calling POST /pre-assessment/chat. "
            f"There are {PRE_ASSESSMENT_TURNS} questions total."
        ),
    }



async def handle_pre_assessment(session: LearnerSession, user_answer: str) -> dict:
    """
    Handles one turn of pre-assessment Q&A.
    On the 4th turn, auto-classifies the learner level and transitions to learning.
    """
    session.pre_assessment_turn += 1
    session.add_message('user', user_answer)

    all_competencies = ', '.join(session.competencies)

    result = PreAssessCrew().crew().kickoff(inputs={
        'topic': session.topic,
        'competencies': all_competencies,
        'chat_history': session.format_recent_history(),
        'user_message': user_answer,
        'turn_number': session.pre_assessment_turn,
    })
    ai_response = result.raw
    session.add_message('assistant', ai_response)

    response = {
        'session_id': session.session_id,
        'phase': 'pre_assessment',
        'turn': session.pre_assessment_turn,
        'message': ai_response,
        'done': False,
    }

    # After enough turns → classify level and move to learning
    if session.pre_assessment_turn >= PRE_ASSESSMENT_TURNS:
        level_result = LevelClassifierCrew().crew().kickoff(inputs={
            'topic': session.topic,
            'chat_history': session.format_recent_history(20),
        })
        try:
            level_data = json.loads(level_result.raw)
            session.user_level = level_data.get('level', 'beginner')
            session.weak_areas = level_data.get('weak_areas', [])
        except (json.JSONDecodeError, AttributeError):
            session.user_level = 'beginner'

        # Prepare first competency
        session.phase = 'learning'
        await _setup_competency(session)

        response['done'] = True
        response['phase'] = 'learning'
        response['level'] = session.user_level
        response['weak_areas'] = session.weak_areas
        response['next_competency'] = session.current_competency
        response['current_subpart_index'] = session.current_subpart_index + 1
        response['current_subpart'] = session.current_subpart
        response['total_subparts'] = len(session.competency_subparts.get(session.current_competency, []))
        response['chat_stage'] = session.chat_stage
        response['bloom_level'] = session.bloom_level
        response['message'] = (
            f"Great work on the pre-assessment! Your level has been set to "
            f"**{session.user_level}**. We'll now begin learning: "
            f"**{session.current_competency}**. Send a message to start!"
        )

    save_session(session)
    return response


async def handle_learning(session: LearnerSession, user_message: str) -> dict:
    """
    Handles one interactive learning turn for the current competency.
    Follows the 22-chat structured learning progression with stage-aware behavior.
    """
    session.learning_turn += 1
    session.add_message('user', user_message)

    comp = session.current_competency
    competency_label = _competency_prompt_label(session, comp)
    subparts = session.competency_subparts.get(comp, [])
    total_subparts = len(subparts)
    current_subpart = session.current_subpart

    # Advance sub-part if learner signals readiness
    msg = user_message.lower().strip()
    advance_signals = ['next', 'continue', 'move on', 'go ahead', 'done', 'got it', 'understood']
    should_advance_subpart = any(signal in msg for signal in advance_signals)

    if should_advance_subpart and total_subparts > 0 and session.current_subpart_index < (total_subparts - 1):
        session.current_subpart_index += 1
        current_subpart = session.current_subpart

    # Clean up sub-part text for the prompt
    cleaned_subpart = re.sub(r"\*\*", "", current_subpart or "").strip()
    subpart_scenario = cleaned_subpart
    subpart_question = ""
    if "Question:" in cleaned_subpart:
        parts = cleaned_subpart.split("Question:", 1)
        subpart_scenario = parts[0].replace("Scenario:", "").strip()
        subpart_question = parts[1].strip()

    competency_is_technical = _is_technical_competency(comp)
    remote_learning_session_id = _ensure_remote_learning_session(session, comp)

    # Get stage-aware context
    chat_stage = session.chat_stage
    bloom_level = session.bloom_level

    result = TutorCrew().crew().kickoff(inputs={
        'topic': session.topic,
        'competency': competency_label,
        'USER_LEVEL': session.user_level,
        'user_level': session.user_level,
        'weak_areas': ', '.join(session.weak_areas),
        'chat_history': session.format_recent_history(),
        'user_message': user_message,
        'turn_number': session.learning_turn,
        'max_turns': session.max_learning_turns,
        'current_subpart': current_subpart,
        'subpart_scenario': subpart_scenario,
        'subpart_question': subpart_question,
        'competency_is_technical': 'yes' if competency_is_technical else 'no',
        'response_depth': 'high' if competency_is_technical else 'medium',
        'subpart_index': session.current_subpart_index + 1,
        'total_subparts': total_subparts,
        'study_material': session.study_materials.get(comp, ''),
        'chat_stage': chat_stage,
        'bloom_level': bloom_level,
    })
    ai_response = result.raw
    session.add_message('assistant', ai_response)

    if remote_learning_session_id:
        try:
            remote_backend_client.record_interaction(
                session_id=remote_learning_session_id,
                interaction_type='teaching',
                ai_prompt=current_subpart or competency_label,
                ai_response=ai_response,
                learner_input=user_message,
                formative_passed=None,
                token=session.remote_auth_token,
            )
        except RemoteBackendError as exc:
            warning = f"Remote interaction sync failed for '{comp}': {exc}"
            if warning not in session.backend_warnings:
                session.backend_warnings.append(warning)

    completed_all_subparts = total_subparts == 0 or session.current_subpart_index >= (total_subparts - 1)
    ready_for_assessment = (
        session.learning_turn >= session.max_learning_turns
        and completed_all_subparts
    )
    if ready_for_assessment:
        session.phase = 'competency_assessment'

    save_session(session)
    return {
        'session_id': session.session_id,
        'phase': session.phase,
        'competency': comp,
        'turn': session.learning_turn,
        'max_turns': session.max_learning_turns,
        'current_subpart_index': session.current_subpart_index + 1,
        'total_subparts': total_subparts,
        'current_subpart': current_subpart,
        'chat_stage': chat_stage,
        'bloom_level': bloom_level,
        'is_doubt_phase': session.is_doubt_phase,
        'message': ai_response,
        'ready_for_assessment': ready_for_assessment,
        'source': session.source,
        'remote_learning_session_id': remote_learning_session_id,
        'backend_warnings': session.backend_warnings,
    }


async def handle_competency_assessment(session: LearnerSession, user_answer: str) -> dict:
    """
    Evaluates a learner's response for the current competency using Claude (rubric-based).
    On pass/fail, either advances to the next competency or triggers final assessment.
    """
    comp = session.current_competency
    competency_label = _competency_prompt_label(session, comp)
    rubric, scenario, rubric_source = _load_assessment_context(session, comp)
    remote_learning_session_id = _ensure_remote_learning_session(session, comp)

    session.add_message('user', user_answer)

    result = AssessmentCrew().crew().kickoff(inputs={
        'competency': competency_label,
        'scenario': scenario,
        'user_response': user_answer,
        'rubric_json': json.dumps(rubric),
    })

    # Parse assessment JSON
    try:
        eval_data = json.loads(result.raw)
    except (json.JSONDecodeError, AttributeError):
        eval_data = {'overall_percent': 0.0, 'pass': False, 'summary': result.raw}

    overall = eval_data.get('overall_percent', 0.0)
    passed = eval_data.get('pass', overall >= PASS_THRESHOLD)
    summary = eval_data.get('summary', '')

    comp_result = CompetencyResult(
        competency=comp,
        score=overall,
        passed=passed,
        feedback=summary,
    )
    session.completed_competencies.append(comp_result)

    ai_response = (
        f"**Competency Assessment — {comp}**\n\n"
        f"Score: {overall:.1f}% | Result: {'✅ Passed' if passed else '❌ Not yet'}\n\n"
        f"{summary}"
    )
    session.add_message('assistant', ai_response)

    if remote_learning_session_id:
        try:
            remote_backend_client.submit_assessment(
                session_id=remote_learning_session_id,
                scenario_question=scenario,
                learner_response=user_answer,
                rubric_score=overall,
                ai_feedback=summary or ai_response,
                token=session.remote_auth_token,
            )
        except RemoteBackendError as exc:
            warning = f"Remote assessment sync failed for '{comp}': {exc}"
            if warning not in session.backend_warnings:
                session.backend_warnings.append(warning)

    # Decide next phase
    if session.is_last_competency:
        session.phase = 'final_assessment'
        next_action = 'final_assessment'
        message_suffix = "\n\n🎉 You've completed all competencies! Send your answer to the **Final Assessment** to finish."
    else:
        session.advance_to_next_competency()
        await _setup_competency(session)
        next_action = 'next_competency'
        message_suffix = f"\n\n➡️ Moving on to next competency: **{session.current_competency}**. Send a message to start learning!"

    save_session(session)
    return {
        'session_id': session.session_id,
        'phase': session.phase,
        'assessed_competency': comp,
        'score': overall,
        'passed': passed,
        'message': ai_response + message_suffix,
        'next_action': next_action,
        'next_competency': session.current_competency if not session.is_last_competency else None,
        'assessment_detail': eval_data,
        'rubric_source': rubric_source,
        'remote_learning_session_id': remote_learning_session_id,
        'backend_warnings': session.backend_warnings,
    }


async def handle_final_assessment(session: LearnerSession, user_answer: str) -> dict:
    """
    Evaluates the learner's final assessment covering all competencies completed.
    """
    all_competencies = ', '.join(session.competencies)
    rubric = _load_rubric('default', session.user_level)

    session.add_message('user', user_answer)

    result = AssessmentCrew().crew().kickoff(inputs={
        'competency': f"Final Assessment covering: {all_competencies}",
        'scenario': f"Comprehensive assessment for {session.topic}",
        'user_response': user_answer,
        'rubric_json': json.dumps(rubric),
    })

    try:
        eval_data = json.loads(result.raw)
    except (json.JSONDecodeError, AttributeError):
        eval_data = {'overall_percent': 0.0, 'pass': False, 'summary': result.raw}

    overall = eval_data.get('overall_percent', 0.0)
    passed = eval_data.get('pass', overall >= PASS_THRESHOLD)
    summary = eval_data.get('summary', '')

    session.final_assessment_result = eval_data
    session.phase = 'completed'

    # Compute competency summary
    comp_scores = [
        {'competency': r.competency, 'score': r.score, 'passed': r.passed}
        for r in session.completed_competencies
    ]

    ai_response = (
        f"**🏆 Final Assessment Complete — {session.topic}**\n\n"
        f"Score: {overall:.1f}% | Result: {'✅ Passed' if passed else '❌ Not yet'}\n\n"
        f"{summary}"
    )
    session.add_message('assistant', ai_response)

    save_session(session)
    return {
        'session_id': session.session_id,
        'phase': 'completed',
        'topic': session.topic,
        'user_level': session.user_level,
        'final_score': overall,
        'passed': passed,
        'message': ai_response,
        'competency_breakdown': comp_scores,
        'assessment_detail': eval_data,
        'backend_warnings': session.backend_warnings,
    }
