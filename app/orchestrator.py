import json
from app.state import LearnerSession, CompetencyResult
from app.session_manager import get_session, save_session
from app.crews.pre_assessment_crew import PreAssessCrew
from app.crews.level_classifier_crew import LevelClassifierCrew
from app.crews.learning_path_planner import PathPlnner
from app.crews.studey_materils_crew import StudyMeterial
from app.crews.ai_tutor_agents_crew import TutorCrew
from app.crews.assessment_crew import AssessmentCrew

PRE_ASSESSMENT_TURNS = 4

RUBRICS = {
    'default': {
        'criteria': [
            {'name': 'Accuracy',     'weight': 0.40},
            {'name': 'Depth',        'weight': 0.35},
            {'name': 'Application',  'weight': 0.25},
        ]
    }
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_rubric(competency: str) -> dict:
    """Return a rubric dict for the given competency (fallback to default)."""
    return RUBRICS.get(competency.lower().replace(' ', '_'), RUBRICS['default'])


async def _setup_competency(session: LearnerSession):
    """Generate study material + learning plan for current competency (if not cached)."""
    comp = session.current_competency
    if comp not in session.study_materials:
        material = StudyMeterial().crew().kickoff(inputs={
            'topic': session.topic,
            'competency': comp,
            'user_level': session.user_level,
        })
        session.study_materials[comp] = material.raw

    if comp not in session.learning_plans:
        plan = PathPlnner().crew().kickoff(inputs={
            'topic': session.topic,
            'competency': comp,
            'user_level': session.user_level,
            'weak_areas': ', '.join(session.weak_areas),
        })
        session.learning_plans[comp] = plan.raw


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

    result = PreAssessCrew().crew().kickoff(inputs={
        'topic': session.topic,
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
    After max_learning_turns, signals the learner is ready for competency assessment.
    """
    session.learning_turn += 1
    session.add_message('user', user_message)

    comp = session.current_competency
    result = TutorCrew().crew().kickoff(inputs={
        'topic': session.topic,
        'competency': comp,
        'user_level': session.user_level,
        'weak_areas': ', '.join(session.weak_areas),
        'chat_history': session.format_recent_history(),
        'user_message': user_message,
        'turn_number': session.learning_turn,
        'max_turns': session.max_learning_turns,
        'study_material': session.study_materials.get(comp, ''),
    })
    ai_response = result.raw
    session.add_message('assistant', ai_response)

    ready_for_assessment = session.learning_turn >= session.max_learning_turns
    if ready_for_assessment:
        session.phase = 'competency_assessment'

    save_session(session)
    return {
        'session_id': session.session_id,
        'phase': session.phase,
        'competency': comp,
        'turn': session.learning_turn,
        'max_turns': session.max_learning_turns,
        'message': ai_response,
        'ready_for_assessment': ready_for_assessment,
    }


async def handle_competency_assessment(session: LearnerSession, user_answer: str) -> dict:
    """
    Evaluates a learner's response for the current competency using Claude (rubric-based).
    On pass/fail, either advances to the next competency or triggers final assessment.
    """
    comp = session.current_competency
    rubric = _load_rubric(comp)

    session.add_message('user', user_answer)

    result = AssessmentCrew().crew().kickoff(inputs={
        'competency': comp,
        'scenario': session.study_materials.get(comp, 'N/A'),
        'user_response': user_answer,
        'rubric_json': json.dumps(rubric),
    })

    # Parse assessment JSON
    try:
        eval_data = json.loads(result.raw)
    except (json.JSONDecodeError, AttributeError):
        eval_data = {'overall_percent': 0.0, 'pass': False, 'summary': result.raw}

    overall = eval_data.get('overall_percent', 0.0)
    passed = eval_data.get('pass', overall >= 70.0)
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
    }


async def handle_final_assessment(session: LearnerSession, user_answer: str) -> dict:
    """
    Evaluates the learner's final assessment covering all competencies completed.
    """
    all_competencies = ', '.join(session.competencies)
    rubric = _load_rubric('default')

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
    passed = eval_data.get('pass', overall >= 70.0)
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
    }