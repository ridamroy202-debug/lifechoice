from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.session_manager import create_session, get_session
from app.orchestrator import (
    handle_pre_assessment_start,
    handle_pre_assessment,
    handle_learning,
    handle_competency_assessment,
    handle_final_assessment,
)

app = FastAPI(
    title="LifeChoice AI Engine",
    version="0.2.0",
    description="AI-powered micro-credential learning platform with adaptive pre-assessment, interactive tutoring, and rubric-based evaluation.",
)


# ── Request Schemas ───────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    topic: str
    competencies: List[str]

class PreAssessmentChatRequest(BaseModel):
    session_id: str
    answer: str

class LearnChatRequest(BaseModel):
    session_id: str
    message: str

class StartPreAssessmentRequest(BaseModel):
    session_id: str

class AssessmentSubmitRequest(BaseModel):
    session_id: str
    answer: str


# ── Guards ────────────────────────────────────────────────────────────────

def _get_or_404(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session

def _require_phase(session, expected: str):
    if session.phase != expected:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires phase '{expected}', but session is in phase '{session.phase}'."
        )


# ── Routes ────────────────────────────────────────────────────────────────

@app.post("/session/start", summary="Start a new learning session")
def start_session(req: StartSessionRequest):
    """
    Creates a new learner session.
    Provide the topic and the full list of competencies to cover (in order).
    Returns a session_id to use in all subsequent calls.
    """
    if not req.competencies:
        raise HTTPException(status_code=400, detail="At least one competency is required.")

    session = create_session(topic=req.topic, competencies=req.competencies)
    return {
        "session_id": session.session_id,
        "topic": session.topic,
        "competencies": session.competencies,
        "total_competencies": len(session.competencies),
        "phase": session.phase,
        "message": (
            f"Session started for '{req.topic}' with {len(req.competencies)} competencies. "
            "Call POST /pre-assessment/chat to begin the pre-assessment."
        ),
    }


@app.post("/pre-assessment/start", summary="Generate the first pre-assessment question")
async def pre_assessment_start(req: StartPreAssessmentRequest):
    """
    Call this ONCE right after /session/start.
    Generates and returns the first scenario-based pre-assessment question
    based on the topic and competencies. No user input required.
    After receiving this question, use POST /pre-assessment/chat to submit answers.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'pre_assessment')
    return await handle_pre_assessment_start(session)


@app.post("/pre-assessment/chat", summary="Submit a pre-assessment answer")
async def pre_assessment_chat(req: PreAssessmentChatRequest):
    """
    Submit an answer to the current pre-assessment question.
    The AI will return the next question, OR — after 4 turns —
    classify the learner's level and transition to the learning phase.

    Response includes `done: true` and `level` when pre-assessment is complete.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'pre_assessment')
    return await handle_pre_assessment(session, req.answer)


@app.post("/learn/chat", summary="Send a message in the learning chat")
async def learn_chat(req: LearnChatRequest):
    """
    Send a message to the AI Tutor for the current competency.
    The tutor responds with Socratic feedback + next scenario.
    After 9 turns, `ready_for_assessment: true` signals you should call
    POST /assessment/competency next.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'learning')
    return await handle_learning(session, req.message)


@app.post("/assessment/competency", summary="Submit competency assessment answer")
async def competency_assessment(req: AssessmentSubmitRequest):
    """
    Submit the learner's answer for the current competency's assessment.
    The AI evaluates using a rubric (powered by Claude Haiku).

    On completion:
    - If more competencies remain → advances to the next competency (phase: learning)
    - If last competency → transitions to final assessment (phase: final_assessment)
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'competency_assessment')
    return await handle_competency_assessment(session, req.answer)


@app.post("/assessment/final", summary="Submit final assessment answer")
async def final_assessment(req: AssessmentSubmitRequest):
    """
    Submit the learner's answer for the final assessment (covers all competencies).
    Returns overall score, pass/fail, and a competency-by-competency breakdown.
    Session phase becomes 'completed' after this call.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'final_assessment')
    return await handle_final_assessment(session, req.answer)


@app.get("/session/{session_id}", summary="Get session status")
def get_session_status(session_id: str):
    """
    Returns the full current state of a learner session:
    phase, current competency, turn counts, level, scores so far.
    """
    session = _get_or_404(session_id)
    return {
        "session_id": session.session_id,
        "topic": session.topic,
        "phase": session.phase,
        "user_level": session.user_level,
        "weak_areas": session.weak_areas,
        "current_competency": session.current_competency,
        "competency_index": session.current_competency_index,
        "total_competencies": len(session.competencies),
        "pre_assessment_turn": session.pre_assessment_turn,
        "learning_turn": session.learning_turn,
        "max_learning_turns": session.max_learning_turns,
        "completed_competencies": [
            {"competency": r.competency, "score": r.score, "passed": r.passed}
            for r in session.completed_competencies
        ],
        "message_count": len(session.messages),
    }
