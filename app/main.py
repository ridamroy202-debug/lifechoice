from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from app.session_manager import create_session, get_session
from app.orchestrator import (
    build_competency_intro,
    handle_pre_assessment_start,
    handle_pre_assessment,
    handle_learning,
    handle_competency_assessment,
    handle_final_assessment,
)
from app.remote_backend import RemoteBackendError, remote_backend_client

openapi_tags = [
    {
        "name": "Sessions",
        "description": "Create a learner session and inspect its current state.",
    },
    {
        "name": "Pre-Assessment",
        "description": "Run the competency pre-assessment used to personalize teaching difficulty.",
    },
    {
        "name": "Learning",
        "description": "Run the guided tutoring chat for the active competency.",
    },
    {
        "name": "Assessments",
        "description": "Submit competency assessment answers and inspect completion state.",
    },
    {
        "name": "Remote Backend",
        "description": "Proxy authentication and lesson data from the LifeChoice backend.",
    },
]

app = FastAPI(
    title="LifeChoice AI Engine",
    version="1.0.0",
    description="AI teaching engine connected to the remote LifeChoice backend for lesson context, rubrics, enrollment access, and learning session sync.",
    openapi_tags=openapi_tags,
)


# ── Request Schemas ───────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    learner_id: Optional[str] = None
    topic: Optional[str] = None
    competencies: List[str] = Field(default_factory=list)
    domain_id: Optional[int] = None
    micro_credential_id: Optional[int] = None
    auth_token: Optional[str] = None

class PreAssessmentChatRequest(BaseModel):
    session_id: str
    answer: str

class LearnChatRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_text(self):
        if not (self.message or self.answer):
            raise ValueError("Provide either 'message' or 'answer'.")
        return self

    @property
    def text(self) -> str:
        return str(self.message or self.answer or "")

class StartPreAssessmentRequest(BaseModel):
    session_id: str

class AssessmentSubmitRequest(BaseModel):
    session_id: str
    answer: Optional[str] = None
    message: Optional[str] = None

    @model_validator(mode="after")
    def validate_text(self):
        if not (self.answer or self.message):
            raise ValueError("Provide either 'answer' or 'message'.")
        return self

    @property
    def text(self) -> str:
        return str(self.answer or self.message or "")


class PreAssessmentQuestionsRequest(BaseModel):
    topic: str
    competencies: List[str]


class BackendLoginRequest(BaseModel):
    email: str
    password: str

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


def _extract_remote_catalog(payload: dict, domain_id: int, micro_credential_id: int) -> tuple[dict, dict]:
    domains = payload.get("data", {}).get("domains", [])
    domain_entry = next((item for item in domains if int(item.get("id", -1)) == int(domain_id)), None)
    if domain_entry is None:
        raise HTTPException(status_code=404, detail="Remote domain not found")
    micro_credentials = domain_entry.get("micro_credentials", [])
    micro_entry = next((item for item in micro_credentials if int(item.get("id", -1)) == int(micro_credential_id)), None)
    if micro_entry is None:
        raise HTTPException(status_code=404, detail="Remote micro-credential not found")
    return domain_entry, micro_entry


# ── Routes ────────────────────────────────────────────────────────────────


@app.post("/session/start", summary="Start a new learning session", tags=["Sessions"])
def start_session(req: StartSessionRequest):
    """
    Creates a new learner session.
    Provide the topic and the full list of competencies to cover (in order).
    Returns a session_id to use in all subsequent calls.
    """
    if req.domain_id is not None and req.micro_credential_id is not None:
        effective_token = req.auth_token or remote_backend_client.default_token or None
        try:
            payload = remote_backend_client.fetch_lesson_competencies(
                domain_id=req.domain_id,
                micro_credential_id=req.micro_credential_id,
            )
        except RemoteBackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        domain_entry, micro_entry = _extract_remote_catalog(payload, req.domain_id, req.micro_credential_id)
        ordered_competencies = sorted(
            micro_entry.get("competencies", []),
            key=lambda item: int(item.get("code") or 9999),
        )
        competency_titles = [str(item.get("title")) for item in ordered_competencies if item.get("title")]
        if not competency_titles:
            raise HTTPException(status_code=400, detail="Remote micro-credential has no competencies")

        warnings: list[str] = []
        remote_access_id = None
        if effective_token:
            try:
                access = remote_backend_client.check_access(
                    req.micro_credential_id,
                    token=effective_token,
                )
            except RemoteBackendError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            access_data = access.get("access", {})
            if access_data.get("can_start_session") is False:
                raise HTTPException(status_code=403, detail=access_data.get("message") or "Remote backend denied access")
            enrollment = access_data.get("enrollment") or {}
            remote_access_id = enrollment.get("id")
        else:
            warnings.append("No auth_token provided. Public lesson content is loaded, but enrollment/access and remote session sync will be disabled.")

        competency_details = {
            str(item["title"]): {
                "id": item.get("id"),
                "code": item.get("code"),
                "description": item.get("description"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
            for item in ordered_competencies
            if item.get("title")
        }

        session = create_session(
            topic=str(micro_entry.get("micro_credential") or micro_entry.get("name") or f"MC {req.micro_credential_id}"),
            competencies=competency_titles,
            source="remote",
            learner_id=req.learner_id,
            domain_id=req.domain_id,
            remote_micro_credential_id=req.micro_credential_id,
            remote_micro_credential_level=micro_entry.get("level"),
            remote_source=domain_entry.get("source"),
            remote_auth_token=effective_token,
            remote_access_id=remote_access_id if isinstance(remote_access_id, int) else None,
            backend_warnings=warnings,
            competency_details=competency_details,
        )
        intro_message = build_competency_intro(session)
        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "competencies": session.competencies,
            "total_competencies": len(session.competencies),
            "phase": session.phase,
            "source": session.source,
            "domain_id": session.domain_id,
            "micro_credential_id": session.remote_micro_credential_id,
            "remote_access_id": session.remote_access_id,
            "backend_warnings": session.backend_warnings,
            "current_competency": session.current_competency,
            "interaction_number": 1,
            "message": intro_message,
        }

    if not req.topic or not req.competencies:
        raise HTTPException(
            status_code=400,
            detail="Provide either topic + competencies, or domain_id + micro_credential_id",
        )

    session = create_session(topic=req.topic, competencies=req.competencies, source="manual", learner_id=req.learner_id)
    intro_message = build_competency_intro(session)
    return {
        "session_id": session.session_id,
        "topic": session.topic,
        "competencies": session.competencies,
        "total_competencies": len(session.competencies),
        "phase": session.phase,
        "source": session.source,
        "current_competency": session.current_competency,
        "interaction_number": 1,
        "message": intro_message,
    }


@app.post(
    "/pre-assessment/questions",
    summary="Create session and get the first pre-assessment question",
    tags=["Pre-Assessment"],
)
async def pre_assessment_questions(req: PreAssessmentQuestionsRequest):
    """
    Convenience endpoint: creates a session *and* immediately returns the first
    scenario-based pre-assessment question. Use the returned session_id for
    follow-up answers via POST /pre-assessment/chat.
    """
    if not req.competencies:
        raise HTTPException(status_code=400, detail="At least one competency is required.")

    session = create_session(topic=req.topic, competencies=req.competencies)
    intro_message = build_competency_intro(session)
    first_question = await handle_pre_assessment_start(session)
    first_question["intro_message"] = intro_message
    return first_question


@app.post("/backend/auth/login", summary="Login to the remote backend", tags=["Remote Backend"])
def backend_login(req: BackendLoginRequest):
    try:
        payload = remote_backend_client.login(req.email, req.password)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/lesson/competencies",
    summary="Proxy lesson competencies from the remote backend",
    tags=["Remote Backend"],
)
def backend_lesson_competencies(
    domain_id: int = Query(...),
    micro_credential_id: int = Query(...),
    competency_id: int | None = Query(None),
):
    try:
        payload = remote_backend_client.fetch_lesson_competencies(
            domain_id=domain_id,
            micro_credential_id=micro_credential_id,
            competency_id=competency_id,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/lesson/rubric/{competency_id}",
    summary="Proxy rubric from the remote backend",
    tags=["Remote Backend"],
)
def backend_lesson_rubric(competency_id: int):
    try:
        payload = remote_backend_client.fetch_rubric(competency_id)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not payload:
        raise HTTPException(status_code=404, detail="Remote rubric not found")
    return payload

@app.post(
    "/pre-assessment/start",
    summary="Generate the first pre-assessment question",
    tags=["Pre-Assessment"],
)
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


@app.post("/pre-assessment/chat", summary="Submit a pre-assessment answer", tags=["Pre-Assessment"])
async def pre_assessment_chat(req: PreAssessmentChatRequest):
    """
    Submit an answer to the current pre-assessment question.
    The AI classifies competency readiness from this answer and immediately
    starts the first teaching interaction for the competency.

    Response includes `done: true` and `level` when pre-assessment is complete.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'pre_assessment')
    return await handle_pre_assessment(session, req.answer)


@app.post("/learn/chat", summary="Send a message in the learning chat", tags=["Learning"])
async def learn_chat(req: LearnChatRequest):
    """
    Send a message to the AI Tutor for the current competency.
    The tutor now follows a per-competency interaction engine:
    - Interaction 1: intro
    - Interaction 2: pre-assessment
    - Interactions 3-8: personalized teaching with mastery checks
    - Extra revision turns appear only when the learner fails the mastery gate

    Response includes `chat_stage`, `bloom_level`, and `is_doubt_phase` fields.
    `ready_for_assessment: true` means the response already contains the
    assessment prompt and the learner should answer it with POST /assessment/competency.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'learning')
    return await handle_learning(session, req.text)


@app.post(
    "/assessment/competency",
    summary="Submit competency assessment answer",
    tags=["Assessments"],
)
async def competency_assessment(req: AssessmentSubmitRequest):
    """
    Submit the learner's answer for the current competency's assessment.
    The AI evaluates using the locked rubric with a fixed 75% pass threshold.

    On completion:
    - If more competencies remain -> advances to the next competency intro
    - If the learner fails -> teaching restarts from interaction 3
    - If the last competency passes -> the session is completed
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'competency_assessment')
    return await handle_competency_assessment(session, req.text)


@app.post("/assessment/final", summary="Submit final assessment answer", tags=["Assessments"])
async def final_assessment(req: AssessmentSubmitRequest):
    """
    Legacy endpoint kept for compatibility. The current engine completes the
    learner journey after the final competency assessment rather than a separate final.
    """
    session = _get_or_404(req.session_id)
    return await handle_final_assessment(session, req.text)


@app.get("/session/{session_id}", summary="Get session status", tags=["Sessions"])
def get_session_status(session_id: str):
    """
    Returns the full current state of a learner session:
    phase, current competency, turn counts, level, scores so far.
    """
    session = _get_or_404(session_id)
    gamification = None
    if session.current_remote_learning_session_id and session.remote_auth_token:
        try:
            gamification = remote_backend_client.fetch_gamification_progress(
                session.current_remote_learning_session_id,
                token=session.remote_auth_token,
            )
        except RemoteBackendError:
            gamification = None

    return {
        "session_id": session.session_id,
        "topic": session.topic,
        "learner_id": session.learner_id,
        "source": session.source,
        "domain_id": session.domain_id,
        "remote_micro_credential_id": session.remote_micro_credential_id,
        "remote_micro_credential_level": session.remote_micro_credential_level,
        "remote_access_id": session.remote_access_id,
        "phase": session.phase,
        "user_level": session.user_level,
        "weak_areas": session.weak_areas,
        "interaction_number": session.competency_interaction,
        "pre_assessment_completed": session.pre_assessment_completed,
        "current_difficulty": session.current_difficulty,
        "formative_check_results": session.formative_check_results,
        "awaiting_formative_response": session.awaiting_formative_response,
        "revision_required": session.revision_required,
        "final_assessment_unlocked": session.final_assessment_unlocked,
        "current_assessment_attempts": session.current_assessment_attempts,
        "current_assessment_prompt": session.current_assessment_prompt,
        "personalization_state": session.personalization_state,
        "current_competency": session.current_competency,
        "current_remote_competency_id": session.current_remote_competency_id,
        "current_remote_learning_session_id": session.current_remote_learning_session_id,
        "competency_index": session.current_competency_index,
        "current_subpart_index": session.current_subpart_index + 1,
        "current_subpart": session.current_subpart,
        "total_subparts": len(session.competency_subparts.get(session.current_competency, [])),
        "total_competencies": len(session.competencies),
        "pre_assessment_turn": session.pre_assessment_turn,
        "learning_turn": session.learning_turn,
        "max_learning_turns": session.max_learning_turns,
        "max_competency_interactions": session.max_competency_interactions,
        "chat_stage": session.chat_stage,
        "bloom_level": session.bloom_level,
        "is_doubt_phase": session.is_doubt_phase,
        "backend_warnings": session.backend_warnings,
        "competency_details": session.competency_details.get(session.current_competency, {}),
        "gamification_progress": gamification,
        "completed_competencies": [
            {"competency": r.competency, "score": r.score, "passed": r.passed}
            for r in session.completed_competencies
        ],
        "message_count": len(session.messages),
    }


