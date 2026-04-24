from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Optional
from app.session_manager import create_session, get_session
from fastapi.responses import HTMLResponse, Response
from app.certificates import (
    get_certificate,
    issue_certificate,
    render_certificate_html,
    render_certificate_pdf,
    render_qr_png,
)
from app.db import init_db
from app.orchestrator import (
    build_competency_intro,
    handle_pre_assessment_start,
    handle_pre_assessment,
    handle_learning,
    handle_competency_assessment,
    handle_final_assessment,
)
from app.persistence import (
    append_event_log,
    get_locked_rubric,
    get_unresolved_anomalies,
    list_badges,
    missing_locked_rubrics,
    seed_locked_rubrics_from_yaml,
    upsert_learner,
    upsert_locked_rubric,
)
from app.policy import build_gamification_payload, build_session_runtime_payload
from app.remote_backend import RemoteBackendError, remote_backend_client
from app.settings import configure_logging, settings

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
    {
        "name": "Certificates",
        "description": "Issue and verify learner certificates after successful completion.",
    },
]

app = FastAPI(
    title="LifeChoice AI Engine",
    version="1.0.0",
    description="AI teaching engine connected to the remote LifeChoice backend for lesson context, rubrics, enrollment access, and learning session sync.",
    openapi_tags=openapi_tags,
)

configure_logging()


@app.on_event("startup")
def startup_event():
    init_db()
    seed_locked_rubrics_from_yaml()


def _log_event(session_id: str | None, learner_id: str | None, route: str, event_type: str, payload: dict[str, Any]) -> None:
    append_event_log(session_id, learner_id, route, event_type, payload)


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


class CertificateGenerateRequest(BaseModel):
    session_id: str
    auth_token: Optional[str] = None


class BackendLearningSessionStartRequest(BaseModel):
    competency_id: int
    auth_token: Optional[str] = None


class BackendLearningInteractionRequest(BaseModel):
    interaction_type: str = "teaching"
    ai_prompt: str
    ai_response: str
    learner_input: Optional[str] = None
    formative_passed: Optional[bool] = None
    auth_token: Optional[str] = None


class BackendLearningAssessmentRequest(BaseModel):
    scenario_question: str
    learner_response: str
    rubric_score: float
    ai_feedback: str
    auth_token: Optional[str] = None


class LockedRubricRegisterRequest(BaseModel):
    competency_name: str
    rubric_json: dict[str, Any]
    version: int = 1
    display_name: Optional[str] = None


class CertificateViewResponse(BaseModel):
    certificate_id: str
    learner_name: str
    learner_email: Optional[str] = None
    micro_credential_title: str
    issue_date: str
    verification_url: str
    qr_code_url: str
    pdf_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    competencies: List[dict[str, Any]] = Field(default_factory=list)
    certificate_html: Optional[str] = None

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


def _public_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _certificate_response(record, include_html: bool = True) -> CertificateViewResponse:
    return CertificateViewResponse(
        certificate_id=record.certificate_id,
        learner_name=record.learner_name,
        learner_email=record.learner_email,
        micro_credential_title=record.micro_credential_title,
        issue_date=record.issue_date,
        verification_url=record.verification_url,
        qr_code_url=record.qr_code_url,
        pdf_url=record.pdf_url,
        metadata=record.metadata,
        competencies=record.competencies,
        certificate_html=render_certificate_html(record) if include_html else None,
    )


def _require_rubric_admin_key(provided_key: Optional[str]) -> None:
    expected_key = settings.rubric_admin_key
    if not expected_key:
        raise HTTPException(status_code=503, detail="Rubric administration is disabled until RUBRIC_ADMIN_KEY is configured.")
    if provided_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid rubric administration key.")


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
        if not effective_token:
            raise HTTPException(status_code=400, detail="auth_token is required for remote-backed learning sessions.")
        try:
            payload = remote_backend_client.fetch_lesson_competencies(
                domain_id=req.domain_id,
                micro_credential_id=req.micro_credential_id,
            )
            access = remote_backend_client.check_access(
                req.micro_credential_id,
                token=effective_token,
            )
            profile_payload = remote_backend_client.fetch_profile(token=effective_token)
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
        missing_rubrics = missing_locked_rubrics(competency_titles)
        if missing_rubrics:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Locked rubrics are missing for this micro-credential. Provision them before launch.",
                    "micro_credential_id": req.micro_credential_id,
                    "missing_competencies": missing_rubrics,
                },
            )

        profile_root = profile_payload.get("data") if isinstance(profile_payload.get("data"), dict) else profile_payload
        profile_id = str(profile_root.get("id") or profile_root.get("user_id") or "").strip()
        if req.learner_id and profile_id and str(req.learner_id) != profile_id:
            _log_event(None, req.learner_id, "/session/start", "identity_mismatch", {"provided_learner_id": req.learner_id, "profile_id": profile_id})
            raise HTTPException(status_code=403, detail="Learner ID does not match the authenticated learner profile.")
        learner_id = profile_id or req.learner_id
        if not learner_id:
            raise HTTPException(status_code=400, detail="Could not determine learner identity from the authenticated profile.")

        access_data = access.get("access", {})
        can_access = access_data.get("can_access")
        can_start_session = access_data.get("can_start_session")
        if can_access is False or can_start_session is False:
            raise HTTPException(status_code=403, detail=access_data.get("message") or "Remote backend denied access")
        enrollment = access_data.get("enrollment") or {}
        remote_access_id = enrollment.get("id")

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

        upsert_learner(learner_id, profile_payload, verified=True)
        session = create_session(
            topic=str(micro_entry.get("micro_credential") or micro_entry.get("name") or f"MC {req.micro_credential_id}"),
            competencies=competency_titles,
            source="remote",
            learner_id=learner_id,
            domain_id=req.domain_id,
            remote_micro_credential_id=req.micro_credential_id,
            remote_micro_credential_level=micro_entry.get("level"),
            remote_source=domain_entry.get("source"),
            remote_auth_token=effective_token,
            remote_access_id=remote_access_id if isinstance(remote_access_id, int) else None,
            backend_warnings=[],
            competency_details=competency_details,
            learner_profile=profile_payload,
            identity_verified=True,
            identity_verified_at=str(profile_root.get("verified_at") or profile_root.get("updated_at") or profile_root.get("created_at") or ""),
        )
        intro_message = build_competency_intro(session)
        _log_event(session.session_id, session.learner_id, "/session/start", "session_started", {
            "topic": session.topic,
            "competency_count": len(session.competencies),
            "identity_verified": True,
        })
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
            "identity_verified": session.identity_verified,
            "gamification": build_gamification_payload(session),
            **build_session_runtime_payload(session),
        }

    if not req.topic or not req.competencies:
        raise HTTPException(
            status_code=400,
            detail="Provide either topic + competencies, or domain_id + micro_credential_id",
        )

    learner_id = req.learner_id or f"manual-{req.topic.lower().replace(' ', '-')[:16]}"
    upsert_learner(learner_id, {"learner_id": learner_id, "source": "manual"}, verified=False)
    session = create_session(topic=req.topic, competencies=req.competencies, source="manual", learner_id=learner_id)
    intro_message = build_competency_intro(session)
    _log_event(session.session_id, session.learner_id, "/session/start", "session_started_manual", {"topic": session.topic, "competency_count": len(session.competencies)})
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
        "identity_verified": session.identity_verified,
        "gamification": build_gamification_payload(session),
        **build_session_runtime_payload(session),
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


@app.get("/backend/auth/profile/me", summary="Proxy learner profile from the remote backend", tags=["Remote Backend"])
def backend_profile_me(auth_token: Optional[str] = Query(None)):
    effective_token = auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.fetch_profile(token=effective_token)
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
    summary="Remote rubric proxy is currently unsupported",
    tags=["Remote Backend"],
)
def backend_lesson_rubric(competency_id: int):
    raise HTTPException(
        status_code=501,
        detail=(
            "Remote rubric lookup is disabled because the live LifeChoice backend Swagger schema "
            "does not currently expose a rubric-by-competency route."
        ),
    )


@app.get(
    "/backend/enrollment/check-access/{mc_id}",
    summary="Proxy enrollment access check from the remote backend",
    tags=["Remote Backend"],
)
def backend_enrollment_check_access(mc_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.check_access(mc_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.post(
    "/backend/learning/sessions/start",
    summary="Start a remote learning session",
    tags=["Remote Backend"],
)
def backend_learning_session_start(req: BackendLearningSessionStartRequest):
    effective_token = req.auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.start_learning_session(
            competency_id=req.competency_id,
            token=effective_token,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/learning/sessions/{session_id}",
    summary="Get a remote learning session",
    tags=["Remote Backend"],
)
def backend_learning_session_detail(session_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.fetch_learning_session(session_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.post(
    "/backend/learning/sessions/{session_id}/interact",
    summary="Record a remote learning interaction",
    tags=["Remote Backend"],
)
def backend_learning_session_interact(session_id: int, req: BackendLearningInteractionRequest):
    effective_token = req.auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.record_interaction(
            session_id=session_id,
            interaction_type=req.interaction_type,
            ai_prompt=req.ai_prompt,
            ai_response=req.ai_response,
            learner_input=req.learner_input,
            formative_passed=req.formative_passed,
            token=effective_token,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.post(
    "/backend/learning/sessions/{session_id}/assess",
    summary="Submit a remote learning assessment",
    tags=["Remote Backend"],
)
def backend_learning_session_assess(session_id: int, req: BackendLearningAssessmentRequest):
    effective_token = req.auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.submit_assessment(
            session_id=session_id,
            scenario_question=req.scenario_question,
            learner_response=req.learner_response,
            rubric_score=req.rubric_score,
            ai_feedback=req.ai_feedback,
            token=effective_token,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/gamification/progress/{session_id}",
    summary="Get remote gamification progress",
    tags=["Remote Backend"],
)
def backend_gamification_progress(session_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = auth_token or remote_backend_client.default_token or None
    try:
        payload = remote_backend_client.fetch_gamification_progress(session_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/micro-credential/readiness",
    summary="Check whether a micro-credential is launch-ready in the AI engine",
    tags=["Remote Backend"],
)
def backend_micro_credential_readiness(
    domain_id: int = Query(...),
    micro_credential_id: int = Query(...),
):
    try:
        payload = remote_backend_client.fetch_lesson_competencies(
            domain_id=domain_id,
            micro_credential_id=micro_credential_id,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    domain_entry, micro_entry = _extract_remote_catalog(payload, domain_id, micro_credential_id)
    ordered_competencies = sorted(
        micro_entry.get("competencies", []),
        key=lambda item: int(item.get("code") or 9999),
    )
    competency_titles = [str(item.get("title")) for item in ordered_competencies if item.get("title")]
    missing = missing_locked_rubrics(competency_titles)
    provisioned = [title for title in competency_titles if title not in missing]
    return {
        "domain_id": domain_id,
        "micro_credential_id": micro_credential_id,
        "micro_credential_title": micro_entry.get("micro_credential") or micro_entry.get("name"),
        "source": domain_entry.get("source"),
        "total_competencies": len(competency_titles),
        "launch_ready": len(missing) == 0,
        "provisioned_competencies": provisioned,
        "missing_competencies": missing,
    }


@app.post(
    "/admin/rubrics/register",
    summary="Register or replace a locked rubric for one competency",
    tags=["Assessments"],
)
def register_locked_rubric(req: LockedRubricRegisterRequest, x_rubric_admin_key: Optional[str] = Header(None)):
    _require_rubric_admin_key(x_rubric_admin_key)
    try:
        stored = upsert_locked_rubric(
            req.competency_name,
            req.rubric_json,
            version=req.version,
            display_name=req.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "competency_name": req.competency_name,
        "locked": True,
        "version": req.version,
        "rubric": stored,
    }

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
    response = await handle_pre_assessment_start(session)
    _log_event(session.session_id, session.learner_id, '/pre-assessment/start', 'pre_assessment_started', {'competency': session.current_competency})
    return response


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
    response = await handle_pre_assessment(session, req.answer)
    _log_event(session.session_id, session.learner_id, '/pre-assessment/chat', 'pre_assessment_answered', {'competency': session.current_competency, 'level': response.get('level')})
    return response


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
    response = await handle_learning(session, req.text)
    _log_event(session.session_id, session.learner_id, '/learn/chat', 'learning_turn', {'competency': session.current_competency, 'interaction_number': response.get('interaction_number'), 'phase': response.get('phase')})
    return response


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
    - If the last competency passes -> the session advances to the final assessment
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'competency_assessment')
    response = await handle_competency_assessment(session, req.text)
    _log_event(session.session_id, session.learner_id, '/assessment/competency', 'competency_assessed', {'competency': response.get('assessed_competency'), 'passed': response.get('passed'), 'score': response.get('score')})
    return response


@app.post("/assessment/final", summary="Submit final micro-credential assessment answer", tags=["Assessments"])
async def final_assessment(req: AssessmentSubmitRequest):
    """
    Submit the final integrated assessment after all competency assessments pass.
    Certificate generation is available only after this route moves the session to `completed`.
    """
    session = _get_or_404(req.session_id)
    _require_phase(session, 'final_assessment')
    response = await handle_final_assessment(session, req.text)
    _log_event(session.session_id, session.learner_id, '/assessment/final', 'final_assessment_submitted', {'passed': response.get('passed'), 'score': response.get('score')})
    return response


@app.post("/certificate/generate", summary="Generate a learner certificate", tags=["Certificates"], response_model=CertificateViewResponse)
def generate_certificate(req: CertificateGenerateRequest, request: Request):
    session = _get_or_404(req.session_id)
    if session.phase != "completed":
        raise HTTPException(status_code=400, detail="Certificate can only be generated after the session is completed.")
    if not session.identity_verified:
        raise HTTPException(status_code=400, detail="Learner identity must be verified before certificate generation.")
    anomalies = get_unresolved_anomalies(session.session_id)
    if any(item.get('severity') == 'critical' for item in anomalies):
        raise HTTPException(status_code=409, detail="Certificate generation blocked by unresolved critical compliance flags.")

    effective_token = req.auth_token or session.remote_auth_token or remote_backend_client.default_token or None
    if not effective_token:
        raise HTTPException(status_code=400, detail="auth_token is required to fetch learner profile details.")

    try:
        profile_payload = remote_backend_client.fetch_profile(token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    record = issue_certificate(session, profile_payload, _public_base_url(request))
    _log_event(session.session_id, session.learner_id, '/certificate/generate', 'certificate_generated', {'certificate_id': record.certificate_id})
    return _certificate_response(record)


@app.get("/certificate/verify/{certificate_id}", summary="Verify a generated certificate", tags=["Certificates"], response_model=CertificateViewResponse)
def verify_certificate(certificate_id: str):
    record = get_certificate(certificate_id)
    if not record:
        raise HTTPException(status_code=404, detail="Certificate not found.")
    return _certificate_response(record, include_html=False)


@app.get("/certificate/{certificate_id}/html", summary="Render certificate HTML", tags=["Certificates"], response_class=HTMLResponse)
def certificate_html(certificate_id: str):
    record = get_certificate(certificate_id)
    if not record:
        raise HTTPException(status_code=404, detail="Certificate not found.")
    return HTMLResponse(render_certificate_html(record))


@app.get("/certificate/{certificate_id}/qr.png", summary="Render certificate QR PNG", tags=["Certificates"])
def certificate_qr(certificate_id: str):
    record = get_certificate(certificate_id)
    if not record:
        raise HTTPException(status_code=404, detail="Certificate not found.")
    return Response(content=render_qr_png(record), media_type="image/png")


@app.get("/certificate/{certificate_id}/pdf", summary="Render certificate PDF", tags=["Certificates"])
def certificate_pdf(certificate_id: str):
    record = get_certificate(certificate_id)
    if not record:
        raise HTTPException(status_code=404, detail="Certificate not found.")
    headers = {"Content-Disposition": f"inline; filename={certificate_id}.pdf"}
    return Response(content=render_certificate_pdf(record), media_type="application/pdf", headers=headers)


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
        "identity_verified": session.identity_verified,
        "identity_verified_at": session.identity_verified_at,
        "remote_sync": session.remote_sync_status,
        "remote_last_event_at": session.remote_last_event_at,
        "phase": session.phase,
        "user_level": session.user_level,
        "weak_areas": session.weak_areas,
        "interaction_number": session.competency_interaction,
        "interaction_type": session.current_interaction_type,
        "delivery_format": session.current_delivery_format,
        "formative_slot": session.formative_slot_number,
        "required_next_action": session.required_next_action,
        "pre_assessment_completed": session.pre_assessment_completed,
        "current_difficulty": session.current_difficulty,
        "formative_check_results": session.formative_check_results,
        "awaiting_formative_response": session.awaiting_formative_response,
        "revision_required": session.revision_required,
        "final_assessment_unlocked": session.final_assessment_unlocked,
        "current_assessment_attempts": session.current_assessment_attempts,
        "current_assessment_prompt": session.current_assessment_prompt,
        "final_assessment_prompt": session.final_assessment_prompt,
        "final_assessment_attempts": session.final_assessment_attempts,
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
        "delivery_format_history": session.delivery_format_history,
        "concept_history": session.concept_history,
        "completion_badge": session.completion_badge,
        "gamification_progress": gamification,
        "gamification": build_gamification_payload(session),
        "earned_badges": list_badges(session.session_id),
        "anomaly_flags": get_unresolved_anomalies(session.session_id),
        "session_summary": session.session_summary,
        "completed_competencies": [
            {"competency": r.competency, "score": r.score, "passed": r.passed}
            for r in session.completed_competencies
        ],
        "message_count": len(session.messages),
    }
