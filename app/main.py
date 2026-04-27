from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Optional
import re
from app.session_manager import create_session, get_session, save_session
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
    create_remote_learning_session_ref,
    get_locked_rubric,
    get_learner_competency_progress,
    get_remote_session_mapping,
    get_remote_learning_session_ref,
    get_unresolved_anomalies,
    list_badges,
    list_learner_competency_progress,
    missing_locked_rubrics,
    seed_locked_rubrics_from_yaml,
    update_remote_learning_session_ref,
    upsert_learner,
    upsert_learner_competency_progress,
    upsert_locked_rubric,
    upsert_remote_session_mapping,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logging()

_PLACEHOLDER_MESSAGES = {
    "string",
    "message",
    "answer",
    "response",
    "text",
    "your interaction content",
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
}


def _is_meaningful_message(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in _PLACEHOLDER_MESSAGES:
        return False
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if len(tokens) <= 1 and len(text) < 12:
        return False
    return True


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


class SessionInteractRequest(BaseModel):
    message: Optional[str] = None
    answer: Optional[str] = None
    response: Optional[str] = None
    auth_token: Optional[str] = None

    @property
    def text(self) -> str:
        for candidate in (self.message, self.answer, self.response):
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    @property
    def has_text(self) -> bool:
        for candidate in (self.message, self.answer, self.response):
            if _is_meaningful_message(candidate):
                return True
        return False


class PreAssessmentQuestionsRequest(BaseModel):
    topic: str
    competencies: List[str]


class BackendLoginRequest(BaseModel):
    email: str
    password: str


class BackendProfileRequest(BaseModel):
    micro_credential_id: int
    domain_id: Optional[int] = None
    auth_token: Optional[str] = None


class BackendLessonCompetenciesRequest(BaseModel):
    micro_credential_id: int
    domain_id: Optional[int] = None
    competency_id: Optional[int] = None


class BackendEnrollmentAccessRequest(BaseModel):
    micro_credential_id: int
    auth_token: Optional[str] = None


class CertificateGenerateRequest(BaseModel):
    session_id: str
    auth_token: Optional[str] = None


class BackendLearningSessionStartRequest(BaseModel):
    micro_credential_id: int
    competency_id: int
    domain_id: Optional[int] = None
    auth_token: Optional[str] = None


class BackendLearningInteractionRequest(BaseModel):
    auth_token: Optional[str] = None
    message: Optional[str] = Field(
        default=None,
        description="Simplified frontend field. When provided, it is used as the AI response payload if ai_response is omitted.",
    )
    interaction_type: Optional[str] = Field(
        default=None,
        description="Optional. Defaults to 'teaching' when omitted.",
    )
    ai_prompt: Optional[str] = Field(
        default=None,
        description="Optional legacy/internal field. Defaults to a backend-generated placeholder when omitted.",
    )
    ai_response: Optional[str] = Field(
        default=None,
        description="Optional legacy/internal field. If omitted, the route uses `message`.",
    )
    learner_input: Optional[str] = None
    formative_passed: Optional[bool] = None

    @model_validator(mode="after")
    def validate_payload(self):
        if not (self.ai_response or self.message):
            raise ValueError("Provide either 'message' or 'ai_response'.")
        return self

    @property
    def resolved_interaction_type(self) -> str:
        return (self.interaction_type or "teaching").strip() or "teaching"

    @property
    def resolved_ai_prompt(self) -> str:
        prompt = (self.ai_prompt or "").strip()
        if prompt:
            return prompt
        return "Interaction forwarded by the AI engine backend proxy."

    @property
    def resolved_ai_response(self) -> str:
        return str(self.ai_response or self.message or "").strip()


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


def _session_history_payload(session, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at,
        }
        for msg in session.messages[-limit:]
    ]


def _current_pending_prompt(session) -> Optional[str]:
    if session.phase == "pre_assessment":
        return session.pre_assessment_prompt
    if session.phase == "learning" and session.awaiting_formative_response:
        return session.current_formative_prompt
    if session.phase == "competency_assessment":
        return session.current_assessment_prompt
    if session.phase == "final_assessment":
        return session.final_assessment_prompt
    return None


def _build_session_payload(session):
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
        "academic_stage": session.academic_stage,
        "academic_guidance": session.academic_guidance,
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
        "current_aip_code": session.current_aip_code,
        "current_aip_name": session.current_aip_name,
        "aip_budget_total": session.aip_budget_total,
        "competency_aip_count": session.competency_aip_count,
        "competency_aip_history": session.competency_aip_history,
        "live_aip_call_count": len(session.live_aip_call_history),
        "mc_completion_reflection_fired": session.mc_completion_reflection_fired,
        "latest_binary_outcome": session.latest_binary_outcome,
        "developing_competency_active": session.developing_competency_active,
        "developing_competency_reason": session.developing_competency_reason,
        "local_assessment_passed": session.local_assessment_passed,
        "remote_assessment_synced": session.remote_assessment_synced,
        "remote_assessment_passed": session.remote_assessment_passed,
        "current_assessment_sync_error": session.current_assessment_sync_error,
        "formative_slot": session.formative_slot_number,
        "required_next_action": session.required_next_action,
        "unlimited_attempts": True,
        "pre_assessment_completed": session.pre_assessment_completed,
        "current_difficulty": session.current_difficulty,
        "formative_check_results": session.formative_check_results,
        "awaiting_formative_response": session.awaiting_formative_response,
        "current_formative_prompt": session.current_formative_prompt,
        "revision_required": session.revision_required,
        "final_assessment_unlocked": session.final_assessment_unlocked,
        "current_assessment_attempts": session.current_assessment_attempts,
        "current_assessment_prompt": session.current_assessment_prompt,
        "final_assessment_prompt": session.final_assessment_prompt,
        "final_assessment_attempts": session.final_assessment_attempts,
        "personalization_state": session.personalization_state,
        "competency": session.current_competency,
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


def _build_interact_response(session, response: Optional[dict[str, Any]], *, counted_as_interaction: bool):
    payload = _build_session_payload(session)
    response = response or {}
    for key, value in response.items():
        if key in {"counted_as_interaction", "history", "current_prompt", "interaction_result"}:
            continue
        if key not in payload:
            payload[key] = value

    if "message" in response:
        payload["message"] = response["message"]
    if "interaction_result" in response:
        payload["interaction_result"] = response["interaction_result"]
    else:
        payload["interaction_result"] = None

    payload["counted_as_interaction"] = counted_as_interaction
    payload["history"] = _session_history_payload(session)
    payload["current_prompt"] = _current_pending_prompt(session)
    return payload


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


def _resolve_remote_catalog(
    *,
    micro_credential_id: int,
    domain_id: Optional[int] = None,
) -> tuple[dict, dict, dict]:
    try:
        payload = remote_backend_client.fetch_lesson_competencies(
            domain_id=domain_id,
            micro_credential_id=micro_credential_id,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    domains = payload.get("data", {}).get("domains", [])
    if not domains:
        raise HTTPException(status_code=404, detail="Remote lesson catalog did not return any domains.")

    if domain_id is not None:
        domain_entry, micro_entry = _extract_remote_catalog(payload, domain_id, micro_credential_id)
        return payload, domain_entry, micro_entry

    for domain_entry in domains:
        for micro_entry in domain_entry.get("micro_credentials", []):
            if int(micro_entry.get("id", -1)) == int(micro_credential_id):
                return payload, domain_entry, micro_entry
    raise HTTPException(status_code=404, detail="Remote micro-credential not found")


def _resolve_effective_token(auth_token: Optional[str]) -> str:
    token = auth_token or remote_backend_client.default_token or None
    if not token:
        raise HTTPException(status_code=400, detail="auth_token is required.")
    return token


def _extract_profile_root(profile_payload: dict[str, Any]) -> dict[str, Any]:
    profile = profile_payload.get("profile")
    if isinstance(profile, dict):
        return profile
    data = profile_payload.get("data")
    if isinstance(data, dict):
        return data
    return profile_payload


def _extract_learner_id(profile_payload: dict[str, Any]) -> str:
    profile_root = _extract_profile_root(profile_payload)
    learner_id = str(profile_root.get("id") or profile_root.get("user_id") or "").strip()
    if not learner_id:
        raise HTTPException(status_code=400, detail="Could not determine learner identity from the authenticated profile.")
    return learner_id


def _ordered_competencies(micro_entry: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(micro_entry.get("competencies", []), key=lambda item: int(item.get("code") or 9999))


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


def _hydrate_session_from_remote_backend_session(remote_session_id: int, auth_token: Optional[str] = None):
    mapping = get_remote_session_mapping(remote_session_id)
    if mapping:
        existing = get_session(mapping["local_session_id"])
        if existing:
            return existing

    effective_token = auth_token or remote_backend_client.default_token or None
    if not effective_token:
        raise HTTPException(
            status_code=400,
            detail="auth_token is required to hydrate a backend session when no default backend token is configured.",
        )

    try:
        remote_payload = remote_backend_client.fetch_learning_session(remote_session_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    remote_session = remote_payload.get("session") if isinstance(remote_payload.get("session"), dict) else remote_payload
    micro_credential_id = int(remote_session.get("micro_credential") or 0)
    competency_id = int(remote_session.get("competency") or 0)
    if not micro_credential_id or not competency_id:
        raise HTTPException(status_code=502, detail="Remote backend session did not return micro-credential and competency identifiers.")

    local_ref = get_remote_learning_session_ref(remote_session_id)
    learner_id = local_ref.get("learner_id") if local_ref else None
    domain_hint = local_ref.get("domain_id") if local_ref else None

    try:
        profile_payload = remote_backend_client.fetch_profile(token=effective_token)
        profile_root = _extract_profile_root(profile_payload)
        learner_id = learner_id or _extract_learner_id(profile_payload)
        upsert_learner(learner_id, profile_root, verified=bool(learner_id))
    except RemoteBackendError:
        profile_root = {}
    except HTTPException:
        profile_root = {}

    _, domain_entry, micro_entry = _resolve_remote_catalog(
        micro_credential_id=micro_credential_id,
        domain_id=domain_hint,
    )
    ordered_competencies = _ordered_competencies(micro_entry)
    target = next((item for item in ordered_competencies if int(item.get("id", -1)) == competency_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Remote competency was not found in the lesson catalog.")

    competency_title = str(target.get("title") or competency_id)
    details = {
        competency_title: {
            "id": int(target.get("id", competency_id)),
            "code": target.get("code"),
            "description": target.get("description"),
            "created_at": target.get("created_at"),
            "updated_at": target.get("updated_at"),
        }
    }
    topic = str(micro_entry.get("micro_credential") or micro_entry.get("name") or f"Micro-Credential {micro_credential_id}")
    session = create_session(
        topic=topic,
        competencies=[competency_title],
        source="remote",
        domain_id=int(domain_entry.get("id", 0)) if domain_entry.get("id") is not None else None,
        remote_micro_credential_id=micro_credential_id,
        remote_micro_credential_level=micro_entry.get("level"),
        remote_source=domain_entry.get("source"),
        remote_auth_token=effective_token,
        learner_profile=profile_root,
        learner_id=learner_id,
        identity_verified=bool(learner_id),
        identity_verified_at=profile_root.get("updated_at") if profile_root else None,
        competency_details=details,
        remote_learning_sessions={competency_title: int(remote_session_id)},
    )
    if isinstance(remote_session.get("attempt_number"), int):
        session.competency_attempts[competency_title] = int(remote_session["attempt_number"])
    if remote_session.get("interactions"):
        session.backend_warnings.append(
            "Hydrated from backend session with existing remote interactions; local learning flow is restarting from the beginning for authoritative progression."
        )
    create_remote_learning_session_ref(
        remote_session_id,
        learner_id or f"remote-session-{remote_session_id}",
        micro_credential_id,
        competency_id,
        competency_title,
        domain_id=int(domain_entry.get("id", 0)) if domain_entry.get("id") is not None else None,
    )
    upsert_remote_session_mapping(remote_session_id, session.session_id)
    save_session(session)
    _log_event(
        session.session_id,
        session.learner_id,
        f"/session/{remote_session_id}/interact",
        "remote_session_hydrated",
        {
            "remote_session_id": remote_session_id,
            "micro_credential_id": micro_credential_id,
            "competency_id": competency_id,
            "competency": competency_title,
        },
    )
    return session


def _get_session_for_interact(session_id: str, auth_token: Optional[str] = None):
    session = get_session(session_id)
    if session:
        return session
    if session_id.isdigit():
        return _hydrate_session_from_remote_backend_session(int(session_id), auth_token)
    raise HTTPException(status_code=404, detail="Session not found.")


async def _ensure_interact_bootstrap(session) -> bool:
    if session.phase != "pre_assessment":
        return False
    if session.intro_delivered and session.pre_assessment_prompt:
        return False
    await handle_pre_assessment_start(session)
    return True


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
        unresolved_missing_rubrics = list(missing_rubrics)
        if missing_rubrics:
            unresolved_missing_rubrics = []
            title_to_remote_id = {
                str(item.get("title")): int(item.get("id"))
                for item in ordered_competencies
                if item.get("title") and item.get("id") is not None
            }
            for competency_title in missing_rubrics:
                remote_competency_id = title_to_remote_id.get(competency_title)
                if not remote_competency_id:
                    unresolved_missing_rubrics.append(competency_title)
                    continue
                try:
                    rubric_payload = remote_backend_client.fetch_competency_rubric(
                        remote_competency_id,
                        token=effective_token,
                    )
                except RemoteBackendError:
                    unresolved_missing_rubrics.append(competency_title)
                    continue
                if not (rubric_payload.get("rubric_rules") or {}).get("rubric_rules"):
                    unresolved_missing_rubrics.append(competency_title)
        if unresolved_missing_rubrics:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Locked rubrics are missing for this micro-credential and no remote rubric rules were available.",
                    "micro_credential_id": req.micro_credential_id,
                    "missing_competencies": unresolved_missing_rubrics,
                },
            )

        profile_root = _extract_profile_root(profile_payload)
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
            "remote_micro_credential_level": session.remote_micro_credential_level,
            "academic_stage": session.academic_stage,
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


@app.post("/backend/auth/profile/me", summary="Fetch learner profile with micro-credential context", tags=["Remote Backend"])
def backend_profile_me(req: BackendProfileRequest):
    effective_token = _resolve_effective_token(req.auth_token)
    try:
        payload = remote_backend_client.fetch_profile(token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _, domain_entry, micro_entry = _resolve_remote_catalog(
        micro_credential_id=req.micro_credential_id,
        domain_id=req.domain_id,
    )
    competencies = _ordered_competencies(micro_entry)
    return {
        "profile": payload,
        "micro_credential": {
            "id": int(micro_entry.get("id", req.micro_credential_id)),
            "title": micro_entry.get("micro_credential") or micro_entry.get("name"),
            "domain_id": int(domain_entry.get("id", 0)),
            "domain_name": domain_entry.get("domain"),
            "level": micro_entry.get("level"),
            "competency_count": len(competencies),
        },
        "competencies": competencies,
    }


@app.get("/backend/auth/profile/me", summary="Proxy learner profile from the remote backend", tags=["Remote Backend"])
def backend_profile_me_get(auth_token: Optional[str] = Query(None)):
    effective_token = _resolve_effective_token(auth_token)
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
    micro_credential_id: int = Query(...),
    domain_id: int | None = Query(None),
    competency_id: int | None = Query(None),
):
    payload, _, _ = _resolve_remote_catalog(
        micro_credential_id=micro_credential_id,
        domain_id=domain_id,
    )
    if competency_id is not None:
        domains = payload.get("data", {}).get("domains", [])
        for domain_entry in domains:
            for micro_entry in domain_entry.get("micro_credentials", []):
                if int(micro_entry.get("id", -1)) != int(micro_credential_id):
                    continue
                filtered = [item for item in micro_entry.get("competencies", []) if int(item.get("id", -1)) == int(competency_id)]
                micro_entry["competencies"] = filtered
    return payload


@app.post(
    "/backend/lesson/competencies",
    summary="Proxy lesson competencies from the remote backend using a request body",
    tags=["Remote Backend"],
)
def backend_lesson_competencies_post(req: BackendLessonCompetenciesRequest):
    payload, _, _ = _resolve_remote_catalog(
        micro_credential_id=req.micro_credential_id,
        domain_id=req.domain_id,
    )
    if req.competency_id is not None:
        domains = payload.get("data", {}).get("domains", [])
        for domain_entry in domains:
            for micro_entry in domain_entry.get("micro_credentials", []):
                if int(micro_entry.get("id", -1)) != int(req.micro_credential_id):
                    continue
                filtered = [item for item in micro_entry.get("competencies", []) if int(item.get("id", -1)) == int(req.competency_id)]
                micro_entry["competencies"] = filtered
    return payload


@app.get(
    "/backend/lesson/rubric/{competency_id}",
    summary="Proxy rubric rules for a remote competency",
    tags=["Remote Backend"],
)
def backend_lesson_rubric(competency_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = _resolve_effective_token(auth_token)
    try:
        payload = remote_backend_client.fetch_competency_rubric(competency_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/enrollment/check-access/{mc_id}",
    summary="Proxy enrollment access check from the remote backend",
    tags=["Remote Backend"],
)
def backend_enrollment_check_access(mc_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = _resolve_effective_token(auth_token)
    try:
        payload = remote_backend_client.check_access(mc_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.post(
    "/backend/enrollment/check-access",
    summary="Proxy enrollment access check from the remote backend using a request body",
    tags=["Remote Backend"],
)
def backend_enrollment_check_access_post(req: BackendEnrollmentAccessRequest):
    effective_token = _resolve_effective_token(req.auth_token)
    try:
        payload = remote_backend_client.check_access(req.micro_credential_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.post(
    "/backend/learning/sessions/start",
    summary="Start a remote learning session",
    tags=["Remote Backend"],
)
def backend_learning_session_start(req: BackendLearningSessionStartRequest):
    effective_token = _resolve_effective_token(req.auth_token)
    try:
        profile_payload = remote_backend_client.fetch_profile(token=effective_token)
        access_payload = remote_backend_client.check_access(req.micro_credential_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    profile_root = _extract_profile_root(profile_payload)
    learner_id = _extract_learner_id(profile_payload)
    access_data = access_payload.get("access", {})
    if not bool(access_data.get("can_access", access_data.get("can_start_session", False))):
        raise HTTPException(status_code=403, detail=access_data.get("message") or "Remote backend denied access")

    _, domain_entry, micro_entry = _resolve_remote_catalog(
        micro_credential_id=req.micro_credential_id,
        domain_id=req.domain_id,
    )
    ordered_competencies = _ordered_competencies(micro_entry)
    target_index = next((idx for idx, item in enumerate(ordered_competencies) if int(item.get("id", -1)) == int(req.competency_id)), None)
    if target_index is None:
        raise HTTPException(status_code=404, detail="Requested competency does not belong to the selected micro-credential.")

    blocking_competencies: list[dict[str, Any]] = []
    for item in ordered_competencies[:target_index]:
        progress = get_learner_competency_progress(learner_id, req.micro_credential_id, int(item.get("id", -1)))
        if not progress or not bool(progress.get("passed")):
            blocking_competencies.append(
                {
                    "id": int(item.get("id", -1)),
                    "code": item.get("code"),
                    "title": item.get("title"),
                }
            )
    if blocking_competencies:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Previous competencies must be passed before starting this learning session.",
                "blocking_competencies": blocking_competencies,
            },
        )

    try:
        payload = remote_backend_client.start_learning_session(
            competency_id=req.competency_id,
            token=effective_token,
        )
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    session_payload = payload.get("session") if isinstance(payload.get("session"), dict) else payload
    remote_session_id = int(session_payload.get("id") or session_payload.get("session_id") or 0)
    if not remote_session_id:
        raise HTTPException(status_code=502, detail="Remote backend did not return a learning session id.")
    target_competency = ordered_competencies[target_index]
    create_remote_learning_session_ref(
        remote_session_id,
        learner_id,
        req.micro_credential_id,
        req.competency_id,
        str(target_competency.get("title") or req.competency_id),
        domain_id=int(domain_entry.get("id", 0)) if domain_entry.get("id") is not None else req.domain_id,
    )
    return {
        "profile": profile_payload,
        "access": access_payload,
        "micro_credential": {
            "id": int(micro_entry.get("id", req.micro_credential_id)),
            "title": micro_entry.get("micro_credential") or micro_entry.get("name"),
            "domain_id": int(domain_entry.get("id", 0)),
            "domain_name": domain_entry.get("domain"),
        },
        "current_competency": target_competency,
        "previous_competencies_passed": True,
        "learning_session": payload,
        "learner_id": learner_id,
        "identity_verified": bool(profile_root.get("id") or profile_root.get("user_id")),
    }


@app.get(
    "/backend/learning/sessions/{session_id}",
    summary="Get a remote learning session",
    tags=["Remote Backend"],
)
def backend_learning_session_detail(session_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = _resolve_effective_token(auth_token)
    try:
        payload = remote_backend_client.fetch_learning_session(session_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    local_state = get_remote_learning_session_ref(session_id)
    return {"remote": payload, "local": local_state}


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
            interaction_type=req.resolved_interaction_type,
            ai_prompt=req.resolved_ai_prompt,
            ai_response=req.resolved_ai_response,
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
    effective_token = _resolve_effective_token(req.auth_token)
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
    passed = bool(payload.get("pass", payload.get("passed", req.rubric_score >= 75.0)))
    local_state = get_remote_learning_session_ref(session_id)
    if local_state:
        update_remote_learning_session_ref(
            session_id,
            status="passed" if passed else "failed",
            latest_score=req.rubric_score,
        )
        upsert_learner_competency_progress(
            local_state["learner_id"],
            local_state["micro_credential_id"],
            local_state["competency_id"],
            local_state["competency_name"],
            passed=passed,
            latest_session_id=str(session_id),
            latest_score=req.rubric_score,
        )
    return payload


@app.get(
    "/backend/gamification/progress/{session_id}",
    summary="Get remote gamification progress",
    tags=["Remote Backend"],
)
def backend_gamification_progress(session_id: int, auth_token: Optional[str] = Query(None)):
    effective_token = _resolve_effective_token(auth_token)
    try:
        payload = remote_backend_client.fetch_gamification_progress(session_id, token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get(
    "/backend/learner/micro-credential/progress",
    summary="Get learner progress for one micro-credential from the AI engine database",
    tags=["Remote Backend"],
)
def backend_learner_micro_credential_progress(
    micro_credential_id: int = Query(...),
    domain_id: int | None = Query(None),
    auth_token: Optional[str] = Query(None),
):
    effective_token = _resolve_effective_token(auth_token)
    try:
        profile_payload = remote_backend_client.fetch_profile(token=effective_token)
    except RemoteBackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    learner_id = _extract_learner_id(profile_payload)
    _, domain_entry, micro_entry = _resolve_remote_catalog(
        micro_credential_id=micro_credential_id,
        domain_id=domain_id,
    )
    ordered_competencies = _ordered_competencies(micro_entry)
    progress_map = {
        int(item["competency_id"]): item
        for item in list_learner_competency_progress(learner_id, micro_credential_id)
    }
    completed_count = 0
    next_competency_id: int | None = None
    previous_passed = True
    competency_items: list[dict[str, Any]] = []
    for index, competency in enumerate(ordered_competencies, start=1):
        competency_id = int(competency.get("id", -1))
        progress = progress_map.get(competency_id)
        passed = bool(progress and progress.get("passed"))
        if passed:
            status = "passed"
            availability = "completed"
            completed_count += 1
        else:
            status = "not_started" if progress is None else "failed"
            availability = "available" if previous_passed and next_competency_id is None else ("blocked" if not previous_passed else "available")
            if availability == "available" and next_competency_id is None:
                next_competency_id = competency_id
        competency_items.append(
            {
                "sequence": index,
                "competency_id": competency_id,
                "code": competency.get("code"),
                "title": competency.get("title"),
                "status": status,
                "availability": availability,
                "passed": passed,
                "latest_score": progress.get("latest_score") if progress else None,
                "latest_session_id": progress.get("latest_session_id") if progress else None,
                "updated_at": progress.get("updated_at") if progress else None,
            }
        )
        previous_passed = previous_passed and passed
    total = len(ordered_competencies)
    progress_percent = round((completed_count / total) * 100, 2) if total else 0.0
    return {
        "profile": profile_payload,
        "micro_credential": {
            "id": int(micro_entry.get("id", micro_credential_id)),
            "title": micro_entry.get("micro_credential") or micro_entry.get("name"),
            "domain_id": int(domain_entry.get("id", 0)),
            "domain_name": domain_entry.get("domain"),
            "total_competencies": total,
        },
        "progress": {
            "completed_competencies": completed_count,
            "total_competencies": total,
            "progress_percent": progress_percent,
            "next_available_competency_id": next_competency_id,
            "all_competencies_passed": total > 0 and completed_count == total,
        },
        "competencies": competency_items,
    }


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
    "/session/{session_id}/interact",
    summary="Unified learner interaction route",
    tags=["Sessions"],
)
async def session_interact(session_id: str, req: Optional[SessionInteractRequest] = None):
    payload = req or SessionInteractRequest()
    session = _get_session_for_interact(session_id, payload.auth_token)
    bootstrapped = await _ensure_interact_bootstrap(session)

    if not payload.has_text or bootstrapped:
        return _build_interact_response(session, None, counted_as_interaction=False)

    if session.phase == "pre_assessment":
        response = await handle_pre_assessment(session, payload.text)
        event_type = "pre_assessment_answered_via_interact"
    elif session.phase == "learning":
        response = await handle_learning(session, payload.text)
        event_type = "learning_turn_via_interact"
    elif session.phase == "competency_assessment":
        response = await handle_competency_assessment(session, payload.text)
        event_type = "competency_assessed_via_interact"
    elif session.phase == "final_assessment":
        response = await handle_final_assessment(session, payload.text)
        event_type = "final_assessment_submitted_via_interact"
    else:
        raise HTTPException(status_code=400, detail="Session is already completed. No further interactions are allowed.")

    counted_as_interaction = bool(response.pop("counted_as_interaction", True))

    _log_event(
        session.session_id,
        session.learner_id,
        f"/session/{session_id}/interact",
        event_type,
        {
            "phase": session.phase,
            "competency": session.current_competency if session.phase != "completed" else None,
            "counted_as_interaction": counted_as_interaction,
        },
    )
    return _build_interact_response(session, response, counted_as_interaction=counted_as_interaction)


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
    return _build_session_payload(session)
