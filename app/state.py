from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CompetencyResult(BaseModel):
    competency: str
    score: float
    passed: bool
    feedback: str = ""


InteractionType = Literal['intro', 'diagnostic', 'teach', 'formative', 'revision', 'competency_assessment', 'final_assessment']


STAGE_MAP = {
    'foundation': {'turns': (1, 2), 'bloom': 'Understand'},
    'guided_application': {'turns': (3, 4), 'bloom': 'Apply'},
    'mastery_gate': {'turns': (5, 6), 'bloom': 'Analyze/Evaluate'},
    'revision': {'turns': (7, 8), 'bloom': 'Apply/Evaluate'},
}


POINTS_PER_FORMATIVE_PASS = 10
STREAK_BONUS_POINTS = 5
STREAK_BONUS_THRESHOLD = 3


AIP_NAME_MAP: Dict[str, str] = {
    "AIP-01": "Introduction",
    "AIP-02": "Diagnostic Check",
    "AIP-03": "Core Teaching",
    "AIP-04": "Worked Example",
    "AIP-05": "FA1 Generation",
    "AIP-06": "FA1 Feedback",
    "AIP-07": "FA2 Generation",
    "AIP-08": "FA2 Feedback",
    "AIP-09": "FA3 Generation",
    "AIP-10": "FA3 Feedback",
    "AIP-11": "Summative Assessment",
    "AIP-12": "Summative Feedback",
    "AIP-13": "Competency Summary",
    "AIP-14": "MC Completion Reflection",
}
MAX_NAMED_AIPS_PER_COMPETENCY = 14


def _get_stage_info(turn: int) -> tuple[str, str]:
    for stage_name, info in STAGE_MAP.items():
        low, high = info['turns']
        if low <= turn <= high:
            return stage_name, info['bloom']
    return 'revision', 'Apply/Evaluate'


class LearnerSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    competencies: List[str]
    source: Literal['manual', 'remote'] = 'manual'
    domain_id: Optional[int] = None
    remote_micro_credential_id: Optional[int] = None
    remote_micro_credential_level: Optional[str] = None
    remote_source: Optional[str] = None
    remote_auth_token: Optional[str] = None
    remote_access_id: Optional[int] = None
    backend_warnings: List[str] = Field(default_factory=list)
    competency_details: Dict[str, dict] = Field(default_factory=dict)
    remote_learning_sessions: Dict[str, int] = Field(default_factory=dict)
    learner_profile: Dict[str, Any] = Field(default_factory=dict)
    identity_verified: bool = False
    identity_verified_at: Optional[str] = None
    learner_id: Optional[str] = None
    current_competency_index: int = 0
    user_level: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'
    weak_areas: List[str] = Field(default_factory=list)
    phase: Literal['pre_assessment', 'learning', 'competency_assessment', 'final_assessment', 'completed'] = 'pre_assessment'
    pre_assessment_turn: int = 0
    competency_interaction: int = 0
    learning_turn: int = 0
    max_learning_turns: int = 6
    max_revision_turns: int = 2
    max_competency_interactions: int = 12
    intro_delivered: bool = False
    pre_assessment_prompt: Optional[str] = None
    pre_assessment_completed: bool = False
    formative_check_results: List[bool] = Field(default_factory=list)
    formative_slots: List[Optional[bool]] = Field(default_factory=list)
    current_difficulty: Literal['support', 'standard', 'stretch'] = 'standard'
    consecutive_formative_passes: int = 0
    consecutive_formative_fails: int = 0
    awaiting_formative_response: bool = False
    current_formative_slot: int = -1
    current_formative_prompt: Optional[str] = None
    revision_required: bool = False
    revision_turns_used: int = 0
    formative_feedback_log: List[dict] = Field(default_factory=list)
    current_assessment_prompt: Optional[str] = None
    final_assessment_unlocked: bool = False
    current_assessment_attempts: int = 0
    final_assessment_prompt: Optional[str] = None
    final_assessment_attempts: int = 0
    personalization_state: Dict[str, str] = Field(default_factory=dict)
    delivery_history: List[str] = Field(default_factory=list)
    delivery_format_history: List[str] = Field(default_factory=list)
    concept_history: List[str] = Field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_interaction_type: InteractionType = 'intro'
    current_delivery_format: Optional[str] = None
    consecutive_easy_passes: int = 0
    messages: List[ChatMessage] = Field(default_factory=list)
    study_materials: Dict[str, str] = Field(default_factory=dict)
    learning_plans: Dict[str, str] = Field(default_factory=dict)
    rubric_cache: Dict[str, dict] = Field(default_factory=dict)
    competency_subparts: Dict[str, List[str]] = Field(default_factory=dict)
    current_subpart_index: int = 0
    completed_competencies: List[CompetencyResult] = Field(default_factory=list)
    competency_attempts: Dict[str, int] = Field(default_factory=dict)
    points_total: int = 0
    last_points_delta: int = 0
    streak_count: int = 0
    streak_bonus_awarded: bool = False
    earned_badges: List[dict] = Field(default_factory=list)
    completion_badge: Optional[dict] = None
    last_delivery_format: Optional[str] = None
    last_feedback_message: Optional[str] = None
    remote_sync_status: Dict[str, Any] = Field(
        default_factory=lambda: {
            "backend_session_id": None,
            "last_sync_at": None,
            "last_sync_outcome": "not_synced",
            "warning": None,
            "warning_list": [],
        }
    )
    remote_last_event_at: Optional[str] = None
    session_summary: Dict[str, Any] = Field(default_factory=dict)
    current_aip_code: Optional[str] = None
    current_aip_name: Optional[str] = None
    competency_aip_history: List[Dict[str, Any]] = Field(default_factory=list)
    all_aip_history: List[Dict[str, Any]] = Field(default_factory=list)
    competency_aip_codes_fired: List[str] = Field(default_factory=list)
    live_aip_call_history: List[Dict[str, Any]] = Field(default_factory=list)
    mc_completion_reflection_fired: bool = False
    latest_binary_outcome: Optional[str] = None
    developing_competency_active: bool = False
    developing_competency_reason: Optional[str] = None
    local_assessment_passed: Optional[bool] = None
    remote_assessment_synced: bool = False
    remote_assessment_passed: Optional[bool] = None
    current_assessment_sync_error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    @property
    def current_competency(self) -> str:
        return self.competencies[self.current_competency_index]

    @property
    def current_subpart(self) -> str:
        parts = self.competency_subparts.get(self.current_competency, [])
        if not parts:
            return ""
        safe_index = min(self.current_subpart_index, len(parts) - 1)
        return parts[safe_index]

    @property
    def current_remote_competency_id(self) -> Optional[int]:
        details = self.competency_details.get(self.current_competency, {})
        remote_id = details.get('id')
        return int(remote_id) if isinstance(remote_id, int) else remote_id

    @property
    def current_remote_learning_session_id(self) -> Optional[int]:
        remote_id = self.remote_learning_sessions.get(self.current_competency)
        return int(remote_id) if isinstance(remote_id, int) else remote_id

    @property
    def eqf_band(self) -> Optional[int]:
        raw = (self.remote_micro_credential_level or "").upper()
        if not raw:
            return None
        for token in raw.replace("-", " ").split():
            if token.isdigit():
                return int(token)
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None

    @property
    def academic_stage(self) -> str:
        raw = (self.remote_micro_credential_level or "").strip().lower()
        mapping = {
            6: "bachelor",
            7: "masters",
            8: "phd",
        }
        if self.eqf_band in mapping:
            return mapping[self.eqf_band]
        textual_mapping = {
            "foundation": "bachelor",
            "intermediate": "bachelor",
            "advance": "masters",
            "advanced": "masters",
            "doctoral": "phd",
        }
        return textual_mapping.get(raw, "professional")

    @property
    def academic_guidance(self) -> str:
        guidance = {
            "bachelor": (
                "Use bachelor-level teaching depth: strong conceptual grounding, clear mechanisms, "
                "scaffolded examples, and professional application without assuming research specialization."
            ),
            "masters": (
                "Use masters-level teaching depth: expect stronger prior knowledge, analyze tradeoffs, "
                "justify design choices, and connect concepts to professional decision-making."
            ),
            "phd": (
                "Use PhD-level teaching depth: emphasize advanced reasoning, edge cases, limitations, "
                "research-style critique, and expert synthesis across complex scenarios."
            ),
            "professional": (
                "Use professional micro-credential depth: practical, rigorous, and clearly structured for workplace application."
            ),
        }
        return guidance[self.academic_stage]

    @property
    def chat_stage(self) -> str:
        stage, _ = _get_stage_info(self.learning_turn)
        return stage

    @property
    def bloom_level(self) -> str:
        _, bloom = _get_stage_info(self.learning_turn)
        return bloom

    @property
    def is_doubt_phase(self) -> bool:
        return self.revision_required

    @property
    def is_last_competency(self) -> bool:
        return self.current_competency_index >= len(self.competencies) - 1

    @property
    def max_learning_window(self) -> int:
        return self.max_learning_turns + self.max_revision_turns

    @property
    def competency_attempt_number(self) -> int:
        return self.competency_attempts.get(self.current_competency, 1)

    @property
    def competency_progress_percent(self) -> float:
        if self.phase in {'competency_assessment', 'final_assessment', 'completed'}:
            return 100.0
        return min(100.0, round((self.competency_interaction / max(1, self.max_competency_interactions)) * 100, 2))

    @property
    def overall_progress_percent(self) -> float:
        completed = len(self.completed_competencies)
        total = max(1, len(self.competencies))
        if self.phase == 'completed':
            return 100.0
        return round(((completed + (self.competency_progress_percent / 100.0)) / total) * 100, 2)

    @property
    def required_next_action(self) -> str:
        if self.phase == 'pre_assessment':
            return 'answer_diagnostic' if self.pre_assessment_prompt else 'start_pre_assessment'
        if self.phase == 'learning':
            return 'answer_formative_check' if self.awaiting_formative_response else 'continue_learning'
        if self.phase == 'competency_assessment':
            return 'submit_competency_assessment'
        if self.phase == 'final_assessment':
            return 'submit_final_assessment'
        if self.phase == 'completed' and self.identity_verified:
            return 'generate_certificate'
        return 'session_complete'

    @property
    def formative_slot_number(self) -> Optional[int]:
        if self.current_formative_slot < 0:
            return None
        return self.current_formative_slot + 1

    @property
    def competency_aip_count(self) -> int:
        return len(self.competency_aip_codes_fired)

    @property
    def aip_budget_total(self) -> int:
        return MAX_NAMED_AIPS_PER_COMPETENCY

    def format_recent_history(self, n: int = 10) -> str:
        if not self.messages:
            return 'No conversation yet.'
        lines = []
        for msg in self.messages[-n:]:
            prefix = 'Learner' if msg.role == 'user' else 'Tutor'
            lines.append(f'{prefix}: {msg.content}')
        return '\n'.join(lines)

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_aip_event(
        self,
        *,
        aip_code: str,
        trigger: str,
        scope: Literal["cc", "mc"] = "cc",
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if scope == "cc" and aip_code in self.competency_aip_codes_fired:
            return
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "aip_code": aip_code,
            "aip_name": AIP_NAME_MAP.get(aip_code, aip_code),
            "trigger": trigger,
            "scope": scope,
            "competency": self.current_competency,
            "phase": self.phase,
            "interaction_number": self.competency_interaction,
            "outcome": outcome,
            "created_at": now,
            "metadata": metadata or {},
        }
        self.current_aip_code = entry["aip_code"]
        self.current_aip_name = entry["aip_name"]
        self.all_aip_history.append(entry)
        if scope == "cc":
            self.competency_aip_history.append(entry)
            self.competency_aip_codes_fired.append(aip_code)
        else:
            self.mc_completion_reflection_fired = True
        self.updated_at = now

    def record_live_aip_call(
        self,
        *,
        aip_code: str,
        purpose: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.live_aip_call_history.append(
            {
                "aip_code": aip_code,
                "aip_name": AIP_NAME_MAP.get(aip_code, aip_code),
                "purpose": purpose,
                "competency": self.current_competency,
                "phase": self.phase,
                "interaction_number": self.competency_interaction,
                "created_at": now,
                "metadata": metadata or {},
            }
        )
        self.updated_at = now

    def record_interaction_event(
        self,
        *,
        interaction_type: InteractionType,
        concept: str,
        delivery_format: Optional[str],
        interaction_number: int,
        phase: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.current_interaction_type = interaction_type
        self.current_delivery_format = delivery_format
        if delivery_format:
            self.last_delivery_format = delivery_format
            self.delivery_history.append(delivery_format)
            self.delivery_format_history.append(delivery_format)
        if concept:
            self.concept_history.append(concept)
        self.interaction_history.append(
            {
                "interaction_number": interaction_number,
                "phase": phase,
                "interaction_type": interaction_type,
                "concept": concept,
                "delivery_format": delivery_format,
                "created_at": now,
            }
        )
        self.updated_at = now

    def set_remote_sync(
        self,
        *,
        outcome: str,
        backend_session_id: Optional[int] = None,
        warning: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        warnings = list(self.remote_sync_status.get("warning_list") or [])
        if warning and warning not in warnings:
            warnings.append(warning)
        self.remote_sync_status = {
            "backend_session_id": backend_session_id or self.current_remote_learning_session_id,
            "last_sync_at": now,
            "last_sync_outcome": outcome,
            "warning": warning,
            "warning_list": warnings,
        }
        self.remote_last_event_at = now
        self.updated_at = now

    def award_points_for_formative(self, passed: bool) -> int:
        self.streak_bonus_awarded = False
        if passed:
            self.streak_count += 1
            delta = POINTS_PER_FORMATIVE_PASS
            if self.streak_count >= STREAK_BONUS_THRESHOLD:
                delta += STREAK_BONUS_POINTS
                self.streak_bonus_awarded = True
            self.points_total += delta
            self.last_points_delta = delta
            return delta

        self.streak_count = 0
        self.last_points_delta = 0
        return 0

    def build_session_summary(self) -> Dict[str, Any]:
        end_time = self.completed_at or datetime.now(timezone.utc).isoformat()
        started = datetime.fromisoformat(self.created_at)
        finished = datetime.fromisoformat(end_time)
        duration_seconds = max(0, int((finished - started).total_seconds()))
        self.session_summary = {
            'total_points': self.points_total,
            'completed_competencies': [item.competency for item in self.completed_competencies],
            'completed_competency_count': len(self.completed_competencies),
            'badges_earned': [
                {
                    'badge_name': badge.get('badge_name'),
                    'awarded_at': badge.get('awarded_at'),
                    'competency_name': badge.get('competency_name'),
                }
                for badge in self.earned_badges
            ],
            'completion_badge': self.completion_badge,
            'time_taken_seconds': duration_seconds,
        }
        return self.session_summary

    def advance_to_next_competency(self):
        self.current_competency_index += 1
        self.reset_competency_cycle()
        self.phase = 'pre_assessment'

    def reset_competency_cycle(self):
        self.pre_assessment_turn = 0
        self.competency_interaction = 0
        self.learning_turn = 0
        self.intro_delivered = False
        self.pre_assessment_prompt = None
        self.pre_assessment_completed = False
        self.formative_check_results = []
        self.formative_slots = []
        self.current_difficulty = 'standard'
        self.consecutive_formative_passes = 0
        self.consecutive_formative_fails = 0
        self.awaiting_formative_response = False
        self.current_formative_slot = -1
        self.current_formative_prompt = None
        self.revision_required = False
        self.revision_turns_used = 0
        self.formative_feedback_log = []
        self.current_assessment_prompt = None
        self.final_assessment_unlocked = False
        self.current_assessment_attempts = 0
        self.final_assessment_prompt = None
        self.final_assessment_attempts = 0
        self.current_subpart_index = 0
        self.delivery_history = []
        self.delivery_format_history = []
        self.personalization_state = {}
        self.last_points_delta = 0
        self.streak_bonus_awarded = False
        self.last_delivery_format = None
        self.last_feedback_message = None
        self.current_interaction_type = 'intro'
        self.current_delivery_format = None
        self.consecutive_easy_passes = 0
        self.current_aip_code = None
        self.current_aip_name = None
        self.competency_aip_history = []
        self.competency_aip_codes_fired = []
        self.live_aip_call_history = []
        self.latest_binary_outcome = None
        self.developing_competency_active = False
        self.developing_competency_reason = None
        self.local_assessment_passed = None
        self.remote_assessment_synced = False
        self.remote_assessment_passed = None
        self.current_assessment_sync_error = None
        self.updated_at = datetime.now(timezone.utc).isoformat()
