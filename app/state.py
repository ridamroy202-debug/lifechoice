from datetime import datetime
from typing import Dict, List, Literal, Optional
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class CompetencyResult(BaseModel):
    competency: str
    score: float
    passed: bool
    feedback: str = ""


STAGE_MAP = {
    'foundation': {'turns': (1, 2), 'bloom': 'Understand'},
    'guided_application': {'turns': (3, 4), 'bloom': 'Apply'},
    'mastery_gate': {'turns': (5, 6), 'bloom': 'Analyze/Evaluate'},
    'revision': {'turns': (7, 8), 'bloom': 'Apply/Evaluate'},
}


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
    rubric_cache: Dict[str, dict] = Field(default_factory=dict)
    learner_id: Optional[str] = None
    current_competency_index: int = 0
    user_level: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'
    weak_areas: List[str] = Field(default_factory=list)
    phase: Literal['pre_assessment', 'learning', 'competency_assessment', 'completed'] = 'pre_assessment'
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
    personalization_state: Dict[str, str] = Field(default_factory=dict)
    delivery_history: List[str] = Field(default_factory=list)
    messages: List[ChatMessage] = Field(default_factory=list)
    study_materials: Dict[str, str] = Field(default_factory=dict)
    learning_plans: Dict[str, str] = Field(default_factory=dict)
    competency_subparts: Dict[str, List[str]] = Field(default_factory=dict)
    current_subpart_index: int = 0
    completed_competencies: List[CompetencyResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

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
        self.current_subpart_index = 0
        self.delivery_history = []
        self.personalization_state = {}
