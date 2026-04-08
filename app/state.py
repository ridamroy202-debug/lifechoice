from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class CompetencyResult(BaseModel):
    competency: str
    score: float
    passed: bool
    feedback: str = ""


# ── Chat Stage Mapping ───────────────────────────────────────────────────────
# Maps chat turn numbers (1-22) to learning stages and Bloom's levels
STAGE_MAP = {
    'introduction':  {'turns': (1, 2),   'bloom': 'Remember'},
    'core_concepts': {'turns': (3, 6),   'bloom': 'Understand'},
    'examples':      {'turns': (7, 10),  'bloom': 'Apply/Analyze'},
    'practice':      {'turns': (11, 14), 'bloom': 'Apply/Evaluate'},
    'doubt_solving': {'turns': (15, 22), 'bloom': 'Evaluate/Create'},
}


def _get_stage_info(turn: int) -> tuple[str, str]:
    """Return (stage_name, bloom_level) for a given learning turn number."""
    for stage_name, info in STAGE_MAP.items():
        low, high = info['turns']
        if low <= turn <= high:
            return stage_name, info['bloom']
    # Fallback: if beyond 22, stay in doubt_solving
    return 'doubt_solving', 'Evaluate/Create'


class LearnerSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    competencies: List[str]                  # ALL competencies for this micro-credential
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
    current_competency_index: int = 0        # which competency we're on right now
    user_level: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'
    weak_areas: List[str] = Field(default_factory=list)
    phase: Literal[
        'pre_assessment',
        'learning',
        'competency_assessment',
        'final_assessment',
        'completed'
    ] = 'pre_assessment'
    pre_assessment_turn: int = 0             # counts pre-assessment Q&A turns (max 4)
    learning_turn: int = 0                   # counts turns within current competency (max 22)
    max_learning_turns: int = 22             # 22 chats per competency
    teaching_turns: int = 16                 # chats 1-16 are structured teaching
    doubt_turns: int = 6                     # chats 17-22 are doubt solving
    messages: List[ChatMessage] = Field(default_factory=list)         # full conversation history
    study_materials: Dict[str, str] = Field(default_factory=dict)     # { competency_name: material_text }
    learning_plans: Dict[str, str] = Field(default_factory=dict)      # { competency_name: plan_text }
    competency_subparts: Dict[str, List[str]] = Field(default_factory=dict)
    current_subpart_index: int = 0
    completed_competencies: List[CompetencyResult] = Field(default_factory=list)
    final_assessment_result: Optional[dict] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def current_competency(self) -> str:
        """Return the competency currently being learned/assessed."""
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
        """Return the current learning stage name based on learning_turn."""
        stage, _ = _get_stage_info(self.learning_turn)
        return stage

    @property
    def bloom_level(self) -> str:
        """Return the Bloom's taxonomy level for the current learning stage."""
        _, bloom = _get_stage_info(self.learning_turn)
        return bloom

    @property
    def is_doubt_phase(self) -> bool:
        """True if the learner is in the doubt-solving phase (chats 15-22)."""
        return self.learning_turn > self.teaching_turns

    @property
    def is_last_competency(self) -> bool:
        return self.current_competency_index >= len(self.competencies) - 1

    def format_recent_history(self, n: int = 10) -> str:
        """Last n messages formatted as readable string for crew inputs."""
        if not self.messages:
            return "No conversation yet."
        lines = []
        for msg in self.messages[-n:]:
            prefix = 'Learner' if msg.role == 'user' else 'Tutor'
            lines.append(f'{prefix}: {msg.content}')
        return '\n'.join(lines)

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def advance_to_next_competency(self):
        """Move to next competency and reset learning turn counter."""
        self.current_competency_index += 1
        self.learning_turn = 0
        self.current_subpart_index = 0
        self.phase = 'learning'
