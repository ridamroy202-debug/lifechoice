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


class LearnerSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    competencies: List[str]                  # ALL competencies for this micro-credential
    current_competency_index: int = 0        # which competency we're on right now
    user_level: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'
    weak_areas: List[str] = []
    phase: Literal[
        'pre_assessment',
        'learning',
        'competency_assessment',
        'final_assessment',
        'completed'
    ] = 'pre_assessment'
    pre_assessment_turn: int = 0             # counts pre-assessment Q&A turns (max 4)
    learning_turn: int = 0                   # counts turns within current competency (max 9)
    max_learning_turns: int = 9
    messages: List[ChatMessage] = []         # full conversation history
    study_materials: Dict[str, str] = {}     # { competency_name: material_text }
    learning_plans: Dict[str, str] = {}      # { competency_name: plan_text }
    completed_competencies: List[CompetencyResult] = []
    final_assessment_result: Optional[dict] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def current_competency(self) -> str:
        """Return the competency currently being learned/assessed."""
        return self.competencies[self.current_competency_index]

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
        self.phase = 'learning'