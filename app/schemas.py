from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional

class UserLevel(BaseModel):
    level: Literal['beginner', 'intermediate','advance']
    weak_areas : List[str] = Field(default_factory=list)

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str

class PreAssessmentRequest(BaseModel):
    competency : str
    message: List[ChatMessage] = Field(default_factory=list)

class PreeAssesmentResponse(BaseModel):
    level: UserLevel
    done: bool = False
    next_question: Optional[str] = None

class LearningChatRequest(BaseModel):
    competency : str
    user_level : UserLevel
    messages : List[ChatMessage]
    interaction_count : int = 0

class LearningChatResponse(BaseModel):
    message: str
    ready_for_assessment: bool = False
    summary: Optional[str] = None

class AssessmentRequest(BaseModel):
    competency: str
    rubic_json: dict
    scenario: str
    user_response: str

class AssessmentResult(BaseModel):
    criteria_score: List[Dict[str,any]]
    overall_percent: float
    pass_: bool = Field(alias='pass')

