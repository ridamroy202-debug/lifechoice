from typing import Dict, List, Optional
from app.state import LearnerSession

_sessions: Dict[str, LearnerSession] = {}


def create_session(topic: str, competencies: List[str]) -> LearnerSession:
    session = LearnerSession(topic=topic, competencies=competencies)
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[LearnerSession]:
    return _sessions.get(session_id)


def save_session(session: LearnerSession):
    _sessions[session.session_id] = session


def delete_session(session_id: str):
    _sessions.pop(session_id, None)