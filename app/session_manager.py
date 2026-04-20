from typing import List, Optional

from app.persistence import create_session_record, delete_session_record, get_session_record, save_session_record
from app.state import LearnerSession


def create_session(topic: str, competencies: List[str], **kwargs) -> LearnerSession:
    session = LearnerSession(topic=topic, competencies=competencies, **kwargs)
    if session.current_competency not in session.competency_attempts:
        session.competency_attempts[session.current_competency] = 1
    create_session_record(session)
    return session


def get_session(session_id: str) -> Optional[LearnerSession]:
    return get_session_record(session_id)


def save_session(session: LearnerSession):
    save_session_record(session)


def delete_session(session_id: str):
    delete_session_record(session_id)
