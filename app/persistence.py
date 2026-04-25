from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import yaml

from app.db import get_connection
from app.state import LearnerSession

PROMPT_ENGINEERING_COMPETENCIES = [
    "Write structured prompts",
    "Optimize outputs iteratively",
    "Control tone & format",
    "Chain prompts logically",
    "Handle edge cases",
    "Reduce hallucinations",
    "Apply task decomposition",
    "Validate outputs",
    "Use system instructions",
    "Build reusable prompt templates",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_rubric_key(name: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in name).split())


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _criteria_to_binary(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(criteria, start=1):
        weight = float(item.get("weight", 0.0) or 0.0)
        normalized.append(
            {
                "criterion_id": item.get("criterion_id") or f"c{index}",
                "name": item.get("name") or f"Criterion {index}",
                "description": item.get("description") or "",
                "weight": weight,
                "binary": True,
            }
        )
    total_weight = sum(item["weight"] for item in normalized)
    if total_weight <= 0:
        equal = round(1 / max(1, len(normalized)), 4)
        for item in normalized:
            item["weight"] = equal
    else:
        for item in normalized:
            item["weight"] = round(item["weight"] / total_weight, 4)
    return normalized


def seed_locked_rubrics_from_yaml(config_path: str = "app/config/rubrics.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    prompt_template = raw.get("prompt_engineering") or {}
    prompt_criteria = _criteria_to_binary(prompt_template.get("criteria", []))
    if not prompt_criteria:
        raise RuntimeError("Prompt engineering rubric criteria missing from app/config/rubrics.yaml")

    now = utc_now_iso()
    records: list[tuple[str, str, int, str, str, int, str]] = []
    for competency in PROMPT_ENGINEERING_COMPETENCIES:
        rubric_key = normalize_rubric_key(competency)
        rubric_json = {
            "rubric_key": rubric_key,
            "display_name": competency,
            "criteria": prompt_criteria,
            "pass_threshold": 75.0,
            "binary_scoring": True,
            "locked": True,
        }
        payload = _json_dumps(rubric_json)
        source_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        records.append((rubric_key, competency, 1, payload, source_hash, 1, now))

    with get_connection() as conn:
        conn.executemany(
            '''
            INSERT OR IGNORE INTO locked_rubrics
            (rubric_key, display_name, version, rubric_json, source_hash, is_locked, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            records,
        )


def upsert_learner(learner_id: str, profile_payload: dict[str, Any], *, verified: bool) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO learners (learner_id, profile_json, identity_verified, verified_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id) DO UPDATE SET
                profile_json=excluded.profile_json,
                identity_verified=excluded.identity_verified,
                verified_at=excluded.verified_at,
                updated_at=excluded.updated_at
            ''',
            (
                learner_id,
                _json_dumps(profile_payload),
                1 if verified else 0,
                now if verified else None,
                now,
                now,
            ),
        )


def create_session_record(session: LearnerSession) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO learning_sessions
            (session_id, learner_id, topic, source, phase, state_json, created_at, updated_at, completed_at, final_assessment_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                session.session_id,
                session.learner_id,
                session.topic,
                session.source,
                session.phase,
                session.model_dump_json(),
                now,
                now,
                None,
                0,
            ),
        )


def save_session_record(session: LearnerSession) -> None:
    now = utc_now_iso()
    completed_at = now if session.phase == "completed" else None
    with get_connection() as conn:
        conn.execute(
            '''
            UPDATE learning_sessions
            SET learner_id=?, topic=?, source=?, phase=?, state_json=?, updated_at=?, completed_at=COALESCE(completed_at, ?), final_assessment_passed=?
            WHERE session_id=?
            ''',
            (
                session.learner_id,
                session.topic,
                session.source,
                session.phase,
                session.model_dump_json(),
                now,
                completed_at,
                1 if session.phase == "completed" else 0,
                session.session_id,
            ),
        )


def get_session_record(session_id: str) -> LearnerSession | None:
    with get_connection() as conn:
        row = conn.execute(
            'SELECT state_json FROM learning_sessions WHERE session_id = ?',
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return LearnerSession.model_validate_json(row["state_json"])


def delete_session_record(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute('DELETE FROM learning_sessions WHERE session_id = ?', (session_id,))


def get_locked_rubric(competency_name: str) -> dict[str, Any] | None:
    key = normalize_rubric_key(competency_name)
    with get_connection() as conn:
        row = conn.execute(
            'SELECT rubric_json FROM locked_rubrics WHERE rubric_key = ? AND is_locked = 1',
            (key,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["rubric_json"])


def get_rubric_version(competency_name: str) -> int | None:
    key = normalize_rubric_key(competency_name)
    with get_connection() as conn:
        row = conn.execute(
            'SELECT version FROM locked_rubrics WHERE rubric_key = ? AND is_locked = 1',
            (key,),
        ).fetchone()
    return int(row["version"]) if row else None


def get_rubric_source_hash(competency_name: str) -> str | None:
    key = normalize_rubric_key(competency_name)
    with get_connection() as conn:
        row = conn.execute(
            'SELECT source_hash FROM locked_rubrics WHERE rubric_key = ? AND is_locked = 1',
            (key,),
        ).fetchone()
    return str(row["source_hash"]) if row else None


def upsert_locked_rubric(
    competency_name: str,
    rubric_payload: dict[str, Any],
    *,
    version: int = 1,
    display_name: str | None = None,
) -> dict[str, Any]:
    rubric_key = normalize_rubric_key(competency_name)
    criteria = _criteria_to_binary(rubric_payload.get("criteria", []))
    if not criteria:
        raise ValueError("Locked rubric must include at least one criterion.")

    normalized_payload = {
        "rubric_key": rubric_key,
        "display_name": display_name or competency_name,
        "criteria": criteria,
        "pass_threshold": 75.0,
        "binary_scoring": True,
        "locked": True,
    }
    payload_json = _json_dumps(normalized_payload)
    source_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO locked_rubrics
            (rubric_key, display_name, version, rubric_json, source_hash, is_locked, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(rubric_key) DO UPDATE SET
                display_name=excluded.display_name,
                version=excluded.version,
                rubric_json=excluded.rubric_json,
                source_hash=excluded.source_hash,
                is_locked=1
            ''',
            (
                rubric_key,
                display_name or competency_name,
                version,
                payload_json,
                source_hash,
                utc_now_iso(),
            ),
        )
    return normalized_payload


def missing_locked_rubrics(competencies: list[str]) -> list[str]:
    missing: list[str] = []
    for competency in competencies:
        if not get_locked_rubric(competency):
            missing.append(competency)
    return missing


def append_event_log(session_id: str | None, learner_id: str | None, route: str, event_type: str, payload: dict[str, Any]) -> int:
    now = utc_now_iso()
    payload_json = _json_dumps(payload)
    with get_connection() as conn:
        previous = conn.execute('SELECT entry_hash FROM event_logs ORDER BY id DESC LIMIT 1').fetchone()
        previous_hash = previous['entry_hash'] if previous else ''
        entry_hash = hashlib.sha256(f'{previous_hash}|{now}|{route}|{event_type}|{payload_json}'.encode('utf-8')).hexdigest()
        cursor = conn.execute(
            '''
            INSERT INTO event_logs (session_id, learner_id, route, event_type, payload_json, created_at, previous_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (session_id, learner_id, route, event_type, payload_json, now, previous_hash, entry_hash),
        )
        return int(cursor.lastrowid)


def add_anomaly_flag(session_id: str | None, learner_id: str | None, flag_type: str, severity: str, details: dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            '''
            INSERT INTO anomaly_flags (session_id, learner_id, flag_type, severity, details_json, created_at, resolved)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ''',
            (session_id, learner_id, flag_type, severity, _json_dumps(details), utc_now_iso()),
        )
        return int(cursor.lastrowid)


def get_unresolved_anomalies(session_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            'SELECT id, flag_type, severity, details_json, created_at FROM anomaly_flags WHERE session_id = ? AND resolved = 0 ORDER BY id ASC',
            (session_id,),
        ).fetchall()
    return [
        {
            'id': int(row['id']),
            'flag_type': row['flag_type'],
            'severity': row['severity'],
            'details': json.loads(row['details_json']),
            'created_at': row['created_at'],
        }
        for row in rows
    ]


def record_competency_attempt(session_id: str, competency_name: str, attempt_number: int, status: str, *, score: float | None = None, rubric_key: str | None = None, evaluation: dict[str, Any] | None = None) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO competency_attempts (session_id, competency_name, attempt_number, status, started_at, updated_at, assessment_score, rubric_key, evaluation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, competency_name, attempt_number) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                assessment_score=excluded.assessment_score,
                rubric_key=excluded.rubric_key,
                evaluation_json=excluded.evaluation_json
            ''',
            (session_id, competency_name, attempt_number, status, now, now, score, rubric_key, _json_dumps(evaluation or {})),
        )


def count_competency_attempts(session_id: str, competency_name: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            'SELECT COUNT(*) AS count FROM competency_attempts WHERE session_id = ? AND competency_name = ?',
            (session_id, competency_name),
        ).fetchone()
    return int(row['count']) if row else 0


def record_formative_check(session_id: str, competency_name: str, attempt_number: int, slot_index: int, *, passed: bool, score: float, learner_response: str, feedback: str, difficulty_tier: str, delivery_format: str) -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO formative_checks (session_id, competency_name, attempt_number, slot_index, passed, score, learner_response, feedback, difficulty_tier, delivery_format, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (session_id, competency_name, attempt_number, slot_index, 1 if passed else 0, score, learner_response, feedback, difficulty_tier, delivery_format, utc_now_iso()),
        )


def record_final_assessment(session_id: str, attempt_number: int, prompt: str, learner_response: str, evaluation: dict[str, Any], overall_percent: float, passed: bool) -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO final_assessments (session_id, attempt_number, prompt, learner_response, evaluation_json, overall_percent, passed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (session_id, attempt_number, prompt, learner_response, _json_dumps(evaluation), overall_percent, 1 if passed else 0, utc_now_iso()),
        )


def create_badge(session_id: str, learner_id: str | None, competency_name: str, badge_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    awarded_at = utc_now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            '''
            INSERT INTO badges (session_id, learner_id, competency_name, badge_name, awarded_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (session_id, learner_id, competency_name, badge_name, awarded_at, _json_dumps(metadata)),
        )
        badge_id = int(cursor.lastrowid)
    return {
        'id': badge_id,
        'session_id': session_id,
        'learner_id': learner_id,
        'competency_name': competency_name,
        'badge_name': badge_name,
        'awarded_at': awarded_at,
        'metadata': metadata,
    }


def list_badges(session_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            'SELECT id, competency_name, badge_name, awarded_at, metadata_json FROM badges WHERE session_id = ? ORDER BY id ASC',
            (session_id,),
        ).fetchall()
    return [
        {
            'id': int(row['id']),
            'competency_name': row['competency_name'],
            'badge_name': row['badge_name'],
            'awarded_at': row['awarded_at'],
            'metadata': json.loads(row['metadata_json']),
        }
        for row in rows
    ]


def create_certificate_record(certificate_id: str, session_id: str, learner_id: str | None, html_file_path: str, pdf_file_path: str, verification_url: str, qr_code_url: str, metadata: dict[str, Any], issued_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT OR REPLACE INTO certificates (certificate_id, session_id, learner_id, html_file_path, pdf_file_path, verification_url, qr_code_url, metadata_json, issued_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (certificate_id, session_id, learner_id, html_file_path, pdf_file_path, verification_url, qr_code_url, _json_dumps(metadata), issued_at),
        )


def get_certificate_record(certificate_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            'SELECT * FROM certificates WHERE certificate_id = ?',
            (certificate_id,),
        ).fetchone()
    if not row:
        return None
    return {
        'certificate_id': row['certificate_id'],
        'session_id': row['session_id'],
        'learner_id': row['learner_id'],
        'html_file_path': row['html_file_path'],
        'pdf_file_path': row['pdf_file_path'],
        'verification_url': row['verification_url'],
        'qr_code_url': row['qr_code_url'],
        'metadata': json.loads(row['metadata_json']),
        'issued_at': row['issued_at'],
    }


def upsert_learner_competency_progress(
    learner_id: str,
    micro_credential_id: int,
    competency_id: int,
    competency_name: str,
    *,
    passed: bool,
    latest_session_id: str | None = None,
    latest_score: float | None = None,
) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO learner_competency_progress
            (learner_id, micro_credential_id, competency_id, competency_name, passed, latest_session_id, latest_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id, micro_credential_id, competency_id) DO UPDATE SET
                competency_name=excluded.competency_name,
                passed=excluded.passed,
                latest_session_id=excluded.latest_session_id,
                latest_score=excluded.latest_score,
                updated_at=excluded.updated_at
            ''',
            (
                learner_id,
                int(micro_credential_id),
                int(competency_id),
                competency_name,
                1 if passed else 0,
                latest_session_id,
                latest_score,
                now,
            ),
        )


def list_learner_competency_progress(learner_id: str, micro_credential_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            '''
            SELECT learner_id, micro_credential_id, competency_id, competency_name, passed, latest_session_id, latest_score, updated_at
            FROM learner_competency_progress
            WHERE learner_id = ? AND micro_credential_id = ?
            ORDER BY competency_id ASC
            ''',
            (learner_id, int(micro_credential_id)),
        ).fetchall()
    return [
        {
            "learner_id": row["learner_id"],
            "micro_credential_id": int(row["micro_credential_id"]),
            "competency_id": int(row["competency_id"]),
            "competency_name": row["competency_name"],
            "passed": bool(row["passed"]),
            "latest_session_id": row["latest_session_id"],
            "latest_score": row["latest_score"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def get_learner_competency_progress(
    learner_id: str,
    micro_credential_id: int,
    competency_id: int,
) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            '''
            SELECT learner_id, micro_credential_id, competency_id, competency_name, passed, latest_session_id, latest_score, updated_at
            FROM learner_competency_progress
            WHERE learner_id = ? AND micro_credential_id = ? AND competency_id = ?
            ''',
            (learner_id, int(micro_credential_id), int(competency_id)),
        ).fetchone()
    if not row:
        return None
    return {
        "learner_id": row["learner_id"],
        "micro_credential_id": int(row["micro_credential_id"]),
        "competency_id": int(row["competency_id"]),
        "competency_name": row["competency_name"],
        "passed": bool(row["passed"]),
        "latest_session_id": row["latest_session_id"],
        "latest_score": row["latest_score"],
        "updated_at": row["updated_at"],
    }


def create_remote_learning_session_ref(
    remote_session_id: int,
    learner_id: str,
    micro_credential_id: int,
    competency_id: int,
    competency_name: str,
    *,
    domain_id: int | None = None,
) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT OR REPLACE INTO remote_learning_session_refs
            (remote_session_id, learner_id, micro_credential_id, competency_id, competency_name, domain_id, status, latest_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM remote_learning_session_refs WHERE remote_session_id = ?), ?), ?)
            ''',
            (
                int(remote_session_id),
                learner_id,
                int(micro_credential_id),
                int(competency_id),
                competency_name,
                domain_id,
                "started",
                None,
                int(remote_session_id),
                now,
                now,
            ),
        )


def get_remote_learning_session_ref(remote_session_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            '''
            SELECT remote_session_id, learner_id, micro_credential_id, competency_id, competency_name, domain_id, status, latest_score, created_at, updated_at
            FROM remote_learning_session_refs
            WHERE remote_session_id = ?
            ''',
            (int(remote_session_id),),
        ).fetchone()
    if not row:
        return None
    return {
        "remote_session_id": int(row["remote_session_id"]),
        "learner_id": row["learner_id"],
        "micro_credential_id": int(row["micro_credential_id"]),
        "competency_id": int(row["competency_id"]),
        "competency_name": row["competency_name"],
        "domain_id": row["domain_id"],
        "status": row["status"],
        "latest_score": row["latest_score"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def update_remote_learning_session_ref(
    remote_session_id: int,
    *,
    status: str,
    latest_score: float | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            UPDATE remote_learning_session_refs
            SET status = ?, latest_score = ?, updated_at = ?
            WHERE remote_session_id = ?
            ''',
            (status, latest_score, utc_now_iso(), int(remote_session_id)),
        )
