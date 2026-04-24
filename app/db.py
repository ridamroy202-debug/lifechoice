from __future__ import annotations

import sqlite3
from pathlib import Path

from app.settings import settings

_DB_PATH = Path(settings.ai_engine_db_path)


def db_path() -> Path:
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS learners (
                learner_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL DEFAULT '{}',
                identity_verified INTEGER NOT NULL DEFAULT 0,
                verified_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                learner_id TEXT,
                topic TEXT NOT NULL,
                source TEXT NOT NULL,
                phase TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                final_assessment_passed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (learner_id) REFERENCES learners (learner_id)
            );

            CREATE TABLE IF NOT EXISTS locked_rubrics (
                rubric_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                rubric_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                is_locked INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS competency_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                competency_name TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                assessment_score REAL,
                rubric_key TEXT,
                evaluation_json TEXT,
                UNIQUE(session_id, competency_name, attempt_number),
                FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
            );

            CREATE TABLE IF NOT EXISTS formative_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                competency_name TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                slot_index INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL,
                learner_response TEXT NOT NULL,
                feedback TEXT NOT NULL,
                difficulty_tier TEXT NOT NULL,
                delivery_format TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
            );

            CREATE TABLE IF NOT EXISTS final_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                learner_response TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                overall_percent REAL NOT NULL,
                passed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
            );

            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                learner_id TEXT,
                route TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                previous_hash TEXT,
                entry_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS anomaly_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                learner_id TEXT,
                flag_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS badges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                learner_id TEXT,
                competency_name TEXT NOT NULL,
                badge_name TEXT NOT NULL,
                awarded_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
            );

            CREATE TABLE IF NOT EXISTS certificates (
                certificate_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                learner_id TEXT,
                html_file_path TEXT NOT NULL,
                pdf_file_path TEXT NOT NULL,
                verification_url TEXT NOT NULL,
                qr_code_url TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
            );
            '''
        )
