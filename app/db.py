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
            CREATE TABLE IF NOT EXISTS question_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                remote_question_id INTEGER,
                core_competency_id INTEGER,
                competency_title TEXT NOT NULL,
                micro_credential TEXT,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                difficulty_level TEXT,
                aip_phase TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                times_served INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS aip_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT,
                core_competency_id INTEGER,
                micro_credential_id INTEGER,
                question_source TEXT NOT NULL,
                question_bank_id INTEGER,
                aip_phase TEXT NOT NULL,
                aip_count_at_time INTEGER NOT NULL DEFAULT 0,
                attempt_number INTEGER NOT NULL DEFAULT 1,
                triggered_at TEXT NOT NULL,
                api_cost_estimate_usd REAL NOT NULL DEFAULT 0.00
            );

            CREATE TABLE IF NOT EXISTS platform_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS integrity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT,
                micro_credential_id INTEGER,
                core_competency_id INTEGER,
                event_type TEXT NOT NULL,
                event_detail TEXT NOT NULL DEFAULT '{}',
                session_id TEXT,
                triggered_at TEXT NOT NULL,
                reviewed_by_admin INTEGER NOT NULL DEFAULT 0,
                admin_action_taken TEXT
            );

            CREATE TABLE IF NOT EXISTS reference_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                core_competency_id INTEGER NOT NULL,
                competency_title TEXT NOT NULL,
                response_text TEXT NOT NULL,
                embedding_json TEXT,
                generated_at TEXT NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS trg_aip_count_exceeded
            AFTER UPDATE OF aip_count_used ON learner_competency_progress
            WHEN NEW.aip_count_used > 14
            BEGIN
                INSERT INTO aip_usage_log (
                    learner_id, core_competency_id, micro_credential_id,
                    question_source, aip_phase, aip_count_at_time,
                    attempt_number, triggered_at, api_cost_estimate_usd
                ) VALUES (
                    NEW.learner_id, NEW.competency_id, NEW.micro_credential_id,
                    'alert_aip_cap_exceeded', 'ALERT', NEW.aip_count_used,
                    NEW.attempts_count, datetime('now'), 0.00
                );
            END;

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

            CREATE TABLE IF NOT EXISTS learner_competency_progress (
                learner_id TEXT NOT NULL,
                micro_credential_id INTEGER NOT NULL,
                competency_id INTEGER NOT NULL,
                competency_name TEXT NOT NULL,
                passed INTEGER NOT NULL DEFAULT 0,
                latest_session_id TEXT,
                latest_score REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (learner_id, micro_credential_id, competency_id)
            );

            CREATE TABLE IF NOT EXISTS remote_learning_session_refs (
                remote_session_id INTEGER PRIMARY KEY,
                learner_id TEXT NOT NULL,
                micro_credential_id INTEGER NOT NULL,
                competency_id INTEGER NOT NULL,
                competency_name TEXT NOT NULL,
                domain_id INTEGER,
                status TEXT NOT NULL DEFAULT 'started',
                latest_score REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS remote_session_mappings (
                remote_session_id INTEGER PRIMARY KEY,
                local_session_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (local_session_id) REFERENCES learning_sessions (session_id)
            );
            '''
        )

        # Safe ALTER TABLE migrations — SQLite ignores duplicate column errors via try/except
        _add_column_if_missing(conn, "learner_competency_progress", "aip_count_used", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "learner_competency_progress", "aip11_attempts", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "learner_competency_progress", "attempts_count", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "learner_competency_progress", "status", "TEXT NOT NULL DEFAULT 'in_progress'")
        _add_column_if_missing(conn, "learner_competency_progress", "enrolled_at", "TEXT")
        _add_column_if_missing(conn, "learner_competency_progress", "last_attempt_at", "TEXT")
        _add_column_if_missing(conn, "learner_competency_progress", "competent_achieved_at", "TEXT")
        # aip_usage_log: add session_id + competency_name for session-level fallback counting
        _add_column_if_missing(conn, "aip_usage_log", "session_id", "TEXT")
        _add_column_if_missing(conn, "aip_usage_log", "competency_name", "TEXT")

        # Seed platform_settings defaults
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO platform_settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("aip_cap_per_cc", "14", now),
        )


def _add_column_if_missing(conn, table: str, column: str, definition: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except Exception:
        pass  # Column already exists
