"""Quick test script for new features."""
from app.db import init_db, get_connection

init_db()

print("=== Testing DB Tables ===")
with get_connection() as conn:
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    print("Tables:", sorted(table_names))

    # Check platform_settings
    cap = conn.execute("SELECT value FROM platform_settings WHERE key='aip_cap_per_cc'").fetchone()
    print("AIP cap:", cap[0] if cap else "NOT FOUND")

    # Check new columns in learner_competency_progress
    cols = conn.execute("PRAGMA table_info(learner_competency_progress)").fetchall()
    col_names = [c[1] for c in cols]
    print("learner_competency_progress columns:", col_names)

print("\n=== Testing Imports ===")
from app.state import LearnerSession, AIP_NAME_MAP
print("AIP_NAME_MAP:", AIP_NAME_MAP)
s = LearnerSession(topic="Test", competencies=["Test CC"])
print("New session fields:", s.current_fa_question, s.injected_context)

from app.gate_service import get_aip_cap, build_injected_context
print("AIP cap from DB:", get_aip_cap())
print("Injected context sample:", build_injected_context("Test CC"))

print("\n=== All tests passed! ===")