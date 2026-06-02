"""Test if everything is working."""
from app.db import init_db, get_connection

init_db()

print("=== Testing DB Structure ===")
with get_connection() as conn:
    # Check new tables exist
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    print("Tables:", sorted(table_names))
    
    # Check platform_settings seeded
    cap = conn.execute("SELECT value FROM platform_settings WHERE key='aip_cap_per_cc'").fetchone()
    print("AIP cap:", cap[0] if cap else "NOT FOUND")
    
    # Check trigger exists
    triggers = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    trigger_names = [t[0] for t in triggers]
    print("Triggers:", trigger_names)

print("\n=== Testing Imports ===")
from app.state import LearnerSession, AIP_NAME_MAP
print("AIP_NAME_MAP keys:", list(AIP_NAME_MAP.keys())[:5], "...")
s = LearnerSession(topic="Test", competencies=["Test CC"])
print("Session new fields:", s.current_fa_question, s.injected_context)

from app.gate_service import get_aip_cap, build_injected_context
print("AIP cap from DB:", get_aip_cap())
print("Injected context sample:", build_injected_context("Test CC"))

from app.persistence import get_platform_setting, cache_question, get_cached_questions
print("Platform setting:", get_platform_setting("aip_cap_per_cc"))

print("\n=== Testing Orchestrator Imports ===")
from app.orchestrator import handle_competency_assessment
print("Orchestrator imports OK")

print("\n=== Testing Main.py Admin Endpoints ===")
from app.main import app
print("FastAPI app routes:", len(app.routes))
admin_routes = [r.path for r in app.routes if '/admin' in str(r.path)]
print("Admin routes:", admin_routes[:5])

print("\n✅ All systems operational!")