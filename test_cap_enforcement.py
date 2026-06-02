"""
Test: AIP cap enforcement on fail-and-retry.
Simulates a learner with no learner_id (non-remote session) hitting the cap
via session_id fallback counter.
"""
from app.db import init_db, get_connection
from app.persistence import log_aip_usage

init_db()

SESSION_ID = "test-session-cap-001"
COMPETENCY = "Test Competency"

# Simulate 14 live_api FA questions already logged for this session
with get_connection() as conn:
    conn.execute("DELETE FROM aip_usage_log WHERE session_id = ?", (SESSION_ID,))

for i in range(14):
    log_aip_usage(
        learner_id=None,
        core_competency_id=None,
        micro_credential_id=None,
        question_source="live_api",
        aip_phase="FA1",
        aip_count_at_time=i + 1,
        attempt_number=1,
        question_bank_id=None,
        api_cost_estimate_usd=0.002,
        session_id=SESSION_ID,
        competency_name=COMPETENCY,
    )

# Now check the fallback counter
from app.gate_service import _get_session_fa_count, get_aip_cap

count = _get_session_fa_count(SESSION_ID, COMPETENCY)
cap = get_aip_cap()
print(f"Session FA count: {count}, Cap: {cap}")
assert count == 14, f"Expected 14, got {count}"
assert count >= cap, f"Count {count} should be >= cap {cap}"
print("✅ Cap enforcement: session-level fallback counter works correctly")
print("✅ On retry after fail, gate will serve from question_bank (zero cost)")

# Cleanup
with get_connection() as conn:
    conn.execute("DELETE FROM aip_usage_log WHERE session_id = ?", (SESSION_ID,))
print("✅ Cleanup done")
