from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from scripts.live_single_competency_test import (  # noqa: E402
    build_assessment_answer,
    build_formative_answer,
    build_learning_answer,
    build_preassessment_answer,
    generate_targeted_answer,
)


def build_final_assessment_answer(topic: str, prompt_text: str) -> str:
    return generate_targeted_answer(
        topic,
        (
            f"{prompt_text}\n\n"
            "Answer as the learner with these headings: Goal, Integrated Plan, Why This Works, "
            "Quality Check, Risk, Final Outcome. Use multiple competencies together in a concrete scenario."
        ),
        label="final-assessment",
    )


def run_learning_cycle(client: TestClient, session_id: str, *, sleep_seconds: float) -> dict:
    pre_start = client.post("/pre-assessment/start", json={"session_id": session_id})
    pre_start.raise_for_status()

    status = client.get(f"/session/{session_id}")
    status.raise_for_status()
    state = status.json()
    competency = state["competency"]

    pre_answer = client.post(
        "/pre-assessment/chat",
        json={
            "session_id": session_id,
            "answer": build_preassessment_answer(competency),
        },
    )
    pre_answer.raise_for_status()
    payload = pre_answer.json()
    print(f"competency={competency} phase_after_preassessment={payload.get('phase')}")

    turn_guard = 0
    while payload.get("phase") == "learning" and not payload.get("ready_for_assessment"):
        turn_guard += 1
        if turn_guard > 14:
            raise RuntimeError(f"Learning loop did not unlock assessment for {competency} within 14 learner turns.")
        time.sleep(sleep_seconds)
        status = client.get(f"/session/{session_id}")
        status.raise_for_status()
        state = status.json()
        competency = state["competency"]
        current_subpart = str(state.get("current_subpart") or competency)
        formative_prompt = str(state.get("current_formative_prompt") or "")
        interaction = int(state.get("interaction_number") or turn_guard)
        if state.get("awaiting_formative_response") and formative_prompt:
            answer = build_formative_answer(competency, formative_prompt, interaction)
        else:
            answer = build_learning_answer(competency, current_subpart, interaction)
        learn = client.post("/learn/chat", json={"session_id": session_id, "message": answer})
        learn.raise_for_status()
        payload = learn.json()
        print(
            f"competency={competency} learn_turn={payload.get('interaction_number')} "
            f"phase={payload.get('phase')} awaiting_formative={payload.get('awaiting_formative_response')} "
            f"ready_for_assessment={payload.get('ready_for_assessment')}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full live micro-credential cycle against the local AI engine.")
    parser.add_argument("--token", required=True, help="Backend access token")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--micro-credential-id", type=int, default=1)
    parser.add_argument("--learner-id", default="4")
    parser.add_argument("--sleep-seconds", type=float, default=2.2)
    parser.add_argument("--max-assessment-attempts", type=int, default=3)
    args = parser.parse_args()

    client = TestClient(app)

    start_response = client.post(
        "/session/start",
        json={
            "learner_id": args.learner_id,
            "domain_id": args.domain_id,
            "micro_credential_id": args.micro_credential_id,
            "auth_token": args.token,
        },
    )
    start_response.raise_for_status()
    start_payload = start_response.json()
    session_id = start_payload["session_id"]
    total_competencies = int(start_payload.get("total_competencies") or len(start_payload.get("competencies") or []))
    print(f"session_id={session_id}")
    print(f"total_competencies={total_competencies}")

    competency_counter = 0
    while True:
        status = client.get(f"/session/{session_id}")
        status.raise_for_status()
        session_state = status.json()
        phase = session_state["phase"]

        if phase == "completed":
            break
        if phase == "final_assessment":
            prompt = str(session_state.get("final_assessment_prompt") or "")
            answer = build_final_assessment_answer(session_state["topic"], prompt)
            final = client.post("/assessment/final", json={"session_id": session_id, "answer": answer})
            final.raise_for_status()
            final_payload = final.json()
            print("final_assessment_result=", json.dumps(final_payload, ensure_ascii=True))
            continue

        competency_counter += 1
        if competency_counter > total_competencies + 3:
            raise RuntimeError("Unexpected competency-loop length; aborting full run.")

        run_learning_cycle(client, session_id, sleep_seconds=args.sleep_seconds)

        attempt = 0
        while True:
            attempt += 1
            if attempt > args.max_assessment_attempts:
                raise RuntimeError("Competency assessment did not pass within the configured retry budget.")
            status = client.get(f"/session/{session_id}")
            status.raise_for_status()
            state = status.json()
            competency = state["competency"]
            assessment_prompt = str(state.get("current_assessment_prompt") or "")
            assessment = client.post(
                "/assessment/competency",
                json={
                    "session_id": session_id,
                    "answer": build_assessment_answer(competency, assessment_prompt),
                },
            )
            assessment.raise_for_status()
            result = assessment.json()
            print("assessment_result=", json.dumps(result, ensure_ascii=True))
            if result.get("passed"):
                break
            run_learning_cycle(client, session_id, sleep_seconds=args.sleep_seconds)

    status = client.get(f"/session/{session_id}")
    status.raise_for_status()
    final_status = status.json()
    print("final_status=", json.dumps(final_status, ensure_ascii=True))

    certificate = client.post(
        "/certificate/generate",
        json={"session_id": session_id, "auth_token": args.token},
    )
    certificate.raise_for_status()
    print("certificate=", json.dumps(certificate.json(), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
