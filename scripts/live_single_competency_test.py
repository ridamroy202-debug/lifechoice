from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from anthropic import Anthropic


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.persistence import get_locked_rubric  # noqa: E402
from app.settings import settings  # noqa: E402


STOPWORDS = {
    "about",
    "after",
    "also",
    "apply",
    "because",
    "build",
    "clear",
    "concept",
    "current",
    "example",
    "explain",
    "first",
    "from",
    "guide",
    "help",
    "idea",
    "into",
    "just",
    "learner",
    "make",
    "model",
    "needs",
    "prompt",
    "scenario",
    "simple",
    "steps",
    "task",
    "that",
    "this",
    "through",
    "using",
    "would",
}

ANTHROPIC_CLIENT = Anthropic(api_key=settings.anthropic_api_key)


def extract_keywords(text: str, limit: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\\-]{3,}", text.lower())
    keywords: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords or ["goal", "context", "format"]


def build_preassessment_answer(competency: str) -> str:
    return (
        f"For {competency.lower()}, I set the goal, context, output format, "
        "and constraints so the response stays useful."
    )


def generate_targeted_answer(competency: str, prompt_text: str, *, label: str) -> str:
    response = ANTHROPIC_CLIENT.messages.create(
        model=settings.anthropic_model,
        max_tokens=280,
        temperature=0.2,
        system=(
            "You are simulating a serious learner. "
            "Answer the teacher's prompt directly in 2-3 concise sentences. "
            "Be applied, competency-specific, and explain why the action makes sense."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Competency: {competency}\n"
                    f"Prompt type: {label}\n"
                    f"Teacher prompt:\n{prompt_text}\n\n"
                    "Write only the learner answer."
                ),
            }
        ],
    )
    parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    return "".join(parts).strip()


def build_learning_answer(competency: str, current_subpart: str, turn: int) -> str:
    keywords = extract_keywords(current_subpart)
    keyword_phrase = ", ".join(keywords[:2])
    answer_bank = [
        f"For {competency.lower()}, I would use {keyword_phrase} to keep the answer precise.",
        f"I would apply {keyword_phrase} in the scenario and explain why each step matters.",
        f"I would tie {keyword_phrase} to the user's goal so the output is consistent.",
        f"I would add context, format, and checks around {keyword_phrase} to reduce ambiguity.",
        f"I would use {keyword_phrase} with a concrete example so the model follows the intent.",
        f"I would test {keyword_phrase}, inspect the gaps, and refine the prompt with clearer rules.",
    ]
    return answer_bank[(turn - 1) % len(answer_bank)]


def build_formative_answer(competency: str, formative_prompt: str, turn: int) -> str:
    return generate_targeted_answer(competency, formative_prompt, label=f"formative-{turn}")


def build_assessment_answer(competency: str, assessment_prompt: str) -> str:
    rubric = get_locked_rubric(competency) or {}
    criteria_lines = []
    for criterion in rubric.get("criteria", []):
        criteria_lines.append(f"- {criterion.get('name')}: {criterion.get('description')}")
    criteria_text = "\n".join(criteria_lines) or "- Meet the locked rubric criteria."
    return generate_targeted_answer(
        competency,
        (
            f"{assessment_prompt}\n\n"
            f"Locked rubric criteria:\n{criteria_text}\n\n"
            "Answer as the learner with these headings: Objective, Structured Prompt Example, "
            "Why This Works, Iteration Plan, Risk. In Structured Prompt Example, include a literal "
            "prompt that specifies role, context, constraints, output format, and at least one example "
            "or step-by-step instruction when relevant."
        ),
        label="assessment",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one live competency cycle against the local AI engine.")
    parser.add_argument("--token", required=True, help="Backend access token")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--micro-credential-id", type=int, default=1)
    parser.add_argument("--learner-id", default="4")
    parser.add_argument("--sleep-seconds", type=float, default=2.2)
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
    print(f"session_id={session_id}")
    print(f"current_competency={start_payload.get('current_competency')}")

    pre_start = client.post("/pre-assessment/start", json={"session_id": session_id})
    pre_start.raise_for_status()

    status = client.get(f"/session/{session_id}")
    status.raise_for_status()
    competency = status.json()["competency"]

    pre_answer = client.post(
        "/pre-assessment/chat",
        json={
            "session_id": session_id,
            "answer": build_preassessment_answer(competency),
        },
    )
    pre_answer.raise_for_status()
    payload = pre_answer.json()
    print(f"phase_after_preassessment={payload.get('phase')}")

    turn_guard = 0
    while payload.get("phase") == "learning" and not payload.get("ready_for_assessment"):
        turn_guard += 1
        if turn_guard > 12:
            raise RuntimeError("Learning loop did not unlock assessment within 12 learner turns.")
        time.sleep(args.sleep_seconds)
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
            f"learn_turn={payload.get('interaction_number')} "
            f"phase={payload.get('phase')} "
            f"awaiting_formative={payload.get('awaiting_formative_response')} "
            f"ready_for_assessment={payload.get('ready_for_assessment')}"
        )

    assessment_state = client.get(f"/session/{session_id}")
    assessment_state.raise_for_status()
    assessment_prompt = str(assessment_state.json().get("current_assessment_prompt") or "")

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

    final_status = client.get(f"/session/{session_id}")
    final_status.raise_for_status()
    print("final_status=", json.dumps(final_status.json(), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
