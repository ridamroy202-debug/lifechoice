import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class Result:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class FakeCrew:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def crew(self):
        return self

    def kickoff(self, inputs):
        if self.kind == "pre":
            return Result("1. What makes a prompt structured?\n2. How would you control output format?")
        if self.kind == "level":
            return Result(json.dumps({"level": "intermediate", "weak_areas": ["iteration awareness", "contextual awareness"]}))
        if self.kind == "study":
            return Result("Study material for competency.")
        if self.kind == "plan":
            return Result(
                "\n".join(
                    [
                        "1. Define the core concept.",
                        "2. Show the key structure.",
                        "3. Apply it in a simple scenario.",
                        "4. Compare strong vs weak prompting.",
                        "5. Practice refinement.",
                        "6. Prepare for assessment.",
                    ]
                )
            )
        if self.kind == "tutor":
            base = (
                "## Title\n"
                f"Teaching {inputs['competency']} using {inputs['delivery_mode']} at {inputs['difficulty_tier']} difficulty.\n\n"
                "## Learner Feedback\n"
                "You identified the main direction correctly and now need a clearer mechanism-level explanation.\n\n"
                "## What This Concept Means\n"
                "This concept explains one practical idea in a clear, applied way for the learner.\n\n"
                "## How It Works\n"
                "Step 1 explains the setup, step 2 shows the decision process, and step 3 shows how to validate the outcome.\n\n"
                "## Visual Aid\n"
                "| Step | Why it matters |\n| --- | --- |\n| 1 | Establish context |\n| 2 | Apply the rule |\n| 3 | Validate the result |\n\n"
                "## Example\n"
                "In a workplace scenario, the learner applies the concept to a realistic prompt-design problem and explains the reasoning.\n\n"
                "## Key Takeaway\n"
                "Use one clear concept, one explicit decision path, and one validation step.\n\n"
                "## Next Learner Action\n"
                "Apply the concept to the next scenario and justify your choice.\n"
            )
            if inputs.get("include_formative_check") == "yes":
                return Result(base + "\n\n## Formative Check\nApply this concept to a realistic scenario and justify your design.")
            return Result(base)
        if self.kind == "assessment":
            competency = inputs.get("competency", "")
            if "Formative check" in competency:
                learner_response = inputs.get("user_response", "")
                if "fail formative" in learner_response:
                    return Result(
                        json.dumps(
                            {
                                "criteria_scores": [
                                    {"criterion_id": "formative_accuracy", "met": False, "evidence": "Inaccurate"},
                                    {"criterion_id": "formative_application", "met": False, "evidence": "Did not apply"},
                                    {"criterion_id": "formative_explanation", "met": False, "evidence": "Unclear"},
                                ],
                                "overall_percent": 0.0,
                                "pass": False,
                                "summary": "Formative check failed.",
                            }
                        )
                    )
                return Result(
                    json.dumps(
                        {
                            "criteria_scores": [
                                {"criterion_id": "formative_accuracy", "met": True, "evidence": "Accurate"},
                                {"criterion_id": "formative_application", "met": True, "evidence": "Applied"},
                                {"criterion_id": "formative_explanation", "met": True, "evidence": "Clear"},
                            ],
                            "overall_percent": 100.0,
                            "pass": True,
                            "summary": "Formative check passed.",
                        }
                    )
                )
            if "Final micro-credential assessment" in competency:
                learner_response = inputs.get("user_response", "")
                if "force final fail" in learner_response:
                    return Result(
                        json.dumps(
                            {
                                "criteria_scores": [
                                    {"criterion_id": "c1", "met": False, "evidence": "Integrated application missing"},
                                    {"criterion_id": "c2", "met": False, "evidence": "Weak reasoning"},
                                    {"criterion_id": "c3", "met": False, "evidence": "Execution vague"},
                                    {"criterion_id": "c4", "met": False, "evidence": "No risk awareness"},
                                ],
                                "overall_percent": 0.0,
                                "pass": False,
                                "summary": "Final assessment failed.",
                            }
                        )
                    )
                return Result(
                    json.dumps(
                        {
                            "criteria_scores": [
                                {"criterion_id": "c1", "met": True, "evidence": "Integrated application shown"},
                                {"criterion_id": "c2", "met": True, "evidence": "Scenario reasoning shown"},
                                {"criterion_id": "c3", "met": True, "evidence": "Execution detail shown"},
                                {"criterion_id": "c4", "met": True, "evidence": "Risk awareness shown"},
                            ],
                            "overall_percent": 100.0,
                            "pass": True,
                            "summary": "Final assessment passed.",
                        }
                    )
                )
            return Result(
                json.dumps(
                    {
                        "criteria_scores": [
                            {"criterion_id": "c1", "met": True, "evidence": "Structure met"},
                            {"criterion_id": "c2", "met": True, "evidence": "Output control met"},
                            {"criterion_id": "c3", "met": True, "evidence": "Iteration met"},
                            {"criterion_id": "c4", "met": True, "evidence": "Context met"},
                        ],
                        "overall_percent": 100.0,
                        "pass": True,
                        "summary": "Competency assessment passed.",
                    }
                )
            )
        raise AssertionError(self.kind)


class EngineFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parent.parent
        cls.db_file = cls.repo / "data" / "test_engine.db"
        os.environ["AI_ENGINE_DB_PATH"] = str(cls.db_file)
        os.environ["RUBRIC_ADMIN_KEY"] = "test-admin-key"
        os.chdir(cls.repo)

        from app.main import app
        import app.main as main_mod
        import app.orchestrator as orch

        cls.app = app
        cls.main_mod = main_mod
        cls.orch = orch

    def setUp(self):
        from app.db import get_connection, init_db
        from app.persistence import seed_locked_rubrics_from_yaml

        init_db()
        with get_connection() as conn:
            conn.executescript(
                """
                DELETE FROM certificates;
                DELETE FROM badges;
                DELETE FROM anomaly_flags;
                DELETE FROM event_logs;
                DELETE FROM final_assessments;
                DELETE FROM formative_checks;
                DELETE FROM competency_attempts;
                DELETE FROM remote_session_mappings;
                DELETE FROM learner_competency_progress;
                DELETE FROM remote_learning_session_refs;
                DELETE FROM learning_sessions;
                DELETE FROM learners;
                DELETE FROM locked_rubrics;
                """
            )
        seed_locked_rubrics_from_yaml()
        self._patch_dependencies()

    def _patch_dependencies(self):
        self.orch.PreAssessCrew = lambda: FakeCrew("pre")
        self.orch.LevelClassifierCrew = lambda: FakeCrew("level")
        self.orch.StudyMeterial = lambda: FakeCrew("study")
        self.orch.PathPlnner = lambda: FakeCrew("plan")
        self.orch.TutorCrew = lambda: FakeCrew("tutor")
        self.orch.AssessmentCrew = lambda: FakeCrew("assessment")

        payload = {
            "success": True,
            "data": {
                "domains": [
                    {
                        "id": 22,
                        "source": "IKON",
                        "micro_credentials": [
                            {
                                "id": 197,
                                "micro_credential": "AI Prompt Engineer",
                                "level": "Intermediate",
                                "competencies": [
                                    {"id": 61, "code": 1, "title": "Write structured prompts", "description": "Create well-structured prompts."},
                                    {"id": 62, "code": 2, "title": "Optimize outputs iteratively", "description": "Iteratively improve prompt outputs."},
                                ],
                            }
                        ],
                    }
                ]
            },
        }

        backend = self.main_mod.remote_backend_client
        backend.fetch_lesson_competencies = lambda **kwargs: payload
        backend.check_access = lambda *args, **kwargs: {"success": True, "access": {"can_access": True, "message": "Access granted", "micro_credential_id": "197"}}
        backend.fetch_profile = lambda **kwargs: {
            "success": True,
            "profile": {
                "id": "7",
                "full_name": "Ridam Test",
                "email": "ridam@example.com",
                "updated_at": "2026-04-18T00:00:00+00:00",
            }
        }
        backend.start_learning_session = lambda **kwargs: {"success": True, "session": {"id": 500 + int(kwargs["competency_id"]), "status": "in_progress"}}
        backend.record_interaction = lambda **kwargs: {"ok": True}
        backend.submit_assessment = lambda **kwargs: {"ok": True}
        backend.fetch_learning_session = lambda session_id, **kwargs: {"success": True, "session": {"id": session_id, "status": "active"}}
        backend.fetch_gamification_progress = lambda *args, **kwargs: {"success": True, "progress": {"points_earned": 0}}
        backend.fetch_competency_rubric = lambda competency_id, **kwargs: {
            "success": True,
            "rubric_rules": {
                "competency_id": competency_id,
                "competency_title": "Write structured prompts" if int(competency_id) == 61 else "Optimize outputs iteratively",
                "rubric_rules": [
                    {
                        "criterion_id": "c1",
                        "criterion_name": "Objective framing",
                        "criterion_descriptor": "Identify the correct objective.",
                        "weight": 0.25,
                    },
                    {
                        "criterion_id": "c2",
                        "criterion_name": "Applied execution",
                        "criterion_descriptor": "Apply the competency in context.",
                        "weight": 0.30,
                    },
                    {
                        "criterion_id": "c3",
                        "criterion_name": "Reasoned justification",
                        "criterion_descriptor": "Explain why the approach fits.",
                        "weight": 0.25,
                    },
                    {
                        "criterion_id": "c4",
                        "criterion_name": "Risk and quality control",
                        "criterion_descriptor": "Identify risks and checks.",
                        "weight": 0.20,
                    },
                ],
            },
        }

        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies
        self.orch.remote_backend_client.check_access = backend.check_access
        self.orch.remote_backend_client.fetch_profile = backend.fetch_profile
        self.orch.remote_backend_client.start_learning_session = backend.start_learning_session
        self.orch.remote_backend_client.record_interaction = backend.record_interaction
        self.orch.remote_backend_client.submit_assessment = backend.submit_assessment
        self.orch.remote_backend_client.fetch_learning_session = backend.fetch_learning_session
        self.orch.remote_backend_client.fetch_gamification_progress = backend.fetch_gamification_progress
        self.orch.remote_backend_client.fetch_competency_rubric = backend.fetch_competency_rubric

    def _start_remote_session(self, client: TestClient) -> str:
        response = client.post(
            "/session/start",
            json={
                "learner_id": "7",
                "domain_id": 22,
                "micro_credential_id": 197,
                "auth_token": "token",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_competencies"], 2)
        self.assertTrue(payload["identity_verified"])
        return payload["session_id"]

    def _finish_current_competency(self, client: TestClient, session_id: str) -> dict:
        response = client.post("/pre-assessment/start", json={"session_id": session_id})
        self.assertEqual(response.status_code, 200, response.text)

        response = client.post(
            "/pre-assessment/chat",
            json={"session_id": session_id, "answer": "I know the basics and can explain structure and constraints."},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["phase"], "learning")

        while True:
            response = client.post(
                "/learn/chat",
                json={"session_id": session_id, "message": "My answer with practical application and justification."},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            if payload["phase"] == "competency_assessment":
                break

        response = client.post(
            "/assessment/competency",
            json={"session_id": session_id, "answer": "Applied scenario answer with steps, rationale, and risk control."},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_full_flow_to_certificate(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)

            first = self._finish_current_competency(client, session_id)
            self.assertEqual(first["phase"], "pre_assessment")

            second = self._finish_current_competency(client, session_id)
            self.assertEqual(second["phase"], "final_assessment")

            final_response = client.post(
                "/assessment/final",
                json={"session_id": session_id, "answer": "Integrated answer combining the completed competencies into one delivery plan."},
            )
            self.assertEqual(final_response.status_code, 200, final_response.text)
            self.assertEqual(final_response.json()["phase"], "completed")

            certificate = client.post(
                "/certificate/generate",
                json={"session_id": session_id, "auth_token": "token"},
            )
            self.assertEqual(certificate.status_code, 200, certificate.text)
            cert_payload = certificate.json()

            self.assertEqual(client.get(f"/certificate/verify/{cert_payload['certificate_id']}").status_code, 200)
            self.assertEqual(client.get(f"/certificate/{cert_payload['certificate_id']}/pdf").status_code, 200)
            self.assertEqual(client.get(f"/certificate/{cert_payload['certificate_id']}/html").status_code, 200)

            status = client.get(f"/session/{session_id}")
            self.assertEqual(status.status_code, 200, status.text)
            status_payload = status.json()
            self.assertEqual(status_payload["phase"], "completed")
            self.assertEqual(len(status_payload["completed_competencies"]), 2)
            self.assertGreater(status_payload["gamification"]["points_total"], 0)
            self.assertEqual(status_payload["remote_sync"]["last_sync_outcome"], "synced")
            self.assertIsNotNone(status_payload["gamification"]["completion_badge"])
            self.assertEqual(status_payload["required_next_action"], "generate_certificate")
            self.assertIn("completed_competencies", status_payload["session_summary"])

    def test_unified_interact_route_handles_context_and_full_progression(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)

            context = client.post(f"/session/{session_id}/interact")
            self.assertEqual(context.status_code, 200, context.text)
            context_payload = context.json()
            self.assertFalse(context_payload["counted_as_interaction"])
            self.assertEqual(context_payload["phase"], "pre_assessment")
            self.assertTrue(context_payload["current_prompt"])

            diagnostic = client.post(
                f"/session/{session_id}/interact",
                json={"message": "I understand prompt basics and can describe constraints and output structure."},
            )
            self.assertEqual(diagnostic.status_code, 200, diagnostic.text)
            self.assertEqual(diagnostic.json()["phase"], "learning")
            self.assertTrue(diagnostic.json()["counted_as_interaction"])

            while True:
                status = client.get(f"/session/{session_id}")
                self.assertEqual(status.status_code, 200, status.text)
                state = status.json()

                if state["phase"] == "completed":
                    break

                if state["phase"] == "pre_assessment":
                    prompt_payload = client.post(f"/session/{session_id}/interact")
                    self.assertEqual(prompt_payload.status_code, 200, prompt_payload.text)
                    answer = "I understand prompt basics and can describe constraints and output structure."
                elif state["phase"] == "learning":
                    if state["awaiting_formative_response"]:
                        answer = "I would apply the concept in a realistic scenario and justify the design choices clearly."
                    else:
                        answer = "I would structure the prompt with goal, constraints, output format, and a worked example."
                elif state["phase"] == "competency_assessment":
                    answer = "Applied scenario answer with steps, rationale, validation, and risk controls."
                elif state["phase"] == "final_assessment":
                    answer = "Integrated answer combining all competencies into one end-to-end delivery plan with governance and validation."
                else:
                    self.fail(f"Unexpected phase during unified interaction test: {state['phase']}")

                step = client.post(
                    f"/session/{session_id}/interact",
                    json={"message": answer},
                )
                self.assertEqual(step.status_code, 200, step.text)

            final_status = client.get(f"/session/{session_id}")
            self.assertEqual(final_status.status_code, 200, final_status.text)
            final_payload = final_status.json()
            self.assertEqual(final_payload["phase"], "completed")
            self.assertEqual(len(final_payload["completed_competencies"]), 2)
            self.assertEqual(final_payload["required_next_action"], "generate_certificate")

    def test_unified_interact_route_can_hydrate_remote_backend_session_id(self):
        remote_payload = {
            "success": True,
            "data": {
                "domains": [
                    {
                        "id": 5,
                        "name": "Technology",
                        "source": "IKON",
                        "micro_credentials": [
                            {
                                "id": 61,
                                "micro_credential": "Digital Transformation Specialist",
                                "level": "EQF 7",
                                "competencies": [
                                    {
                                        "id": 601,
                                        "code": 1,
                                        "title": "Digital maturity assessment",
                                        "description": "Assess digital maturity across people, process, data, and technology.",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        }
        backend = self.main_mod.remote_backend_client
        backend.fetch_learning_session = lambda session_id, **kwargs: {
            "success": True,
            "session": {
                "id": int(session_id),
                "micro_credential": 61,
                "competency": 601,
                "attempt_number": 1,
                "status": "in_progress",
                "interaction_count": 0,
                "interactions": [],
            },
        }
        backend.fetch_lesson_competencies = lambda **kwargs: remote_payload
        self.orch.remote_backend_client.fetch_learning_session = backend.fetch_learning_session
        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies

        with TestClient(self.app) as client:
            context = client.post("/session/1122/interact", json={"auth_token": "token"})
            self.assertEqual(context.status_code, 200, context.text)
            context_payload = context.json()
            self.assertFalse(context_payload["counted_as_interaction"])
            self.assertEqual(context_payload["source"], "remote")
            self.assertEqual(context_payload["remote_micro_credential_id"], 61)
            self.assertEqual(context_payload["current_remote_learning_session_id"], 1122)
            self.assertEqual(context_payload["current_competency"], "Digital maturity assessment")
            self.assertTrue(context_payload["current_prompt"])
            local_session_id = context_payload["session_id"]
            self.assertNotEqual(local_session_id, "1122")

            step = client.post(
                "/session/1122/interact",
                json={
                    "auth_token": "token",
                    "message": "Digital maturity assessment evaluates organisational readiness across leadership, process, data, technology, and capability so transformation priorities can be set on evidence.",
                },
            )
            self.assertEqual(step.status_code, 200, step.text)
            step_payload = step.json()
            self.assertTrue(step_payload["counted_as_interaction"])
            self.assertEqual(step_payload["phase"], "learning")
            self.assertEqual(step_payload["session_id"], local_session_id)

    def test_remote_sync_uses_extended_interaction_types(self):
        recorded_types: list[str] = []
        backend = self.main_mod.remote_backend_client
        backend.record_interaction = lambda **kwargs: recorded_types.append(kwargs["interaction_type"]) or {"ok": True}
        self.orch.remote_backend_client.record_interaction = backend.record_interaction

        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)

            intro_context = client.get(f"/session/{session_id}")
            self.assertEqual(intro_context.status_code, 200, intro_context.text)

            pre_start = client.post("/pre-assessment/start", json={"session_id": session_id})
            self.assertEqual(pre_start.status_code, 200, pre_start.text)

            pre_answer = client.post(
                "/pre-assessment/chat",
                json={"session_id": session_id, "answer": "I understand structured prompting and can apply constraints clearly."},
            )
            self.assertEqual(pre_answer.status_code, 200, pre_answer.text)

            while True:
                state = client.get(f"/session/{session_id}").json()
                if state["phase"] == "competency_assessment":
                    break
                answer = (
                    "I would apply the concept in a realistic scenario and justify the design clearly."
                    if state["awaiting_formative_response"]
                    else "I would define the goal, audience, constraints, and output structure with one applied example."
                )
                step = client.post("/learn/chat", json={"session_id": session_id, "message": answer})
                self.assertEqual(step.status_code, 200, step.text)

            assess = client.post(
                "/assessment/competency",
                json={"session_id": session_id, "answer": "Applied scenario answer with rationale, validation, and explicit risk controls."},
            )
            self.assertEqual(assess.status_code, 200, assess.text)

            self.assertIn("intro", recorded_types)
            self.assertIn("diagnostic", recorded_types)
            self.assertIn("teaching", recorded_types)
            self.assertIn("formative_check", recorded_types)
            self.assertIn("competency_assessment", recorded_types)

    def test_guards_block_out_of_phase_actions(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)

            learn = client.post("/learn/chat", json={"session_id": session_id, "message": "test"})
            self.assertEqual(learn.status_code, 400)

            final_assessment = client.post("/assessment/final", json={"session_id": session_id, "answer": "test"})
            self.assertEqual(final_assessment.status_code, 400)

            certificate = client.post("/certificate/generate", json={"session_id": session_id, "auth_token": "token"})
            self.assertEqual(certificate.status_code, 400)

    def test_intro_message_stays_within_three_sentences(self):
        with TestClient(self.app) as client:
            response = client.post(
                "/session/start",
                json={
                    "learner_id": "7",
                    "domain_id": 22,
                    "micro_credential_id": 197,
                    "auth_token": "token",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            intro = response.json()["message"]
            sentences = [part.strip() for part in intro.replace("\n", " ").split(".") if part.strip()]
            self.assertLessEqual(len(sentences), 3)

    def test_session_start_blocks_microcredential_without_locked_rubrics(self):
        payload = {
            "success": True,
            "data": {
                "domains": [
                    {
                        "id": 22,
                        "source": "IKON",
                        "micro_credentials": [
                            {
                                "id": 198,
                                "micro_credential": "AI Project Manager",
                                "level": "Intermediate",
                                "competencies": [
                                    {"id": 71, "code": 1, "title": "Define AI scope", "description": "Define project scope for AI work."},
                                    {"id": 72, "code": 2, "title": "Prioritise AI use cases", "description": "Prioritise cases by value and feasibility."},
                                ],
                            }
                        ],
                    }
                ]
            },
        }
        backend = self.main_mod.remote_backend_client
        backend.fetch_lesson_competencies = lambda **kwargs: payload
        backend.fetch_competency_rubric = lambda competency_id, **kwargs: {"success": True, "rubric_rules": {"competency_id": competency_id, "rubric_rules": []}}
        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies
        self.orch.remote_backend_client.fetch_competency_rubric = backend.fetch_competency_rubric

        with TestClient(self.app) as client:
            response = client.post(
                "/session/start",
                json={
                    "learner_id": "7",
                    "domain_id": 22,
                    "micro_credential_id": 198,
                    "auth_token": "token",
                },
            )
            self.assertEqual(response.status_code, 409, response.text)
            detail = response.json()["detail"]
            self.assertIn("missing_competencies", detail)
            self.assertIn("Define AI scope", detail["missing_competencies"])

    def test_session_start_allows_remote_microcredential_when_remote_rubrics_exist(self):
        payload = {
            "success": True,
            "data": {
                "domains": [
                    {
                        "id": 22,
                        "source": "IKON",
                        "micro_credentials": [
                            {
                                "id": 198,
                                "micro_credential": "AI Project Manager",
                                "level": "Intermediate",
                                "competencies": [
                                    {"id": 71, "code": 1, "title": "Define AI scope", "description": "Define project scope for AI work."},
                                    {"id": 72, "code": 2, "title": "Prioritise AI use cases", "description": "Prioritise cases by value and feasibility."},
                                ],
                            }
                        ],
                    }
                ]
            },
        }
        backend = self.main_mod.remote_backend_client
        backend.fetch_lesson_competencies = lambda **kwargs: payload
        backend.fetch_competency_rubric = lambda competency_id, **kwargs: {
            "success": True,
            "rubric_rules": {
                "competency_id": competency_id,
                "competency_title": "Define AI scope" if int(competency_id) == 71 else "Prioritise AI use cases",
                "rubric_rules": [
                    {"criterion_id": "c1", "criterion_name": "Objective framing", "criterion_descriptor": "Identify scope.", "weight": 0.25},
                    {"criterion_id": "c2", "criterion_name": "Applied execution", "criterion_descriptor": "Apply in context.", "weight": 0.30},
                    {"criterion_id": "c3", "criterion_name": "Reasoned justification", "criterion_descriptor": "Justify choice.", "weight": 0.25},
                    {"criterion_id": "c4", "criterion_name": "Risk and quality control", "criterion_descriptor": "Identify risks.", "weight": 0.20},
                ],
            },
        }
        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies
        self.orch.remote_backend_client.fetch_competency_rubric = backend.fetch_competency_rubric

        with TestClient(self.app) as client:
            response = client.post(
                "/session/start",
                json={
                    "learner_id": "7",
                    "domain_id": 22,
                    "micro_credential_id": 198,
                    "auth_token": "token",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)

    def test_docs_openapi_and_new_backend_proxy_routes_are_exposed(self):
        with TestClient(self.app) as client:
            docs = client.get("/docs")
            self.assertEqual(docs.status_code, 200)

            openapi = client.get("/openapi.json")
            self.assertEqual(openapi.status_code, 200)
            paths = openapi.json()["paths"]
            self.assertIn("/backend/enrollment/check-access/{mc_id}", paths)
            self.assertIn("/backend/enrollment/check-access", paths)
            self.assertIn("/backend/auth/profile/me", paths)
            self.assertIn("/backend/learning/sessions/start", paths)
            self.assertIn("/backend/learning/sessions/{session_id}", paths)
            self.assertIn("/backend/learning/sessions/{session_id}/interact", paths)
            self.assertIn("/backend/learning/sessions/{session_id}/assess", paths)
            self.assertIn("/backend/gamification/progress/{session_id}", paths)
            self.assertIn("/backend/learner/micro-credential/progress", paths)
            self.assertIn("/session/{session_id}/interact", paths)

    def test_cors_preflight_allows_local_frontend_origin(self):
        with TestClient(self.app) as client:
            response = client.options(
                "/backend/learning/sessions/start",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type,authorization",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    def test_remote_rubric_proxy_returns_remote_rules(self):
        with TestClient(self.app) as client:
            response = client.get("/backend/lesson/rubric/61?auth_token=token")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["rubric_rules"]["competency_id"], 61)
            self.assertTrue(payload["rubric_rules"]["rubric_rules"])

    def test_backend_profile_competencies_access_and_learning_session_routes(self):
        with TestClient(self.app) as client:
            profile = client.post(
                "/backend/auth/profile/me",
                json={"micro_credential_id": 197, "auth_token": "token"},
            )
            self.assertEqual(profile.status_code, 200, profile.text)
            profile_payload = profile.json()
            self.assertEqual(profile_payload["micro_credential"]["id"], 197)
            self.assertEqual(len(profile_payload["competencies"]), 2)

            competencies = client.post(
                "/backend/lesson/competencies",
                json={"micro_credential_id": 197},
            )
            self.assertEqual(competencies.status_code, 200, competencies.text)
            self.assertEqual(competencies.json()["data"]["domains"][0]["micro_credentials"][0]["id"], 197)

            access = client.post(
                "/backend/enrollment/check-access",
                json={"micro_credential_id": 197, "auth_token": "token"},
            )
            self.assertEqual(access.status_code, 200, access.text)
            self.assertTrue(access.json()["access"]["can_access"])

            start_first = client.post(
                "/backend/learning/sessions/start",
                json={"micro_credential_id": 197, "competency_id": 61, "auth_token": "token"},
            )
            self.assertEqual(start_first.status_code, 200, start_first.text)
            self.assertTrue(start_first.json()["previous_competencies_passed"])
            remote_session_id = start_first.json()["learning_session"]["session"]["id"]

            progress_initial = client.get(
                "/backend/learner/micro-credential/progress",
                params={"micro_credential_id": 197, "auth_token": "token"},
            )
            self.assertEqual(progress_initial.status_code, 200, progress_initial.text)
            self.assertEqual(progress_initial.json()["progress"]["completed_competencies"], 0)

            block_second = client.post(
                "/backend/learning/sessions/start",
                json={"micro_credential_id": 197, "competency_id": 62, "auth_token": "token"},
            )
            self.assertEqual(block_second.status_code, 409, block_second.text)

            assess = client.post(
                f"/backend/learning/sessions/{remote_session_id}/assess",
                json={
                    "scenario_question": "question",
                    "learner_response": "answer",
                    "rubric_score": 80.0,
                    "ai_feedback": "good",
                    "auth_token": "token",
                },
            )
            self.assertEqual(assess.status_code, 200, assess.text)

            start_second = client.post(
                "/backend/learning/sessions/start",
                json={"micro_credential_id": 197, "competency_id": 62, "auth_token": "token"},
            )
            self.assertEqual(start_second.status_code, 200, start_second.text)

            detail = client.get(
                f"/backend/learning/sessions/{remote_session_id}",
                params={"auth_token": "token"},
            )
            self.assertEqual(detail.status_code, 200, detail.text)
            self.assertEqual(detail.json()["local"]["status"], "passed")

            progress_final = client.get(
                "/backend/learner/micro-credential/progress",
                params={"micro_credential_id": 197, "auth_token": "token"},
            )
            self.assertEqual(progress_final.status_code, 200, progress_final.text)
            progress_payload = progress_final.json()
            self.assertEqual(progress_payload["progress"]["completed_competencies"], 1)
            self.assertEqual(progress_payload["progress"]["next_available_competency_id"], 62)

    def test_revision_flow_requires_two_extra_interactions_after_two_failed_formatives(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)
            client.post("/pre-assessment/start", json={"session_id": session_id})
            start_learning = client.post(
                "/pre-assessment/chat",
                json={"session_id": session_id, "answer": "I know the basics and can explain structure and constraints."},
            )
            self.assertEqual(start_learning.status_code, 200, start_learning.text)

            interaction_4 = client.post("/learn/chat", json={"session_id": session_id, "message": "advance one"})
            self.assertEqual(interaction_4.status_code, 200, interaction_4.text)
            self.assertEqual(interaction_4.json()["interaction_number"], 4)

            after_fail_1 = client.post("/learn/chat", json={"session_id": session_id, "message": "fail formative one"})
            self.assertEqual(after_fail_1.status_code, 200, after_fail_1.text)
            self.assertEqual(after_fail_1.json()["interaction_number"], 5)

            interaction_6 = client.post("/learn/chat", json={"session_id": session_id, "message": "advance two"})
            self.assertEqual(interaction_6.status_code, 200, interaction_6.text)
            self.assertEqual(interaction_6.json()["interaction_number"], 6)

            after_fail_2 = client.post("/learn/chat", json={"session_id": session_id, "message": "fail formative two"})
            self.assertEqual(after_fail_2.status_code, 200, after_fail_2.text)
            self.assertTrue(after_fail_2.json()["revision_required"])

            revision_1 = client.post("/learn/chat", json={"session_id": session_id, "message": "revision step one"})
            self.assertEqual(revision_1.status_code, 200, revision_1.text)
            self.assertEqual(revision_1.json()["interaction_number"], 8)

            revision_2 = client.post("/learn/chat", json={"session_id": session_id, "message": "revision step two"})
            self.assertEqual(revision_2.status_code, 200, revision_2.text)
            self.assertEqual(revision_2.json()["interaction_type"], "revision")

    def test_final_assessment_failure_resets_to_interaction_three(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)
            first = self._finish_current_competency(client, session_id)
            self.assertEqual(first["phase"], "pre_assessment")
            second = self._finish_current_competency(client, session_id)
            self.assertEqual(second["phase"], "final_assessment")

            final_response = client.post(
                "/assessment/final",
                json={"session_id": session_id, "answer": "force final fail"},
            )
            self.assertEqual(final_response.status_code, 200, final_response.text)
            payload = final_response.json()
            self.assertEqual(payload["phase"], "learning")
            self.assertEqual(payload["interaction_number"], 3)
            self.assertTrue(payload["mastery_reset"])


class RemoteBackendContractTests(unittest.TestCase):
    def test_remote_backend_client_uses_live_swagger_paths(self):
        from app.remote_backend import RemoteBackendClient

        client = RemoteBackendClient()

        def fake_request(method, url, **kwargs):
            class FakeResponse:
                status_code = 200
                content = b"{}"
                text = "{}"

                def json(self):
                    return {}

            calls.append((method, url))
            return FakeResponse()

        calls = []
        with patch("app.remote_backend.requests.request", side_effect=fake_request):
            client.login("a@example.com", "secret")
            client.fetch_profile(token="token")
            client.fetch_lesson_competencies(domain_id=22, micro_credential_id=197)
            client.check_access(197, token="token")
            client.start_learning_session(competency_id=61, token="token")
            client.fetch_learning_session(501, token="token")
            client.fetch_gamification_progress(501, token="token")

        urls = [url for _, url in calls]
        self.assertIn("https://lifechoice.duckdns.org/auth/login/login/", urls)
        self.assertIn("https://lifechoice.duckdns.org/auth/profile/me/", urls)
        self.assertIn("https://lifechoice.duckdns.org/lesson/competencies/", urls)
        self.assertIn("https://lifechoice.duckdns.org/enrollment/enrollments/check-access/197/", urls)
        self.assertIn("https://lifechoice.duckdns.org/learning/sessions/start/", urls)
        self.assertIn("https://lifechoice.duckdns.org/learning/sessions/501/", urls)
        self.assertIn("https://lifechoice.duckdns.org/gamification/progress/501/", urls)

    def test_record_interaction_falls_back_for_extended_types(self):
        from app.remote_backend import RemoteBackendClient

        client = RemoteBackendClient()
        calls: list[dict] = []

        class FakeResponse:
            def __init__(self, status_code: int, payload: dict[str, object]):
                self.status_code = status_code
                self._payload = payload
                self.text = json.dumps(payload)
                self.content = self.text.encode("utf-8")

            def json(self):
                return self._payload

        def fake_request(method, url, **kwargs):
            calls.append({"method": method, "url": url, "json": kwargs.get("json")})
            if len(calls) == 1:
                return FakeResponse(400, {"detail": "invalid choice"})
            return FakeResponse(201, {"success": True})

        with patch("app.remote_backend.requests.request", side_effect=fake_request):
            payload = client.record_interaction(
                session_id=1125,
                interaction_type="competency_assessment",
                ai_prompt="Assessment prompt",
                ai_response="Assessment feedback",
                learner_input="Learner answer",
                formative_passed=True,
                token="token",
            )

        self.assertEqual(payload["success"], True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["json"]["interaction_type"], "competency_assessment")
        self.assertEqual(calls[1]["json"]["interaction_type"], "teaching")
        self.assertIn("[AI_ENGINE_FALLBACK]", calls[1]["json"]["ai_prompt"])


class EvaluationNormalizationTests(unittest.TestCase):
    def test_normalize_binary_evaluation_uses_positional_scores_when_ids_do_not_match(self):
        import app.orchestrator as orch

        rubric = {
            "criteria": [
                {"criterion_id": "formative_accuracy", "name": "Concept accuracy", "weight": 0.34},
                {"criterion_id": "formative_application", "name": "Applied reasoning", "weight": 0.33},
                {"criterion_id": "formative_explanation", "name": "Clear explanation", "weight": 0.33},
            ]
        }
        evaluation = {
            "criteria_scores": [
                {"criterion_name": "Concept accuracy", "met": "true", "evidence": "Accurate"},
                {"criterion_name": "Applied reasoning", "met": "true", "evidence": "Applied"},
                {"criterion_name": "Clear explanation", "met": "true", "evidence": "Clear"},
            ],
            "overall_percent": 100.0,
            "pass": True,
            "summary": "Strong response.",
        }

        normalized = orch._normalize_binary_evaluation(evaluation, rubric)
        self.assertEqual(normalized["overall_percent"], 100.0)
        self.assertTrue(normalized["pass"])
        self.assertEqual([item["met"] for item in normalized["criteria_scores"]], [True, True, True])

    def test_normalize_binary_evaluation_falls_back_to_raw_overall_when_names_drift(self):
        import app.orchestrator as orch

        rubric = {
            "criteria": [
                {"criterion_id": "formative_accuracy", "name": "Concept accuracy", "weight": 0.34},
                {"criterion_id": "formative_application", "name": "Applied reasoning", "weight": 0.33},
                {"criterion_id": "formative_explanation", "name": "Clear explanation", "weight": 0.33},
            ]
        }
        evaluation = {
            "criteria_scores": [
                {"criterion_name": "Accuracy of budgeting logic", "met": "yes", "evidence": "Strong"},
                {"criterion_name": "Use of evidence in prioritization", "met": "yes", "evidence": "Applied"},
                {"criterion_name": "Reasoning clarity", "met": "yes", "evidence": "Clear"},
            ],
            "overall_percent": 82.0,
            "pass": True,
            "summary": "Learner demonstrated solid applied reasoning.",
        }

        normalized = orch._normalize_binary_evaluation(evaluation, rubric)
        self.assertGreaterEqual(normalized["overall_percent"], 82.0)
        self.assertTrue(normalized["pass"])

    def test_formative_heuristics_accept_specific_brand_identity_answer(self):
        import app.orchestrator as orch

        prompt = 'For a startup with the mission to "develop renewable energy solutions," what key visual elements would you focus on, and how would they align with the mission statement?'
        answer = (
            "I would focus on a green and blue color palette, clean typography, modular iconography, and optimistic imagery of renewable systems "
            "because those elements communicate sustainability, trust, and technological progress. These choices align with the mission by making "
            "the brand look credible, future-focused, and clearly tied to renewable energy outcomes."
        )

        heuristics = orch._build_formative_heuristics(prompt, answer, "Construct comprehensive brand identity systems")
        self.assertTrue(heuristics["scenario_relevance"])
        self.assertTrue(heuristics["concrete_application"])
        self.assertTrue(heuristics["explanation_quality"])
        self.assertTrue(heuristics["pass"])

    def test_evaluate_formative_response_uses_heuristic_override_when_model_is_too_harsh(self):
        import app.orchestrator as orch
        from app.state import LearnerSession

        class FailingAssessmentCrew:
            def crew(self):
                return self

            def kickoff(self, inputs):
                return Result(
                    json.dumps(
                        {
                            "criteria_scores": [
                                {"criterion_id": "formative_accuracy", "met": False, "evidence": "Too generic"},
                                {"criterion_id": "formative_application", "met": False, "evidence": "Did not apply"},
                                {"criterion_id": "formative_explanation", "met": False, "evidence": "No reasoning"},
                            ],
                            "overall_percent": 0.0,
                            "pass": False,
                            "summary": "Model judged the answer too harshly.",
                        }
                    )
                )

        session = LearnerSession(
            topic="Brand Identity Architect",
            competencies=["Construct comprehensive brand identity systems"],
        )
        session.current_formative_prompt = 'For a startup with the mission to "develop renewable energy solutions," what key visual elements would you focus on, and how would they align with the mission statement?'

        with patch.object(orch, "AssessmentCrew", return_value=FailingAssessmentCrew()):
            passed, overall, summary, easy_pass = orch._evaluate_formative_response(
                session,
                (
                    "I would use a green and blue color palette, modern typography, sustainability-focused imagery, and simple iconography "
                    "because those visual elements signal renewable energy, trust, and innovation. They align with the mission by making the brand "
                    "feel environmentally responsible, future-focused, and easy for stakeholders to understand."
                ),
            )

        self.assertTrue(passed)
        self.assertGreaterEqual(overall, 75.0)
        self.assertIn("Heuristic validation confirmed", summary)
        self.assertTrue(easy_pass)


if __name__ == "__main__":
    unittest.main()
