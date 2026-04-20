import json
import os
import unittest
from pathlib import Path

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
            base = f"Teaching {inputs['competency']} using {inputs['delivery_mode']} at {inputs['difficulty_tier']} difficulty."
            if inputs.get("include_formative_check") == "yes":
                return Result(base + "\n\n**Formative Check**\nApply this concept to a realistic scenario and justify your design.")
            return Result(base + "\n\nWorked example and immediate feedback.")
        if self.kind == "assessment":
            competency = inputs.get("competency", "")
            if "Formative check" in competency:
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
        backend.check_access = lambda *args, **kwargs: {"access": {"can_start_session": True, "enrollment": {"id": 901}}}
        backend.fetch_profile = lambda **kwargs: {
            "data": {
                "id": "7",
                "full_name": "Ridam Test",
                "email": "ridam@example.com",
                "updated_at": "2026-04-18T00:00:00+00:00",
            }
        }
        backend.start_learning_session = lambda **kwargs: {"id": 500 + int(kwargs["competency_id"])}
        backend.record_interaction = lambda **kwargs: {"ok": True}
        backend.submit_assessment = lambda **kwargs: {"ok": True}
        backend.fetch_gamification_progress = lambda *args, **kwargs: {"points": 0}

        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies
        self.orch.remote_backend_client.check_access = backend.check_access
        self.orch.remote_backend_client.fetch_profile = backend.fetch_profile
        self.orch.remote_backend_client.start_learning_session = backend.start_learning_session
        self.orch.remote_backend_client.record_interaction = backend.record_interaction
        self.orch.remote_backend_client.submit_assessment = backend.submit_assessment
        self.orch.remote_backend_client.fetch_gamification_progress = backend.fetch_gamification_progress

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

    def test_guards_block_out_of_phase_actions(self):
        with TestClient(self.app) as client:
            session_id = self._start_remote_session(client)

            learn = client.post("/learn/chat", json={"session_id": session_id, "message": "test"})
            self.assertEqual(learn.status_code, 400)

            final_assessment = client.post("/assessment/final", json={"session_id": session_id, "answer": "test"})
            self.assertEqual(final_assessment.status_code, 400)

            certificate = client.post("/certificate/generate", json={"session_id": session_id, "auth_token": "token"})
            self.assertEqual(certificate.status_code, 400)

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
        self.orch.remote_backend_client.fetch_lesson_competencies = backend.fetch_lesson_competencies

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


if __name__ == "__main__":
    unittest.main()
