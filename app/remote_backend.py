from __future__ import annotations

import os
from typing import Any

import requests
from requests import RequestException
from dotenv import load_dotenv


load_dotenv()


class RemoteBackendError(RuntimeError):
    pass


class RemoteBackendClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("REMOTE_BACKEND_URL", "http://3.111.49.240:8005").rstrip("/")
        self.default_token = os.getenv("REMOTE_API_TOKEN", "").strip()

    def _headers(self, token: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        final_token = (token or self.default_token or "").strip()
        if final_token:
            headers["Authorization"] = f"Bearer {final_token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(token),
                params=params,
                json=json_body,
                data=data,
                timeout=45,
            )
        except RequestException as exc:
            raise RemoteBackendError(f"Remote backend network error for {path}: {exc}") from exc
        if allow_404 and response.status_code == 404:
            return {}
        if response.status_code >= 400:
            raise RemoteBackendError(
                f"Remote backend request failed {response.status_code} for {path}: {response.text}"
            )
        if not response.content:
            return {}
        return response.json()

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/login/login/",
            data={"email": email, "password": password},
        )

    def fetch_lesson_competencies(
        self,
        *,
        domain_id: int,
        micro_credential_id: int,
        competency_id: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "domain_id": domain_id,
            "micro_credential_id": micro_credential_id,
        }
        if competency_id is not None:
            params["competency_id"] = competency_id
        return self._request("GET", "/lesson/competencies/", params=params)

    def fetch_rubric(self, competency_id: int) -> dict[str, Any] | None:
        payload = self._request(
            "GET",
            f"/lesson/rubrics/by-competency/{competency_id}/",
            allow_404=True,
        )
        return payload or None

    def check_access(self, micro_credential_id: int, *, token: str | None) -> dict[str, Any]:
        if not (token or self.default_token):
            raise RemoteBackendError("Remote backend auth token is required to check enrollment access")
        return self._request(
            "GET",
            f"/enrollment/test/has-access/{micro_credential_id}/",
            token=token,
        )

    def start_learning_session(
        self,
        *,
        mc_access_id: int,
        competency_id: int,
        token: str | None,
    ) -> dict[str, Any]:
        if not (token or self.default_token):
            raise RemoteBackendError("Remote backend auth token is required to start a learning session")
        return self._request(
            "POST",
            "/learning/sessions/start/",
            token=token,
            json_body={"mc_access_id": mc_access_id, "competency_id": competency_id},
        )

    def record_interaction(
        self,
        *,
        session_id: int,
        interaction_type: str,
        ai_prompt: str,
        ai_response: str,
        learner_input: str | None,
        formative_passed: bool | None,
        token: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "interaction_type": interaction_type,
            "ai_prompt": ai_prompt,
            "ai_response": ai_response,
        }
        if learner_input:
            body["learner_input"] = learner_input
        if formative_passed is not None:
            body["formative_passed"] = formative_passed
        return self._request(
            "POST",
            f"/learning/sessions/{session_id}/interact/",
            token=token,
            json_body=body,
        )

    def submit_assessment(
        self,
        *,
        session_id: int,
        scenario_question: str,
        learner_response: str,
        rubric_score: float,
        ai_feedback: str,
        token: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/learning/sessions/{session_id}/assess/",
            token=token,
            json_body={
                "scenario_question": scenario_question,
                "learner_response": learner_response,
                "rubric_score": rubric_score,
                "ai_feedback": ai_feedback,
            },
        )

    def fetch_learning_session(self, session_id: int, *, token: str | None) -> dict[str, Any]:
        return self._request("GET", f"/learning/sessions/{session_id}/", token=token)

    def fetch_gamification_progress(self, session_id: int, *, token: str | None) -> dict[str, Any]:
        return self._request("GET", f"/gamification/progress/{session_id}/", token=token)


remote_backend_client = RemoteBackendClient()
