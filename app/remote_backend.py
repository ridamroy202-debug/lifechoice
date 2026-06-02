from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urljoin

import requests
from requests import RequestException

from app.settings import settings


logger = logging.getLogger(__name__)


class RemoteBackendError(RuntimeError):
    pass


class RemoteBackendClient:
    _interaction_fallback_map = {
        "intro": "teaching",
        "diagnostic": "teaching",
        "competency_assessment": "teaching",
        "final_assessment": "teaching",
    }

    def __init__(self) -> None:
        self.base_url = settings.remote_backend_url
        self.default_token = settings.remote_api_token
        self.timeout_seconds = settings.remote_api_timeout_seconds

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
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = request_timeout if request_timeout is not None else self.timeout_seconds
        started = time.perf_counter()
        try:
            response = requests.request(
                method,
                url,
                headers=self._headers(token),
                params=params,
                json=json_body,
                data=data,
                timeout=timeout,
            )
        except RequestException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "remote_backend_error method=%s path=%s timeout=%.1fs elapsed_ms=%.1f error=%s",
                method,
                path,
                timeout,
                elapsed_ms,
                exc,
            )
            raise RemoteBackendError(f"Remote backend network error for {path}: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms >= 1500:
            logger.info(
                "remote_backend_slow_call method=%s path=%s status=%s elapsed_ms=%.1f",
                method,
                path,
                response.status_code,
                elapsed_ms,
            )

        if allow_404 and response.status_code == 404:
            return {}
        if response.status_code >= 400:
            detail = response.text
            if len(detail) > 800:
                detail = f"{detail[:800]}...[truncated]"
            raise RemoteBackendError(
                f"Remote backend request failed {response.status_code} for {path}: {detail}"
            )
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _unwrap_payload(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
        for key in keys:
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        return payload

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/login/login/",
            data={"email": email, "password": password},
        )

    def fetch_profile(self, *, token: str | None) -> dict[str, Any]:
        if not (token or self.default_token):
            raise RemoteBackendError("Remote backend auth token is required to fetch learner profile")
        return self._request(
            "GET",
            "/auth/profile/me/",
            token=token,
        )

    def fetch_lesson_competencies(
        self,
        *,
        domain_id: int | None = None,
        micro_credential_id: int | None = None,
        competency_id: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if domain_id is not None:
            params["domain_id"] = domain_id
        if micro_credential_id is not None:
            params["micro_credential_id"] = micro_credential_id
        if competency_id is not None:
            params["competency_id"] = competency_id
        return self._request("GET", "/lesson/competencies/", params=params)

    def fetch_competency_rubric(self, competency_id: int, *, token: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/lesson/competencies/{competency_id}/rubric-rules/",
            token=token,
        )

    def check_access(self, mc_id: int, *, token: str | None) -> dict[str, Any]:
        if not (token or self.default_token):
            raise RemoteBackendError("Remote backend auth token is required to check enrollment access")
        return self._request(
            "GET",
            f"/enrollment/enrollments/check-access/{mc_id}/",
            token=token,
        )

    def start_learning_session(
        self,
        *,
        competency_id: int,
        token: str | None,
    ) -> dict[str, Any]:
        if not (token or self.default_token):
            raise RemoteBackendError("Remote backend auth token is required to start a learning session")
        return self._request(
            "POST",
            "/learning/sessions/start/",
            token=token,
            json_body={"competency_id": competency_id},
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
        path = f"/learning/sessions/{session_id}/interact/"
        try:
            return self._request(
                "POST",
                path,
                token=token,
                json_body=body,
            )
        except RemoteBackendError as exc:
            fallback_type = self._interaction_fallback_map.get(interaction_type)
            if not fallback_type:
                raise
            fallback_prompt = (
                f"{ai_prompt}\n\n[AI_ENGINE_FALLBACK]"
                f"{json.dumps({'original_interaction_type': interaction_type}, separators=(',', ':'))}"
            )
            fallback_body = dict(body)
            fallback_body["interaction_type"] = fallback_type
            fallback_body["ai_prompt"] = fallback_prompt
            try:
                return self._request(
                    "POST",
                    path,
                    token=token,
                    json_body=fallback_body,
                )
            except RemoteBackendError as fallback_exc:
                raise RemoteBackendError(
                    f"{exc} | fallback interaction sync failed: {fallback_exc}"
                ) from fallback_exc

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

    def fetch_question_bank(
        self,
        competency_title: str,
        assessment_type: str,
        count: int = 5,
        *,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch MCQ questions from remote question bank. assessment_type: FA1, FA2, or FA3."""
        return self._request(
            "GET",
            "/questionbank/questions/",
            token=token,
            params={"competency": competency_title, "assessment_type": assessment_type, "count": count},
            request_timeout=min(self.timeout_seconds, 10.0),
        )

    def fetch_gamification_progress(self, session_id: int, *, token: str | None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/gamification/progress/{session_id}/",
            token=token,
            request_timeout=min(self.timeout_seconds, 6.0),
        )

    def absolute_url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))


remote_backend_client = RemoteBackendClient()
