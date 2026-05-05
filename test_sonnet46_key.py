#!/usr/bin/env python3
"""
Validate an Anthropic API key and check Sonnet model access.

Usage examples:
  python test_sonnet46_key.py
  python test_sonnet46_key.py --api-key "<key>"
  python test_sonnet46_key.py --model "claude-sonnet-4-6"
  python test_sonnet46_key.py --skip-model-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _parse_error(resp: requests.Response) -> str:
    try:
        payload: dict[str, Any] = resp.json()
        err = payload.get("error")
        if isinstance(err, dict):
            err_type = err.get("type")
            err_msg = err.get("message")
            if err_type or err_msg:
                return f"{err_type or 'error'}: {err_msg or ''}".strip()
        return json.dumps(payload, ensure_ascii=True)
    except Exception:
        return resp.text.strip() or f"HTTP {resp.status_code}"


def check_key(api_key: str, timeout_s: float) -> tuple[bool, str]:
    url = f"{ANTHROPIC_BASE_URL}/models"
    try:
        resp = requests.get(url, headers=_headers(api_key), timeout=timeout_s)
    except requests.RequestException as exc:
        return False, f"Network/API error while checking key: {exc}"

    if resp.status_code == 200:
        return True, "Key accepted by Anthropic /v1/models."

    if resp.status_code in (401, 403):
        return False, f"Invalid or unauthorized API key ({resp.status_code}): {_parse_error(resp)}"

    return False, f"Key check failed ({resp.status_code}): {_parse_error(resp)}"


def check_model_access(api_key: str, model: str, timeout_s: float) -> tuple[bool, str]:
    url = f"{ANTHROPIC_BASE_URL}/messages"
    body = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with: OK"}],
    }
    try:
        resp = requests.post(url, headers=_headers(api_key), json=body, timeout=timeout_s)
    except requests.RequestException as exc:
        return False, f"Network/API error while checking model access: {exc}"

    if resp.status_code == 200:
        return True, f"Model access confirmed for '{model}'."

    if resp.status_code in (401, 403):
        return False, f"Unauthorized while calling model ({resp.status_code}): {_parse_error(resp)}"

    if resp.status_code == 400:
        return False, (
            f"Key is valid but model call failed for '{model}' (400). "
            f"Likely invalid model id or no access: {_parse_error(resp)}"
        )

    if resp.status_code == 429:
        return False, f"Rate limit / quota issue (429): {_parse_error(resp)}"

    return False, f"Model check failed ({resp.status_code}): {_parse_error(resp)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Anthropic Sonnet API key validity and model access.")
    parser.add_argument("--api-key", help="Anthropic API key. If omitted, reads ANTHROPIC_API_KEY from env.")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model id to test after key validation (default: claude-sonnet-4-6).",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds (default: 20).")
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Only validate key via /v1/models; skip /v1/messages model call.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to dotenv file to load before reading env vars (default: .env).",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)

    api_key = (args.api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if not api_key:
        print("ERROR: Missing API key. Provide --api-key or set ANTHROPIC_API_KEY.")
        return 2

    print("1) Checking API key with /v1/models ...")
    key_ok, key_msg = check_key(api_key, args.timeout)
    print(key_msg)
    if not key_ok:
        return 2

    if args.skip_model_check:
        print("Model check skipped.")
        return 0

    print(f"2) Checking model access with /v1/messages using model '{args.model}' ...")
    model_ok, model_msg = check_model_access(api_key, args.model, args.timeout)
    print(model_msg)
    if not model_ok:
        return 3

    print("SUCCESS: API key and model access are both valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
