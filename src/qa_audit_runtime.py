"""Shared, auditable QA requests. Importing this module never creates a client.

Runtime policy is versioned separately from the unchanged QA prompt protocol.
No SDK-internal retries; both experiment modes use the policy below.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone

RUNTIME_VERSION = "qa_audit_runtime_v1"
GENERATION = {
    "temperature": 0,
    "max_tokens": 600,
    "response_format": {"type": "json_object"},
    "extra_body": {"thinking": {"type": "disabled"}},
}
POLICY = {"max_attempts": 3, "backoff_seconds": [2, 4],
          "sdk_max_retries": 0, "timeout_seconds": 60.0}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def object_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class SchemaError(ValueError):
    """Only QA schema-validation failures should be wrapped in this class."""


def request_payload(model, messages):
    return {"model": model, "messages": copy.deepcopy(messages), **copy.deepcopy(GENERATION)}


def plain_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return copy.deepcopy(usage)
    return {key: getattr(usage, key, None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def usage_summary(attempts):
    """Missing usage is unknown, never a claim that the failed call was free."""
    result = {"attempts_with_usage": sum(a.get("usage") is not None for a in attempts),
              "attempt_count": len(attempts)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(a.get("usage") or {}).get(key) for a in attempts]
        known = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
        result[key + "_known_sum"] = sum(known) if known else None
        result[key + "_complete"] = len(known) == len(attempts)
    return result


def classify_exception(exc):
    if isinstance(exc, (json.JSONDecodeError, SchemaError)):
        return "invalid_json_or_schema", True, False
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return "transient_api_error", True, False
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, ConnectionError)) or name in {"APITimeoutError", "APIConnectionError"}:
        return "connection_or_timeout", True, False
    # Auth/permission/configuration errors halt the batch instead of repeated
    # attempts or retries through another endpoint. Unknown bugs also halt.
    return "fatal_api_or_runtime_error", False, True


def execute_question(client, request, validate, emit, *, clock=time.perf_counter,
                     sleep=time.sleep, policy=None):
    """Send one request using the shared retry policy and an injected validator.

    emit must persist events immediately. Log failures are not swallowed.
    Full requests are stored by the runner once, before this function is called.
    Returned raw text and usage include failed JSON/schema responses too.
    """
    policy = copy.deepcopy(POLICY if policy is None else policy)
    if policy["max_attempts"] < 1 or len(policy["backoff_seconds"]) < policy["max_attempts"] - 1:
        raise ValueError("Invalid retry policy")
    whole_start = clock()
    attempts, backoff_actual = [], 0.0
    answer, fatal = None, False
    request_hash = object_hash(request)
    for number in range(1, policy["max_attempts"] + 1):
        event = {"attempt": number, "request_sha256": request_hash, "started_at_utc": utc_now()}
        emit({"event": "attempt_started", **event})
        attempt_start = clock()
        event.update(usage=None, response_id=None, response_model=None,
                     system_fingerprint=None, finish_reason=None, raw_response_text=None)
        retry = False
        api_finished = None
        try:
            response = client.chat.completions.create(**copy.deepcopy(request))
            api_finished = clock()
            event.update(response_id=getattr(response, "id", None),
                         response_model=getattr(response, "model", None),
                         system_fingerprint=getattr(response, "system_fingerprint", None),
                         usage=plain_usage(response))
            if not response.choices:
                raise SchemaError("No response choices")
            choice = response.choices[0]
            event["finish_reason"] = getattr(choice, "finish_reason", None)
            event["raw_response_text"] = getattr(choice.message, "content", None)
            if event["finish_reason"] in {"length", "content_filter"}:
                # A repeated identical output cap is not a truncation remedy.
                event.update(status="failed", error_kind="truncated_or_filtered", retryable=False)
            else:
                content = event["raw_response_text"]
                if not isinstance(content, str) or not content.strip():
                    raise SchemaError("No response text")
                answer = validate(json.loads(content))
                event.update(status="succeeded", retryable=False)
        except Exception as exc:
            kind, retry, fatal = classify_exception(exc)
            event.update(status="failed", error_kind=kind,
                         error_class=type(exc).__name__, http_status=getattr(exc, "status_code", None),
                         retryable=retry)
            # Do not serialize exception messages/HTTP headers: they can carry
            # credentials or provider internals. Class/status are sufficient.
        event["api_seconds"] = (api_finished if api_finished is not None else clock()) - attempt_start
        event["attempt_seconds"] = clock() - attempt_start
        event["finished_at_utc"] = utc_now()
        attempts.append(event)
        emit({"event": "attempt_finished", **event})
        if answer is not None or not retry or number == policy["max_attempts"]:
            break
        delay = policy["backoff_seconds"][number - 1]
        emit({"event": "backoff_started", "after_attempt": number, "planned_seconds": delay})
        wait_start = clock()
        sleep(delay)
        elapsed = clock() - wait_start
        backoff_actual += elapsed
        emit({"event": "backoff_finished", "after_attempt": number, "actual_seconds": elapsed})
    return {
        "status": "succeeded" if answer is not None else "failed",
        "halt_batch": fatal, "model_response": answer, "attempts": attempts,
        "attempt_count": len(attempts), "usage_all_attempts": usage_summary(attempts),
        "timing": {
            "generation_total_seconds": clock() - whole_start,
            "api_seconds_sum": sum(a["api_seconds"] for a in attempts),
            "successful_api_seconds": next((a["api_seconds"] for a in attempts if a["status"] == "succeeded"), None),
            "retry_backoff_seconds": backoff_actual,
        },
    }
