"""Patch E2B Python SDK so ``POST /sandboxes`` succeeds are parsed on HTTP 200.

Official OpenAPI client only unmarshals on **201**; many private deployments return **200**,
which yields ``parsed is None`` and ``Exception("Body of the request is None")``.

Use from minimal scripts::

    from siirl.execution.rollout.agentflow.swe.environment.e2b_http200_compat import apply_e2b_http200_sdk_patch
    apply_e2b_http200_sdk_patch()
    from e2b import Sandbox
    Sandbox.beta_create(...)
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

_done = False


def _response_status_code(response: Any) -> int:
    code = getattr(response, "status_code", None)
    if code is None:
        return 0
    return int(getattr(code, "value", code))


def _http_json_body_preview(response: Any, limit: int = 800) -> str:
    raw = getattr(response, "content", b"") or b""
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _parse_create_sandbox_payload(response: Any) -> dict[str, Any] | None:
    raw = getattr(response, "content", b"") or b""
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _coerce_sandbox_create_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    inner = out.get("data")
    if isinstance(inner, dict) and "sandboxID" not in out and "sandbox_id" not in out:
        out = {**inner}
    aliases = (
        ("sandbox_id", "sandboxID"),
        ("sandboxId", "sandboxID"),
        ("envd_version", "envdVersion"),
        ("client_id", "clientID"),
        ("template_id", "templateID"),
        ("envd_access_token", "envdAccessToken"),
        ("traffic_access_token", "trafficAccessToken"),
    )
    for alt, canon in aliases:
        if canon not in out and alt in out:
            out[canon] = out[alt]
    return out


def _load_sandbox_response_model() -> Any | None:
    for name in (
        "e2b.api.client.models.sandbox",
        "e2b.api.client.models",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        cls = getattr(mod, "Sandbox", None)
        if cls is not None and hasattr(cls, "from_dict"):
            return cls
    return None


def apply_e2b_http200_sdk_patch(log: logging.Logger | None = None) -> None:
    """Idempotent: monkey-patch ``post_sandboxes._parse_response`` for HTTP 200 + JSON body shapes."""
    global _done
    if _done:
        return
    _log = log or logging.getLogger(__name__)

    model = _load_sandbox_response_model()
    if model is None:
        _log.warning(
            "e2b_http200_compat: could not import Sandbox API model; skipping POST /sandboxes patch."
        )
        _done = True
        return

    mod_names = ("e2b.api.client.api.sandboxes.post_sandboxes",)
    patched = False
    for mod_name in mod_names:
        try:
            post_mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if not hasattr(post_mod, "_parse_response"):
            continue

        orig = post_mod._parse_response

        def _parse_response(
            *,
            client: Any,
            response: Any,
            _orig: Any = orig,
            _model: Any = model,
        ) -> Any:
            if _response_status_code(response) == 200:
                payload = _parse_create_sandbox_payload(response)
                if payload is None:
                    _log.warning(
                        "E2B POST /sandboxes HTTP 200 but body empty or non-JSON. Prefix: %r",
                        _http_json_body_preview(response, 400),
                    )
                else:
                    coerced = _coerce_sandbox_create_dict(payload)
                    try:
                        return _model.from_dict(coerced)
                    except Exception as e:
                        _log.warning(
                            "E2B POST /sandboxes HTTP 200 but Sandbox.from_dict failed (%s). Prefix: %r",
                            e,
                            _http_json_body_preview(response, 600),
                        )
            return _orig(client=client, response=response)

        post_mod._parse_response = _parse_response  # type: ignore[assignment]
        patched = True

    _done = True
    if patched:
        _log.info("e2b_http200_compat: patched POST /sandboxes for HTTP 200 / alternate JSON.")
    else:
        _log.warning(
            "e2b_http200_compat: module %r not found; patch not applied.",
            mod_names[0],
        )
