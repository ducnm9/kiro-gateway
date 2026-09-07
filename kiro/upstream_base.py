# -*- coding: utf-8 -*-

"""Upstream backend routing for the Kiro Gateway.

Kiro Gateway can proxy to multiple upstream providers (Kiro, Command Code,
ChatGPT/Codex). This module owns the routing decision: given a raw client model
name, decide which upstream backend should handle the request.
"""

from kiro import config


def _is_codex_model(raw_model: str) -> bool:
    """Return True if a raw model name maps to the ChatGPT (Codex) upstream.

    Codex model ids are bare names present in ``config.CHATGPT_MODEL_IDS``
    (e.g. ``gpt-5.5``). Bare names avoid conflict with Command Code, whose
    models are provider-qualified (contain ``/``).

    Args:
        raw_model: The client-supplied model name, before normalization.

    Returns:
        True if the model is a configured Codex model, else False.
    """
    return raw_model in config.CHATGPT_MODEL_IDS


def resolve_upstream(raw_model: str) -> str:
    """Resolve which upstream backend handles a raw model name.

    Routing rules (checked in order):
    1. ChatGPT/Codex: when ``CHATGPT_ENABLED`` and the model is a configured
       Codex model id (bare name in ``CHATGPT_MODEL_IDS``) → ``"chatgpt"``.
    2. Command Code: when ``COMMAND_CODE_ENABLED`` and the model is
       provider-qualified (contains ``/``) → ``"command_code"``.
    3. Everything else:
       - when ``KIRO_ENABLED`` → ``"kiro"`` (native passthrough, default).
       - when Kiro is disabled → ``"kiro_disabled"``. The caller must turn this
         into a clear client-facing error instead of routing to a non-existent
         Kiro backend (transparency: we never silently reroute a user's model
         to a different upstream).

    The Codex check runs first and matches only bare, registered ids, so it
    never collides with Command Code's ``/``-qualified models or with arbitrary
    Kiro model names.

    IMPORTANT: this must be called on the RAW client model string, before any
    model-name normalization (normalize_model_name strips the provider prefix).

    Args:
        raw_model: The client-supplied model name, before normalization.

    Returns:
        Backend name: ``"chatgpt"``, ``"command_code"``, ``"kiro"``, or
        ``"kiro_disabled"`` (unmatched model while the Kiro upstream is off).
    """
    if config.CHATGPT_ENABLED and _is_codex_model(raw_model):
        return "chatgpt"
    if config.COMMAND_CODE_ENABLED and "/" in raw_model:
        return "command_code"
    if config.KIRO_ENABLED:
        return "kiro"
    return "kiro_disabled"


def enabled_upstreams() -> list:
    """Return the list of currently enabled upstream provider names.

    Order reflects routing precedence and user-facing listing:
    ``["kiro", "command_code", "chatgpt"]`` filtered by their enable flags.

    Returns:
        A list of enabled upstream names (may be empty if all are disabled).
    """
    upstreams = []
    if config.KIRO_ENABLED:
        upstreams.append("kiro")
    if config.COMMAND_CODE_ENABLED:
        upstreams.append("command_code")
    if config.CHATGPT_ENABLED:
        upstreams.append("chatgpt")
    return upstreams


def kiro_disabled_error_message(raw_model: str) -> str:
    """Build a clear, actionable error message for an unroutable model.

    Used when ``resolve_upstream`` returns ``"kiro_disabled"``: the Kiro upstream
    is off and the requested model matched no other enabled upstream.

    Args:
        raw_model: The client-supplied model name that could not be routed.

    Returns:
        A user-friendly message explaining why the model is unavailable and how
        to fix it.
    """
    enabled = [u for u in enabled_upstreams()]
    if enabled:
        enabled_str = ", ".join(enabled)
        hint = (
            f"The Kiro upstream is disabled (KIRO_ENABLED=false). "
            f"Currently enabled upstream(s): {enabled_str}. "
            f"Use a model served by an enabled upstream, or set KIRO_ENABLED=true "
            f"to route this model to Kiro."
        )
    else:
        hint = (
            "No upstream is enabled. Enable at least one of KIRO_ENABLED, "
            "COMMAND_CODE_ENABLED, or CHATGPT_ENABLED."
        )
    return f"Model '{raw_model}' is not available. {hint}"
