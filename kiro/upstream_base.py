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
    3. Everything else → ``"kiro"``.

    The Codex check runs first and matches only bare, registered ids, so it
    never collides with Command Code's ``/``-qualified models or with arbitrary
    Kiro model names.

    IMPORTANT: this must be called on the RAW client model string, before any
    model-name normalization (normalize_model_name strips the provider prefix).

    Args:
        raw_model: The client-supplied model name, before normalization.

    Returns:
        Backend name: ``"chatgpt"``, ``"command_code"``, or ``"kiro"``.
    """
    if config.CHATGPT_ENABLED and _is_codex_model(raw_model):
        return "chatgpt"
    if config.COMMAND_CODE_ENABLED and "/" in raw_model:
        return "command_code"
    return "kiro"
