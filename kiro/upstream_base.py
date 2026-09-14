# -*- coding: utf-8 -*-

"""Upstream backend routing for the Kiro Gateway.

Kiro Gateway can proxy to multiple upstream providers (Kiro, Command Code,
Antigravity). This module owns the routing decision: given a raw client model
name, decide which upstream backend should handle the request.
"""

from kiro import config


def resolve_upstream(raw_model: str) -> str:
    """Resolve which upstream backend handles a raw model name.

    Routing rules (evaluated in order):
    1. Models prefixed ``antigravity/`` route to the Antigravity backend
       (Google Cloud Code Assist) when ``ANTIGRAVITY_ENABLED`` is true.
    2. Models containing ``/`` (provider-qualified, e.g. ``deepseek/deepseek-v4-pro``)
       route to Command Code when ``COMMAND_CODE_ENABLED`` is true.
    3. Everything else routes to Kiro (native passthrough).

    IMPORTANT: this must be called on the RAW client model string, before any
    model-name normalization (normalize_model_name strips the provider prefix).

    Args:
        raw_model: The client-supplied model name, before normalization.

    Returns:
        Backend name: ``"antigravity"``, ``"command_code"``, or ``"kiro"``.
    """
    if config.ANTIGRAVITY_ENABLED and raw_model.startswith("antigravity/"):
        return "antigravity"
    if config.COMMAND_CODE_ENABLED and "/" in raw_model:
        return "command_code"
    return "kiro"
