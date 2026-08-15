# -*- coding: utf-8 -*-

"""Upstream backend routing for the Kiro Gateway.

Kiro Gateway can proxy to multiple upstream providers (Kiro, Command Code).
This module owns the routing decision: given a raw client model name, decide
which upstream backend should handle the request.
"""

from kiro import config


def resolve_upstream(raw_model: str) -> str:
    """Resolve which upstream backend handles a raw model name.

    Routing rule: Command Code models are provider-qualified
    (e.g. ``deepseek/deepseek-v4-pro``) and contain ``/``; Kiro models are
    bare names (e.g. ``claude-haiku-4.5``). A model containing ``/`` routes to
    Command Code, everything else to Kiro. Disabled entirely when
    ``COMMAND_CODE_ENABLED`` is false.

    IMPORTANT: this must be called on the RAW client model string, before any
    model-name normalization (normalize_model_name strips the provider prefix).

    Args:
        raw_model: The client-supplied model name, before normalization.

    Returns:
        Backend name: ``"command_code"`` or ``"kiro"``.
    """
    if config.COMMAND_CODE_ENABLED and "/" in raw_model:
        return "command_code"
    return "kiro"
