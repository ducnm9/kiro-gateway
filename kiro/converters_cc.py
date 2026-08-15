# -*- coding: utf-8 -*-

"""OpenAI to Command Code request converter.

Clean-room implementation of the Command Code request envelope, based solely
on observed wire behavior. No source was copied from third-party tools.
"""

import datetime
import json
from typing import Any, Dict, List

from loguru import logger

from kiro.config import COMMAND_CODE_MAX_TOKENS, COMMAND_CODE_MAX_TOKENS_CAP
from kiro.converters_anthropic import extract_system_prompt
from kiro.converters_core import extract_text_content
from kiro.converters_openai import convert_openai_tools_to_unified
from kiro.models_anthropic import AnthropicMessagesRequest
from kiro.models_openai import ChatCompletionRequest


def _build_config() -> Dict[str, Any]:
    """Build the fixed config block required by the Command Code envelope."""
    return {
        "workingDir": "/",
        "date": datetime.date.today().isoformat(),
        "environment": "Python proxy",
        "structure": [],
        "isGitRepo": False,
        "currentBranch": "",
        "mainBranch": "",
        "gitStatus": "",
        "recentCommits": [],
    }


def build_cc_payload(request_data: ChatCompletionRequest) -> Dict[str, Any]:
    """Build the Command Code request envelope from an OpenAI request.

    Command Code nests all generation parameters under ``params``. The
    ``config``/``memory``/``taste``/``skills`` fields are fixed boilerplate
    required by the upstream. The upstream is streaming-only, so ``stream``
    is always true. Tools are forwarded in Command Code's flat function form.

    Args:
        request_data: OpenAI chat completion request.

    Returns:
        Command Code request envelope as a dict.
    """
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []

    for msg in request_data.messages:
        if msg.role in ("system", "developer"):
            text = extract_text_content(msg.content)
            if text:
                system_parts.append(text)
        elif msg.role == "tool":
            text = extract_text_content(msg.content)
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        else:
            text = extract_text_content(msg.content)
            messages.append({"role": msg.role, "content": [{"type": "text", "text": text}]})

    max_tokens = request_data.max_tokens or request_data.max_completion_tokens
    if not max_tokens or max_tokens <= 0:
        max_tokens = COMMAND_CODE_MAX_TOKENS
    max_tokens = min(max_tokens, COMMAND_CODE_MAX_TOKENS_CAP)

    tools: List[Dict[str, Any]] = []
    unified_tools = convert_openai_tools_to_unified(request_data.tools) or []
    for tool in unified_tools:
        schema = tool.input_schema or {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": schema,
        })

    params: Dict[str, Any] = {
        "model": request_data.model,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if system_parts:
        params["system"] = "\n".join(system_parts)

    return {
        "config": _build_config(),
        "memory": "",
        "taste": "",
        "skills": None,
        "permissionMode": "standard",
        "params": params,
    }


def _get(block: Any, name: str, default: Any = None) -> Any:
    """Get an attribute from a dict or Pydantic model block."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def build_cc_payload_anthropic(request_data: AnthropicMessagesRequest) -> Dict[str, Any]:
    """Build the Command Code envelope from an Anthropic Messages request.

    Maps Anthropic content blocks to CC messages (tool_use → flattened text,
    tool_result → flattened text, image → image part), forwards tools with
    ``input_schema``, and clamps ``max_tokens``.

    Args:
        request_data: Anthropic Messages request.

    Returns:
        Command Code request envelope as a dict.
    """
    system_prompt = extract_system_prompt(request_data.system)

    messages: List[Dict[str, Any]] = []
    for msg in request_data.messages:
        role = msg.role
        if isinstance(msg.content, str):
            messages.append({"role": role, "content": [{"type": "text", "text": msg.content}]})
            continue

        parts: List[Dict[str, Any]] = []
        for block in msg.content:
            block_type = _get(block, "type", "")
            if block_type == "text":
                parts.append({"type": "text", "text": _get(block, "text", "") or ""})
            elif block_type == "image":
                source = _get(block, "source")
                source_type = _get(source, "type", "") if source is not None else ""
                if source_type == "base64":
                    parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _get(source, "media_type", ""),
                            "data": _get(source, "data", ""),
                        },
                    })
                elif source_type == "url":
                    logger.warning(
                        f"URL-based images are not supported by Command Code, skipping: "
                        f"{_get(source, 'url', '')[:80]}"
                    )
            elif block_type == "tool_use":
                name = _get(block, "name", "") or ""
                call_id = _get(block, "id", "") or ""
                tool_input = _get(block, "input", {}) or {}
                args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
                text = f"Assistant requested tool {name} ({call_id}) with arguments: {args_str}"
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                })
            elif block_type == "tool_result":
                tool_use_id = _get(block, "tool_use_id", "") or ""
                result_content = _get(block, "content", "") or ""
                if isinstance(result_content, list):
                    result_content = extract_text_content(result_content)
                text = f"Tool result ({tool_use_id}):\n{result_content}"
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                })
            # thinking and tool_reference blocks are skipped

        if parts:
            messages.append({"role": role, "content": parts})

    tools: List[Dict[str, Any]] = []
    if request_data.tools:
        for tool in request_data.tools:
            if tool.type is not None:
                logger.warning(f"Skipping server-side tool '{tool.name}' (type={tool.type})")
                continue
            tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema or {"type": "object", "properties": {}},
            })

    max_tokens = min(request_data.max_tokens, COMMAND_CODE_MAX_TOKENS_CAP)

    params: Dict[str, Any] = {
        "model": request_data.model,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if system_prompt:
        params["system"] = system_prompt

    return {
        "config": _build_config(),
        "memory": "",
        "taste": "",
        "skills": None,
        "permissionMode": "standard",
        "params": params,
    }
