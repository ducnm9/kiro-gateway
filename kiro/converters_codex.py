# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Request converters for the ChatGPT (Codex) upstream.

Clean-room implementation of the Codex Responses API request envelope, based
solely on observed wire behavior. Translates OpenAI Chat Completions and
Anthropic Messages requests into the Codex Responses format.

Key Codex Responses requirements handled here:
- ``input`` is an array of message items; each item has a ``role`` and a list
  of typed content parts (``input_text``/``input_image`` for user/developer,
  ``output_text`` for assistant).
- System prompts are carried as ``role="developer"`` (kept in the cacheable
  prefix), never as ``role="system"``.
- ``store`` is always false and ``stream`` is always true.
- ``reasoning.effort`` is set from the client's reasoning hint (default low).
- Tools are flattened to Responses function form ``{type, name, description,
  parameters}``; a ``tool_choice`` referencing an unknown tool is dropped.
- Only an allowlisted set of top-level fields is emitted; anything else is
  stripped to avoid upstream "routing_unsupported" errors.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.converters_anthropic import extract_system_prompt
from kiro.converters_core import extract_text_content
from kiro.converters_openai import convert_openai_tools_to_unified
from kiro.models_anthropic import AnthropicMessagesRequest
from kiro.models_openai import ChatCompletionRequest


# Default instructions injected when the client provides none. Kept short and
# neutral; the Codex backend requires a non-empty instructions field.
CODEX_DEFAULT_INSTRUCTIONS: str = "You are a helpful coding assistant."

# Reasoning effort levels accepted by the Codex Responses API.
_CODEX_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")

# Top-level fields accepted by the Codex Responses API. Anything else is stripped.
_RESPONSES_ALLOWLIST = {
    "model", "input", "instructions", "tools", "tool_choice",
    "stream", "store", "reasoning", "include",
}


def _normalize_effort(value: Optional[str]) -> str:
    """Normalize a reasoning-effort hint to a Codex-accepted level.

    Args:
        value: Client-provided effort hint (may be None or non-standard).

    Returns:
        A valid Codex effort level; unknown values map to "low", and the
        aggressive "ultra" alias maps to "xhigh".
    """
    if not value:
        return "low"
    v = str(value).lower()
    if v in _CODEX_EFFORT_LEVELS:
        return v
    if v == "ultra" or v == "max":
        return "xhigh"
    return "low"


def _flatten_tools(unified_tools: Optional[list]) -> List[Dict[str, Any]]:
    """Flatten unified tools into Codex Responses function-tool form.

    Args:
        unified_tools: List of UnifiedTool, or None.

    Returns:
        A list of Responses function-tool dicts (possibly empty).
    """
    tools: List[Dict[str, Any]] = []
    for tool in unified_tools or []:
        name = (tool.name or "").strip()
        if not name:
            continue
        parameters = tool.input_schema or {"type": "object", "properties": {}}
        entry: Dict[str, Any] = {
            "type": "function",
            "name": name[:128],
            "parameters": parameters,
        }
        if tool.description:
            entry["description"] = tool.description
        tools.append(entry)
    return tools


def _apply_tool_choice(payload: Dict[str, Any], tool_choice: Any, valid_names: set) -> None:
    """Attach a validated tool_choice to the payload, or drop it.

    A function tool_choice referencing an unknown tool name is dropped so the
    upstream does not reject the request.

    Args:
        payload: The Codex payload being built (mutated in place).
        tool_choice: The client-provided tool_choice (str or dict), or None.
        valid_names: Set of valid function tool names.
    """
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        payload["tool_choice"] = tool_choice
        return
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = ((tool_choice.get("function") or {}).get("name")
                    or tool_choice.get("name") or "")
            if name and name in valid_names:
                payload["tool_choice"] = tool_choice
            # else: drop unknown function tool_choice
        else:
            payload["tool_choice"] = tool_choice


def _finalize_payload(
    model: str,
    input_items: List[Dict[str, Any]],
    instructions: str,
    tools: List[Dict[str, Any]],
    tool_choice: Any,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    """Assemble the final Codex Responses payload with allowlist filtering.

    Args:
        model: The upstream model id.
        input_items: The Codex ``input`` array.
        instructions: System/instructions text (developer-role prefix).
        tools: Flattened function tools.
        tool_choice: Optional client tool_choice.
        reasoning_effort: Optional reasoning-effort hint.

    Returns:
        A Codex Responses request dict containing only allowlisted fields.
    """
    if not input_items:
        # Codex rejects an empty input array.
        input_items = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "..."}],
        }]

    effort = _normalize_effort(reasoning_effort)
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "instructions": instructions or CODEX_DEFAULT_INSTRUCTIONS,
        "stream": True,
        "store": False,
        "reasoning": {"effort": effort, "summary": "auto"},
    }
    if effort != "none":
        payload["include"] = ["reasoning.encrypted_content"]

    if tools:
        payload["tools"] = tools
        valid_names = {t["name"] for t in tools}
        _apply_tool_choice(payload, tool_choice, valid_names)

    # Allowlist filter — strip anything unexpected.
    return {k: v for k, v in payload.items() if k in _RESPONSES_ALLOWLIST}


def _user_content_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI user/tool content into Codex input parts.

    Text becomes ``input_text``; image_url parts become ``input_image``. Remote
    image URLs are passed through as-is here (inlining to base64 is handled by
    the request layer before dispatch); data URLs are forwarded directly.

    Args:
        content: OpenAI message content (str or list of parts).

    Returns:
        A list of Codex input content parts.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}] if content else []

    parts: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append({"type": "input_text", "text": item})
                continue
            itype = item.get("type") if isinstance(item, dict) else None
            if itype == "text":
                text = item.get("text", "")
                if text:
                    parts.append({"type": "input_text", "text": text})
            elif itype == "image_url":
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    parts.append({"type": "input_image", "image_url": url})
    else:
        text = extract_text_content(content)
        if text:
            parts.append({"type": "input_text", "text": text})
    return parts


def build_codex_payload(request_data: ChatCompletionRequest) -> Dict[str, Any]:
    """Build a Codex Responses request from an OpenAI Chat Completions request.

    System/developer messages are concatenated into ``instructions``; user and
    tool messages become ``input`` items with ``input_text``/``input_image``
    parts; assistant messages become ``output_text`` items. Tool calls in
    assistant messages and tool results are flattened into text to keep the
    Codex ``input`` well-formed without server-generated item ids.

    Args:
        request_data: OpenAI chat completion request.

    Returns:
        A Codex Responses request dict.
    """
    system_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []

    for msg in request_data.messages:
        role = msg.role
        if role in ("system", "developer"):
            text = extract_text_content(msg.content)
            if text:
                system_parts.append(text)
        elif role == "assistant":
            text = extract_text_content(msg.content)
            # Represent tool calls as readable text (Codex cannot resolve
            # client-side tool_call ids when store=false).
            if msg.tool_calls:
                for call in msg.tool_calls:
                    fn = (call.get("function") if isinstance(call, dict) else None) or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    text = (text + f"\n[called tool {name} with {args}]").strip()
            if text:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
        elif role == "tool":
            text = extract_text_content(msg.content)
            call_id = msg.tool_call_id or ""
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"[tool result {call_id}]\n{text}"}],
            })
        else:  # user
            parts = _user_content_parts(msg.content)
            if parts:
                input_items.append({"type": "message", "role": "user", "content": parts})

    instructions = "\n".join(system_parts) if system_parts else CODEX_DEFAULT_INSTRUCTIONS
    tools = _flatten_tools(convert_openai_tools_to_unified(request_data.tools))

    reasoning_effort = getattr(request_data, "reasoning_effort", None)

    return _finalize_payload(
        model=request_data.model,
        input_items=input_items,
        instructions=instructions,
        tools=tools,
        tool_choice=getattr(request_data, "tool_choice", None),
        reasoning_effort=reasoning_effort,
    )


def _get(block: Any, name: str, default: Any = None) -> Any:
    """Get an attribute from a dict or Pydantic model block."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _anthropic_content_parts(content: Any, is_assistant: bool) -> List[Dict[str, Any]]:
    """Convert Anthropic content blocks into Codex input parts.

    Args:
        content: Anthropic message content (str or list of blocks).
        is_assistant: Whether the message role is assistant (→ output_text).

    Returns:
        A list of Codex input content parts.
    """
    text_type = "output_text" if is_assistant else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []

    parts: List[Dict[str, Any]] = []
    for block in content or []:
        btype = _get(block, "type", "")
        if btype == "text":
            text = _get(block, "text", "") or ""
            if text:
                parts.append({"type": text_type, "text": text})
        elif btype == "image" and not is_assistant:
            source = _get(block, "source")
            stype = _get(source, "type", "") if source is not None else ""
            if stype == "base64":
                media_type = _get(source, "media_type", "")
                data = _get(source, "data", "")
                if data:
                    parts.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{data}",
                    })
            elif stype == "url":
                url = _get(source, "url", "")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
        elif btype == "tool_use":
            name = _get(block, "name", "") or ""
            tool_input = _get(block, "input", {}) or {}
            args = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
            parts.append({"type": text_type, "text": f"[called tool {name} with {args}]"})
        elif btype == "tool_result":
            tool_use_id = _get(block, "tool_use_id", "") or ""
            result = _get(block, "content", "") or ""
            if isinstance(result, list):
                result = extract_text_content(result)
            # Anthropic tool_result appears in a user-role message.
            parts.append({"type": "input_text", "text": f"[tool result {tool_use_id}]\n{result}"})
        # thinking / tool_reference blocks are skipped
    return parts


def build_codex_payload_anthropic(request_data: AnthropicMessagesRequest) -> Dict[str, Any]:
    """Build a Codex Responses request from an Anthropic Messages request.

    The Anthropic system prompt becomes ``instructions``; message blocks are
    mapped to Codex input parts (text, images, flattened tool_use/tool_result).

    Args:
        request_data: Anthropic Messages request.

    Returns:
        A Codex Responses request dict.
    """
    instructions = extract_system_prompt(request_data.system) or CODEX_DEFAULT_INSTRUCTIONS

    input_items: List[Dict[str, Any]] = []
    for msg in request_data.messages:
        is_assistant = msg.role == "assistant"
        parts = _anthropic_content_parts(msg.content, is_assistant)
        if parts:
            input_items.append({
                "type": "message",
                "role": "assistant" if is_assistant else "user",
                "content": parts,
            })

    tools: List[Dict[str, Any]] = []
    if request_data.tools:
        for tool in request_data.tools:
            if getattr(tool, "type", None) is not None:
                logger.warning(f"Skipping server-side tool for Codex (type set): {getattr(tool, 'name', '?')}")
                continue
            name = (tool.name or "").strip()
            if not name:
                continue
            entry: Dict[str, Any] = {
                "type": "function",
                "name": name[:128],
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }
            if tool.description:
                entry["description"] = tool.description
            tools.append(entry)

    return _finalize_payload(
        model=request_data.model,
        input_items=input_items,
        instructions=instructions,
        tools=tools,
        tool_choice=None,
        reasoning_effort=None,
    )


async def inline_remote_images(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inline remote image URLs in a Codex payload as base64 data URLs.

    The Codex backend cannot fetch remote images, so any ``input_image`` part
    whose ``image_url`` is a non-data URL is fetched and replaced with a
    ``data:<mime>;base64,<...>`` URL. ``data:`` URLs are left untouched.

    This is fail-safe: if a fetch fails, the original URL is kept (the request
    proceeds and the upstream decides), and the payload is never corrupted.

    Args:
        payload: A Codex Responses payload (mutated in place and returned).

    Returns:
        The same payload with remote images inlined where possible.
    """
    import base64 as _base64

    import httpx

    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return payload

    # Collect (part) references that need fetching.
    pending: List[Dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "input_image":
                continue
            url = part.get("image_url")
            if isinstance(url, str) and url and not url.startswith("data:"):
                pending.append(part)

    if not pending:
        return payload

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for part in pending:
            url = part["image_url"]
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning(f"Codex image inline skipped (HTTP {resp.status_code}): {url[:80]}")
                    continue
                mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                b64 = _base64.b64encode(resp.content).decode("ascii")
                part["image_url"] = f"data:{mime};base64,{b64}"
            except (httpx.HTTPError, ValueError) as e:
                # Fail-safe: keep the original URL; never corrupt the payload.
                logger.warning(f"Codex image inline failed ({type(e).__name__}): {url[:80]}")

    return payload
