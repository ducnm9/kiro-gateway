# -*- coding: utf-8 -*-

"""OpenAI/Anthropic to Antigravity (Gemini) request converters.

Clean-room implementation of the Cloud Code Assist request envelope, based
solely on observed wire behavior. Converts OpenAI and Anthropic chat formats
to the Gemini-style request structure used by Cloud Code Assist.

Key differences from standard Gemini API:
- Request is wrapped in an envelope with project, model, requestType, userAgent
- Tools use parametersJsonSchema (Gemini) or parameters (Claude/GPT-OSS)
- Thinking effort is controlled via generationConfig.thinkingConfig for Gemini 3.7
"""

import json
import secrets
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import ANTIGRAVITY_MAX_TOKENS, ANTIGRAVITY_MAX_TOKENS_CAP
from kiro.converters_core import extract_text_content
from kiro.models_anthropic import AnthropicMessagesRequest
from kiro.models_openai import ChatCompletionRequest
from kiro.upstream_antigravity import AntigravityBackend


# ==============================================================================
# Constants
# ==============================================================================

# Tool schema fields allowed by the Cloud Code Assist custom-tool bridge
# for Claude/GPT-OSS models (Protobuf Schema subset).
_CUSTOM_TOOL_SCHEMA_ALLOW = frozenset([
    "type", "description", "properties", "required", "items", "enum",
])


# ==============================================================================
# OpenAI → Antigravity Converter
# ==============================================================================


def build_antigravity_payload(
    request_data: ChatCompletionRequest,
    runtime_model: str,
    project_id: str,
    thinking_effort: str = "off",
    use_legacy_parameters: bool = False,
) -> Dict[str, Any]:
    """Build the Antigravity request envelope from an OpenAI chat completion request.

    Converts OpenAI message format to Gemini contents, extracts system prompt,
    converts tools, and wraps everything in the Cloud Code Assist envelope.

    Args:
        request_data: OpenAI chat completion request.
        runtime_model: Resolved Antigravity runtime model ID.
        project_id: Cloud Code Assist project ID.
        thinking_effort: Thinking effort level (off, low, medium, high).
        use_legacy_parameters: True for Claude/GPT-OSS (strict schema subset).

    Returns:
        Complete Antigravity request envelope as a dict.
    """
    system_parts: List[Dict[str, Any]] = []
    contents: List[Dict[str, Any]] = []

    # Convert messages.
    for msg in request_data.messages:
        if msg.role in ("system", "developer"):
            text = extract_text_content(msg.content)
            if text:
                system_parts.append({"text": text})

        elif msg.role == "assistant":
            parts = _convert_assistant_parts(msg)
            _append_turn(contents, "model", parts)

        elif msg.role == "tool":
            # Tool results become functionResponse under user role.
            part = _convert_tool_result_openai(msg)
            _append_turn(contents, "user", [part])

        else:  # user
            parts = _convert_user_parts_openai(msg)
            _append_turn(contents, "user", parts)

    # Ensure first turn is user (Gemini requirement).
    contents = _ensure_first_turn_is_user(contents)

    # Build the inner request body.
    request_body: Dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {
            "role": "user",
            "parts": system_parts if system_parts else [{"text": "You are a helpful assistant."}],
        },
    }

    # Generation config.
    gen_config = _build_generation_config(
        max_tokens=request_data.max_tokens or request_data.max_completion_tokens,
        temperature=request_data.temperature,
        runtime_model=runtime_model,
        thinking_effort=thinking_effort,
    )
    if gen_config:
        request_body["generationConfig"] = gen_config

    # Tools.
    tools = _convert_openai_tools(request_data.tools, use_legacy_parameters)
    if tools:
        request_body["tools"] = tools
        tool_config = _map_tool_choice_openai(request_data.tool_choice, runtime_model)
        if tool_config:
            request_body["toolConfig"] = tool_config

    # Wrap in envelope.
    return {
        "project": project_id,
        "model": runtime_model,
        "request": request_body,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": _generate_request_id(),
    }


# ==============================================================================
# Anthropic → Antigravity Converter
# ==============================================================================


def build_antigravity_payload_anthropic(
    request_data: AnthropicMessagesRequest,
    runtime_model: str,
    project_id: str,
    thinking_effort: str = "off",
    use_legacy_parameters: bool = False,
) -> Dict[str, Any]:
    """Build the Antigravity request envelope from an Anthropic Messages request.

    Converts Anthropic content blocks to Gemini parts, extracts system prompt,
    converts tools, and wraps in the Cloud Code Assist envelope.

    Args:
        request_data: Anthropic Messages request.
        runtime_model: Resolved Antigravity runtime model ID.
        project_id: Cloud Code Assist project ID.
        thinking_effort: Thinking effort level.
        use_legacy_parameters: True for Claude/GPT-OSS.

    Returns:
        Complete Antigravity request envelope as a dict.
    """
    # Extract system prompt.
    system_parts: List[Dict[str, Any]] = []
    system_text = _extract_anthropic_system(request_data.system)
    if system_text:
        system_parts.append({"text": system_text})

    # Convert messages.
    contents: List[Dict[str, Any]] = []
    for msg in request_data.messages:
        role = msg.role
        if isinstance(msg.content, str):
            gemini_role = "model" if role == "assistant" else "user"
            _append_turn(contents, gemini_role, [{"text": msg.content}])
            continue

        parts: List[Dict[str, Any]] = []
        for block in msg.content:
            block_type = _get(block, "type", "")

            if block_type == "text":
                text = _get(block, "text", "")
                if text:
                    parts.append({"text": text})

            elif block_type == "thinking":
                thinking_text = _get(block, "thinking", "")
                if thinking_text:
                    part: Dict[str, Any] = {"thought": True, "text": thinking_text}
                    signature = _get(block, "signature")
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)

            elif block_type == "image":
                image_part = _convert_image_block(block)
                if image_part:
                    parts.append(image_part)

            elif block_type == "tool_use":
                # Assistant's tool call → functionCall.
                name = _get(block, "name", "")
                tool_id = _get(block, "id", "")
                input_data = _get(block, "input", {})
                if name:
                    fc_part: Dict[str, Any] = {
                        "functionCall": {
                            "name": name,
                            "args": input_data if isinstance(input_data, dict) else {},
                        }
                    }
                    if tool_id and _needs_tool_call_id(runtime_model):
                        fc_part["functionCall"]["id"] = _sanitize_tool_call_id(tool_id, name)
                    parts.append(fc_part)

            elif block_type == "tool_result":
                # Tool result → functionResponse under user role.
                tool_name = _get(block, "name", "") or "tool"
                tool_id = _get(block, "tool_use_id", "")
                content_blocks = _get(block, "content", [])
                is_error = _get(block, "is_error", False)
                response_text = _extract_tool_result_text(content_blocks)
                fr_part: Dict[str, Any] = {
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"error": response_text} if is_error else {"output": response_text},
                    }
                }
                if tool_id and _needs_tool_call_id(runtime_model):
                    fr_part["functionResponse"]["id"] = _sanitize_tool_call_id(tool_id, tool_name)
                parts.append(fr_part)

        if parts:
            gemini_role = "model" if role == "assistant" else "user"
            _append_turn(contents, gemini_role, parts)

    # Ensure first turn is user.
    contents = _ensure_first_turn_is_user(contents)

    # Build inner request body.
    request_body: Dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {
            "role": "user",
            "parts": system_parts if system_parts else [{"text": "You are a helpful assistant."}],
        },
    }

    # Generation config.
    gen_config = _build_generation_config(
        max_tokens=request_data.max_tokens,
        temperature=getattr(request_data, "temperature", None),
        runtime_model=runtime_model,
        thinking_effort=thinking_effort,
    )
    if gen_config:
        request_body["generationConfig"] = gen_config

    # Tools.
    tools = _convert_anthropic_tools(request_data.tools, use_legacy_parameters)
    if tools:
        request_body["tools"] = tools
        tool_choice = getattr(request_data, "tool_choice", None)
        tool_config = _map_tool_choice_anthropic(tool_choice, runtime_model)
        if tool_config:
            request_body["toolConfig"] = tool_config

    # Wrap in envelope.
    return {
        "project": project_id,
        "model": runtime_model,
        "request": request_body,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": _generate_request_id(),
    }


# ==============================================================================
# Message Conversion Helpers
# ==============================================================================


def _convert_user_parts_openai(msg: Any) -> List[Dict[str, Any]]:
    """Convert an OpenAI user message to Gemini parts.

    Handles string content, array content with text and image_url blocks.

    Args:
        msg: OpenAI ChatMessage object.

    Returns:
        List of Gemini part dicts.
    """
    content = msg.content
    if isinstance(content, str):
        return [{"text": content}] if content else []

    if not isinstance(content, list):
        text = extract_text_content(content)
        return [{"text": text}] if text else []

    parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"text": block})
            continue
        if isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if url and url.startswith("data:"):
                    # Parse data URI: data:<mime>;base64,<data>
                    mime, _, b64data = url.partition(";base64,")
                    mime = mime.replace("data:", "")
                    if b64data:
                        parts.append({
                            "inlineData": {"mimeType": mime or "image/png", "data": b64data}
                        })
        else:
            # Pydantic model block
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append({"text": text})
            elif block_type == "image_url":
                image_url = getattr(block, "image_url", None)
                url = getattr(image_url, "url", "") if image_url else ""
                if url and url.startswith("data:"):
                    mime, _, b64data = url.partition(";base64,")
                    mime = mime.replace("data:", "")
                    if b64data:
                        parts.append({
                            "inlineData": {"mimeType": mime or "image/png", "data": b64data}
                        })

    return parts if parts else [{"text": ""}]


def _convert_assistant_parts(msg: Any) -> List[Dict[str, Any]]:
    """Convert an OpenAI assistant message to Gemini model parts.

    Handles text content, reasoning_content (thinking), and tool_calls.

    Args:
        msg: OpenAI ChatMessage object.

    Returns:
        List of Gemini part dicts.
    """
    parts: List[Dict[str, Any]] = []

    # Thinking/reasoning content.
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        parts.append({"thought": True, "text": reasoning})

    # Text content.
    text = extract_text_content(msg.content)
    if text:
        parts.append({"text": text})

    # Tool calls.
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                tc_id = tc.get("id", "")
            else:
                func = getattr(tc, "function", None)
                name = getattr(func, "name", "") if func else ""
                args_str = getattr(func, "arguments", "{}") if func else "{}"
                tc_id = getattr(tc, "id", "")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, ValueError):
                args = {}
            if not isinstance(args, dict):
                args = {}

            fc_part: Dict[str, Any] = {"functionCall": {"name": name, "args": args}}
            if tc_id:
                fc_part["functionCall"]["id"] = _sanitize_tool_call_id(tc_id, name)
            parts.append(fc_part)

    return parts if parts else [{"text": ""}]


def _convert_tool_result_openai(msg: Any) -> Dict[str, Any]:
    """Convert an OpenAI tool result message to a Gemini functionResponse part.

    Args:
        msg: OpenAI ChatMessage with role="tool".

    Returns:
        Gemini functionResponse part dict.
    """
    text = extract_text_content(msg.content)
    name = getattr(msg, "name", "") or "tool"
    tool_call_id = getattr(msg, "tool_call_id", "")

    part: Dict[str, Any] = {
        "functionResponse": {
            "name": name,
            "response": {"output": text or ""},
        }
    }
    if tool_call_id:
        part["functionResponse"]["id"] = _sanitize_tool_call_id(tool_call_id, name)
    return part


def _convert_image_block(block: Any) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic image block to a Gemini inlineData part.

    Args:
        block: Anthropic image content block.

    Returns:
        Gemini inlineData part dict, or None if conversion fails.
    """
    source = _get(block, "source")
    if source is None:
        return None

    source_type = _get(source, "type", "")
    if source_type == "base64":
        media_type = _get(source, "media_type", "image/png")
        data = _get(source, "data", "")
        if data:
            return {"inlineData": {"mimeType": media_type, "data": data}}
    elif source_type == "url":
        logger.debug("URL-based images are not supported by Antigravity; skipping")

    return None


# ==============================================================================
# Tool Conversion
# ==============================================================================


def _convert_openai_tools(
    tools: Optional[List[Any]], use_legacy: bool
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI tool definitions to Gemini functionDeclarations.

    Args:
        tools: OpenAI tools list from the request.
        use_legacy: If True, use strict ``parameters`` format for Claude/GPT-OSS.

    Returns:
        Gemini tools array, or None if no tools.
    """
    if not tools:
        return None

    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {"type": "object", "properties": {}})
        else:
            if getattr(tool, "type", "") != "function":
                continue
            func = getattr(tool, "function", None)
            if not func:
                continue
            name = getattr(func, "name", "")
            description = getattr(func, "description", "")
            parameters = getattr(func, "parameters", {"type": "object", "properties": {}})

        if not name:
            continue

        # Process schema.
        schema = _dereference_schema(parameters)
        schema = _ensure_root_object_schema(schema)
        schema = _strip_meta_schema(schema)

        decl: Dict[str, Any] = {"name": name, "description": description or ""}
        if use_legacy:
            decl["parameters"] = _normalize_custom_tool_schema(schema)
        else:
            decl["parametersJsonSchema"] = schema

        declarations.append(decl)

    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _convert_anthropic_tools(
    tools: Optional[List[Any]], use_legacy: bool
) -> Optional[List[Dict[str, Any]]]:
    """Convert Anthropic tool definitions to Gemini functionDeclarations.

    Args:
        tools: Anthropic tools list from the request.
        use_legacy: If True, use strict ``parameters`` format.

    Returns:
        Gemini tools array, or None if no tools.
    """
    if not tools:
        return None

    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        name = _get(tool, "name", "")
        description = _get(tool, "description", "")
        input_schema = _get(tool, "input_schema", {"type": "object", "properties": {}})

        if not name:
            continue

        schema = _dereference_schema(input_schema)
        schema = _ensure_root_object_schema(schema)
        schema = _strip_meta_schema(schema)

        decl: Dict[str, Any] = {"name": name, "description": description or ""}
        if use_legacy:
            decl["parameters"] = _normalize_custom_tool_schema(schema)
        else:
            decl["parametersJsonSchema"] = schema

        declarations.append(decl)

    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


# ==============================================================================
# Schema Processing
# ==============================================================================


def _dereference_schema(
    schema: Any,
    root_defs: Optional[Dict[str, Any]] = None,
    visited: Optional[set] = None,
) -> Any:
    """Recursively resolve $ref references in a JSON Schema.

    Args:
        schema: JSON Schema (dict, list, or primitive).
        root_defs: Root-level $defs/definitions for resolution.
        visited: Set of visited objects to prevent infinite recursion.

    Returns:
        Schema with all $ref pointers resolved.
    """
    if not schema or not isinstance(schema, dict):
        return schema
    if isinstance(schema, list):
        return [_dereference_schema(item, root_defs, visited) for item in schema]

    if visited is None:
        visited = set()
    obj_id = id(schema)
    if obj_id in visited:
        return schema
    visited.add(obj_id)

    # Gather definitions.
    defs = dict(root_defs or {})
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema["$defs"])
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema["definitions"])

    # Resolve $ref.
    ref = schema.get("$ref")
    if isinstance(ref, str):
        import re
        match = re.match(r"^#/(?:\$defs|definitions)/(.+)$", ref)
        if match and match.group(1) in defs:
            resolved = _dereference_schema(defs[match.group(1)], defs, visited)
            if isinstance(resolved, dict):
                # Merge any sibling keys (e.g., description alongside $ref).
                rest = {k: v for k, v in schema.items() if k != "$ref"}
                rest_resolved = _dereference_schema(rest, defs, visited)
                if isinstance(rest_resolved, dict):
                    return {**resolved, **rest_resolved}
                return resolved
            return resolved

    # Recurse into all values.
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            out[key] = _dereference_schema(value, defs, visited)
        elif isinstance(value, list):
            out[key] = [_dereference_schema(item, defs, visited) for item in value]
        else:
            out[key] = value
    return out


def _ensure_root_object_schema(schema: Any) -> Dict[str, Any]:
    """Ensure a schema has type=object at the root level.

    Args:
        schema: Input schema.

    Returns:
        Schema dict with type=object guaranteed.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if "type" not in schema:
        return {**schema, "type": "object", "properties": schema.get("properties", {})}
    return schema


def _strip_meta_schema(schema: Any) -> Any:
    """Remove JSON Schema meta keywords that are not relevant to the API.

    Args:
        schema: Input schema.

    Returns:
        Schema with meta keywords removed.
    """
    if not isinstance(schema, dict):
        return schema

    omit = {"$schema", "$id", "$anchor", "$dynamicAnchor", "$vocabulary", "$comment", "$defs", "definitions"}
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in omit:
            continue
        out[key] = _strip_meta_schema(value)
    return out


def _normalize_custom_tool_schema(schema: Any) -> Any:
    """Normalize a schema for the Cloud Code Assist custom-tool bridge.

    Claude and GPT-OSS models use a Protobuf-based tool bridge that only
    accepts a strict subset of JSON Schema fields. This function strips
    everything else to prevent 400 errors.

    Allowed fields: type, description, properties, required, items, enum.

    Args:
        schema: Input schema.

    Returns:
        Normalized schema with only allowed fields.
    """
    if not schema or not isinstance(schema, dict):
        return schema
    if isinstance(schema, list):
        return [_normalize_custom_tool_schema(item) for item in schema]

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _CUSTOM_TOOL_SCHEMA_ALLOW:
            continue

        if key == "type":
            # Handle union types like ["string", "null"] → first non-null.
            if isinstance(value, list):
                scalar = next(
                    (v for v in value if isinstance(v, str) and v != "null"), None
                )
                if scalar:
                    out["type"] = scalar
            elif isinstance(value, str):
                out["type"] = value
            continue

        if key == "properties" and isinstance(value, dict):
            props: Dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                props[prop_name] = _normalize_custom_tool_schema(prop_schema)
            out["properties"] = props
            continue

        if key == "enum" and isinstance(value, list):
            # Only string enums are supported.
            if all(isinstance(v, str) for v in value):
                out["enum"] = value
            continue

        out[key] = _normalize_custom_tool_schema(value)

    return out


# ==============================================================================
# Generation Config
# ==============================================================================


def _build_generation_config(
    max_tokens: Optional[int],
    temperature: Optional[float],
    runtime_model: str,
    thinking_effort: str,
) -> Dict[str, Any]:
    """Build the generationConfig for the Antigravity request.

    Args:
        max_tokens: Requested max output tokens (from client).
        temperature: Requested temperature (from client).
        runtime_model: Resolved runtime model ID.
        thinking_effort: Thinking effort level.

    Returns:
        Generation config dict (may be empty).
    """
    gen_config: Dict[str, Any] = {}

    if temperature is not None:
        gen_config["temperature"] = temperature

    # Thinking config for Gemini 3.7 (tiered model).
    if runtime_model == "gemini-3.7-flash-tiered" and thinking_effort != "off":
        level_map = {"minimal": "LOW", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
        level = level_map.get(thinking_effort, "LOW")
        gen_config["thinkingConfig"] = {"thinkingLevel": level}

    # Max output tokens.
    from kiro.upstream_antigravity import AntigravityBackend

    # Use a temporary instance just for max token lookup.
    backend = AntigravityBackend.__new__(AntigravityBackend)
    max_allowed = backend.get_max_output_tokens(runtime_model)

    if max_tokens and max_tokens > 0:
        gen_config["maxOutputTokens"] = min(max_tokens, max_allowed)
    else:
        gen_config["maxOutputTokens"] = min(ANTIGRAVITY_MAX_TOKENS, max_allowed)

    return gen_config


# ==============================================================================
# Tool Choice Mapping
# ==============================================================================


def _map_tool_choice_openai(
    tool_choice: Any, runtime_model: str
) -> Optional[Dict[str, Any]]:
    """Map OpenAI tool_choice to Gemini toolConfig.

    Args:
        tool_choice: OpenAI tool_choice value (str or dict).
        runtime_model: Runtime model ID.

    Returns:
        Gemini toolConfig dict, or None if no explicit choice.
    """
    if not tool_choice:
        # Default: use VALIDATED for Claude, AUTO for others.
        if runtime_model.startswith("claude-"):
            return {"functionCallingConfig": {"mode": "VALIDATED"}}
        return None

    if isinstance(tool_choice, str):
        mode_map = {"none": "NONE", "auto": "AUTO", "required": "ANY"}
        mode = mode_map.get(tool_choice)
        if mode:
            return {"functionCallingConfig": {"mode": mode}}

    # tool_choice = {"type": "function", "function": {"name": "..."}}
    # Gemini doesn't support forcing a specific function; use ANY.
    return {"functionCallingConfig": {"mode": "ANY"}}


def _map_tool_choice_anthropic(
    tool_choice: Any, runtime_model: str
) -> Optional[Dict[str, Any]]:
    """Map Anthropic tool_choice to Gemini toolConfig.

    Args:
        tool_choice: Anthropic tool_choice value.
        runtime_model: Runtime model ID.

    Returns:
        Gemini toolConfig dict, or None.
    """
    if not tool_choice:
        if runtime_model.startswith("claude-"):
            return {"functionCallingConfig": {"mode": "VALIDATED"}}
        return None

    choice_type = _get(tool_choice, "type", "")
    if choice_type == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if choice_type in ("any", "required"):
        return {"functionCallingConfig": {"mode": "ANY"}}
    if choice_type == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}

    return None


# ==============================================================================
# Utility Helpers
# ==============================================================================


def _append_turn(
    contents: List[Dict[str, Any]], role: str, parts: List[Dict[str, Any]]
) -> None:
    """Append parts to contents, merging consecutive same-role turns.

    Gemini requires alternating user/model turns. If the last turn has the
    same role, merge the parts instead of creating a new turn.

    Args:
        contents: Existing contents list (modified in place).
        role: Gemini role ("user" or "model").
        parts: Parts to append.
    """
    if not parts:
        return
    if contents and contents[-1].get("role") == role:
        contents[-1]["parts"].extend(parts)
    else:
        contents.append({"role": role, "parts": parts})


def _ensure_first_turn_is_user(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure the first turn in contents is from the user role.

    Google Antigravity / Gemini requires the first turn to be from 'user'.
    If the conversation starts with 'model', prepend a minimal user message.

    Args:
        contents: Conversation contents.

    Returns:
        Contents with guaranteed user-first ordering.
    """
    if contents and contents[0].get("role") == "model":
        contents.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})
    return contents


def _extract_anthropic_system(system: Any) -> str:
    """Extract system prompt text from Anthropic request.

    Handles both string and list-of-blocks formats.

    Args:
        system: Anthropic system field (str, list, or None).

    Returns:
        System prompt text.
    """
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
            else:
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(system)


def _extract_tool_result_text(content: Any) -> str:
    """Extract text from an Anthropic tool_result content field.

    Args:
        content: Content field (str, list of blocks, or None).

    Returns:
        Concatenated text content.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
            else:
                text = getattr(block, "text", "")
                if text:
                    texts.append(text)
        return "\n".join(texts)
    return str(content)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Get an attribute from a dict or Pydantic model.

    Args:
        obj: Dict or Pydantic model instance.
        name: Attribute/key name.
        default: Default value if not found.

    Returns:
        Attribute value or default.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _needs_tool_call_id(runtime_model: str) -> bool:
    """Check if a runtime model requires tool call IDs.

    Claude and GPT-OSS models need explicit tool call IDs.

    Args:
        runtime_model: Runtime model ID.

    Returns:
        True if tool call IDs should be included.
    """
    return runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-")


def _sanitize_tool_call_id(call_id: str, fallback_name: str = "tool") -> str:
    """Sanitize a tool call ID to only allowed characters.

    Cloud Code Assist accepts only alphanumeric, underscore, and dash.
    Max length: 64 characters.

    Args:
        call_id: Raw tool call ID.
        fallback_name: Name to use if ID is empty after sanitization.

    Returns:
        Sanitized ID string.
    """
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
    capped = cleaned[:64]
    return capped or f"{fallback_name}_{int(time.time())}_{secrets.token_hex(4)}"


def _generate_request_id() -> str:
    """Generate a unique request ID for the Antigravity envelope.

    Returns:
        UUID-like request ID string.
    """
    return f"{int(time.time() * 1000)}-{secrets.token_hex(8)}"
