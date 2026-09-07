#!/usr/bin/env python3
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

"""Import ChatGPT (Codex) OAuth tokens into the gateway credentials file.

Reads OAuth tokens produced by a ChatGPT login in either:
- the official Codex CLI (``~/.codex/auth.json``), or
- OpenCode (``~/.local/share/opencode/auth.json``)

and appends/updates an account entry in the gateway's Codex credentials file
(``chatgpt_credentials.json`` by default). The gateway then consumes these
tokens and refreshes them automatically.

The two source formats differ in field names and nesting; this importer probes
several known shapes and never guesses silently — if it cannot find the tokens,
it prints the JSON key structure it *did* find so you can report it.

SECURITY:
- Tokens are written only to the local credentials file (never uploaded).
- The credentials file is created with owner-only permissions (0600).
- Token values are never printed; only masked previews are shown.

Examples:
    # Auto-detect the source file (Codex CLI first, then OpenCode):
    python scripts/import_codex_auth.py

    # Explicit source + custom label for this account:
    python scripts/import_codex_auth.py --source ~/.codex/auth.json --label acc1

    # Custom output file:
    python scripts/import_codex_auth.py --output /path/to/chatgpt_credentials.json

    # Inspect the source structure without writing anything:
    python scripts/import_codex_auth.py --dry-run
"""

import argparse
import base64
import binascii
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default source locations, tried in order for auto-detection.
DEFAULT_SOURCES = [
    Path("~/.codex/auth.json").expanduser(),                       # Codex CLI
    Path("~/.local/share/opencode/auth.json").expanduser(),        # OpenCode (Linux)
    Path("~/Library/Application Support/opencode/auth.json").expanduser(),  # OpenCode (macOS)
]

DEFAULT_OUTPUT = Path("chatgpt_credentials.json")


def _mask(token: Optional[str]) -> str:
    """Return a masked preview of a token that never reveals the full value."""
    if not token:
        return "<none>"
    if len(token) <= 10:
        return "****"
    return f"{token[:6]}…{token[-4:]}"


def _decode_jwt_claims(token: Optional[str]) -> Dict[str, Any]:
    """Best-effort decode of a JWT payload (no signature check). Never raises."""
    if not token or not isinstance(token, str):
        return {}
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(decoded.decode("utf-8"))
        return claims if isinstance(claims, dict) else {}
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _extract_account_id(id_token: Optional[str], access_token: Optional[str]) -> Optional[str]:
    """Extract chatgpt_account_id from JWT claims of the id/access token."""
    for token in (id_token, access_token):
        claims = _decode_jwt_claims(token)
        auth = claims.get("https://api.openai.com/auth")
        if isinstance(auth, dict):
            acc = auth.get("chatgpt_account_id") or auth.get("account_id")
            if isinstance(acc, str) and acc:
                return acc
        for key in ("chatgpt_account_id", "account_id"):
            val = claims.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _first_str(container: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty string value among the given keys."""
    for k in keys:
        v = container.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _find_token_bag(data: Any) -> Optional[Dict[str, Any]]:
    """Locate the dict holding the OAuth tokens across known source shapes.

    Handles:
    - Codex CLI: {"tokens": {"access_token", "refresh_token", "id_token", "account_id"}}
      or the same fields at the top level.
    - OpenCode: {"openai": {"type": "oauth", "access", "refresh", ...}} or a
      similar per-provider nesting; also a top-level oauth object.

    Returns the dict that actually contains an access/refresh token, or None.
    """
    def looks_like_tokens(d: Any) -> bool:
        if not isinstance(d, dict):
            return False
        has_access = _first_str(d, ("access_token", "access", "accessToken")) is not None
        has_refresh = _first_str(d, ("refresh_token", "refresh", "refreshToken")) is not None
        return has_access and has_refresh

    if not isinstance(data, dict):
        return None

    # 1. Top level directly holds tokens.
    if looks_like_tokens(data):
        return data

    # 2. Codex CLI nests under "tokens".
    if looks_like_tokens(data.get("tokens")):
        return data["tokens"]

    # 3. OpenCode nests per provider (e.g. "openai"); scan one level deep.
    for key in ("openai", "chatgpt", "codex"):
        entry = data.get(key)
        if looks_like_tokens(entry):
            return entry
        if isinstance(entry, dict) and looks_like_tokens(entry.get("tokens")):
            return entry["tokens"]

    # 4. Last resort: scan all one-level-deep dicts.
    for value in data.values():
        if looks_like_tokens(value):
            return value
        if isinstance(value, dict) and looks_like_tokens(value.get("tokens")):
            return value["tokens"]

    return None


def _describe_structure(data: Any, depth: int = 0, max_depth: int = 2) -> str:
    """Render the key structure of a JSON object (keys only, no values)."""
    indent = "  " * depth
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, dict) and depth < max_depth:
                lines.append(f"{indent}{k}:")
                lines.append(_describe_structure(v, depth + 1, max_depth))
            else:
                type_name = type(v).__name__
                lines.append(f"{indent}{k}: <{type_name}>")
        return "\n".join(lines)
    return f"{indent}<{type(data).__name__}>"


def extract_account(source_path: Path) -> Dict[str, Any]:
    """Extract a gateway credential entry from a source auth.json file.

    Args:
        source_path: Path to a Codex CLI or OpenCode auth.json file.

    Returns:
        A gateway credential entry dict (provider/accessToken/refreshToken/...).

    Raises:
        SystemExit: If the file is missing/invalid or tokens cannot be found
            (with a structure dump to aid debugging).
    """
    if not source_path.exists():
        sys.exit(f"ERROR: source file not found: {source_path}")

    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: cannot read/parse {source_path}: {e}")

    bag = _find_token_bag(raw)
    if bag is None:
        print(f"ERROR: could not locate OAuth tokens in {source_path}", file=sys.stderr)
        print("Found this key structure (values hidden):", file=sys.stderr)
        print(_describe_structure(raw), file=sys.stderr)
        sys.exit(
            "\nPlease report the key structure above so the importer can be updated."
        )

    access_token = _first_str(bag, ("access_token", "access", "accessToken"))
    refresh_token = _first_str(bag, ("refresh_token", "refresh", "refreshToken"))
    id_token = _first_str(bag, ("id_token", "id", "idToken"))
    account_id = _first_str(bag, ("account_id", "accountId", "chatgpt_account_id"))
    if not account_id:
        account_id = _extract_account_id(id_token, access_token)

    entry: Dict[str, Any] = {
        "provider": "chatgpt",
        "enabled": True,
        "accessToken": access_token,
        "refreshToken": refresh_token,
    }
    if id_token:
        entry["idToken"] = id_token
    if account_id:
        entry["chatgptAccountId"] = account_id
    return entry


def _load_output(output_path: Path) -> List[Dict[str, Any]]:
    """Load the existing credentials list, or return an empty list."""
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        print(f"WARNING: {output_path} is not a JSON list; starting fresh.", file=sys.stderr)
        return []
    except (OSError, json.JSONDecodeError):
        print(f"WARNING: {output_path} is unreadable; starting fresh.", file=sys.stderr)
        return []


def _write_output(output_path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write the credentials list with owner-only permissions (0600)."""
    fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Import ChatGPT (Codex) OAuth tokens into the gateway credentials file."
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to a Codex CLI or OpenCode auth.json. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Gateway credentials file to write (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional comment/label stored on the account entry.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be imported without writing the file.",
    )
    args = parser.parse_args()

    # Resolve the source file.
    if args.source:
        source_path = Path(args.source).expanduser()
    else:
        source_path = next((p for p in DEFAULT_SOURCES if p.exists()), None)
        if source_path is None:
            tried = "\n  ".join(str(p) for p in DEFAULT_SOURCES)
            sys.exit(
                "ERROR: no source auth.json found. Log in with Codex CLI or OpenCode "
                f"first, or pass --source.\nTried:\n  {tried}"
            )
        print(f"Auto-detected source: {source_path}")

    entry = extract_account(source_path)
    if args.label:
        entry["comment"] = args.label

    if not entry.get("accessToken") or not entry.get("refreshToken"):
        sys.exit("ERROR: source is missing accessToken or refreshToken.")

    # Report (masked).
    print("Imported account:")
    print(f"  chatgptAccountId : {entry.get('chatgptAccountId', '<will backfill from JWT at runtime>')}")
    print(f"  accessToken      : {_mask(entry.get('accessToken'))}")
    print(f"  refreshToken     : {_mask(entry.get('refreshToken'))}")
    print(f"  idToken          : {_mask(entry.get('idToken'))}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    output_path = Path(args.output).expanduser()
    entries = _load_output(output_path)

    # De-dupe by chatgptAccountId (or refreshToken when id is unknown): update in place.
    def same_account(existing: Dict[str, Any]) -> bool:
        if entry.get("chatgptAccountId") and existing.get("chatgptAccountId"):
            return existing["chatgptAccountId"] == entry["chatgptAccountId"]
        return existing.get("refreshToken") == entry.get("refreshToken")

    replaced = False
    for i, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("provider") == "chatgpt" and same_account(existing):
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    _write_output(output_path, entries)
    action = "updated" if replaced else "added"
    codex_count = sum(1 for e in entries if isinstance(e, dict) and e.get("provider") == "chatgpt")
    print(f"\nOK: {action} account in {output_path} (0600). Total Codex accounts: {codex_count}.")
    print("Next: set CHATGPT_ENABLED=true (and CHATGPT_CREDENTIALS_FILE if not the default) in .env.")


if __name__ == "__main__":
    main()
