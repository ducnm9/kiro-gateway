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
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default JSON source locations, tried in order for auto-detection.
DEFAULT_SOURCES = [
    Path("~/.codex/auth.json").expanduser(),                       # Codex CLI
    Path("~/.local/share/opencode/auth.json").expanduser(),        # OpenCode (older, file-based)
    Path("~/Library/Application Support/opencode/auth.json").expanduser(),  # OpenCode (macOS)
]

# OpenCode (v0.x, "opencode2") stores credentials in a SQLite DB, not auth.json.
# The `credential` table row with integration_id='openai' holds the OAuth JSON
# in its `value` column: {"type","refresh","access","expires","metadata":{"accountID"}}.
DEFAULT_OPENCODE_DB = Path("~/.local/share/opencode/opencode.db").expanduser()

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

    return _bag_to_entry(access_token, refresh_token, id_token, account_id)


def _bag_to_entry(
    access_token: Optional[str],
    refresh_token: Optional[str],
    id_token: Optional[str],
    account_id: Optional[str],
) -> Dict[str, Any]:
    """Build a gateway credential entry from extracted token fields."""
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


def extract_accounts_from_opencode_db(db_path: Path) -> List[Dict[str, Any]]:
    """Extract Codex credential entries from an OpenCode SQLite database.

    OpenCode (v0.x, "opencode2") stores OAuth credentials in the ``credential``
    table. The row(s) with ``integration_id='openai'`` carry a JSON blob in the
    ``value`` column of the shape::

        {"type","methodID","refresh","access","expires","metadata":{"accountID"}}

    Args:
        db_path: Path to ``opencode.db``.

    Returns:
        A list of gateway credential entries (one per OpenAI credential row).

    Raises:
        SystemExit: If the DB/table is missing or holds no OpenAI credential.
    """
    if not db_path.exists():
        sys.exit(f"ERROR: OpenCode database not found: {db_path}")

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        sys.exit(f"ERROR: cannot open {db_path}: {e}")

    try:
        cur = con.cursor()
        has_table = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='credential'"
        ).fetchone()
        if not has_table:
            sys.exit(f"ERROR: no 'credential' table in {db_path} (unexpected OpenCode schema)")
        rows = cur.execute(
            "SELECT value FROM credential WHERE integration_id = 'openai' AND active = 1"
        ).fetchall()
    except sqlite3.Error as e:
        sys.exit(f"ERROR: reading credentials from {db_path}: {e}")
    finally:
        con.close()

    if not rows:
        sys.exit(
            f"ERROR: no active OpenAI credential in {db_path}. "
            "Log in to ChatGPT in OpenCode first."
        )

    entries: List[Dict[str, Any]] = []
    for (value,) in rows:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace")
        try:
            data = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            print("WARNING: skipping an OpenAI credential row with non-JSON value.", file=sys.stderr)
            continue
        access_token = _first_str(data, ("access", "access_token", "accessToken"))
        refresh_token = _first_str(data, ("refresh", "refresh_token", "refreshToken"))
        id_token = _first_str(data, ("id", "id_token", "idToken"))
        account_id = None
        meta = data.get("metadata")
        if isinstance(meta, dict):
            account_id = _first_str(meta, ("accountID", "account_id", "chatgpt_account_id"))
        if not (access_token and refresh_token):
            print("WARNING: skipping an OpenAI credential row missing access/refresh.", file=sys.stderr)
            continue
        entries.append(_bag_to_entry(access_token, refresh_token, id_token, account_id))

    if not entries:
        sys.exit(f"ERROR: OpenAI credential rows in {db_path} had no usable tokens.")
    return entries


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


def _same_account(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two entries are the same account by chatgptAccountId, else by refreshToken."""
    if a.get("chatgptAccountId") and b.get("chatgptAccountId"):
        return a["chatgptAccountId"] == b["chatgptAccountId"]
    return a.get("refreshToken") == b.get("refreshToken")


def _merge_entry(entries: List[Dict[str, Any]], new_entry: Dict[str, Any]) -> str:
    """Merge one entry into the list (update in place if same account)."""
    for i, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("provider") == "chatgpt" and _same_account(existing, new_entry):
            entries[i] = new_entry
            return "updated"
    entries.append(new_entry)
    return "added"


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
        "--opencode-db", type=str, default=None,
        help=f"Path to OpenCode's SQLite DB (default: {DEFAULT_OPENCODE_DB}). "
             "Use this when OpenCode stores auth in opencode.db instead of auth.json.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Gateway credentials file to write (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional comment/label stored on the account entry (single-source imports).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be imported without writing the file.",
    )
    args = parser.parse_args()

    # Determine the new entries from the chosen source.
    new_entries: List[Dict[str, Any]] = []

    if args.opencode_db:
        db_path = Path(args.opencode_db).expanduser()
        print(f"Reading OpenCode database: {db_path}")
        new_entries = extract_accounts_from_opencode_db(db_path)
    elif args.source:
        source_path = Path(args.source).expanduser()
        new_entries = [extract_account(source_path)]
    else:
        # Auto-detect: JSON sources first, then the OpenCode SQLite DB.
        source_path = next((p for p in DEFAULT_SOURCES if p.exists()), None)
        if source_path is not None:
            print(f"Auto-detected source: {source_path}")
            new_entries = [extract_account(source_path)]
        elif DEFAULT_OPENCODE_DB.exists():
            print(f"Auto-detected OpenCode database: {DEFAULT_OPENCODE_DB}")
            new_entries = extract_accounts_from_opencode_db(DEFAULT_OPENCODE_DB)
        else:
            tried = "\n  ".join(str(p) for p in DEFAULT_SOURCES + [DEFAULT_OPENCODE_DB])
            sys.exit(
                "ERROR: no source found. Log in to ChatGPT with Codex CLI or OpenCode "
                f"first, or pass --source / --opencode-db.\nTried:\n  {tried}"
            )

    # Apply an optional label only when importing exactly one account.
    if args.label and len(new_entries) == 1:
        new_entries[0]["comment"] = args.label

    # Validate + report (masked).
    for idx, entry in enumerate(new_entries):
        if not entry.get("accessToken") or not entry.get("refreshToken"):
            sys.exit(f"ERROR: entry #{idx} is missing accessToken or refreshToken.")
        print(f"Account #{idx + 1}:")
        print(f"  chatgptAccountId : {entry.get('chatgptAccountId', '<will backfill from JWT at runtime>')}")
        print(f"  accessToken      : {_mask(entry.get('accessToken'))}")
        print(f"  refreshToken     : {_mask(entry.get('refreshToken'))}")
        print(f"  idToken          : {_mask(entry.get('idToken'))}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    output_path = Path(args.output).expanduser()
    entries = _load_output(output_path)
    added = updated = 0
    for entry in new_entries:
        if _merge_entry(entries, entry) == "updated":
            updated += 1
        else:
            added += 1

    _write_output(output_path, entries)
    codex_count = sum(1 for e in entries if isinstance(e, dict) and e.get("provider") == "chatgpt")
    print(f"\nOK: {added} added, {updated} updated in {output_path} (0600). "
          f"Total Codex accounts: {codex_count}.")
    print("Next: set CHATGPT_ENABLED=true (and CHATGPT_CREDENTIALS_FILE if not the default) in .env.")


if __name__ == "__main__":
    main()
