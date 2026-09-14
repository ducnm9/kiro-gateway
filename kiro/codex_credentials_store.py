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

"""Persistence for ChatGPT (Codex) OAuth credentials.

OpenAI rotates refresh tokens on every ``refresh_token`` grant: the previous
refresh token is invalidated and a new one is returned. If the rotated token is
kept only in memory, the on-disk ``chatgpt_credentials.json`` becomes stale and
the next process start fails with ``refresh_token_reused`` (HTTP 401).

This module writes rotated tokens back to the credentials file so refresh
survives restarts, mirroring what ``KiroAuthManager`` does for the Kiro upstream.

Design:
- ``CodexAuthManager`` stays free of file I/O; it only invokes an optional
  callback after a successful refresh. This module supplies that callback via
  :func:`build_persist_callback`, keeping the auth manager unit-testable.
- The credentials file is a JSON list of account entries (same shape produced by
  ``scripts/import_codex_auth.py``). A single account is merged in place, matched
  by ``chatgptAccountId`` and falling back to ``refreshToken``; all other
  entries and unknown fields are preserved.
- Writes are atomic (write to a temp file in the same directory, then
  ``os.replace``) with owner-only permissions (0600), so a crash mid-write can
  never truncate the credentials file.

SECURITY: Token values are never logged; only masked previews are emitted.
"""

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


def _mask(token: Optional[str]) -> str:
    """Return a masked preview of a token for safe logging.

    Args:
        token: The token to mask (may be None/empty).

    Returns:
        A masked string that never reveals the full token value.
    """
    if not token:
        return "<none>"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}…{token[-4:]}"


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    """Load the credentials file as a list of account entries.

    A missing file yields an empty list (a fresh file will be created on write).
    A malformed or non-list file raises, because silently discarding a user's
    other accounts would be worse than surfacing the error to the caller.

    Args:
        path: Path to the credentials file.

    Returns:
        The list of account entries currently on disk (possibly empty).

    Raises:
        ValueError: If the file exists but is not valid JSON or not a JSON list.
        OSError: If the file exists but cannot be read.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Codex credentials file is not valid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("Codex credentials file must contain a JSON list of accounts")
    return data


def _same_account(entry: Dict[str, Any], account_id: Optional[str], refresh_token: Optional[str]) -> bool:
    """Return True if a stored entry refers to the same Codex account.

    Matching prefers the stable ``chatgptAccountId``; when either side lacks it,
    it falls back to the refresh token. The refresh-token fallback matches
    against the entry's *current* (pre-rotation) token, which is exactly what is
    on disk before this update is applied.

    Args:
        entry: A stored account entry.
        account_id: The ChatGPT account id of the account being updated.
        refresh_token: The pre-rotation refresh token of the account being
            updated (used only when account ids are unavailable).

    Returns:
        True if ``entry`` is the account being updated.
    """
    if entry.get("provider", "chatgpt") != "chatgpt":
        return False
    entry_account_id = entry.get("chatgptAccountId")
    if account_id and entry_account_id:
        return entry_account_id == account_id
    if refresh_token and entry.get("refreshToken"):
        return entry.get("refreshToken") == refresh_token
    return False


def _serialize(entries: List[Dict[str, Any]]) -> str:
    """Render the credentials list as pretty-printed JSON with a trailing newline."""
    return json.dumps(entries, indent=2, ensure_ascii=False) + "\n"


def _write_in_place(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write entries directly to ``path`` (non-atomic fallback).

    Used when an atomic rename is not possible because ``path`` is itself a
    bind-mount (common with ``podman/docker -v file:file``): renaming onto the
    mount point fails with ``EBUSY``/``EXDEV``. Truncating and rewriting the
    existing inode works because the inode, not a directory entry, is mounted.

    This sacrifices atomicity: a crash mid-write could leave a partial file.
    That is an accepted trade-off — persisting the rotated token is strictly
    more important than the small crash window, and the credentials file is
    small enough that a single ``write`` almost always completes in one go.

    Args:
        path: Destination credentials file (an existing bind-mounted file).
        entries: Full list of account entries to persist.

    Raises:
        OSError: On any filesystem failure.
    """
    data = _serialize(entries)
    # Open for truncate-write on the existing inode; do not recreate the file
    # (which is what triggers EBUSY on a bind-mounted path).
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    # Best-effort permission tightening; ignore on filesystems that refuse it.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _atomic_write(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write the entries to ``path`` with 0600 permissions, atomically if possible.

    Preferred path: write a sibling temp file, fsync it, then ``os.replace`` it
    onto the target — a crash-safe atomic swap on the same filesystem.

    Fallback path: when the target is a single-file bind-mount (as with
    ``podman/docker -v file:file``), the rename step fails with ``EBUSY`` (and,
    across filesystems, ``EXDEV``). In that case, write in place on the existing
    inode instead, trading atomicity for the ability to persist at all. Without
    this fallback, rotated Codex tokens would never reach disk inside a
    file-mounted container and the ``refresh_token_reused`` bug would recur.

    Args:
        path: Destination credentials file.
        entries: Full list of account entries to persist.

    Raises:
        OSError: On any filesystem failure that is not a recoverable rename
            limitation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_serialize(entries))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError as e:
            # EBUSY: target is a bind-mount point (podman/docker -v file:file).
            # EXDEV: temp and target ended up on different filesystems.
            # Both mean the atomic rename cannot land on the target; fall back
            # to an in-place rewrite of the existing inode.
            if e.errno in (errno.EBUSY, errno.EXDEV):
                logger.debug(
                    f"Atomic rename onto {path} failed ({e.errno}); "
                    f"falling back to in-place write."
                )
                _write_in_place(path, entries)
            else:
                raise
    finally:
        # Always clean up the temp file (it is either consumed by replace or
        # left behind by the fallback / an error).
        try:
            tmp_path.unlink()
        except OSError:
            pass


def persist_codex_tokens(
    path: Path,
    *,
    chatgpt_account_id: Optional[str],
    match_refresh_token: Optional[str],
    access_token: str,
    refresh_token: str,
    id_token: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> None:
    """Merge one account's refreshed tokens into the credentials file on disk.

    Reads the current file, updates the matching account entry in place (or
    appends a new one when no match exists), and writes the result atomically.
    Other accounts and any unknown fields on the matched entry are preserved.

    Args:
        path: Path to the Codex credentials file.
        chatgpt_account_id: ChatGPT account id used to locate the entry.
        match_refresh_token: Pre-rotation refresh token, used to locate the
            entry when account ids are unavailable on either side.
        access_token: New access token to store.
        refresh_token: New (rotated) refresh token to store.
        id_token: New id token to store, when present.
        expires_at: New ISO-8601 expiry timestamp, when known.

    Raises:
        ValueError: If the existing file is malformed (not JSON / not a list).
        OSError: On filesystem read/write failure.
    """
    entries = _load_entries(path)

    updated_fields: Dict[str, Any] = {
        "provider": "chatgpt",
        "accessToken": access_token,
        "refreshToken": refresh_token,
    }
    if id_token is not None:
        updated_fields["idToken"] = id_token
    if chatgpt_account_id is not None:
        updated_fields["chatgptAccountId"] = chatgpt_account_id
    if expires_at is not None:
        updated_fields["expiresAt"] = expires_at

    matched = False
    for entry in entries:
        if isinstance(entry, dict) and _same_account(entry, chatgpt_account_id, match_refresh_token):
            # Update only our fields; preserve everything else on the entry
            # (enabled flag, comment/label, and any future fields).
            entry.update(updated_fields)
            matched = True
            break

    if not matched:
        new_entry: Dict[str, Any] = {"enabled": True}
        new_entry.update(updated_fields)
        entries.append(new_entry)

    _atomic_write(path, entries)
    logger.info(
        f"Persisted rotated Codex tokens "
        f"(account={chatgpt_account_id or '<unknown>'}, "
        f"refresh={_mask(refresh_token)}, "
        f"{'updated existing' if matched else 'appended new'} entry) to {path}"
    )


def build_persist_callback(credentials_file: str) -> Callable[[Dict[str, Any]], None]:
    """Build an ``on_token_refreshed`` callback bound to a credentials file.

    The returned callback matches the signature expected by
    ``CodexAuthManager`` (a single dict of refreshed token fields) and persists
    those tokens to ``credentials_file``. Persistence failures are logged but
    never raised: a failed write must not break an otherwise-successful refresh
    (the in-memory token still works for the current process lifetime).

    Args:
        credentials_file: Path (possibly using ``~``) to the Codex credentials
            file to keep in sync.

    Returns:
        A callback ``(payload: dict) -> None`` suitable for
        ``CodexAuthManager(on_token_refreshed=...)``.

    The payload dict is expected to contain:
        ``chatgpt_account_id``, ``match_refresh_token``, ``access_token``,
        ``refresh_token``, ``id_token``, ``expires_at``.
    """
    path = Path(credentials_file).expanduser()

    def _callback(payload: Dict[str, Any]) -> None:
        try:
            persist_codex_tokens(
                path,
                chatgpt_account_id=payload.get("chatgpt_account_id"),
                match_refresh_token=payload.get("match_refresh_token"),
                access_token=payload["access_token"],
                refresh_token=payload["refresh_token"],
                id_token=payload.get("id_token"),
                expires_at=payload.get("expires_at"),
            )
        except (OSError, ValueError, KeyError) as e:
            # Do not break refresh on a persistence failure; the current
            # process keeps working with the in-memory token. Next restart may
            # still hit the stale token, but that is strictly better than
            # crashing a working request path.
            logger.error(
                f"Failed to persist rotated Codex tokens to {path}: "
                f"{type(e).__name__}: {e}"
            )

    return _callback
