
"""Unit tests for the Codex (ChatGPT) credentials persistence store.

Covers persist_codex_tokens + build_persist_callback:
- Updating an existing account entry in place (matched by chatgptAccountId).
- Matching by pre-rotation refresh token when account ids are unavailable.
- Preserving other accounts and unknown fields on the matched entry.
- Appending a new entry when no match exists.
- Creating a fresh file when none exists.
- Atomic write: file permissions (0600) and no leftover temp files.
- Malformed / non-list existing files raising ValueError.
- The callback swallowing persistence errors (never breaking refresh).

These tests use real temp files (tmp_path); no network is involved, so they are
compatible with the global network-isolation fixture.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from kiro.codex_credentials_store import (
    persist_codex_tokens,
    build_persist_callback,
    _same_account,
    _mask,
)


def _read(path: Path):
    """Read and parse the JSON credentials file."""
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# persist_codex_tokens — update existing entry
# =============================================================================

class TestPersistUpdateExisting:
    """Updating an existing account entry."""

    def test_updates_matching_entry_by_account_id(self, tmp_path):
        """
        What it does: Rotates tokens on the entry matched by chatgptAccountId.
        Purpose: The core case — refresh writes the new tokens back.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "enabled": True, "chatgptAccountId": "acc1",
             "accessToken": "old_at", "refreshToken": "old_rt"},
        ]), encoding="utf-8")

        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token="old_rt",
            access_token="new_at", refresh_token="new_rt",
            id_token="new_id", expires_at="2030-01-01T00:00:00+00:00",
        )

        entries = _read(path)
        assert len(entries) == 1
        assert entries[0]["accessToken"] == "new_at"
        assert entries[0]["refreshToken"] == "new_rt"
        assert entries[0]["idToken"] == "new_id"
        assert entries[0]["expiresAt"] == "2030-01-01T00:00:00+00:00"

    def test_matches_by_refresh_token_when_no_account_id(self, tmp_path):
        """
        What it does: Falls back to refresh-token match when ids are absent.
        Purpose: Support entries imported without a chatgptAccountId.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "enabled": True,
             "accessToken": "old_at", "refreshToken": "match_this_rt"},
        ]), encoding="utf-8")

        persist_codex_tokens(
            path, chatgpt_account_id=None, match_refresh_token="match_this_rt",
            access_token="new_at", refresh_token="rotated_rt",
        )

        entries = _read(path)
        assert len(entries) == 1
        assert entries[0]["refreshToken"] == "rotated_rt"

    def test_preserves_other_accounts_and_unknown_fields(self, tmp_path):
        """
        What it does: Only the matched entry changes; others and extra fields stay.
        Purpose: Never clobber sibling accounts or user metadata (comment, etc.).
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "enabled": True, "chatgptAccountId": "acc1",
             "accessToken": "at1", "refreshToken": "rt1", "comment": "primary"},
            {"provider": "chatgpt", "enabled": False, "chatgptAccountId": "acc2",
             "accessToken": "at2", "refreshToken": "rt2"},
        ]), encoding="utf-8")

        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token="rt1",
            access_token="new_at1", refresh_token="new_rt1",
        )

        entries = _read(path)
        assert len(entries) == 2
        acc1 = next(e for e in entries if e["chatgptAccountId"] == "acc1")
        acc2 = next(e for e in entries if e["chatgptAccountId"] == "acc2")
        # Matched entry updated, unknown field preserved.
        assert acc1["accessToken"] == "new_at1"
        assert acc1["refreshToken"] == "new_rt1"
        assert acc1["comment"] == "primary"
        # Sibling untouched.
        assert acc2["accessToken"] == "at2"
        assert acc2["refreshToken"] == "rt2"
        assert acc2["enabled"] is False

    def test_does_not_write_none_id_or_expiry(self, tmp_path):
        """
        What it does: Omitting id_token/expires_at leaves existing values intact.
        Purpose: A refresh response without those fields must not erase them.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "chatgptAccountId": "acc1",
             "accessToken": "old_at", "refreshToken": "old_rt",
             "idToken": "keep_id", "expiresAt": "keep_expiry"},
        ]), encoding="utf-8")

        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token="old_rt",
            access_token="new_at", refresh_token="new_rt",
            id_token=None, expires_at=None,
        )

        entry = _read(path)[0]
        assert entry["idToken"] == "keep_id"
        assert entry["expiresAt"] == "keep_expiry"


# =============================================================================
# persist_codex_tokens — append / create
# =============================================================================

class TestPersistAppendCreate:
    """Appending new entries and creating the file."""

    def test_appends_when_no_match(self, tmp_path):
        """
        What it does: Adds a new entry when the account is not present.
        Purpose: A newly discovered account is persisted without dropping others.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "chatgptAccountId": "acc1",
             "accessToken": "at1", "refreshToken": "rt1"},
        ]), encoding="utf-8")

        persist_codex_tokens(
            path, chatgpt_account_id="acc2", match_refresh_token="rt2",
            access_token="at2", refresh_token="rt2",
        )

        entries = _read(path)
        assert len(entries) == 2
        new = next(e for e in entries if e.get("chatgptAccountId") == "acc2")
        assert new["enabled"] is True
        assert new["refreshToken"] == "rt2"

    def test_creates_file_when_missing(self, tmp_path):
        """
        What it does: Creates the credentials file if it does not exist.
        Purpose: First-ever persistence should not fail on a missing file.
        """
        path = tmp_path / "nested" / "creds.json"
        assert not path.exists()

        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token=None,
            access_token="at", refresh_token="rt",
        )

        entries = _read(path)
        assert len(entries) == 1
        assert entries[0]["chatgptAccountId"] == "acc1"


# =============================================================================
# Atomic write guarantees
# =============================================================================

class TestAtomicWrite:
    """File permission and temp-file guarantees."""

    def test_file_has_owner_only_permissions(self, tmp_path):
        """
        What it does: The written file is 0600.
        Purpose: Credentials must not be world/group readable.
        """
        path = tmp_path / "creds.json"
        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token=None,
            access_token="at", refresh_token="rt",
        )
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600

    def test_no_leftover_temp_files(self, tmp_path):
        """
        What it does: No .tmp files remain after a successful write.
        Purpose: The atomic write must clean up its temp file (via rename).
        """
        path = tmp_path / "creds.json"
        persist_codex_tokens(
            path, chatgpt_account_id="acc1", match_refresh_token=None,
            access_token="at", refresh_token="rt",
        )
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "creds.json"]
        assert leftovers == []

    def test_falls_back_to_in_place_write_on_ebusy(self, tmp_path):
        """
        What it does: When os.replace raises EBUSY (target is a bind-mount, as
            with podman/docker -v file:file), the write falls back to an
            in-place rewrite and the file is still updated.
        Purpose: Rotated tokens must persist inside file-mounted containers,
            where an atomic rename onto the mount point is rejected by the kernel.
        """
        import errno as _errno
        from unittest.mock import patch

        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "chatgptAccountId": "acc1",
             "accessToken": "old_at", "refreshToken": "old_rt"},
        ]), encoding="utf-8")
        original_inode = os.stat(path).st_ino

        def _raise_ebusy(src, dst):
            raise OSError(_errno.EBUSY, "Device or resource busy")

        with patch("kiro.codex_credentials_store.os.replace", side_effect=_raise_ebusy):
            persist_codex_tokens(
                path, chatgpt_account_id="acc1", match_refresh_token="old_rt",
                access_token="new_at", refresh_token="new_rt",
            )

        entry = _read(path)[0]
        assert entry["accessToken"] == "new_at"
        assert entry["refreshToken"] == "new_rt"
        # In-place write keeps the same inode (the bind-mounted file itself).
        assert os.stat(path).st_ino == original_inode
        # Temp file must still be cleaned up.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "creds.json"]
        assert leftovers == []

    def test_falls_back_on_exdev(self, tmp_path):
        """
        What it does: An EXDEV (cross-device rename) also triggers the in-place
            fallback rather than propagating.
        Purpose: Temp file and target on different filesystems must not break
            persistence.
        """
        import errno as _errno
        from unittest.mock import patch

        path = tmp_path / "creds.json"

        def _raise_exdev(src, dst):
            raise OSError(_errno.EXDEV, "Invalid cross-device link")

        with patch("kiro.codex_credentials_store.os.replace", side_effect=_raise_exdev):
            persist_codex_tokens(
                path, chatgpt_account_id="acc1", match_refresh_token=None,
                access_token="at", refresh_token="rt",
            )

        assert _read(path)[0]["refreshToken"] == "rt"

    def test_non_recoverable_oserror_propagates(self, tmp_path):
        """
        What it does: A non-EBUSY/EXDEV OSError from os.replace propagates.
        Purpose: Genuine filesystem failures must not be silently swallowed by
            the fallback path.
        """
        import errno as _errno
        from unittest.mock import patch

        path = tmp_path / "creds.json"

        def _raise_eacces(src, dst):
            raise OSError(_errno.EACCES, "Permission denied")

        with patch("kiro.codex_credentials_store.os.replace", side_effect=_raise_eacces):
            with pytest.raises(OSError):
                persist_codex_tokens(
                    path, chatgpt_account_id="acc1", match_refresh_token=None,
                    access_token="at", refresh_token="rt",
                )


# =============================================================================
# Malformed input
# =============================================================================

class TestMalformedInput:
    """Malformed existing files must raise rather than silently discard data."""

    def test_invalid_json_raises(self, tmp_path):
        """
        What it does: A non-JSON existing file raises ValueError.
        Purpose: Avoid overwriting a user's file we cannot understand.
        """
        path = tmp_path / "creds.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            persist_codex_tokens(
                path, chatgpt_account_id="acc1", match_refresh_token=None,
                access_token="at", refresh_token="rt",
            )

    def test_non_list_json_raises(self, tmp_path):
        """
        What it does: A JSON object (not a list) raises ValueError.
        Purpose: The credentials file must be a list of accounts.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps({"provider": "chatgpt"}), encoding="utf-8")
        with pytest.raises(ValueError):
            persist_codex_tokens(
                path, chatgpt_account_id="acc1", match_refresh_token=None,
                access_token="at", refresh_token="rt",
            )


# =============================================================================
# _same_account matching
# =============================================================================

class TestSameAccount:
    """Direct tests of the entry-matching predicate."""

    def test_matches_by_account_id(self):
        entry = {"chatgptAccountId": "acc1", "refreshToken": "rtX"}
        assert _same_account(entry, "acc1", "different") is True

    def test_no_match_different_account_id(self):
        entry = {"chatgptAccountId": "acc1", "refreshToken": "rt1"}
        # Account ids present on both sides but differ → no match even if rt equal.
        assert _same_account(entry, "acc2", "rt1") is False

    def test_matches_by_refresh_token_fallback(self):
        entry = {"refreshToken": "rt1"}
        assert _same_account(entry, None, "rt1") is True

    def test_ignores_non_chatgpt_provider(self):
        entry = {"provider": "kiro", "chatgptAccountId": "acc1"}
        assert _same_account(entry, "acc1", None) is False

    def test_no_match_when_nothing_comparable(self):
        entry = {"provider": "chatgpt"}
        assert _same_account(entry, None, None) is False


# =============================================================================
# build_persist_callback
# =============================================================================

class TestBuildPersistCallback:
    """Tests for the callback factory wired into CodexAuthManager."""

    def test_callback_persists_payload(self, tmp_path):
        """
        What it does: The callback writes the payload's tokens to the file.
        Purpose: End-to-end wiring from refresh payload to disk.
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps([
            {"provider": "chatgpt", "chatgptAccountId": "acc1",
             "accessToken": "old_at", "refreshToken": "old_rt"},
        ]), encoding="utf-8")

        callback = build_persist_callback(str(path))
        callback({
            "chatgpt_account_id": "acc1",
            "match_refresh_token": "old_rt",
            "access_token": "new_at",
            "refresh_token": "new_rt",
            "id_token": None,
            "expires_at": "2030-01-01T00:00:00+00:00",
        })

        entry = _read(path)[0]
        assert entry["accessToken"] == "new_at"
        assert entry["refreshToken"] == "new_rt"

    def test_callback_swallows_malformed_file_error(self, tmp_path):
        """
        What it does: The callback does not raise when the file is malformed.
        Purpose: A persistence failure must never propagate into the refresh path.
        """
        path = tmp_path / "creds.json"
        path.write_text("{not json", encoding="utf-8")

        callback = build_persist_callback(str(path))
        # Must not raise despite the malformed file.
        callback({
            "chatgpt_account_id": "acc1",
            "match_refresh_token": "old_rt",
            "access_token": "new_at",
            "refresh_token": "new_rt",
        })

    def test_callback_swallows_missing_required_key(self, tmp_path):
        """
        What it does: A payload missing access_token/refresh_token is ignored safely.
        Purpose: Defensive — a malformed payload must not crash the caller.
        """
        path = tmp_path / "creds.json"
        callback = build_persist_callback(str(path))
        callback({"chatgpt_account_id": "acc1"})  # missing tokens → KeyError caught
        # File should not have been created by a failed write.
        assert not path.exists()


# =============================================================================
# Masking
# =============================================================================

class TestMask:
    """Token masking used in log messages."""

    def test_mask_none(self):
        assert _mask(None) == "<none>"

    def test_mask_short(self):
        assert _mask("short") == "****"

    def test_mask_long_hides_middle(self):
        masked = _mask("abcdefghijklmnop")
        assert "defghijkl" not in masked
        assert masked.startswith("abcd")
