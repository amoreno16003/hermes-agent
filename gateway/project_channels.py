"""Mirror Hermes projects as Discord channels.

Each project becomes one text channel (``#proj-<slug>``) under a dedicated
category; each Hermes session inside it becomes a thread (Discord's own
auto-thread behaviour already provides that half — see
``discord.auto_thread``). This module owns only the *project → channel*
direction: provisioning the channel, remembering the binding on the project
row, and resolving a channel back to the project whose working directory the
agent should operate in.

Design constraints this respects:

- **Config-gated, default off.** Nothing here runs unless
  ``discord.project_channels.enabled`` is true. A bug must not be able to
  spray channels into a server that never asked for the feature.
- **Create-only.** We create channels and categories; we never delete, never
  rename, and never touch a channel we did not create. MANAGE_CHANNELS is a
  broad permission and this module deliberately uses the smallest slice of it.
- **Idempotent.** Provisioning an already-bound project is a no-op, and a
  pre-existing channel with the expected name is adopted rather than
  duplicated. Safe to run repeatedly (startup backfill, CLI, retries).
- **Never fatal.** Discord being down, a revoked permission, or a rate limit
  must degrade to "no channel binding" — never break project creation or
  message handling. Every public entry point swallows and logs.

The REST calls go through ``httpx`` directly rather than discord.py because
provisioning runs from the CLI too, where no gateway client is live.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"

# Discord rejects channel names with uppercase/spaces/most punctuation and caps
# them at 100 chars. Slugs from projects_db are already lowercase-hyphenated,
# but a hand-edited DB row could carry anything, so normalise defensively.
_CHANNEL_SAFE_RE = re.compile(r"[^a-z0-9-]+")

DEFAULT_CHANNEL_PREFIX = "proj-"
DEFAULT_CATEGORY_NAME = "Projects"

# Where a project created FROM Discord gets its folder. Discord has no way to
# pick a filesystem path, so new projects land in one configured root as
# ``<root>/<Project Name>``. Overridable via
# ``discord.project_channels.projects_root``.
DEFAULT_PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), "Hermes Projects")

# Discord channel type constants (numeric in the REST API).
_CHANNEL_TYPE_TEXT = 0
_CHANNEL_TYPE_CATEGORY = 4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _discord_config(config: Optional[dict] = None) -> dict:
    """Return the ``discord:`` block from config.yaml (never None)."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception as exc:  # pragma: no cover - config load is best-effort
            logger.debug("project_channels: config load failed: %s", exc)
            return {}
    block = config.get("discord") if isinstance(config, dict) else None
    return block if isinstance(block, dict) else {}


def settings(config: Optional[dict] = None) -> dict:
    """Resolve ``discord.project_channels`` settings with defaults applied."""
    raw = _discord_config(config).get("project_channels")
    raw = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "guild_id": str(raw.get("guild_id") or "").strip(),
        "category_name": str(
            raw.get("category_name") or DEFAULT_CATEGORY_NAME
        ).strip(),
        "channel_prefix": str(
            raw.get("channel_prefix")
            if raw.get("channel_prefix") is not None
            else DEFAULT_CHANNEL_PREFIX
        ).strip(),
        # When true, a message in a project channel makes the agent operate in
        # that project's primary folder instead of the global TERMINAL_CWD.
        "bind_cwd": bool(raw.get("bind_cwd", True)),
        # Root folder for projects created from Discord (see create_project_from_discord).
        "projects_root": str(
            raw.get("projects_root") or DEFAULT_PROJECTS_ROOT
        ).strip(),
    }


def is_enabled(config: Optional[dict] = None) -> bool:
    s = settings(config)
    return bool(s["enabled"] and s["guild_id"])


def channel_name_for(slug: str, config: Optional[dict] = None) -> str:
    """Deterministic Discord channel name for a project slug."""
    prefix = settings(config)["channel_prefix"]
    raw = f"{prefix}{str(slug or '').strip().lower()}"
    name = _CHANNEL_SAFE_RE.sub("-", raw).strip("-")
    return (name[:100] or "proj").strip("-") or "proj"


# ---------------------------------------------------------------------------
# Discord REST
# ---------------------------------------------------------------------------


def _bot_token() -> str:
    """Read the bot token from the environment (loaded from .env at startup)."""
    return (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()


def _headers(token: str) -> Dict[str, str]:
    # Discord blocks the default urllib/httpx UA on some edges; send a real one.
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": "HermesAgent (https://github.com/NousResearch/hermes-agent, 1.0)",
        "Content-Type": "application/json",
    }


def _request(
    method: str, path: str, token: str, payload: Optional[dict] = None
) -> Optional[Any]:
    """Perform one Discord REST call. Returns parsed JSON, or None on failure."""
    import httpx

    try:
        resp = httpx.request(
            method,
            f"{DISCORD_API}{path}",
            headers=_headers(token),
            json=payload,
            timeout=15.0,
        )
    except Exception as exc:
        logger.warning("project_channels: %s %s failed: %s", method, path, exc)
        return None

    if resp.status_code == 403:
        logger.warning(
            "project_channels: Discord denied %s %s (403). The bot is missing "
            "MANAGE_CHANNELS in this guild.",
            method,
            path,
        )
        return None
    if resp.status_code == 429:
        retry_after = ""
        try:
            retry_after = str(resp.json().get("retry_after", ""))
        except Exception:
            pass
        logger.warning(
            "project_channels: rate limited on %s %s (retry_after=%s); "
            "skipping this pass",
            method,
            path,
            retry_after,
        )
        return None
    if resp.status_code >= 400:
        logger.warning(
            "project_channels: %s %s -> HTTP %s: %s",
            method,
            path,
            resp.status_code,
            resp.text[:300],
        )
        return None
    if not resp.content:
        return {}
    try:
        return resp.json()
    except Exception:
        return {}


def list_guild_channels(guild_id: str, token: str) -> List[dict]:
    data = _request("GET", f"/guilds/{guild_id}/channels", token)
    return data if isinstance(data, list) else []


def ensure_category(
    guild_id: str, token: str, name: str, channels: Optional[List[dict]] = None
) -> Optional[str]:
    """Return the id of the category ``name``, creating it if absent."""
    if not name:
        return None
    if channels is None:
        channels = list_guild_channels(guild_id, token)
    for ch in channels:
        if (
            ch.get("type") == _CHANNEL_TYPE_CATEGORY
            and str(ch.get("name", "")).strip().lower() == name.strip().lower()
        ):
            return str(ch.get("id"))
    created = _request(
        "POST",
        f"/guilds/{guild_id}/channels",
        token,
        {"name": name, "type": _CHANNEL_TYPE_CATEGORY},
    )
    if isinstance(created, dict) and created.get("id"):
        logger.info("project_channels: created category '%s'", name)
        return str(created["id"])
    return None


def ensure_channel(
    guild_id: str,
    token: str,
    channel_name: str,
    *,
    category_id: Optional[str] = None,
    topic: str = "",
    channels: Optional[List[dict]] = None,
) -> Optional[str]:
    """Return the id of text channel ``channel_name``, creating it if absent.

    Adopts a pre-existing channel with the same name rather than creating a
    duplicate — re-running provisioning after a manual channel rename or a
    lost DB binding converges instead of spamming.
    """
    if channels is None:
        channels = list_guild_channels(guild_id, token)
    wanted = channel_name.strip().lower()
    for ch in channels:
        if (
            ch.get("type") == _CHANNEL_TYPE_TEXT
            and str(ch.get("name", "")).strip().lower() == wanted
        ):
            logger.info(
                "project_channels: adopting existing channel #%s (%s)",
                channel_name,
                ch.get("id"),
            )
            return str(ch.get("id"))

    payload: Dict[str, Any] = {"name": channel_name, "type": _CHANNEL_TYPE_TEXT}
    if category_id:
        payload["parent_id"] = str(category_id)
    if topic:
        payload["topic"] = topic[:1024]
    created = _request("POST", f"/guilds/{guild_id}/channels", token, payload)
    if isinstance(created, dict) and created.get("id"):
        logger.info(
            "project_channels: created channel #%s (%s)", channel_name, created["id"]
        )
        return str(created["id"])
    return None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def provision_project(
    project_id: str, *, config: Optional[dict] = None, conn: Any = None
) -> Optional[str]:
    """Ensure a Discord channel exists for ``project_id`` and record the binding.

    Returns the channel id (existing or newly created), or None when the
    feature is disabled, the token/guild is missing, or Discord refused. Never
    raises — project creation must succeed even when Discord is unreachable.
    """
    try:
        s = settings(config)
        if not s["enabled"]:
            return None
        guild_id = s["guild_id"]
        if not guild_id:
            logger.warning(
                "project_channels: enabled but discord.project_channels.guild_id "
                "is unset; skipping provisioning"
            )
            return None
        token = _bot_token()
        if not token:
            logger.warning(
                "project_channels: DISCORD_BOT_TOKEN is not set; skipping "
                "provisioning"
            )
            return None

        from hermes_cli import projects_db as pdb

        owns_conn = conn is None
        conn = conn if conn is not None else pdb.connect()
        try:
            project = pdb.get_project(conn, project_id)
            if project is None:
                logger.warning("project_channels: no such project %s", project_id)
                return None
            if project.discord_channel_id:
                return project.discord_channel_id

            channels = list_guild_channels(guild_id, token)
            category_id = ensure_category(
                guild_id, token, s["category_name"], channels=channels
            )
            topic = f"Hermes project: {project.name}"
            if project.primary_path:
                topic += f" — {project.primary_path}"
            channel_id = ensure_channel(
                guild_id,
                token,
                channel_name_for(project.slug, config),
                category_id=category_id,
                topic=topic,
                channels=channels,
            )
            if not channel_id:
                return None
            pdb.update_project(conn, project_id, discord_channel_id=channel_id)
            return channel_id
        finally:
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(
            "project_channels: provisioning failed for %s: %s",
            project_id,
            exc,
            exc_info=True,
        )
        return None


def sync_all_projects(
    *, config: Optional[dict] = None, include_archived: bool = False
) -> List[Tuple[str, Optional[str]]]:
    """Provision channels for every project missing one.

    Returns ``[(slug, channel_id_or_None), ...]``. Used by the startup backfill
    and by ``hermes project sync-discord``.
    """
    results: List[Tuple[str, Optional[str]]] = []
    try:
        if not is_enabled(config):
            return results
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            for project in pdb.list_projects(conn, include_archived=include_archived):
                if project.discord_channel_id:
                    results.append((project.slug, project.discord_channel_id))
                    continue
                cid = provision_project(project.id, config=config, conn=conn)
                results.append((project.slug, cid))
    except Exception as exc:
        logger.warning("project_channels: sync_all_projects failed: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Resolution (channel -> project working directory)
# ---------------------------------------------------------------------------


def create_project_from_discord(
    name: str, *, config: Optional[dict] = None
) -> Dict[str, Any]:
    """Create a project from Discord: make the folder, the row, and the channel.

    Discord can't supply a filesystem path, so the folder is derived as
    ``<projects_root>/<name>``. Returns a result dict the slash handler renders;
    it never raises so a bad name or a Discord failure reports cleanly.

    Ordering matters: the folder is created FIRST, because a project row whose
    primary_path doesn't exist would silently fail cwd binding later.
    """
    result: Dict[str, Any] = {"ok": False, "error": "", "created_folder": False}
    clean = str(name or "").strip()
    if not clean:
        result["error"] = "Project name must not be empty."
        return result
    # Reject path separators / traversal outright — the name becomes a folder.
    if any(ch in clean for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|')) or ".." in clean:
        result["error"] = (
            "Project name can't contain path separators or these characters: "
            r"/ \ : * ? \" < > |"
        )
        return result

    try:
        s = settings(config)
        root = os.path.abspath(os.path.expanduser(s["projects_root"]))
        folder = os.path.join(root, clean)

        # Containment check: the resolved folder must stay under the root even
        # after normalisation, so a crafted name can't escape it.
        if os.path.commonpath([root, os.path.abspath(folder)]) != root:
            result["error"] = "Project name resolves outside the projects root."
            return result

        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            result["created_folder"] = True

        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            existing = pdb.project_for_path(conn, folder)
            if existing is not None:
                result.update(
                    ok=True,
                    already_existed=True,
                    project_id=existing.id,
                    slug=existing.slug,
                    name=existing.name,
                    folder=folder,
                    channel_id=existing.discord_channel_id,
                )
                return result
            pid = pdb.create_project(
                conn, name=clean, folders=[folder], primary_path=folder
            )
            channel_id = provision_project(pid, config=config, conn=conn)
            project = pdb.get_project(conn, pid)

        result.update(
            ok=True,
            already_existed=False,
            project_id=pid,
            slug=project.slug if project else "",
            name=clean,
            folder=folder,
            channel_id=channel_id,
        )
        return result
    except Exception as exc:
        logger.warning(
            "project_channels: create_project_from_discord(%r) failed: %s",
            name, exc, exc_info=True,
        )
        result["error"] = str(exc)
        return result


def cwd_for_channel(
    channel_id: str, parent_id: Optional[str] = None, config: Optional[dict] = None
) -> Optional[str]:
    """Return the project working directory bound to a channel, if any.

    ``parent_id`` lets a *thread* inherit its parent channel's project, which
    is the common case: the channel is the project, each thread inside it is a
    session. Returns None when the feature is off, nothing is bound, or the
    directory no longer exists on disk.
    """
    try:
        s = settings(config)
        if not (s["enabled"] and s["bind_cwd"]):
            return None
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            project = None
            for cid in (channel_id, parent_id):
                if not cid:
                    continue
                project = pdb.project_for_channel(conn, str(cid))
                if project is not None:
                    break
            if project is None:
                return None
            path = project.primary_path
            if path and os.path.isdir(path):
                return path
            if path:
                logger.warning(
                    "project_channels: project '%s' primary path does not exist: %s",
                    project.slug,
                    path,
                )
            return None
    except Exception as exc:
        logger.debug("project_channels: cwd_for_channel failed: %s", exc)
        return None


def channel_for_cwd(cwd: str, config: Optional[dict] = None) -> Optional[str]:
    """Return the Discord channel bound to the project owning ``cwd``.

    The handoff path uses this to deliver a CLI/desktop/TUI session into its
    project's channel instead of the single global home channel.
    """
    try:
        if not is_enabled(config) or not cwd:
            return None
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, str(cwd))
            if project is None:
                return None
            return project.discord_channel_id or None
    except Exception as exc:
        logger.debug("project_channels: channel_for_cwd failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session -> thread mirroring
# ---------------------------------------------------------------------------

# Sessions whose thread we've already created, keyed by session id. Persisted
# in state.db's session row (thread_id) for real bindings; this in-memory set
# only suppresses duplicate work within one gateway process.
_MIRRORED_SESSIONS: set = set()


def mirror_state_path():
    """Path of the JSON file recording session_id -> mirrored thread id."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "gateway" / "project_session_threads.json"


def _read_mirror_state() -> Dict[str, str]:
    import json

    try:
        path = mirror_state_path()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_mirror_state(state: Dict[str, str]) -> None:
    import json

    try:
        path = mirror_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("project_channels: could not persist mirror state: %s", exc)


def create_thread_in_channel(
    channel_id: str, name: str, token: Optional[str] = None
) -> Optional[str]:
    """Create a public thread under ``channel_id`` via REST. Returns its id.

    Uses the threads endpoint directly (type 11 = PUBLIC_THREAD) so this works
    from the mirror loop without a live discord.py client object.
    """
    tok = token or _bot_token()
    if not tok:
        return None
    clean = (name or "session").strip()[:100] or "session"
    created = _request(
        "POST",
        f"/channels/{channel_id}/threads",
        tok,
        {"name": clean, "type": 11, "auto_archive_duration": 10080},
    )
    if isinstance(created, dict) and created.get("id"):
        return str(created["id"])
    return None


def post_message(channel_id: str, content: str, token: Optional[str] = None) -> bool:
    """Post a plain message to a channel/thread. Best-effort."""
    tok = token or _bot_token()
    if not tok:
        return False
    res = _request(
        "POST", f"/channels/{channel_id}/messages", tok, {"content": content[:1900]}
    )
    return isinstance(res, dict)


def sessions_needing_threads(session_db: Any, *, limit: int = 500) -> List[dict]:
    """Return session rows that belong to a project but have no mirrored thread.

    A session qualifies when its cwd resolves to a project that has a bound
    Discord channel, and we have not already created a thread for it. Sessions
    that already LIVE in Discord (source='discord') are skipped: they already
    have their own thread from auto-thread, so mirroring them would duplicate.
    """
    from hermes_cli import projects_db as pdb

    out: List[dict] = []
    try:
        rows = session_db.list_sessions_rich(
            source=None, exclude_sources=["tool"], limit=limit
        )
    except Exception as exc:
        logger.debug("project_channels: could not list sessions: %s", exc)
        return out

    mirrored = _read_mirror_state()
    with pdb.connect_closing() as conn:
        for row in rows:
            sid = str(row.get("id") or "")
            if not sid or sid in mirrored or sid in _MIRRORED_SESSIONS:
                continue
            # Discord-native sessions already have a thread of their own.
            if str(row.get("source") or "") == "discord":
                continue
            cwd = row.get("cwd") or row.get("git_repo_root") or ""
            if not cwd:
                continue
            project = pdb.project_for_path(conn, str(cwd))
            if project is None or not project.discord_channel_id:
                continue
            out.append(
                {
                    "session_id": sid,
                    "title": row.get("title") or sid[:8],
                    "source": row.get("source") or "",
                    "channel_id": project.discord_channel_id,
                    "project_name": project.name,
                }
            )
    return out


def mirror_sessions_to_threads(
    session_db: Any, *, config: Optional[dict] = None, limit: int = 50
) -> List[Tuple[str, Optional[str]]]:
    """Create a Discord thread for each project-linked session missing one.

    Returns ``[(session_id, thread_id_or_None), ...]``. Idempotent: a session
    is recorded in the mirror state once its thread exists, so repeat runs are
    no-ops. Never raises.
    """
    results: List[Tuple[str, Optional[str]]] = []
    try:
        if not is_enabled(config):
            return results
        token = _bot_token()
        if not token:
            return results

        pending = sessions_needing_threads(session_db)
        if not pending:
            return results

        state = _read_mirror_state()
        for item in pending[:limit]:
            sid = item["session_id"]
            thread_name = f"{item['title']}"[:100]
            tid = create_thread_in_channel(
                item["channel_id"], thread_name, token=token
            )
            if tid:
                state[sid] = tid
                _MIRRORED_SESSIONS.add(sid)
                post_message(
                    tid,
                    f"🔗 **{item['title']}**\n"
                    f"Session `{sid}` · started on `{item['source']}`\n"
                    f"Continue it here with `/resume {sid}`.",
                    token=token,
                )
                logger.info(
                    "project_channels: mirrored session %s -> thread %s (%s)",
                    sid, tid, item["project_name"],
                )
            results.append((sid, tid))
        _write_mirror_state(state)
    except Exception as exc:
        logger.warning("project_channels: session mirroring failed: %s", exc)
    return results


__all__ = [
    "channel_for_cwd",
    "channel_name_for",
    "create_project_from_discord",
    "create_thread_in_channel",
    "cwd_for_channel",
    "ensure_category",
    "ensure_channel",
    "is_enabled",
    "list_guild_channels",
    "mirror_sessions_to_threads",
    "post_message",
    "provision_project",
    "sessions_needing_threads",
    "settings",
    "sync_all_projects",
]
