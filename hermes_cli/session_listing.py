"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

from typing import Any


def parse_session_listing_args(raw_args: str) -> tuple[bool, bool, str, str | None]:
    """Parse `/sessions`-style args into listing flags, a resume target, and a search query.

    Returns ``(include_all_sources, include_unnamed, target, search_query)``.
    ``list``/``ls`` and ``browse`` are display aliases; ``all``/``--all`` widens
    source scope; ``full``/``--full`` keeps unnamed sessions in the listing.
    ``search``/``find`` makes the remaining words a search query —
    ``search_query`` is ``None`` when search wasn't requested and ``""`` when it
    was requested without a query. Flags are only honored before the first
    positional word, so titles containing e.g. "all" aren't misparsed. Anything
    else is treated as a target so `/sessions <id-or-title>` can delegate to
    `/resume`.
    """
    import shlex

    parts = shlex.split(raw_args or "")
    include_all = False
    include_unnamed = False
    target_parts: list[str] = []
    for i, part in enumerate(parts):
        lower = part.strip().lower()
        if not target_parts:
            if lower in {"list", "ls", "browse"}:
                continue
            if lower in {"all", "--all"}:
                include_all = True
                continue
            if lower in {"full", "--full"}:
                include_unnamed = True
                continue
            if lower in {"search", "find"}:
                query = " ".join(parts[i + 1:]).strip()
                return include_all, include_unnamed, "", query
        target_parts.append(part)
    return include_all, include_unnamed, " ".join(target_parts).strip(), None


def parse_project_listing_args(raw_args: str) -> tuple[bool, str] | None:
    """Detect and parse the ``/sessions project [slug]`` form.

    Returns ``(is_project_mode, slug)`` where ``slug`` is empty to mean "group
    every project". Returns ``None`` when this isn't the project form, so the
    caller falls through to :func:`parse_session_listing_args` unchanged.

    Kept separate from the main parser because project mode changes the SHAPE
    of the output (grouped by project, not one flat list), so it can't just be
    another boolean flag on the existing path.
    """
    import shlex

    try:
        parts = shlex.split(raw_args or "")
    except ValueError:
        return None
    if not parts:
        return None
    head = parts[0].strip().lower()
    if head not in {"project", "projects", "proj"}:
        return None
    slug = " ".join(parts[1:]).strip()
    return True, slug


def query_sessions_by_project(
    session_db: Any,
    *,
    slug: str = "",
    limit_per_project: int = 10,
    scan_limit: int = 500,
) -> list[dict[str, Any]]:
    """Group sessions by the Hermes project owning their working directory.

    Returns ``[{"project": <Project>, "sessions": [row, ...]}, ...]`` ordered
    by project creation. A session belongs to a project when its ``cwd`` (or
    ``git_repo_root``) is the project's folder or nested under it — the same
    longest-prefix rule ``projects_db.project_for_path`` uses everywhere else,
    so this view can't disagree with kanban worktrees or desktop grouping.

    Sessions with no cwd (every gateway session predating cwd binding) simply
    match no project and are omitted rather than guessed at.
    """
    from hermes_cli import projects_db as pdb

    rows = session_db.list_sessions_rich(
        source=None, exclude_sources=["tool"], limit=scan_limit
    )
    out: list[dict[str, Any]] = []
    with pdb.connect_closing() as conn:
        projects = pdb.list_projects(conn)
        wanted = (slug or "").strip().lower()
        if wanted:
            projects = [
                p
                for p in projects
                if p.slug.lower() == wanted or p.name.lower() == wanted
            ]
        # Resolve each session ONCE, then bucket — avoids re-walking the folder
        # table per project per session.
        buckets: dict[str, list[dict[str, Any]]] = {p.id: [] for p in projects}
        for row in rows:
            cwd = row.get("cwd") or row.get("git_repo_root") or ""
            if not cwd:
                continue
            proj = pdb.project_for_path(conn, str(cwd))
            if proj is None or proj.id not in buckets:
                continue
            if len(buckets[proj.id]) < limit_per_project:
                buckets[proj.id].append(row)
        for p in projects:
            out.append({"project": p, "sessions": buckets.get(p.id, [])})
    return out


def format_project_session_listing(
    groups: list[dict[str, Any]],
    *,
    channel_name_fn: Any = None,
) -> str:
    """Render sessions grouped by project for gateway messengers."""
    if not groups:
        return (
            "No projects found.\n"
            "Create one with `hermes project create <name>`."
        )

    lines: list[str] = ["📁 **Sessions by project**", ""]
    total = 0
    for group in groups:
        proj = group["project"]
        sessions = group["sessions"]
        total += len(sessions)
        header = f"**{proj.name}**"
        if channel_name_fn is not None:
            try:
                header += f" · #{channel_name_fn(proj.slug)}"
            except Exception:
                pass
        lines.append(f"{header} — {len(sessions)} session(s)")
        if not sessions:
            lines.append("  _none yet_")
        for row in sessions:
            title = str(row.get("title") or "").strip()
            sid = str(row.get("id") or "")
            src = str(row.get("source") or "")
            label = title or (str(row.get("preview") or "")[:40] or "—")
            lines.append(f"  • {label} `{src}` — `{sid}`")
        lines.append("")

    if total == 0:
        lines.append(
            "_No sessions are linked to a project yet. A session links when it "
            "runs in a project folder — start one in that project's channel._"
        )
    else:
        lines.append("Resume any of them with `/resume <session id>`.")
    return "\n".join(lines)


def query_session_listing(
    session_db: Any,
    *,
    source: str | None,
    session_key: str | None = None,
    current_session_id: str | None = None,
    include_all_sources: bool = False,
    include_unnamed: bool = False,
    search_query: str | None = None,
    limit: int = 10,
    exclude_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return session rows for interactive listing surfaces.

    This is the shared selection policy behind CLI/gateway session browsing:
    source-scoped by default, optionally global, hide unnamed sessions unless
    the caller asks for a full listing, and never include the current session.
    ``session_key`` further restricts gateway callers to one exact conversation
    lane before the database applies its result limit.
    With ``search_query``, rows are filtered by title/id match (SQL-level, see
    ``SessionDB.list_sessions_rich``) and ordered by most-recent activity;
    unnamed sessions stay visible since an id match may be the only handle.
    """
    query_source = None if include_all_sources else source
    fetch_limit = max(limit * 4, limit)
    search = (search_query or "").strip()
    rows = session_db.list_sessions_rich(
        source=query_source,
        session_key=session_key,
        exclude_sources=exclude_sources,
        limit=fetch_limit,
        search_query=search or None,
        order_by_last_active=bool(search),
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        if current_session_id and row.get("id") == current_session_id:
            continue
        if not include_unnamed and not row.get("title") and not search:
            continue
        result.append(row)
        if len(result) >= limit:
            break
    return result


def format_gateway_session_listing(
    rows: list[dict[str, Any]],
    *,
    include_source: bool = False,
    title: str = "Sessions",
) -> str:
    """Render a compact Markdown-ish session list for gateway messengers."""
    if not rows:
        return (
            "No sessions found.\n"
            "Use `/title My Session` to name this chat, or `/sessions full` "
            "to include unnamed sessions."
        )

    lines = [f"📋 **{title}**", ""]
    for idx, row in enumerate(rows, start=1):
        session_id = str(row.get("id") or "")
        title_text = str(row.get("title") or "—")
        preview = str(row.get("preview") or "")[:40]
        source = str(row.get("source") or "")
        source_part = f" `{source}`" if include_source and source else ""
        preview_part = f" — _{preview}_" if preview else ""
        lines.append(f"{idx}. **{title_text}**{source_part} — `{session_id}`{preview_part}")
    lines.append("")
    lines.append("Resume: `/resume <session id>` or `/resume <number>` from `/resume`.")
    lines.append("More: `/sessions all`, `/sessions full`, `/sessions search <query>`.")
    return "\n".join(lines)
