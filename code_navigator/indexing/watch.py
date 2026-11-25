"""Keep an index fresh: a polling file watcher and git hooks. No extra dependencies."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ..config import INDEX_DIR_NAME
from ..ingestion.repo_loader import SKIP_DIRS, _skip_file

HOOK_NAMES = ("post-commit", "post-merge", "post-checkout", "post-rewrite")
_BEGIN, _END = "# >>> code-navigator >>>", "# <<< code-navigator <<<"


# ---------------------------------------------------------------------------- watcher

def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """(mtime_ns, size) of every file the indexer could read."""
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and d != INDEX_DIR_NAME]
        for name in filenames:
            if _skip_file(name):
                continue
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            out[path.relative_to(root).as_posix()] = (st.st_mtime_ns, st.st_size)
    return out


def watch(root: str | Path, on_change: Callable[[list[str]], None], interval: float = 2.0,
          max_cycles: int | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
    """Poll ``root`` and call ``on_change(changed_paths)`` once edits have settled for one interval."""
    root = Path(root).expanduser().resolve()
    last = snapshot(root)
    pending: set[str] = set()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        sleep(interval)
        now = snapshot(root)
        diff = {p for p in set(last) | set(now) if last.get(p) != now.get(p)}
        last = now
        if diff:
            pending |= diff          # still changing - wait for the next quiet interval
        elif pending:
            changed, pending = sorted(pending), set()
            on_change(changed)


# ---------------------------------------------------------------------------- git hooks

def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise ValueError(f"{repo} is not a git repository ({res.stderr.strip() or 'git failed'})")
    return res.stdout.strip()


def hooks_dir(repo: str | Path) -> Path:
    repo = Path(repo).expanduser().resolve()
    path = Path(_git(repo, "rev-parse", "--git-path", "hooks"))
    return path if path.is_absolute() else (repo / path).resolve()


def _block(index_dir: str | None) -> str:
    extra = f' --index-dir "{index_dir}"' if index_dir else ""
    cmd = f'"{sys.executable}" -m code_navigator update "$(git rev-parse --show-toplevel)"{extra} --quiet'
    return (f"{_BEGIN}\n# Re-index in the background after git changes the working tree.\n"
            f"( {cmd} >/dev/null 2>&1 & )\n{_END}\n")


def _strip(text: str) -> str:
    if _BEGIN not in text:
        return text
    head, _, rest = text.partition(_BEGIN)
    _, _, tail = rest.partition(_END)
    return (head.rstrip("\n") + "\n" + tail.lstrip("\n")) if head.strip() else tail.lstrip("\n")


def install_hooks(repo: str | Path, index_dir: str | None = None) -> list[str]:
    d = hooks_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    done = []
    for name in HOOK_NAMES:
        path = d / name
        text = path.read_text(encoding="utf-8") if path.exists() else "#!/bin/sh\n"
        text = _strip(text)
        if not text.startswith("#!"):
            text = "#!/bin/sh\n" + text
        path.write_text(text.rstrip("\n") + "\n\n" + _block(index_dir), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        done.append(str(path))
    return done


def uninstall_hooks(repo: str | Path) -> list[str]:
    d = hooks_dir(repo)
    done = []
    for name in HOOK_NAMES:
        path = d / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if _BEGIN not in text:
            continue
        rest = _strip(text)
        if rest.strip() in ("", "#!/bin/sh"):
            path.unlink()
        else:
            path.write_text(rest, encoding="utf-8")
        done.append(str(path))
    return done


def hook_status(repo: str | Path) -> dict[str, bool]:
    d = hooks_dir(repo)
    return {name: (d / name).exists() and _BEGIN in (d / name).read_text(encoding="utf-8") for name in HOOK_NAMES}
