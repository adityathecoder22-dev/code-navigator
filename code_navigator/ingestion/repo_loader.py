"""Load a repository (local folder or git URL) and classify its files."""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..models import SourceFile
from ..utils import log, safe_project_name, sha1

LANG_BY_EXT = {
    ".py": "python", ".java": "java", ".go": "go",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".cs": "csharp", ".kt": "kotlin", ".rs": "rust", ".rb": "ruby", ".php": "php", ".scala": "scala",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "target", "build", "dist", "out", "bin", "obj",
    "__pycache__", ".venv", "venv", "env", ".idea", ".vscode", ".gradle", ".mvn", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache", ".tox", "genproto", "third_party", ".terraform",
}
TEST_DIRS = {"test", "tests", "__tests__", "spec", "e2e", "integration-tests", "loadgenerator-tests"}

SKIP_FILE_PATTERNS = [
    "*_pb2.py", "*_pb2_grpc.py", "*.pb.go", "*_grpc.pb.go", "*.pb.cc", "*.pb.h", "*Grpc.java",
    "*.min.js", "*.bundle.js", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.sum", "*.lock", ".env", ".env.*", "*.pem", "*.key", "*.crt", "*.p12", "*.jks",
]

MANIFEST_FILES = {
    "pom.xml", "build.gradle", "build.gradle.kts", "go.mod", "requirements.txt", "pyproject.toml",
    "setup.py", "package.json", "Cargo.toml", "Gemfile", "composer.json",
}
COMPOSE_PATTERN = re.compile(r"^(docker-)?compose([.-][\w.-]+)?\.ya?ml$", re.IGNORECASE)


@dataclass
class Repository:
    name: str
    root: Path
    files: list[SourceFile] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[SourceFile]:
        return [f for f in self.files if f.kind == kind]

    def config_files(self, subtype: str | None = None) -> list[SourceFile]:
        return [f for f in self.files if f.kind == "config" and (subtype is None or f.language == subtype)]


def _is_git_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@", "ssh://")) or source.endswith(".git")


def _clone(url: str, dest: Path) -> None:
    if dest.exists() and (dest / ".git").exists():
        log.info("Repository already cloned at %s (pulling latest)", dest)
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Cloning %s -> %s", url, dest)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def classify(rel_path: str, text: str) -> tuple[str, str] | None:
    """Return (kind, language/subtype) or None if the file should be ignored."""
    name = os.path.basename(rel_path)
    lower = name.lower()
    ext = os.path.splitext(lower)[1]

    if COMPOSE_PATTERN.match(name):
        return "config", "compose"
    if lower == "dockerfile" or lower.endswith(".dockerfile") or lower.startswith("dockerfile."):
        return "config", "dockerfile"
    if ext == ".proto":
        return "config", "proto"
    if name in MANIFEST_FILES or ext == ".csproj":
        return "config", "manifest"
    if ext in (".yaml", ".yml"):
        head = text[:4000]
        if re.search(r"^\s*(openapi|swagger)\s*:", head, re.MULTILINE):
            return "config", "openapi"
        if re.search(r"^apiVersion\s*:", text, re.MULTILINE) and re.search(r"^kind\s*:", text, re.MULTILINE):
            return "config", "k8s"
        if lower.startswith(("application", "bootstrap")):
            return "config", "app_config"
        return "config", "yaml"
    if ext == ".json":
        if re.search(r'"(openapi|swagger)"\s*:', text[:2000]):
            return "config", "openapi"
        return None
    if ext == ".properties" and lower.startswith(("application", "bootstrap")):
        return "config", "app_config"
    if ext in (".md", ".rst") or lower in ("readme", "readme.txt"):
        return "doc", "markdown"
    if ext in LANG_BY_EXT:
        return "code", LANG_BY_EXT[ext]
    return None


def _skip_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_FILE_PATTERNS)


def load_repository(source: str, settings: Settings, name: str | None = None) -> Repository:
    """Load from a local directory or clone a git URL into CACHE_DIR/repos/<name>."""
    if _is_git_url(source):
        repo_name = safe_project_name(name or Path(source.rstrip("/")).stem)
        root = settings.cache_dir / "repos" / repo_name
        _clone(source, root)
    else:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repository folder not found: {root}")
        repo_name = safe_project_name(name or root.name)

    repo = Repository(name=repo_name, root=root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
            and (settings.include_tests or d.lower() not in TEST_DIRS)
        )
        for fname in sorted(filenames):
            if _skip_file(fname):
                continue
            abs_path = Path(dirpath) / fname
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > settings.max_file_bytes:
                continue
            try:
                raw = abs_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:2048]:  # binary
                continue
            text = raw.decode("utf-8", errors="replace")
            rel = abs_path.relative_to(root).as_posix()
            cls = classify(rel, text)
            if cls is None:
                continue
            kind, lang = cls
            repo.files.append(SourceFile(
                path=rel, abs_path=str(abs_path), kind=kind, language=lang,
                size=size, sha1=sha1(text), text=text,
            ))
    log.info("Loaded %d files from %s (%d code, %d config, %d docs)", len(repo.files), root,
             len(repo.by_kind("code")), len(repo.by_kind("config")), len(repo.by_kind("doc")))
    return repo
