from __future__ import annotations

import re
import subprocess
from pathlib import Path


from src.env import load_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Credential patterns
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "anthropic_api_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic_bearer": re.compile(r"Bearer\s+[A-Za-z0-9_\-]{32,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
}

# Suffixes worth reading. Binary and data files are skipped.
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".sh", ".ps1", ".ipynb", ".html", ".csv", "",
}

# This file necessarily contains the patterns themselves.
SELF = Path(__file__).name


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=30, check=True,
        )
        paths = [PROJECT_ROOT / line for line in out.stdout.splitlines() if line]
        if paths:
            return paths
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # no git repo: use .gitignore rules
    excluded = {".venv", "data", "results", "results_binary", "__pycache__",
                ".pytest_cache", ".git", ".idea", ".vscode", "artifacts"}
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name == ".env":       # gitignored, and the one place a key lives
            continue
        files.append(path)
    return files


def test_no_secret_appears_in_a_tracked_file():
    offenders: list[str] = []
    for path in _tracked_files():
        if path.name == SELF or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            match = pattern.search(text)
            if match:
                rel = path.relative_to(PROJECT_ROOT)
                offenders.append(f"{rel}: {name} matched {match.group(0)[:12]}...")

    assert not offenders, (
        "Secret-shaped strings found in tracked files. Remove them, put the "
        "value in .env, and rotate the credential if it was ever committed:\n  "
        + "\n  ".join(offenders)
    )


def test_env_file_is_gitignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    entries = {line.strip() for line in gitignore.splitlines()}
    assert ".env" in entries, ".env must be listed in .gitignore"

    try:
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        assert out.returncode != 0, ".env is tracked by git -- remove and rotate"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


def test_no_key_is_hardcoded_in_the_llm_module():
    source = (PROJECT_ROOT / "src" / "models" / "llm.py").read_text(encoding="utf-8")
    assert "api_key=" not in source, (
        "src/models/llm.py passes api_key explicitly; let the SDK resolve it "
        "from the environment instead"
    )


def test_env_loader_does_not_override_a_real_environment_variable(monkeypatch,
                                                                  tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-the-file\n", encoding="utf-8")
    load_env(env)
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_env_loader_reads_a_missing_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nANTHROPIC_API_KEY='quoted-value'\nBAD LINE\n",
        encoding="utf-8",
    )
    loaded = load_env(env)
    import os

    assert "ANTHROPIC_API_KEY" in loaded
    assert os.environ["ANTHROPIC_API_KEY"] == "quoted-value"


def test_missing_env_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == []
