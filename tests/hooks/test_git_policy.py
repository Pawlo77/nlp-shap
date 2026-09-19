"""Tests for Cursor git/PR beforeShellExecution policy."""

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".cursor" / "hooks" / "git_policy.py"


@pytest.fixture(scope="module")
def git_policy():
    """Load git_policy.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("git_policy", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("cmd", "permission"),
    [
        ("git commit --no-verify -m 'feat: x'", "deny"),
        ("git commit -n -m 'feat: x'", "deny"),
        ("git commit", "deny"),
        ("git commit -m 'WIP'", "deny"),
        ("git commit -m 'feat: trailing.'", "deny"),
        ("git commit -m 'feat: add thing'", "ask"),
        (
            "git commit -m \"$(cat <<'EOF'\nfeat: add thing\n\nBody.\nEOF\n)\"",
            "ask",
        ),
        ("git push origin main", "deny"),
        ("git push origin master", "deny"),
        ("git push origin HEAD:main", "deny"),
        ("git push -u origin feature/foo", "ask"),
        ("gh pr create --title 'feat: x' --body 'y'", "ask"),
        ("gh pr merge 1", "ask"),
        ("git status", None),
        ("uv run pytest", None),
    ],
)
def test_decide_git(git_policy, cmd: str, permission: str | None) -> None:
    """Commit/push/PR policy returns deny, ask, or None."""
    result = git_policy.decide_git(cmd)
    if permission is None:
        assert result is None
    else:
        assert result is not None
        assert result["permission"] == permission


def test_extract_heredoc_message(git_policy) -> None:
    """HEREDOC -m body is extracted for validation."""
    cmd = "git commit -m \"$(cat <<'EOF'\nfeat: add thing\n\nMore.\nEOF\n)\""
    assert git_policy.extract_commit_message(cmd) is not None
    assert "feat: add thing" in git_policy.extract_commit_message(cmd)
