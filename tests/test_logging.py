"""Tests for nlp_shap logging bootstrap."""

import logging
import textwrap
from pathlib import Path

import pytest

from nlp_shap._logging import (
    ENV_LOGGING_CONFIG,
    bootstrap_logging,
    find_logging_config,
)


@pytest.fixture
def reset_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset bootstrap state and logging handlers between tests."""
    monkeypatch.setattr("nlp_shap._logging._bootstrapped", False)
    for name in ("nlp_shap", "nlp_shap.tests"):
        package_logger = logging.getLogger(name)
        package_logger.handlers.clear()
        package_logger.setLevel(logging.NOTSET)
        package_logger.propagate = True
    monkeypatch.delenv(ENV_LOGGING_CONFIG, raising=False)


def test_find_logging_config_from_env(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NLP_SHAP_LOGGING_CONFIG selects an explicit TOML file."""
    config_path = tmp_path / "custom.toml"
    config_path.write_text("[tool.logging]\nversion = 1\n", encoding="utf-8")
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))

    assert find_logging_config() == config_path


def test_find_logging_config_from_pyproject(
    reset_bootstrap: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without env override, pyproject.toml in cwd is used."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert find_logging_config() == Path.cwd() / "pyproject.toml"


def test_bootstrap_applies_toml_config(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap_logging loads handlers from [tool.logging]."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [tool.logging]
            version = 1
            disable_existing_loggers = false

            [tool.logging.formatters.standard]
            format = "%(levelname)s %(message)s"

            [tool.logging.handlers.console]
            class = "logging.StreamHandler"
            formatter = "standard"
            stream = "ext://sys.stderr"

            [tool.logging.loggers.nlp_shap]
            level = "INFO"
            handlers = ["console"]
            propagate = false
            """
        ).strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()

    package_logger = logging.getLogger("nlp_shap")
    assert package_logger.level == logging.INFO
    assert any(
        isinstance(handler, logging.StreamHandler) for handler in package_logger.handlers
    )


def test_bootstrap_uses_null_handler_without_config(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing config file falls back to a null handler."""
    monkeypatch.chdir(tmp_path)
    bootstrap_logging()

    package_logger = logging.getLogger("nlp_shap")
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.NullHandler)


def test_bootstrap_uses_null_handler_without_logging_section(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML without [tool.logging] falls back to a null handler."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text("[project]\nname = \"example\"\n", encoding="utf-8")
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()

    package_logger = logging.getLogger("nlp_shap")
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.NullHandler)


def test_bootstrap_is_idempotent(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated bootstrap calls must not reconfigure logging."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [tool.logging]
            version = 1
            disable_existing_loggers = false

            [tool.logging.formatters.standard]
            format = "%(message)s"

            [tool.logging.handlers.console]
            class = "logging.StreamHandler"
            formatter = "standard"
            stream = "ext://sys.stderr"

            [tool.logging.loggers.nlp_shap]
            level = "WARNING"
            handlers = ["console"]
            propagate = false
            """
        ).strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()
    handler_count = len(logging.getLogger("nlp_shap").handlers)

    bootstrap_logging()

    assert len(logging.getLogger("nlp_shap").handlers) == handler_count
