"""Central logging bootstrap for nlp_shap."""

import logging
import os
from pathlib import Path
from typing import Final

import logging518.config

PACKAGE_NAME: Final[str] = "nlp_shap"
"""Root logger namespace for the package."""

ENV_LOGGING_CONFIG: Final[str] = "NLP_SHAP_LOGGING_CONFIG"
"""Environment variable pointing at a TOML file with a ``[tool.logging]`` section."""

PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
"""Default TOML filename searched for logging configuration."""

_bootstrapped: bool = False
"""Whether :func:`bootstrap_logging` has run in this process."""


def _ensure_null_handler() -> None:
    package_logger = logging.getLogger(PACKAGE_NAME)
    if not package_logger.handlers:
        package_logger.addHandler(logging.NullHandler())


def find_logging_config() -> Path | None:
    """Return the TOML file that defines ``[tool.logging]``, if any."""
    env_path = os.getenv(ENV_LOGGING_CONFIG)
    if env_path:
        path = Path(env_path)
        return path if path.is_file() else None

    current = Path.cwd()
    for directory in (current, *current.parents):
        candidate = directory / PYPROJECT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def bootstrap_logging() -> None:
    """Load logging from TOML or keep imports quiet.

    Looks for ``[tool.logging]`` in ``pyproject.toml`` (walked up from the
    current working directory) or in the file named by
    ``NLP_SHAP_LOGGING_CONFIG``. Uses ``logging518`` to pass the section to
    :func:`logging.config.dictConfig`.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    config_path = find_logging_config()
    if config_path is None:
        _ensure_null_handler()
        return

    try:
        logging518.config.fileConfig(config_path)
    except KeyError:
        _ensure_null_handler()
