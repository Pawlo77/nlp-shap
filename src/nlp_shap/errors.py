"""Typed errors raised by nlp-shap pipeline components."""


class BackendUnavailableError(RuntimeError):
    """Raised when a configured generative backend cannot be reached."""
