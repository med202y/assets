from __future__ import annotations

from pathlib import Path

_READY = False


def _posix_portable(inference):
    def _portable(path):
        path = Path(path).resolve()
        root = inference.PROJECT_ROOT
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
    return _portable


def load_ml1():
    """Charge ML 1 une seule fois et retourne la fonction predict_crisis(date) -> float."""
    global _READY
    from src.models.ml1 import inference
    if not _READY:
        inference._portable = _posix_portable(inference)
        inference.configure_inference(trusted=True)
        _READY = True
    return inference.predict_crisis