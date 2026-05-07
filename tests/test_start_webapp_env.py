import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import start_webapp
from start_webapp import configure_runtime_environment, require_current_environment_torch


def test_configure_runtime_environment_sets_mps_fallback(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    configure_runtime_environment()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_require_current_environment_torch_uses_current_python_only(monkeypatch):
    monkeypatch.setattr(start_webapp, "torch_available", lambda: False)

    with pytest.raises(ModuleNotFoundError) as error:
        require_current_environment_torch()

    assert str(sys.executable) in str(error.value)
    assert "current Python environment" in str(error.value)


def test_require_current_environment_torch_passes_when_torch_is_available(monkeypatch):
    monkeypatch.setattr(start_webapp, "torch_available", lambda: True)
    require_current_environment_torch()
