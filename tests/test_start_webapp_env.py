import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from start_webapp import build_conda_reexec_command, configure_runtime_environment, should_auto_reexec_conda


def test_should_auto_reexec_conda_when_torch_missing_outside_target_env(monkeypatch):
    monkeypatch.delenv("MASK_ITERATION_SKIP_CONDA_REEXEC", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    assert should_auto_reexec_conda(torch_available=False, env_name="mask_iteration_sam3") is True


def test_should_not_reexec_inside_target_env(monkeypatch):
    monkeypatch.delenv("MASK_ITERATION_SKIP_CONDA_REEXEC", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "mask_iteration_sam3")
    assert should_auto_reexec_conda(torch_available=False, env_name="mask_iteration_sam3") is False


def test_build_conda_reexec_command_preserves_args():
    command = build_conda_reexec_command(
        conda_bin="/opt/anaconda3/bin/conda",
        env_name="mask_iteration_sam3",
        script_path=Path("/project/start_webapp.py"),
        argv=["--port", "8766"],
    )
    assert command == [
        "/opt/anaconda3/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "mask_iteration_sam3",
        "python",
        "/project/start_webapp.py",
        "--port",
        "8766",
    ]


def test_configure_runtime_environment_sets_mps_fallback(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    configure_runtime_environment()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
