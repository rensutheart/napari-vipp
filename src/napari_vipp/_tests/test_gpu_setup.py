from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_gpu_dev.py"


def _load_setup_module():
    name = "_napari_vipp_setup_gpu_dev_test"
    spec = importlib.util.spec_from_file_location(name, SETUP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


setup_gpu_dev = _load_setup_module()


def _constraint_entries(name: str) -> set[str]:
    path = PROJECT_ROOT / "constraints" / name
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_cuda_constraint_files_pin_the_verified_runtime_components():
    cuda13 = _constraint_entries("gpu-cuda13-py312.txt")
    cuda12 = _constraint_entries("gpu-cuda12-py312.txt")

    assert {
        "cupy-cuda13x==14.1.1",
        "cuda-toolkit==13.2.2",
        "nvidia-cuda-nvrtc==13.2.86",
        "nvidia-cuda-runtime==13.2.86",
        "nvidia-nvjitlink==13.2.86",
        "nvidia-nvimgcodec-cu13==0.8.0.22",
    } <= cuda13
    assert {
        "cupy-cuda12x==14.1.1",
        "cuda-toolkit==12.9.2.0",
        "nvidia-cuda-nvrtc-cu12==12.9.86",
        "nvidia-cuda-runtime-cu12==12.9.79",
        "nvidia-nvjitlink-cu12==12.9.86",
    } <= cuda12
    assert not any("cuda12" in entry for entry in cuda13)
    assert not any("cuda13" in entry for entry in cuda12)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="CUDA setup intentionally refuses macOS before planning.",
)
def test_plan_only_prints_exact_cuda13_plan_without_writing(tmp_path, capsys):
    target = tmp_path / "dedicated-gpu-env"

    result = setup_gpu_dev.main(
        [
            "--track",
            "cuda13",
            "--python",
            sys.executable,
            "--venv",
            str(target),
            "--plan-only",
        ]
    )

    assert result == 0
    assert not target.exists()
    document = json.loads(capsys.readouterr().out)
    assert document["plan_only"] is True
    assert document["track"] == "cuda13"
    assert document["expected_cupy_distribution"] == "cupy-cuda13x"
    assert document["venv_root"] == str(target.resolve())
    assert document["constraint"].endswith("gpu-cuda13-py312.txt")

    actions = {action["name"]: action["argv"] for action in document["actions"]}
    assert actions["create_venv"] == [
        str(Path(sys.executable).resolve()),
        "-m",
        "venv",
        str(target.resolve()),
    ]
    assert actions["ensure_pip"] == [
        document["venv_python"],
        "-m",
        "ensurepip",
        "--upgrade",
    ]
    install = actions["install_project_and_cuda_runtime"]
    assert install[0] == document["venv_python"]
    assert "cupy-cuda13x[ctk]==14.1.1" in install
    assert "--constraint" in install
    assert actions["probe_cuda_runtime"][0] == document["venv_python"]


def test_track_specs_use_separate_default_venvs_and_distributions():
    cuda12 = setup_gpu_dev.TRACKS["cuda12"]
    cuda13 = setup_gpu_dev.TRACKS["cuda13"]

    assert cuda12.venv_name == ".venv-gpu-cu12"
    assert cuda13.venv_name == ".venv-gpu-cu13"
    assert cuda12.cupy_distribution == "cupy-cuda12x"
    assert cuda13.cupy_distribution == "cupy-cuda13x"
    assert cuda12.constraint_name != cuda13.constraint_name


def test_mixed_or_wrong_cupy_distributions_fail_closed():
    with pytest.raises(setup_gpu_dev.SetupError, match="mixed CuPy"):
        setup_gpu_dev._validate_cupy_distributions(
            ("cupy-cuda12x", "cupy-cuda13x"),
            expected="cupy-cuda13x",
            allow_missing=False,
        )
    with pytest.raises(setup_gpu_dev.SetupError, match="selected track"):
        setup_gpu_dev._validate_cupy_distributions(
            ("cupy-cuda12x",),
            expected="cupy-cuda13x",
            allow_missing=False,
        )
    with pytest.raises(setup_gpu_dev.SetupError, match="was not installed"):
        setup_gpu_dev._validate_cupy_distributions(
            (),
            expected="cupy-cuda13x",
            allow_missing=False,
        )


def test_cucim_wheel_requires_cuda13_cp312_and_exact_checksum(tmp_path):
    wheel = tmp_path / "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
    wheel.write_bytes(b"local experimental wheel fixture")
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()

    validated = setup_gpu_dev._validated_cucim_wheel(
        setup_gpu_dev.TRACKS["cuda13"],
        wheel,
        expected.upper(),
    )

    assert validated is not None
    assert validated.path == wheel.resolve()
    assert validated.sha256 == expected
    with pytest.raises(setup_gpu_dev.SetupError, match="does not match"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            "0" * 64,
        )
    with pytest.raises(setup_gpu_dev.SetupError, match="CUDA 13 only"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda12"],
            wheel,
            expected,
        )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="CUDA setup intentionally refuses macOS before planning.",
)
def test_cucim_plan_installs_only_the_checksum_verified_local_wheel(tmp_path):
    wheel = tmp_path / "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
    wheel.write_bytes(b"local experimental wheel fixture")
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()
    target = tmp_path / "gpu-venv"

    plan = setup_gpu_dev.create_setup_plan(
        track_name="cuda13",
        base_python=sys.executable,
        venv_path=target,
        cucim_wheel=wheel,
        cucim_sha256=expected,
    )

    assert not target.exists()
    assert plan.install_cucim is not None
    assert plan.install_cucim.argv[-1] == str(wheel.resolve())
    assert "--no-deps" in plan.install_cucim.argv
    assert "--force-reinstall" in plan.install_cucim.argv
    assert "nvidia-nvimgcodec-cu13==0.8.0.22" in plan.install_project.argv
    assert plan.cucim_probe is not None
    assert plan.actions[-1] is plan.pip_check


def test_cucim_wheel_and_checksum_must_be_provided_together(tmp_path):
    wheel = tmp_path / "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
    wheel.write_bytes(b"fixture")

    with pytest.raises(setup_gpu_dev.SetupError, match="provided together"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            None,
        )


def test_unsafe_venv_targets_and_macos_are_rejected(tmp_path):
    with pytest.raises(setup_gpu_dev.SetupError, match="project root"):
        setup_gpu_dev._validate_venv_target(
            PROJECT_ROOT,
            project_root=PROJECT_ROOT,
        )
    with pytest.raises(setup_gpu_dev.SetupError, match="Windows and Linux"):
        setup_gpu_dev._require_supported_platform("darwin")

    ordinary_directory = tmp_path / "not-a-venv"
    ordinary_directory.mkdir()
    expected_python = setup_gpu_dev._venv_python(
        ordinary_directory,
        platform_name=sys.platform,
    )
    with pytest.raises(setup_gpu_dev.SetupError, match="not a complete"):
        setup_gpu_dev._validate_existing_venv(
            ordinary_directory,
            expected_python,
        )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="CUDA setup intentionally refuses macOS before planning.",
)
def test_all_install_and_probe_actions_target_only_the_dedicated_venv(tmp_path):
    target = tmp_path / "gpu-venv"
    plan = setup_gpu_dev.create_setup_plan(
        track_name="cuda12",
        base_python=sys.executable,
        venv_path=target,
    )

    assert plan.create_venv is not None
    assert plan.create_venv.argv[0] == str(Path(sys.executable).resolve())
    assert plan.create_venv.argv[1:3] == ("-m", "venv")
    dedicated_python = str(plan.venv_python)
    for action in (
        plan.ensure_pip,
        plan.upgrade_tools,
        plan.install_project,
        plan.pip_check,
        plan.gpu_probe,
    ):
        assert action.argv[0] == dedicated_python
        assert action.argv[0] != str(plan.base_python)


def test_platform_wrappers_delegate_to_the_shared_setup_helper():
    powershell = (PROJECT_ROOT / "scripts" / "setup_gpu_dev.ps1").read_text(
        encoding="utf-8"
    )
    shell = (PROJECT_ROOT / "scripts" / "setup_gpu_dev.sh").read_text(
        encoding="utf-8"
    )

    assert "setup_gpu_dev.py" in powershell
    assert 'setup_gpu_dev.py"' in shell
    assert "VIPP_GPU_SETUP_PYTHON" in powershell
    assert "VIPP_GPU_SETUP_PYTHON" in shell
