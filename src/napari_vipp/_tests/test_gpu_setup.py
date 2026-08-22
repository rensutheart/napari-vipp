from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

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

GPU_SETUP_INTEGRATION_SUPPORTED = (
    sys.platform != "darwin"
    and sys.implementation.name == "cpython"
    and sys.version_info[:2] == (3, 12)
)
gpu_setup_integration = pytest.mark.skipif(
    not GPU_SETUP_INTEGRATION_SUPPORTED,
    reason=(
        "The CUDA setup integration uses the real base interpreter and supports "
        "only native Windows/Linux 64-bit CPython 3.12."
    ),
)


def _constraint_entries(name: str) -> set[str]:
    path = PROJECT_ROOT / "constraints" / name
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _project_metadata() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_cuda_constraint_files_pin_the_verified_runtime_components():
    cuda13 = _constraint_entries("gpu-cuda13-py312.txt")
    cuda12 = _constraint_entries("gpu-cuda12-py312.txt")
    scientific_stack = {
        "numpy==2.5.1",
        "scipy==1.18.0",
        "scikit-image==0.26.0",
    }

    assert scientific_stack <= cuda13
    assert scientific_stack <= cuda12
    assert {
        "cupy-cuda13x==14.1.1",
        "cuda-toolkit==13.2.2",
        "nvidia-cuda-nvrtc==13.2.86",
        "nvidia-cuda-runtime==13.2.86",
        "nvidia-nvjitlink==13.2.86",
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


def test_published_gpu_extras_pin_the_admitted_scientific_stack():
    extras = _project_metadata()["optional-dependencies"]

    for track in ("gpu-cuda12", "gpu-cuda13"):
        requirements = tuple(extras[track])
        for pin in ("numpy==2.5.1", "scipy==1.18.0", "scikit-image==0.26.0"):
            assert any(
                requirement.startswith(f"{pin};") for requirement in requirements
            )
        assert all(
            "python_version == '3.12'" in requirement
            for requirement in requirements
        )
        assert all(
            "platform_system == 'Windows'" in requirement
            for requirement in requirements
        )
        assert all(
            "platform_system == 'Linux'" in requirement
            for requirement in requirements
        )


def test_development_extra_pins_the_admitted_scientific_stack():
    requirements = set(_project_metadata()["optional-dependencies"]["dev"])

    assert {
        "numpy==2.5.1",
        "scipy==1.18.0",
        "scikit-image==0.26.0",
    } <= requirements


def test_release_metadata_bounds_cpu_python_support_to_ci_matrix():
    metadata = _project_metadata()

    assert metadata["version"] == "0.13.0a7"
    assert metadata["requires-python"] == ">=3.12,<3.14"
    assert "Programming Language :: Python :: 3.12" in metadata["classifiers"]
    assert "Programming Language :: Python :: 3.13" in metadata["classifiers"]


@gpu_setup_integration
@pytest.mark.parametrize("track_name", ("cuda12", "cuda13"))
def test_each_setup_plan_selects_constraints_with_the_validated_scientific_stack(
    tmp_path,
    track_name,
):
    plan = setup_gpu_dev.create_setup_plan(
        track_name=track_name,
        base_python=sys.executable,
        venv_path=tmp_path / f"{track_name}-venv",
    )

    assert {
        "numpy==2.5.1",
        "scipy==1.18.0",
        "scikit-image==0.26.0",
    } <= _constraint_entries(plan.constraint_path.name)


@gpu_setup_integration
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
    assert document["schema"] == "napari-vipp-gpu-development-plan"
    assert document["schema_version"] == 2
    assert document["plan_only"] is True
    assert document["track"] == "cuda13"
    assert document["expected_cupy_distribution"] == "cupy-cuda13x"
    assert document["venv_root"] == str(target.resolve())
    assert document["constraint"].endswith("gpu-cuda13-py312.txt")
    record = document["environment_record"]
    assert record == {
        "path": str(
            target.resolve() / "share" / "napari-vipp" / "gpu-environment.json"
        ),
        "document": {
            "schema": "napari-vipp-gpu-environment",
            "schema_version": 3,
            "track": "cuda13",
            "cupy_distribution": "cupy-cuda13x",
        },
        "write_after": ["probe_cuda_runtime", "check_dependencies"],
    }

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


def test_setup_probes_use_only_private_memory_pools():
    probe = setup_gpu_dev.GPU_PROBE
    assert "cuda.MemoryPool()" in probe
    assert "cuda.using_allocator(pool.malloc)" in probe
    assert "pool.free_all_blocks()" in probe
    assert "pool.used_bytes()" in probe
    assert "pool.total_bytes()" in probe
    assert "get_default_memory_pool" not in probe


@pytest.mark.parametrize("failure_point", ["operation", "synchronize"])
def test_setup_probes_release_private_pool_without_masking_primary_failure(
    monkeypatch,
    failure_point,
):
    events = []
    primary_error = RuntimeError(f"simulated {failure_point} failure")

    class Array:
        shape = (4, 4)

        def reshape(self, *_shape):
            return self

    class Pool:
        def malloc(self, _size):
            raise AssertionError("The test array does not use the CuPy allocator")

        def free_all_blocks(self):
            events.append("free")

        @staticmethod
        def used_bytes():
            return 0

        @staticmethod
        def total_bytes():
            return 0

    class Stream:
        def synchronize(self):
            events.append("sync")
            if failure_point == "synchronize":
                raise primary_error

    class Cuda:
        MemoryPool = Pool

        @staticmethod
        @contextmanager
        def using_allocator(_allocator):
            yield

        @staticmethod
        def get_current_stream():
            return Stream()

    cupy = ModuleType("cupy")
    cupy.float32 = "float32"
    cupy.cuda = Cuda()
    cupy.arange = lambda *_args, **_kwargs: Array()
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    cupyx = ModuleType("cupyx")
    scipy = ModuleType("cupyx.scipy")
    ndimage = ModuleType("cupyx.scipy.ndimage")

    def gaussian_filter(values, *_args, **_kwargs):
        if failure_point == "operation":
            raise primary_error
        return values

    ndimage.gaussian_filter = gaussian_filter
    ndimage.median_filter = lambda values, **_kwargs: values
    scipy.ndimage = ndimage
    cupyx.scipy = scipy
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", ndimage)

    with pytest.raises(RuntimeError) as exc_info:
        exec(setup_gpu_dev.GPU_PROBE, {})

    assert exc_info.value is primary_error
    assert "free" in events


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


def test_environment_record_path_cannot_escape_venv(tmp_path, monkeypatch):
    venv_root = tmp_path / "gpu-venv"
    venv_root.mkdir()
    expected = venv_root / setup_gpu_dev.GPU_ENVIRONMENT_RECORD_RELATIVE_PATH
    outside = tmp_path / "outside" / "gpu-environment.json"

    with pytest.raises(setup_gpu_dev.SetupError, match="inside its dedicated venv"):
        setup_gpu_dev._validated_environment_record_path(
            outside,
            venv_root=venv_root,
        )

    share = venv_root / "share"
    original_predicate = setup_gpu_dev._is_link_or_junction
    monkeypatch.setattr(
        setup_gpu_dev,
        "_is_link_or_junction",
        lambda path: path == share or original_predicate(path),
    )
    with pytest.raises(setup_gpu_dev.SetupError, match="symlink or junction parent"):
        setup_gpu_dev._validated_environment_record_path(
            expected,
            venv_root=venv_root,
        )


@gpu_setup_integration
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


@gpu_setup_integration
def test_successful_setup_writes_record_only_after_probes_and_pip_check(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "gpu-venv"
    plan = setup_gpu_dev.create_setup_plan(
        track_name="cuda13",
        base_python=sys.executable,
        venv_path=target,
    )
    record_path = plan.environment_record_path
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-gpu-environment",
                "schema_version": 2,
                "track": "cuda13",
                "cupy_distribution": "cupy-cuda13x",
                "obsolete_provider": {"wheel_sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    events = []

    def run(action):
        events.append(action.name)
        if action is plan.pip_check:
            assert not record_path.exists()

    monkeypatch.setattr(setup_gpu_dev, "_run_action", run)
    monkeypatch.setattr(
        setup_gpu_dev,
        "_validate_existing_venv",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        setup_gpu_dev,
        "_installed_cupy_distributions",
        lambda _python: ("cupy-cuda13x",),
    )

    setup_gpu_dev.execute_setup_plan(plan)

    assert events[-2:] == ["probe_cuda_runtime", "check_dependencies"]
    assert json.loads(record_path.read_text(encoding="utf-8")) == {
        "schema": "napari-vipp-gpu-environment",
        "schema_version": 3,
        "track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
    }


@pytest.mark.parametrize(
    "environment",
    (
        {
            "implementation": "CPython",
            "version": [3, 13, 0],
            "pointer_bits": 64,
        },
        {
            "implementation": "PyPy",
            "version": [3, 12, 0],
            "pointer_bits": 64,
        },
        {
            "implementation": "CPython",
            "version": [3, 12, 0],
            "pointer_bits": 32,
        },
    ),
)
def test_gpu_setup_rejects_an_unsupported_base_interpreter(environment):
    with pytest.raises(setup_gpu_dev.SetupError, match="CPython 3.12"):
        setup_gpu_dev._validate_python_environment(
            environment,
            role="base interpreter",
        )


def test_environment_record_replacement_is_atomic_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "share" / "napari-vipp" / "gpu-environment.json"
    path.parent.mkdir(parents=True)
    original = '{"old": true}\n'
    path.write_text(original, encoding="utf-8")
    document = {
        "schema": "napari-vipp-gpu-environment",
        "schema_version": 3,
        "track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
    }

    def fail_replace(source, target):
        assert target == path
        assert json.loads(Path(source).read_text(encoding="utf-8")) == document
        assert path.read_text(encoding="utf-8") == original
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(setup_gpu_dev.os, "replace", fail_replace)

    with pytest.raises(setup_gpu_dev.SetupError, match="Could not write"):
        setup_gpu_dev._write_environment_record_atomic(
            path,
            document,
            venv_root=tmp_path,
        )

    assert path.read_text(encoding="utf-8") == original
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_platform_wrappers_delegate_to_the_shared_setup_helper():
    powershell = (PROJECT_ROOT / "scripts" / "setup_gpu_dev.ps1").read_text(
        encoding="utf-8"
    )
    shell = (PROJECT_ROOT / "scripts" / "setup_gpu_dev.sh").read_text(encoding="utf-8")

    assert "setup_gpu_dev.py" in powershell
    assert 'setup_gpu_dev.py"' in shell
    assert "VIPP_GPU_SETUP_PYTHON" in powershell
    assert "VIPP_GPU_SETUP_PYTHON" in shell
    assert "--existing-environment" not in powershell
    assert "py -3.12" in powershell


def test_setup_helper_has_no_retired_provider_or_existing_environment_mode():
    source = SETUP_SCRIPT.read_text(encoding="utf-8").lower()

    assert "cucim" not in source
    assert "--existing-environment" not in source
