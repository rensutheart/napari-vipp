from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import tomllib
import zipfile
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


def _local_cucim_artifact(tmp_path, monkeypatch=None):
    wheel = tmp_path / "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cucim/__init__.py", '__version__ = "26.06.00"\n')
        archive.writestr(
            "cucim_cu13-26.6.0.dist-info/METADATA",
            "Name: cucim-cu13\nVersion: 26.6.0\n",
        )
        archive.writestr(
            "cucim_cu13-26.6.0.dist-info/RECORD",
            "fixture RECORD is deliberately excluded from the payload hash\n",
        )
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    payload_sha256 = setup_gpu_dev._wheel_payload_sha256(wheel)
    if monkeypatch is not None:
        monkeypatch.setattr(
            setup_gpu_dev,
            "CUCIM_WHEEL_PAYLOAD_SHA256",
            payload_sha256,
        )
    manifest = tmp_path / "cucim-build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-cucim-windows-build",
                "schema_version": 2,
                "distribution": "cucim-cu13",
                "distribution_version": "26.6.0",
                "build_recipe_id": "napari-vipp-cucim-windows-v1",
                "local_build_only": True,
                "source_repository": "https://github.com/rapidsai/cucim.git",
                "source_tag": "v26.06.00",
                "source_commit": (
                    "3c15781c207eab93a317dd9803a6e726fe01f7c4"
                ),
                "source_date_epoch": 1780510583,
                "created_utc": "2026-08-06T12:00:00Z",
                "wheel_payload_hash_algorithm": (
                    "sha256-wheel-payload-length-prefix-v1"
                ),
                "wheel_filename": wheel.name,
                "wheel_size_bytes": wheel.stat().st_size,
                "wheel_sha256": wheel_sha256,
                "wheel_payload_sha256": payload_sha256,
                "wheel_payload_file_count": 2,
                "python": {
                    "implementation": "CPython",
                    "version": "3.12.9",
                    "abi": "cp312",
                    "architecture": "64bit",
                    "platform": "win_amd64",
                },
                "pinned_packages": dict(
                    setup_gpu_dev.CUCIM_BUILD_PINNED_PACKAGES
                ),
                "resolved_packages": dict(
                    setup_gpu_dev.CUCIM_BUILD_PINNED_PACKAGES
                ),
                "cuda": {
                    "track": "cuda13",
                    "nvcc_version": "13.2.86",
                    "runtime_version": "13.2.86",
                    "nvjitlink_version": "13.2.86",
                    "nvimgcodec_version": "0.8.0.22",
                },
                "features": {
                    "cucim_skimage": True,
                    "cucim_clara": False,
                    "console_script": False,
                },
                "adaptations": list(setup_gpu_dev.CUCIM_BUILD_ADAPTATIONS),
                "verification": {
                    "independent_builds": 2,
                    "canonical_payloads_match": True,
                    "archive_sha256_match": True,
                    "metadata_and_licenses": "passed",
                    "real_gpu_probe": "passed",
                    "real_gpu_probe_output": "26.06.00 120",
                    "pip_check": "passed",
                    "exact_package_inventory": "passed",
                },
            }
        ),
        encoding="utf-8",
    )
    return wheel, manifest, wheel_sha256, payload_sha256


def _mock_existing_release_environment(monkeypatch, tmp_path):
    venv_root = tmp_path / "released-vipp"
    python = venv_root / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    (venv_root / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    environment = {
        "executable": str(python),
        "prefix": str(venv_root),
        "base_prefix": str(tmp_path / "base-python"),
        "implementation": "CPython",
        "version": [3, 12, 9],
        "pointer_bits": 64,
        "platform": "win32",
        "machine": "AMD64",
    }
    reports = {
        name: {"name": name, "version": version, "direct_url": None}
        for name, version in setup_gpu_dev.CUDA13_RELEASE_DISTRIBUTIONS.items()
    }
    monkeypatch.setattr(
        setup_gpu_dev,
        "_python_environment",
        lambda _python: environment,
    )
    monkeypatch.setattr(
        setup_gpu_dev,
        "_installed_distribution_reports",
        lambda _python, _names: reports,
    )
    monkeypatch.setattr(
        setup_gpu_dev,
        "_installed_cupy_distributions",
        lambda _python: ("cupy-cuda13x",),
    )
    return venv_root, python, reports


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


def test_release_metadata_bounds_cpu_python_support_to_ci_matrix():
    metadata = _project_metadata()

    assert metadata["version"] == "0.13.0a4"
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
            "schema_version": 2,
            "track": "cuda13",
            "cupy_distribution": "cupy-cuda13x",
            "cucim": None,
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
    for probe in (setup_gpu_dev.GPU_PROBE, setup_gpu_dev.CUCIM_PROBE):
        assert "cuda.MemoryPool()" in probe
        assert "cuda.using_allocator(pool.malloc)" in probe
        assert "pool.free_all_blocks()" in probe
        assert "pool.used_bytes()" in probe
        assert "pool.total_bytes()" in probe
        assert "get_default_memory_pool" not in probe


@pytest.mark.parametrize("probe_name", ["GPU_PROBE", "CUCIM_PROBE"])
@pytest.mark.parametrize("failure_point", ["operation", "synchronize"])
def test_setup_probes_release_private_pool_without_masking_primary_failure(
    monkeypatch,
    probe_name,
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

    if probe_name == "GPU_PROBE":
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
    else:
        cucim = ModuleType("cucim")
        skimage = ModuleType("cucim.skimage")
        restoration = ModuleType("cucim.skimage.restoration")

        def rolling_ball(values, **_kwargs):
            if failure_point == "operation":
                raise primary_error
            return values

        restoration.rolling_ball = rolling_ball
        skimage.restoration = restoration
        cucim.skimage = skimage
        cucim.is_available = lambda component: component == "skimage"
        monkeypatch.setitem(sys.modules, "cucim", cucim)
        monkeypatch.setitem(sys.modules, "cucim.skimage", skimage)
        monkeypatch.setitem(sys.modules, "cucim.skimage.restoration", restoration)

    with pytest.raises(RuntimeError) as exc_info:
        exec(getattr(setup_gpu_dev, probe_name), {})

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


def test_cucim_wheel_requires_pinned_manifest_and_exact_checksums(
    tmp_path,
    monkeypatch,
):
    wheel, manifest, expected, payload = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )

    validated = setup_gpu_dev._validated_cucim_wheel(
        setup_gpu_dev.TRACKS["cuda13"],
        wheel,
        manifest,
    )

    assert validated is not None
    assert validated.path == wheel.resolve()
    assert validated.sha256 == expected
    assert validated.payload_sha256 == payload
    assert validated.source_tag == "v26.06.00"
    assert validated.source_commit == (
        "3c15781c207eab93a317dd9803a6e726fe01f7c4"
    )
    assert validated.build_recipe_id == "napari-vipp-cucim-windows-v1"

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["unknown_audit_field"] = "not in schema v2"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(setup_gpu_dev.SetupError, match="top-level fields"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            manifest,
        )
    del document["unknown_audit_field"]
    document["wheel_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(setup_gpu_dev.SetupError, match="does not match"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            manifest,
        )
    with pytest.raises(setup_gpu_dev.SetupError, match="CUDA 13 only"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda12"],
            wheel,
            manifest,
        )


def test_cucim_manifest_rejects_transitive_lock_and_inventory_tampering(
    tmp_path,
    monkeypatch,
):
    wheel, manifest, _expected, _payload = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["pinned_packages"]["attrs"] = "0.0.0"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(setup_gpu_dev.SetupError, match="pinned recipe"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            manifest,
        )

    document["pinned_packages"] = dict(
        setup_gpu_dev.CUCIM_BUILD_PINNED_PACKAGES
    )
    del document["verification"]["exact_package_inventory"]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(setup_gpu_dev.SetupError, match="verification object"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            manifest,
        )


@gpu_setup_integration
def test_cucim_plan_installs_only_the_manifest_verified_local_wheel(
    tmp_path,
    monkeypatch,
):
    wheel, manifest, expected, payload = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )
    target = tmp_path / "gpu-venv"

    plan = setup_gpu_dev.create_setup_plan(
        track_name="cuda13",
        base_python=sys.executable,
        venv_path=target,
        cucim_wheel=wheel,
        cucim_manifest=manifest,
    )

    assert not target.exists()
    assert plan.install_cucim is not None
    assert plan.install_cucim.argv[-1] == str(wheel.resolve())
    assert "--no-deps" in plan.install_cucim.argv
    assert "--force-reinstall" in plan.install_cucim.argv
    assert {
        "click==8.4.2",
        "lazy-loader==0.5",
        "nvidia-nvimgcodec-cu13==0.8.0.22",
    } <= set(plan.install_project.argv)
    assert plan.cucim_probe is not None
    assert plan.actions[-1] is plan.pip_check
    record = plan.as_dict(plan_only=True)["environment_record"]
    assert record["document"]["cucim"] == {
        "distribution": "cucim-cu13",
        "wheel_sha256": expected,
        "wheel_payload_sha256": payload,
        "source_tag": "v26.06.00",
        "source_commit": "3c15781c207eab93a317dd9803a6e726fe01f7c4",
        "build_recipe_id": "napari-vipp-cucim-windows-v1",
    }
    assert record["write_after"] == [
        "probe_cuda_runtime",
        "verify_installed_cucim_artifact",
        "probe_cucim",
        "check_dependencies",
    ]


def test_cucim_wheel_and_manifest_must_be_provided_together(tmp_path):
    wheel, _manifest, _expected, _payload = _local_cucim_artifact(tmp_path)

    with pytest.raises(setup_gpu_dev.SetupError, match="provided together"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            None,
        )


def test_setup_rejects_an_unapproved_payload_before_install(tmp_path):
    wheel, manifest, _expected, payload = _local_cucim_artifact(tmp_path)
    assert payload != setup_gpu_dev.CUCIM_WHEEL_PAYLOAD_SHA256

    with pytest.raises(setup_gpu_dev.SetupError, match="not approved"):
        setup_gpu_dev._validated_cucim_wheel(
            setup_gpu_dev.TRACKS["cuda13"],
            wheel,
            manifest,
        )


def test_setup_payload_pin_matches_compute_policy():
    from napari_vipp.core.compute_policy import (
        PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
    )

    assert setup_gpu_dev.CUCIM_WHEEL_PAYLOAD_SHA256 == (
        PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256
    )


def test_payload_digest_ignores_zip_metadata_order_and_record(tmp_path):
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    entries = {
        "cucim/a.py": b"a = 1\n",
        "cucim/b.py": b"b = 2\n",
    }
    with zipfile.ZipFile(first, "w") as archive:
        for name, content in entries.items():
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            archive.writestr(info, content)
        archive.writestr("cucim-26.6.0.dist-info/RECORD", "first")
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("cucim-26.6.0.dist-info/RECORD", "second")
        for name, content in reversed(tuple(entries.items())):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 6, 12, 0, 0))
            archive.writestr(info, content)

    assert first.read_bytes() != second.read_bytes()
    assert setup_gpu_dev._wheel_payload_sha256(
        first
    ) == setup_gpu_dev._wheel_payload_sha256(second)


@pytest.mark.parametrize("unsafe_name", ["../escape.py", "/rooted.py", "C:/x.py"])
def test_payload_digest_rejects_unsafe_zip_paths(tmp_path, unsafe_name):
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(unsafe_name, "bad")

    with pytest.raises(setup_gpu_dev.SetupError, match="unsafe ZIP path"):
        setup_gpu_dev._wheel_payload_sha256(wheel)


def test_payload_digest_rejects_symlinks_and_duplicate_paths(tmp_path):
    symlink_wheel = tmp_path / "symlink.whl"
    with zipfile.ZipFile(symlink_wheel, "w") as archive:
        info = zipfile.ZipInfo("cucim/link.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target.py")
    with pytest.raises(setup_gpu_dev.SetupError, match="regular files only"):
        setup_gpu_dev._wheel_payload_sha256(symlink_wheel)

    duplicate_wheel = tmp_path / "duplicate.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_wheel, "w") as archive:
            archive.writestr("cucim/a.py", "first")
            archive.writestr("cucim/a.py", "second")
    with pytest.raises(setup_gpu_dev.SetupError, match="duplicate ZIP path"):
        setup_gpu_dev._wheel_payload_sha256(duplicate_wheel)


def test_existing_environment_plan_never_replaces_released_vipp(
    tmp_path,
    monkeypatch,
):
    venv_root, python, _reports = _mock_existing_release_environment(
        monkeypatch,
        tmp_path,
    )
    wheel, manifest, wheel_hash, payload_hash = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )

    plan = setup_gpu_dev.create_existing_environment_plan(
        track_name="cuda13",
        environment_python=python,
        cucim_wheel=wheel,
        cucim_manifest=manifest,
        platform_name="win32",
    )

    assert plan.venv_root == venv_root.resolve()
    assert all(action.argv[0] == str(python.resolve()) for action in plan.actions)
    assert all("--editable" not in action.argv for action in plan.actions)
    assert not any("install_project" in action.name for action in plan.actions)
    assert not any("upgrade" in action.name for action in plan.actions)
    assert plan.install_cucim_runtime.argv[-3:] == (
        "click==8.4.2",
        "lazy-loader==0.5",
        "nvidia-nvimgcodec-cu13==0.8.0.22",
    )
    assert "--no-deps" in plan.install_cucim_runtime.argv
    assert "--no-deps" in plan.install_cucim.argv
    assert "--force-reinstall" in plan.install_cucim.argv

    document = plan.as_dict(plan_only=True)
    assert document["mode"] == "existing-environment"
    assert document["required_vipp"] == "napari-vipp==0.13.0a4"
    assert document["environment_record"]["document"] == {
        "schema": "napari-vipp-gpu-environment",
        "schema_version": 2,
        "track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
        "cucim": {
            "distribution": "cucim-cu13",
            "wheel_sha256": wheel_hash,
            "wheel_payload_sha256": payload_hash,
            "source_tag": "v26.06.00",
            "source_commit": "3c15781c207eab93a317dd9803a6e726fe01f7c4",
            "build_recipe_id": "napari-vipp-cucim-windows-v1",
        },
    }


def test_existing_environment_requires_released_noneditable_exact_stack(
    tmp_path,
    monkeypatch,
):
    _venv_root, python, reports = _mock_existing_release_environment(
        monkeypatch,
        tmp_path,
    )
    reports["napari-vipp"]["direct_url"] = json.dumps(
        {
            "url": "file:///checkout",
            "dir_info": {"editable": True},
        }
    )
    with pytest.raises(setup_gpu_dev.SetupError, match="released napari-vipp wheel"):
        setup_gpu_dev._validate_existing_release_environment(python)

    reports["napari-vipp"]["direct_url"] = None
    reports["numpy"]["version"] = "2.4.0"
    with pytest.raises(setup_gpu_dev.SetupError, match="numpy==2.5.1"):
        setup_gpu_dev._validate_existing_release_environment(python)


def test_existing_environment_failure_removes_approval_record(
    tmp_path,
    monkeypatch,
):
    _venv_root, python, _reports = _mock_existing_release_environment(
        monkeypatch,
        tmp_path,
    )
    wheel, manifest, _wheel_hash, _payload_hash = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )
    plan = setup_gpu_dev.create_existing_environment_plan(
        track_name="cuda13",
        environment_python=python,
        cucim_wheel=wheel,
        cucim_manifest=manifest,
        platform_name="win32",
    )
    record_path = plan.environment_record_path
    record_path.parent.mkdir(parents=True)
    record_path.write_text('{"stale": true}\n', encoding="utf-8")
    events = []

    def run(action):
        events.append(action.name)
        assert not record_path.exists()
        if action is plan.cucim_probe:
            raise setup_gpu_dev.SetupError("simulated real cuCIM probe failure")

    monkeypatch.setattr(setup_gpu_dev, "_run_action", run)
    monkeypatch.setattr(
        setup_gpu_dev,
        "_validate_installed_cucim_artifact",
        lambda *_args: events.append("verify_installed_cucim_artifact"),
    )

    with pytest.raises(setup_gpu_dev.SetupError, match="simulated"):
        setup_gpu_dev.execute_existing_environment_plan(plan)

    assert events == [
        "install_pinned_cucim_runtime",
        "install_manifest_verified_cucim",
        "verify_installed_cucim_artifact",
        "probe_cuda_runtime",
        "probe_cucim",
    ]
    assert not record_path.exists()


def test_existing_environment_success_writes_record_last(tmp_path, monkeypatch):
    _venv_root, python, _reports = _mock_existing_release_environment(
        monkeypatch,
        tmp_path,
    )
    wheel, manifest, wheel_hash, payload_hash = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )
    plan = setup_gpu_dev.create_existing_environment_plan(
        track_name="cuda13",
        environment_python=python,
        cucim_wheel=wheel,
        cucim_manifest=manifest,
        platform_name="win32",
    )
    record_path = plan.environment_record_path
    record_path.parent.mkdir(parents=True)
    record_path.write_text('{"stale": true}\n', encoding="utf-8")
    events = []

    def run(action):
        assert not record_path.exists()
        events.append(action.name)

    def validate_provenance(*_args):
        assert not record_path.exists()
        events.append("verify_installed_cucim_artifact")

    monkeypatch.setattr(setup_gpu_dev, "_run_action", run)
    monkeypatch.setattr(
        setup_gpu_dev,
        "_validate_installed_cucim_artifact",
        validate_provenance,
    )

    setup_gpu_dev.execute_existing_environment_plan(plan)

    assert events == [
        "install_pinned_cucim_runtime",
        "install_manifest_verified_cucim",
        "verify_installed_cucim_artifact",
        "probe_cuda_runtime",
        "probe_cucim",
        "check_dependencies",
        "verify_installed_cucim_artifact",
    ]
    assert json.loads(record_path.read_text(encoding="utf-8"))["cucim"] == {
        "distribution": "cucim-cu13",
        "wheel_sha256": wheel_hash,
        "wheel_payload_sha256": payload_hash,
        "source_tag": "v26.06.00",
        "source_commit": "3c15781c207eab93a317dd9803a6e726fe01f7c4",
        "build_recipe_id": "napari-vipp-cucim-windows-v1",
    }


def test_pep610_archive_sha256_requires_one_unambiguous_digest():
    digest = "a" * 64
    assert setup_gpu_dev._pep610_archive_sha256(
        json.dumps({"archive_info": {"hash": f"sha256={digest}"}})
    ) == digest
    assert setup_gpu_dev._pep610_archive_sha256(
        json.dumps({"archive_info": {"hashes": {"sha256": digest}}})
    ) == digest

    with pytest.raises(setup_gpu_dev.SetupError, match="unambiguous"):
        setup_gpu_dev._pep610_archive_sha256(
            json.dumps(
                {
                    "archive_info": {
                        "hash": f"sha256={digest}",
                        "hashes": {"sha256": "b" * 64},
                    }
                }
            )
        )


def test_installed_cucim_must_match_verified_wheel_pep610_hash(
    tmp_path,
    monkeypatch,
):
    wheel_path, manifest, wheel_hash, _payload_hash = _local_cucim_artifact(
        tmp_path,
        monkeypatch,
    )
    wheel = setup_gpu_dev._validated_cucim_wheel(
        setup_gpu_dev.TRACKS["cuda13"],
        wheel_path,
        manifest,
    )
    assert wheel is not None
    report = {
        "cucim-cu13": {
            "name": "cucim-cu13",
            "version": "26.6.0",
            "direct_url": json.dumps(
                {"archive_info": {"hash": f"sha256={wheel_hash}"}}
            ),
        }
    }
    monkeypatch.setattr(
        setup_gpu_dev,
        "_run_capture",
        lambda argv: subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(report),
            stderr="",
        ),
    )

    setup_gpu_dev._validate_installed_cucim_artifact(
        setup_gpu_dev._cucim_provenance_action(Path(sys.executable)),
        wheel,
    )

    report["cucim-cu13"]["direct_url"] = json.dumps(
        {"archive_info": {"hash": f"sha256={'b' * 64}"}}
    )
    with pytest.raises(setup_gpu_dev.SetupError, match="does not match"):
        setup_gpu_dev._validate_installed_cucim_artifact(
            setup_gpu_dev._cucim_provenance_action(Path(sys.executable)),
            wheel,
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
                "cucim": {
                    "distribution": "cucim-cu13",
                    "wheel_sha256": "f" * 64,
                },
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
        "schema_version": 2,
        "track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
        "cucim": None,
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
        "schema_version": 2,
        "track": "cuda13",
        "cupy_distribution": "cupy-cuda13x",
        "cucim": None,
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
    assert 'contains "--existing-environment"' in powershell
    assert '& python -c "import sys; print(sys.executable)"' in powershell
