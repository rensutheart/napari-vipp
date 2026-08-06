[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$WorkRoot = (Join-Path $env:TEMP "napari-vipp-cucim-windows"),
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SourceRepository = "https://github.com/rapidsai/cucim.git"
$SourceTag = "v26.06.00"
$SourceCommit = "3c15781c207eab93a317dd9803a6e726fe01f7c4"
$BuildRecipeId = "napari-vipp-cucim-windows-v1"
$ManifestSchema = "napari-vipp-cucim-windows-build"
$ManifestSchemaVersion = 2
$PayloadHashAlgorithm = "sha256-wheel-payload-length-prefix-v1"

# These are the direct scientific, CUDA, and build inputs used for the
# napari-vipp 0.13.0a1 CUDA 13 qualification. Do not make this recipe float.
$PinnedPackages = [ordered]@{
    "pip" = "26.1.2"
    "setuptools" = "83.0.0"
    "wheel" = "0.47.0"
    "build" = "1.5.0"
    "rapids-build-backend" = "0.4.1"
    "numpy" = "2.5.1"
    "scipy" = "1.18.0"
    "scikit-image" = "0.26.0"
    "lazy-loader" = "0.5"
    "click" = "8.4.2"
    "cupy-cuda13x" = "14.1.1"
    "cuda-pathfinder" = "1.6.0"
    "cuda-toolkit" = "13.2.2"
    "nvidia-cublas" = "13.4.1.3"
    "nvidia-cuda-nvrtc" = "13.2.86"
    "nvidia-cuda-runtime" = "13.2.86"
    "nvidia-cufft" = "12.2.0.57"
    "nvidia-curand" = "10.4.2.66"
    "nvidia-cusolver" = "12.2.0.11"
    "nvidia-cusparse" = "12.7.10.12"
    "nvidia-nvjitlink" = "13.2.86"
    "nvidia-cuda-nvcc" = "13.2.86"
    "nvidia-cuda-crt" = "13.2.86"
    "nvidia-nvvm" = "13.2.86"
    "nvidia-nvimgcodec-cu13" = "0.8.0.22"
}

function Write-Utf8Lf {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $normalized = $Text.Replace("`r`n", "`n").Replace("`r", "`n")
    [System.IO.File]::WriteAllText(
        $Path,
        $normalized,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Assert-SafeNonRootDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "A non-empty directory path is required."
    }
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $pathRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.TrimEnd('\', '/') -eq $pathRoot.TrimEnd('\', '/')) {
        throw "Refusing to use a filesystem root as a working or output directory: $fullPath"
    }
    return $fullPath
}

function Invoke-PythonProbe {
    param([Parameter(Mandatory = $true)][string]$Executable)
    $program = @'
import json
import platform
import struct
import sys
import sysconfig

print(json.dumps({
    "executable": sys.executable,
    "implementation": sys.implementation.name,
    "version": platform.python_version(),
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "pointer_bits": struct.calcsize("P") * 8,
    "system": platform.system(),
    "machine": platform.machine(),
    "platform_tag": sysconfig.get_platform(),
}))
'@
    $probeOutput = ($program | & $Executable -) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect Python interpreter: $Executable"
    }
    try {
        return $probeOutput | ConvertFrom-Json
    } catch {
        throw "Python returned an invalid interpreter report: $probeOutput"
    }
}

function Assert-SupportedPython {
    param([Parameter(Mandatory = $true)]$Report)
    if (
        $Report.system -ne "Windows" -or
        $Report.implementation -ne "cpython" -or
        $Report.major -ne 3 -or
        $Report.minor -ne 12 -or
        $Report.pointer_bits -ne 64 -or
        $Report.platform_tag -ne "win-amd64"
    ) {
        throw (
            "This exact recipe requires native 64-bit CPython 3.12 for " +
            "win-amd64. Selected interpreter: " +
            "$($Report.implementation) $($Report.version), " +
            "$($Report.pointer_bits)-bit, $($Report.platform_tag)."
        )
    }
}

function Remove-GeneratedBuildPaths {
    param([Parameter(Mandatory = $true)][string]$PackageRoot)
    $resolvedPackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
    foreach ($relativeTarget in @("build", "dist", "src\cucim_cu13.egg-info")) {
        $target = Join-Path $resolvedPackageRoot $relativeTarget
        if (-not (Test-Path -LiteralPath $target)) {
            continue
        }
        $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
        if (-not $resolvedTarget.StartsWith(
            $resolvedPackageRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to remove a generated path outside the checkout: $resolvedTarget"
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

function Invoke-WheelBuild {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExecutable,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )
    Remove-GeneratedBuildPaths -PackageRoot $PackageRoot
    & $PythonExecutable -m build --wheel --no-isolation $PackageRoot | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "cuCIM wheel build failed."
    }
    $wheels = @(Get-ChildItem -LiteralPath (Join-Path $PackageRoot "dist") `
        -Filter "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl")
    if ($wheels.Count -ne 1) {
        throw "Expected exactly one CPython 3.12 win_amd64 cuCIM 26.6.0 wheel; found $($wheels.Count)."
    }
    return $wheels[0]
}

# Canonical identity for the installable payload. ZIP container timestamps and
# compression are intentionally ignored. For every safe, regular, non-RECORD
# file sorted by UTF-8 name bytes, hash uint64be(name length), name bytes,
# uint64be(content length), and uncompressed content. Directories are skipped.
$wheelPayloadHashProgram = @'
import hashlib
import json
import stat
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

wheel_path = Path(sys.argv[1])
entries = []
seen = set()
with zipfile.ZipFile(wheel_path) as archive:
    for info in archive.infolist():
        name = info.filename
        if name in seen:
            raise ValueError(f"duplicate wheel entry: {name!r}")
        seen.add(name)
        if "\\" in name or "\x00" in name:
            raise ValueError(f"non-POSIX wheel entry: {name!r}")
        is_directory = info.is_dir()
        checked_name = name[:-1] if is_directory and name.endswith("/") else name
        parts = checked_name.split("/")
        if (
            not checked_name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or (len(parts[0]) >= 2 and parts[0][1] == ":")
        ):
            raise ValueError(f"unsafe wheel entry: {name!r}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            raise ValueError(f"symlink wheel entry: {name!r}")
        if is_directory:
            continue
        if file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"non-regular wheel entry: {name!r}")
        if name.endswith(".dist-info/RECORD"):
            continue
        name_bytes = name.encode("utf-8")
        entries.append((name_bytes, archive.read(info)))

digest = hashlib.sha256()
for name_bytes, contents in sorted(entries, key=lambda item: item[0]):
    digest.update(struct.pack(">Q", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack(">Q", len(contents)))
    digest.update(contents)
print(json.dumps({"sha256": digest.hexdigest(), "file_count": len(entries)}))
'@

function Get-WheelPayloadHash {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExecutable,
        [Parameter(Mandatory = $true)][string]$WheelPath
    )
    $output = (
        $wheelPayloadHashProgram | & $PythonExecutable - $WheelPath
    ) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "The wheel failed canonical payload validation: $WheelPath"
    }
    return $output | ConvertFrom-Json
}

if ($env:OS -ne "Windows_NT") {
    throw "This adaptation script targets native Windows only."
}

$hostPython = Invoke-PythonProbe -Executable $Python
Assert-SupportedPython -Report $hostPython

$WorkRoot = Assert-SafeNonRootDirectory -Path $WorkRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $WorkRoot "artifacts"
}
$OutputDirectory = Assert-SafeNonRootDirectory -Path $OutputDirectory
New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $gitCommand) {
    throw "Git for Windows is required. Install it, then rerun this recipe."
}
$git = $gitCommand.Source
$cacheRoot = Join-Path $WorkRoot "source-cache\cucim.git"
if (-not (Test-Path -LiteralPath $cacheRoot)) {
    New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($cacheRoot)) `
        -Force | Out-Null
    & $git init --bare $cacheRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the local cuCIM source cache."
    }
    & $git -C $cacheRoot remote add origin $SourceRepository
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to configure the cuCIM source cache."
    }
}

$origin = (& $git -C $cacheRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -ne $SourceRepository) {
    throw "The source cache origin is not the expected upstream repository: $origin"
}
& $git -C $cacheRoot config core.autocrlf false
& $git -C $cacheRoot config core.eol lf
& $git -C $cacheRoot config core.symlinks false
& $git -C $cacheRoot fetch --force --depth 1 origin `
    "refs/tags/$SourceTag`:refs/tags/$SourceTag"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to fetch the exact cuCIM source tag $SourceTag."
}
$tagCommit = (& $git -C $cacheRoot rev-parse "$SourceTag^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $tagCommit -ne $SourceCommit) {
    throw "Upstream tag $SourceTag does not resolve to the pinned commit $SourceCommit."
}

$runId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-$PID"
$sourceRoot = Join-Path $WorkRoot "runs\$runId\cucim"
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($sourceRoot)) `
    -Force | Out-Null
& $git -c core.autocrlf=false -c core.eol=lf -c core.symlinks=false `
    -C $cacheRoot worktree add --detach $sourceRoot $SourceCommit
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the isolated checkout for $SourceCommit."
}
$actualCommit = (& $git -C $sourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $SourceCommit) {
    throw "The isolated checkout is not the pinned cuCIM commit."
}

$packageRoot = Join-Path $sourceRoot "python\cucim"
$expectedLinks = [ordered]@{
    "CHANGELOG.md" = "CHANGELOG.md"
    "CONTRIBUTING.md" = "CONTRIBUTING.md"
    "LICENSE" = "LICENSE"
    "LICENSE-3rdparty.md" = "LICENSE-3rdparty.md"
    "README.md" = "README.md"
    "VERSION" = "VERSION"
    "src\cucim\VERSION" = "VERSION"
}
$linkEntries = @(& $git -C $sourceRoot ls-files -s -- "python/cucim")
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect upstream symbolic-link entries."
}
$actualLinks = @(
    $linkEntries |
        Where-Object { $_ -match '^120000 ' } |
        ForEach-Object { ($_ -split "`t", 2)[1].Substring("python/cucim/".Length).Replace('/', '\') }
)
$linkDifference = @(Compare-Object @($expectedLinks.Keys) $actualLinks)
if ($linkDifference.Count -ne 0) {
    throw "The pinned source has an unexpected symbolic-link layout; refusing to guess."
}
foreach ($destinationRelative in $expectedLinks.Keys) {
    $sourcePath = Join-Path $sourceRoot $expectedLinks[$destinationRelative]
    $destinationPath = Join-Path $packageRoot $destinationRelative
    $sourceText = [System.IO.File]::ReadAllText($sourcePath)
    if ((Get-Item -LiteralPath $destinationPath).LinkType) {
        Remove-Item -LiteralPath $destinationPath -Force
    }
    Write-Utf8Lf -Path $destinationPath -Text $sourceText
}

$manifestPath = Join-Path $packageRoot "MANIFEST.in"
$manifestText = [System.IO.File]::ReadAllText($manifestPath)
$manifestAnchor = "recursive-include src/cucim *.py *.pyi *.cu *.h *.npy *.txt *.md"
if (-not $manifestText.Contains($manifestAnchor)) {
    throw "Could not find the expected MANIFEST.in anchor."
}
$manifestText = $manifestText.Replace(
    $manifestAnchor,
    "include src/cucim/VERSION`ninclude VIPP-WINDOWS-BUILD-NOTICE.txt`n`n$manifestAnchor"
)
Write-Utf8Lf -Path $manifestPath -Text $manifestText

$padPath = Join-Path $packageRoot "src\cucim\skimage\_vendored\pad.py"
$padText = [System.IO.File]::ReadAllText($padPath)
$oldPadCode = "    x_view = x.view()`n    x_view.shape = (ndim, 2)"
$newPadCode = @'
    # Modified by napari-vipp-cucim-windows-v1 for NumPy 2.5:
    # ndarray.shape assignment is deprecated; reshape preserves this view.
    x_view = x.view().reshape((ndim, 2))
'@
$padText = $padText.Replace("`r`n", "`n")
if (-not $padText.Contains($oldPadCode)) {
    throw "Could not find the expected vendored pad compatibility code."
}
$padText = $padText.Replace($oldPadCode, $newPadCode.TrimEnd())
Write-Utf8Lf -Path $padPath -Text $padText

$pyprojectPath = Join-Path $packageRoot "pyproject.toml"
$pyprojectText = [System.IO.File]::ReadAllText($pyprojectPath).Replace("`r`n", "`n")
$replacements = [ordered]@{
    '"rapids-build-backend>=0.4.0,<0.5.0"' = '"rapids-build-backend==0.4.1"'
    '"setuptools>=80.9.0"' = '"setuptools==83.0.0"'
    'description = "cuCIM - an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging."' = 'description = "cuCIM skimage subset, locally adapted for native Windows CUDA 13 by the napari-vipp build recipe."'
    "    `"LICENSE-3rdparty.md`",`n]" = "    `"LICENSE-3rdparty.md`",`n    `"VIPP-WINDOWS-BUILD-NOTICE.txt`",`n]"
    '    "click",' = '    "click==8.4.2",'
    '    "cupy-cuda13x>=13.6.0,!=14.0.0,!=14.1.0",' = '    "cupy-cuda13x==14.1.1",'
    '    "lazy-loader>=0.4",' = '    "lazy-loader==0.5",'
    '    "numpy>=1.23.4,<3.0",' = '    "numpy==2.5.1",'
    '    "nvidia-nvimgcodec-cu13>=0.8.0,<0.9.0",' = '    "nvidia-nvimgcodec-cu13==0.8.0.22",'
    '    "scikit-image>=0.23.2,<0.27.0",' = '    "scikit-image==0.26.0",'
    '    "scipy>=1.11.2",' = '    "scipy==1.18.0",'
    '    "Operating System :: POSIX :: Linux",' = "    `"Operating System :: Microsoft :: Windows :: Windows 10`",`n    `"Operating System :: Microsoft :: Windows :: Windows 11`","
    '    "Environment :: GPU :: NVIDIA CUDA :: 12",' = '    "Environment :: GPU :: NVIDIA CUDA :: 13",'
    "[project.entry-points.`"console_scripts`"]`ncucim = `"cucim.clara.cli:main`"`n`n" = ''
    '    "wheel",' = '    "wheel==0.47.0",'
}
foreach ($oldText in $replacements.Keys) {
    if (-not $pyprojectText.Contains($oldText)) {
        throw "Could not find an expected pyproject.toml recipe anchor: $oldText"
    }
    $pyprojectText = $pyprojectText.Replace($oldText, $replacements[$oldText])
}
Write-Utf8Lf -Path $pyprojectPath -Text $pyprojectText

# rapids-build-backend regenerates pyproject metadata from dependencies.yaml at
# build time. Pin that authoritative input too, otherwise the wheel would still
# advertise upstream's broad dependency ranges despite the local pyproject.
$dependenciesPath = Join-Path $sourceRoot "dependencies.yaml"
$dependenciesText = [System.IO.File]::ReadAllText($dependenciesPath).Replace("`r`n", "`n")
$dependencyReplacements = [ordered]@{
    '          - wheel' = '          - wheel==0.47.0'
    '          - rapids-build-backend>=0.4.0,<0.5.0' = '          - rapids-build-backend==0.4.1'
    '          - setuptools>=80.9.0' = '          - setuptools==83.0.0'
    '          - click' = '          - click==8.4.2'
    '          - lazy-loader>=0.4' = '          - lazy-loader==0.5'
    '          - numpy>=1.23.4,<3.0' = '          - numpy==2.5.1'
    '          - scikit-image>=0.23.2,<0.27.0' = '          - scikit-image==0.26.0'
    '          - scipy>=1.11.2' = '          - scipy==1.18.0'
    '              - cupy-cuda13x>=13.6.0,!=14.0.0,!=14.1.0' = '              - cupy-cuda13x==14.1.1'
    '              - nvidia-nvimgcodec-cu13>=0.8.0,<0.9.0' = '              - nvidia-nvimgcodec-cu13==0.8.0.22'
}
foreach ($oldText in $dependencyReplacements.Keys) {
    if (-not $dependenciesText.Contains($oldText)) {
        throw "Could not find an expected dependencies.yaml recipe anchor: $oldText"
    }
    $dependenciesText = $dependenciesText.Replace(
        $oldText,
        $dependencyReplacements[$oldText]
    )
}
Write-Utf8Lf -Path $dependenciesPath -Text $dependenciesText

$setupPath = Join-Path $packageRoot "setup.py"
$setupText = [System.IO.File]::ReadAllText($setupPath).Replace("`r`n", "`n")
$oldSetupComment = @'
# As we vendored a shared object that links to a specific Python version,
# make sure it is treated as impure so the wheel is named properly.
'@
$newSetupComment = @'
# The local Windows recipe deliberately narrows the artifact tag to its tested
# CPython 3.12 / win_amd64 target. Do not retag this wheel as universal.
'@
if (-not $setupText.Contains($oldSetupComment.Trim())) {
    throw "Could not find the expected setup.py platform-tag anchor."
}
$setupText = $setupText.Replace($oldSetupComment.Trim(), $newSetupComment.Trim())
Write-Utf8Lf -Path $setupPath -Text $setupText

$noticePath = Join-Path $packageRoot "VIPP-WINDOWS-BUILD-NOTICE.txt"
$notice = @"
Local Windows build adaptation notice

This artifact is built locally from RAPIDS cuCIM $SourceTag at commit
$SourceCommit using recipe $BuildRecipeId. It is not an official NVIDIA or
RAPIDS Windows binary distribution and is not published by napari-vipp.

Recipe changes:
- materialize upstream repository symlinks as regular UTF-8/LF files;
- omit the unusable Clara command-line entry point (Clara is unavailable);
- narrow metadata and dependencies to the qualified Windows CUDA 13 stack;
- pin RAPIDS' authoritative dependency-generator input to that same stack;
- replace deprecated ndarray.shape assignment in
  src/cucim/skimage/_vendored/pad.py with reshape for NumPy 2.5.

The upstream Apache-2.0 and third-party license texts are included unchanged.
"@
Write-Utf8Lf -Path $noticePath -Text $notice

$venvRoot = Join-Path $sourceRoot ".build-venv"
& $Python -m venv $venvRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the isolated CPython 3.12 build environment."
}
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$venvReport = Invoke-PythonProbe -Executable $venvPython
Assert-SupportedPython -Report $venvReport

& $venvPython -m pip install --disable-pip-version-check --no-input `
    "pip==$($PinnedPackages['pip'])" `
    "setuptools==$($PinnedPackages['setuptools'])" `
    "wheel==$($PinnedPackages['wheel'])"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the pinned packaging tools."
}
$installNames = @($PinnedPackages.Keys | Where-Object {
    $_ -notin @("pip", "setuptools", "wheel")
})
$requirements = @($installNames | ForEach-Object { "$_==$($PinnedPackages[$_])" })
& $venvPython -m pip install --disable-pip-version-check --no-input `
    --only-binary=:all: $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the pinned scientific, CUDA, and build inputs."
}

# rapids-build-backend invokes the Unix `which` command. Git for Windows ships
# the required executable, but its usr/bin directory is not normally on PATH.
$whichPath = Join-Path ([System.IO.Path]::GetDirectoryName(
    [System.IO.Path]::GetDirectoryName($git)
)) "usr\bin\which.exe"
if (-not (Test-Path -LiteralPath $whichPath)) {
    $whichCommand = Get-Command which.exe -ErrorAction SilentlyContinue
    if ($null -eq $whichCommand) {
        throw "rapids-build-backend requires which.exe from Git for Windows."
    }
    $whichPath = $whichCommand.Source
}
$cudaBin = Join-Path $venvRoot "Lib\site-packages\nvidia\cu13\bin"
$env:PATH = "$cudaBin;$([System.IO.Path]::GetDirectoryName($whichPath));$env:PATH"
$nvccVersionText = (& nvcc --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $nvccVersionText -notmatch "release 13\.2") {
    throw "The isolated CUDA 13.2.86 nvcc was not selected.`n$nvccVersionText"
}
$sourceEpoch = (& $git -C $sourceRoot show -s --format=%ct $SourceCommit).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceEpoch -notmatch '^\d+$') {
    throw "Could not derive SOURCE_DATE_EPOCH from the pinned commit."
}
$env:SOURCE_DATE_EPOCH = $sourceEpoch
$env:PYTHONHASHSEED = "0"

$firstWheel = Invoke-WheelBuild -PythonExecutable $venvPython -PackageRoot $packageRoot
$firstPayload = Get-WheelPayloadHash -PythonExecutable $venvPython `
    -WheelPath $firstWheel.FullName
$firstArchiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $firstWheel.FullName).Hash.ToLowerInvariant()
$firstCopyDirectory = Join-Path $sourceRoot ".first-build"
New-Item -ItemType Directory -Path $firstCopyDirectory -Force | Out-Null
Copy-Item -LiteralPath $firstWheel.FullName -Destination `
    (Join-Path $firstCopyDirectory $firstWheel.Name)

$wheel = Invoke-WheelBuild -PythonExecutable $venvPython -PackageRoot $packageRoot
$payload = Get-WheelPayloadHash -PythonExecutable $venvPython -WheelPath $wheel.FullName
if ($payload.sha256 -ne $firstPayload.sha256) {
    throw (
        "Two clean builds produced different canonical payloads: " +
        "$($firstPayload.sha256) versus $($payload.sha256)."
    )
}
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $wheel.FullName).Hash.ToLowerInvariant()

$wheelValidationProgram = @'
import email
import json
import re
import sys
import zipfile

wheel_path = sys.argv[1]
with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
    if len(metadata_names) != 1 or len(wheel_names) != 1:
        raise AssertionError("wheel must contain exactly one METADATA and WHEEL")
    if any(name.endswith(".dist-info/entry_points.txt") for name in names):
        raise AssertionError("local skimage wheel must not install a Clara command")
    if any(name.endswith((".so", ".dll", ".pyd")) for name in names):
        raise AssertionError("unexpected prebuilt native library in local wheel")
    metadata = email.message_from_bytes(archive.read(metadata_names[0]))
    if metadata["Name"] != "cucim-cu13" or metadata["Version"] != "26.6.0":
        raise AssertionError("unexpected distribution identity")
    required = set(metadata.get_all("Requires-Dist", []))
    expected_versions = {
        "click": "8.4.2",
        "cupy-cuda13x": "14.1.1",
        "lazy-loader": "0.5",
        "numpy": "2.5.1",
        "nvidia-nvimgcodec-cu13": "0.8.0.22",
        "scikit-image": "0.26.0",
        "scipy": "1.18.0",
    }
    for package, version in expected_versions.items():
        if not any(re.fullmatch(rf"{re.escape(package)}\s*==\s*{re.escape(version)}", item) for item in required):
            raise AssertionError(f"missing exact dependency: {package}=={version}")
    license_names = [name for name in names if ".dist-info/licenses/" in name]
    license_payloads = {name.rsplit("/", 1)[-1]: archive.read(name) for name in license_names}
    if len(license_payloads.get("LICENSE", b"")) < 1000:
        raise AssertionError("Apache license was not materialized")
    if len(license_payloads.get("LICENSE-3rdparty.md", b"")) < 1000:
        raise AssertionError("third-party license was not materialized")
    if "VIPP-WINDOWS-BUILD-NOTICE.txt" not in license_payloads:
        raise AssertionError("local adaptation notice is missing")
    wheel_metadata = archive.read(wheel_names[0]).decode("utf-8")
    if "Root-Is-Purelib: false" not in wheel_metadata or "Tag: cp312-cp312-win_amd64" not in wheel_metadata:
        raise AssertionError("wheel compatibility tag is broader than the qualified target")
print(json.dumps({"distribution": metadata["Name"], "version": metadata["Version"]}))
'@
$wheelValidation = (
    $wheelValidationProgram | & $venvPython - $wheel.FullName
) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "The locally built wheel failed its metadata or license checks."
}

& $venvPython -m pip install --disable-pip-version-check --no-input `
    --no-deps --force-reinstall $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the locally built cuCIM wheel."
}
$gpuProbe = @'
import cucim
import cupy as cp
from cucim.skimage import filters, measure, restoration

x = cp.arange(4096, dtype=cp.float32).reshape(64, 64)
outputs = (
    filters.gaussian(x, 1.0),
    restoration.rolling_ball(x, radius=8),
    measure.label(x > 2048),
)
cp.cuda.get_current_stream().synchronize()
assert cucim.is_available("skimage")
assert not cucim.is_available("clara")
assert all(output.shape == x.shape for output in outputs)
print(cucim.__version__, cp.cuda.Device(0).compute_capability)
'@
$gpuProbeOutput = ($gpuProbe | & $venvPython -) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "The installed wheel failed its real-GPU operation probe."
}
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The isolated cuCIM build environment has broken dependencies."
}

$resolvedVersionsProgram = @'
import base64
import importlib.metadata as metadata
import json
import sys

names = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
print(json.dumps({name: metadata.version(name) for name in names}, sort_keys=True))
'@
$packageNamesJson = @($PinnedPackages.Keys) | ConvertTo-Json -Compress
$packageNamesBase64 = [Convert]::ToBase64String(
    [System.Text.Encoding]::UTF8.GetBytes($packageNamesJson)
)
$resolvedVersionsJson = (
    $resolvedVersionsProgram | & $venvPython - $packageNamesBase64
) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "Could not record resolved package versions."
}
$resolvedVersionsObject = $resolvedVersionsJson | ConvertFrom-Json
$resolvedProperties = @($resolvedVersionsObject.PSObject.Properties)
if ($resolvedProperties.Count -ne $PinnedPackages.Count) {
    throw "The resolved package report has an unexpected number of fields."
}
$resolvedVersions = [ordered]@{}
foreach ($packageName in $PinnedPackages.Keys) {
    $property = $resolvedVersionsObject.PSObject.Properties[$packageName]
    if ($null -eq $property -or -not ($property.Value -is [string])) {
        throw "The resolved package report is missing $packageName."
    }
    $resolvedVersions[$packageName] = $property.Value
}

$outputWheelPath = Join-Path $OutputDirectory $wheel.Name
$outputManifestName = $wheel.BaseName + ".build-manifest.json"
$outputManifestPath = Join-Path $OutputDirectory $outputManifestName
$outputPrefix = $OutputDirectory.TrimEnd('\', '/') + `
    [System.IO.Path]::DirectorySeparatorChar
foreach ($target in @($outputWheelPath, $outputManifestPath)) {
    $fullTarget = [System.IO.Path]::GetFullPath($target)
    if (-not $fullTarget.StartsWith(
        $outputPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to publish outside the exact output directory: $fullTarget"
    }
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite an existing output artifact: $target"
    }
}

$manifest = [ordered]@{
    schema = $ManifestSchema
    schema_version = $ManifestSchemaVersion
    build_recipe_id = $BuildRecipeId
    local_build_only = $true
    source_repository = $SourceRepository
    source_tag = $SourceTag
    source_commit = $SourceCommit
    source_date_epoch = [long]$sourceEpoch
    created_utc = [DateTime]::UtcNow.ToString("o")
    distribution = "cucim-cu13"
    distribution_version = "26.6.0"
    wheel_filename = $wheel.Name
    wheel_size_bytes = $wheel.Length
    wheel_sha256 = $archiveHash
    wheel_payload_hash_algorithm = $PayloadHashAlgorithm
    wheel_payload_sha256 = $payload.sha256
    wheel_payload_file_count = $payload.file_count
    python = [ordered]@{
        implementation = "CPython"
        version = $venvReport.version
        abi = "cp312"
        architecture = "64bit"
        platform = "win_amd64"
    }
    pinned_packages = $PinnedPackages
    resolved_packages = $resolvedVersions
    cuda = [ordered]@{
        track = "cuda13"
        nvcc_version = $PinnedPackages["nvidia-cuda-nvcc"]
        runtime_version = $PinnedPackages["nvidia-cuda-runtime"]
        nvjitlink_version = $PinnedPackages["nvidia-nvjitlink"]
        nvimgcodec_version = $PinnedPackages["nvidia-nvimgcodec-cu13"]
    }
    features = [ordered]@{
        cucim_skimage = $true
        cucim_clara = $false
        console_script = $false
    }
    adaptations = @(
        "materialize-upstream-symlinks-utf8-lf",
        "remove-unavailable-clara-console-entry-point",
        "exact-pin-qualified-scientific-cuda-build-stack",
        "pin-rapids-dependency-generator-input",
        "numpy-2.5-pad-reshape-compatibility"
    )
    verification = [ordered]@{
        independent_builds = 2
        canonical_payloads_match = $true
        archive_sha256_match = ($archiveHash -eq $firstArchiveHash)
        metadata_and_licenses = "passed"
        real_gpu_probe = "passed"
        real_gpu_probe_output = $gpuProbeOutput.Trim()
        pip_check = "passed"
    }
}
$manifestJson = $manifest | ConvertTo-Json -Depth 8
$partialWheelPath = "$outputWheelPath.partial-$runId"
$partialManifestPath = "$outputManifestPath.partial-$runId"
$wheelPublished = $false
$manifestPublished = $false
try {
    # Prepare and fully write both artifacts before either final name appears.
    Copy-Item -LiteralPath $wheel.FullName -Destination $partialWheelPath
    Write-Utf8Lf -Path $partialManifestPath -Text ($manifestJson + "`n")
    Move-Item -LiteralPath $partialWheelPath -Destination $outputWheelPath
    $wheelPublished = $true
    Move-Item -LiteralPath $partialManifestPath -Destination $outputManifestPath
    $manifestPublished = $true
} catch {
    $publicationError = $_
    foreach ($partialPath in @($partialWheelPath, $partialManifestPath)) {
        if (Test-Path -LiteralPath $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
    }
    if ($manifestPublished -and (Test-Path -LiteralPath $outputManifestPath)) {
        Remove-Item -LiteralPath $outputManifestPath -Force
    }
    if ($wheelPublished -and (Test-Path -LiteralPath $outputWheelPath)) {
        Remove-Item -LiteralPath $outputWheelPath -Force
    }
    throw $publicationError
}

[ordered]@{
    wheel = $outputWheelPath
    manifest = $outputManifestPath
    wheel_sha256 = $archiveHash
    wheel_payload_sha256 = $payload.sha256
    build_recipe_id = $BuildRecipeId
} | ConvertTo-Json
