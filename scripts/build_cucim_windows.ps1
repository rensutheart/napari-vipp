[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$WorkRoot = (Join-Path $env:TEMP "napari-vipp-cucim-windows"),
    [string]$CucimTag = "v26.06.00",
    [string]$CupyVersion = "14.1.1",
    [string]$NvccVersion = "13.3.73"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
    throw "This adaptation script targets native Windows only."
}

$sourceRoot = Join-Path $WorkRoot "cucim-$CucimTag"
$venvRoot = Join-Path $WorkRoot "venv"
$packageRoot = Join-Path $sourceRoot "python\cucim"

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    git clone --branch $CucimTag --depth 1 `
        https://github.com/rapidsai/cucim.git $sourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone cuCIM $CucimTag."
    }
}

$expectedCommit = (git -C $sourceRoot rev-list -n 1 $CucimTag).Trim()
$actualCommit = (git -C $sourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Existing checkout is not pinned to $CucimTag ($expectedCommit)."
}

# Git stores this file as a relative symbolic link. Standard Windows checkouts
# materialize the link target as text, so setuptools otherwise reads the literal
# string '../../../../VERSION' as the package version.
$sourceVersion = Join-Path $sourceRoot "VERSION"
$packageVersion = Join-Path $packageRoot "src\cucim\VERSION"
$versionItem = Get-Item -LiteralPath $packageVersion
if ($versionItem.LinkType) {
    throw "Expected a Windows non-symlink checkout for $packageVersion."
}
Copy-Item -LiteralPath $sourceVersion -Destination $packageVersion -Force

# The upstream manifest does not include VERSION in the source-built wheel,
# although cucim._version reads it at runtime.
$manifestPath = Join-Path $packageRoot "MANIFEST.in"
$manifestLines = @(Get-Content -LiteralPath $manifestPath)
$versionManifestEntry = "include src/cucim/VERSION"
if ($manifestLines -notcontains $versionManifestEntry) {
    $anchor = "recursive-include src/cucim *.py *.pyi *.cu *.h *.npy *.txt *.md"
    $anchorIndex = [Array]::IndexOf($manifestLines, $anchor)
    if ($anchorIndex -lt 0) {
        throw "Could not find the expected MANIFEST.in anchor."
    }
    $before = if ($anchorIndex -ge 0) { $manifestLines[0..$anchorIndex] } else { @() }
    $after = if ($anchorIndex + 1 -lt $manifestLines.Count) {
        $manifestLines[($anchorIndex + 1)..($manifestLines.Count - 1)]
    } else {
        @()
    }
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllLines(
        $manifestPath,
        @($before) + $versionManifestEntry + @($after),
        $utf8
    )
}

# NumPy 2.5 deprecates assigning ndarray.shape. cuCIM's strict test settings
# turn that warning into 252 histogram-median failures. reshape(copy=False
# semantics for this view) is the direct compatible replacement.
$padPath = Join-Path $packageRoot "src\cucim\skimage\_vendored\pad.py"
$padText = [System.IO.File]::ReadAllText($padPath)
$oldPadCode = "    x_view = x.view()`r`n    x_view.shape = (ndim, 2)"
if (-not $padText.Contains($oldPadCode)) {
    $oldPadCode = "    x_view = x.view()`n    x_view.shape = (ndim, 2)"
}
if ($padText.Contains($oldPadCode)) {
    $padText = $padText.Replace(
        $oldPadCode,
        "    x_view = x.view().reshape((ndim, 2))"
    )
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($padPath, $padText, $utf8)
} elseif (-not $padText.Contains("x.view().reshape((ndim, 2))")) {
    throw "Could not find the expected vendored pad compatibility code."
}

$venvPython = Join-Path $venvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the isolated Python environment."
    }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install `
    "cupy-cuda13x[ctk]==$CupyVersion" `
    "nvidia-cuda-nvcc==$NvccVersion" `
    "numpy<3" `
    "scipy>=1.11.2" `
    "scikit-image>=0.23.2,<0.27" `
    "lazy-loader>=0.4" `
    click `
    "rapids-build-backend>=0.4,<0.5" `
    build
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the isolated build dependencies."
}

# rapids-build-backend 0.4.1 invokes the Unix `which` command. Git for Windows
# ships a suitable executable, but its usr/bin directory is not normally on
# PowerShell's PATH.
$whichPath = "C:\Program Files\Git\usr\bin\which.exe"
if (-not (Test-Path -LiteralPath $whichPath)) {
    $whichCommand = Get-Command which.exe -ErrorAction SilentlyContinue
    if ($null -eq $whichCommand) {
        throw "rapids-build-backend requires which.exe; install Git for Windows."
    }
    $whichPath = $whichCommand.Source
}
$cudaBin = Join-Path $venvRoot "Lib\site-packages\nvidia\cu13\bin"
$env:PATH = "$cudaBin;$([System.IO.Path]::GetDirectoryName($whichPath));$env:PATH"

$nvccVersionText = (& nvcc --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $nvccVersionText -notmatch "release 13\.") {
    throw "The isolated CUDA 13 nvcc was not selected.`n$nvccVersionText"
}

# Clean only generated paths underneath the pinned disposable package checkout.
$resolvedPackageRoot = (Resolve-Path -LiteralPath $packageRoot).Path
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

& $venvPython -m build --wheel --no-isolation $packageRoot
if ($LASTEXITCODE -ne 0) {
    throw "cuCIM wheel build failed."
}

$wheels = @(Get-ChildItem -LiteralPath (Join-Path $packageRoot "dist") `
    -Filter "cucim_cu13-*-win_amd64.whl")
if ($wheels.Count -ne 1) {
    throw "Expected exactly one native Windows CUDA 13 wheel; found $($wheels.Count)."
}
$wheel = $wheels[0]

# nvidia-nvimgcodec is part of the upstream package metadata but is not needed
# by cucim.skimage. Installing without dependencies makes the intentionally
# absent native cucim.clara/libcucim boundary explicit.
& $venvPython -m pip install --no-deps --force-reinstall $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the source-built wheel."
}

$probe = @"
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
"@
& $venvPython -c $probe
if ($LASTEXITCODE -ne 0) {
    throw "The installed wheel failed its real-GPU operation probe."
}

$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $wheel.FullName
[pscustomobject]@{
    source_tag = $CucimTag
    source_commit = $actualCommit
    wheel = $wheel.FullName
    wheel_size_bytes = $wheel.Length
    wheel_sha256 = $hash.Hash
    python = $venvPython
    cucim_skimage = $true
    cucim_clara = $false
} | ConvertTo-Json
