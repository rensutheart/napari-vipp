$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptPath = Join-Path $PSScriptRoot "setup_gpu_dev.py"
$existingEnvironmentMode = @($args) -contains "--existing-environment"
if ($env:VIPP_GPU_SETUP_PYTHON) {
    $bootstrapPython = $env:VIPP_GPU_SETUP_PYTHON
    $pythonPath = (& $bootstrapPython -c "import sys; print(sys.executable)").Trim()
} elseif ($existingEnvironmentMode) {
    # In this mode, use the active venv rather than the py launcher base
    # interpreter. The Python helper independently proves this is a released
    # VIPP CUDA 13 venv before modifying it.
    $pythonPath = (& python -c "import sys; print(sys.executable)").Trim()
} else {
    $pythonPath = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
}
if ($LASTEXITCODE -ne 0 -or -not $pythonPath) {
    throw (
        "A CPython 3.12 interpreter is required. Activate the target venv " +
        "for --existing-environment or set VIPP_GPU_SETUP_PYTHON to its " +
        "executable path."
    )
}

$forwardArgs = @("--python", $pythonPath) + $args
& $pythonPath $scriptPath @forwardArgs
exit $LASTEXITCODE
