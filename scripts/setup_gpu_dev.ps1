$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptPath = Join-Path $PSScriptRoot "setup_gpu_dev.py"
if ($env:VIPP_GPU_SETUP_PYTHON) {
    $bootstrapPython = $env:VIPP_GPU_SETUP_PYTHON
    $pythonPath = (& $bootstrapPython -c "import sys; print(sys.executable)").Trim()
} else {
    $pythonPath = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
}
if ($LASTEXITCODE -ne 0 -or -not $pythonPath) {
    throw "A CPython 3.12 interpreter is required. Set VIPP_GPU_SETUP_PYTHON to its executable path."
}

$forwardArgs = @("--python", $pythonPath) + $args
& $pythonPath $scriptPath @forwardArgs
exit $LASTEXITCODE
