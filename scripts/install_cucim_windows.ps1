[CmdletBinding()]
param(
    [string]$TargetPython = "",
    [string]$StateRoot = "",
    [string]$WorkRoot = "",
    [string]$ArtifactDirectory = "",
    [switch]$PlanOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$installer = Join-Path $PSScriptRoot "install_cucim_windows.py"
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "The cuCIM installer backend is missing: $installer"
}

if ([string]::IsNullOrWhiteSpace($TargetPython)) {
    if (-not [string]::IsNullOrWhiteSpace($env:VIPP_CUCIM_TARGET_PYTHON)) {
        $TargetPython = $env:VIPP_CUCIM_TARGET_PYTHON
    } elseif (
        -not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV) -and
        (Test-Path -LiteralPath (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe") -PathType Leaf)
    ) {
        $TargetPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    } else {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            $dialog = New-Object System.Windows.Forms.OpenFileDialog
            $dialog.Title = (
                "Select python.exe in the released VIPP CUDA 13 environment"
            )
            $dialog.Filter = "Python interpreter (python.exe)|python.exe"
            $dialog.CheckFileExists = $true
            $dialog.Multiselect = $false
            if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                $dialog.InitialDirectory = $env:USERPROFILE
            }
            $selection = $dialog.ShowDialog()
            if ($selection -ne [System.Windows.Forms.DialogResult]::OK) {
                Write-Host "No VIPP environment was selected. Nothing was changed."
                exit 1
            }
            $TargetPython = $dialog.FileName
        } catch {
            throw (
                "Could not open the Python selector. Rerun with " +
                "-TargetPython 'C:\path\to\VIPP\Scripts\python.exe'. " +
                $_.Exception.Message
            )
        }
    }
}

$arguments = @(
    $installer,
    "--target-python", $TargetPython
)
if (-not [string]::IsNullOrWhiteSpace($StateRoot)) {
    $arguments += @("--state-root", $StateRoot)
}
if (-not [string]::IsNullOrWhiteSpace($WorkRoot)) {
    $arguments += @("--work-root", $WorkRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ArtifactDirectory)) {
    $arguments += @("--artifact-directory", $ArtifactDirectory)
}
if ($PlanOnly) {
    $arguments += "--plan-only"
}
if ($DryRun) {
    $arguments += "--dry-run"
}

Write-Host "VIPP cuCIM installer"
Write-Host "Target: $TargetPython"
Write-Host (
    "The pinned builder and released-environment verifier will perform all " +
    "approval checks automatically."
)
Write-Host ""

& $TargetPython @arguments
exit $LASTEXITCODE
