[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f]{40}$')]
    [string]$CertificateThumbprint,

    [ValidateNotNullOrEmpty()]
    [string]$TimestampUrl = 'http://timestamp.digicert.com'
)

$ErrorActionPreference = 'Stop'
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$expectedSuffix = '-SIGNING-STAGING.exe'
if (-not $resolvedInput.EndsWith($expectedSuffix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Only an official $expectedSuffix artifact may be signed by this hook."
}

$normalizedThumbprint = $CertificateThumbprint.ToUpperInvariant()
$certificate = @(
    Get-ChildItem -Path Cert:\CurrentUser\My, Cert:\LocalMachine\My |
        Where-Object { $_.Thumbprint -eq $normalizedThumbprint }
)
if ($certificate.Count -ne 1) {
    throw 'The requested signing certificate was not found uniquely in the Windows certificate stores.'
}
if (-not $certificate[0].HasPrivateKey) {
    throw 'The requested signing certificate has no accessible private key.'
}
if ($certificate[0].NotAfter.ToUniversalTime() -le [DateTime]::UtcNow) {
    throw 'The requested signing certificate has expired.'
}
$codeSigningOid = '1.3.6.1.5.5.7.3.3'
if (-not ($certificate[0].EnhancedKeyUsageList.ObjectId.Value -contains $codeSigningOid)) {
    throw 'The requested certificate is not authorized for code signing.'
}

$signTool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if ($null -eq $signTool) {
    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $candidate = Get-ChildItem -LiteralPath $kitsRoot -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($null -eq $candidate) {
        throw 'signtool.exe was not found. Install the Windows SDK signing tools.'
    }
    $signToolPath = $candidate.FullName
} else {
    $signToolPath = $signTool.Source
}

if (-not $PSCmdlet.ShouldProcess($resolvedInput, "Authenticode-sign with certificate $normalizedThumbprint")) {
    return
}

& $signToolPath sign /sha1 $normalizedThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 /v $resolvedInput
if ($LASTEXITCODE -ne 0) {
    throw "signtool.exe failed with exit code $LASTEXITCODE."
}
& $signToolPath verify /pa /all /v $resolvedInput
if ($LASTEXITCODE -ne 0) {
    throw "signtool.exe verification failed with exit code $LASTEXITCODE."
}

$signature = Get-AuthenticodeSignature -LiteralPath $resolvedInput
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Authenticode verification reported $($signature.Status): $($signature.StatusMessage)"
}
if ($null -eq $signature.TimeStamperCertificate) {
    throw 'The signature has no independently verifiable timestamp certificate.'
}

[ordered]@{
    path = $resolvedInput
    status = $signature.Status.ToString()
    signer_subject = $signature.SignerCertificate.Subject
    signer_thumbprint = $signature.SignerCertificate.Thumbprint
    timestamp_subject = $signature.TimeStamperCertificate.Subject
    timestamp_thumbprint = $signature.TimeStamperCertificate.Thumbprint
} | ConvertTo-Json -Depth 3
