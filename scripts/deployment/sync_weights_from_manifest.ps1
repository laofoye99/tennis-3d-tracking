[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,
    [string]$RepoPath = $(if ($env:TENNIS_DEPLOY_DIR) { $env:TENNIS_DEPLOY_DIR } else { (Get-Location).Path })
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version 3.0

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Expand-ManifestValue {
    param([AllowNull()][string]$Value)
    if ($null -eq $Value) {
        return $null
    }

    return [regex]::Replace($Value, "\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}", {
        param($match)
        $name = $match.Groups[1].Value
        $expanded = [Environment]::GetEnvironmentVariable($name)
        if ([string]::IsNullOrWhiteSpace($expanded)) {
            throw "Environment variable '$name' is required by weights manifest."
        }
        return $expanded
    })
}

$RepoPath = [System.IO.Path]::GetFullPath($RepoPath)
$ManifestPath = [System.IO.Path]::GetFullPath($ManifestPath)

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Weights manifest not found: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($null -eq $manifest.weights) {
    throw "Manifest must contain a 'weights' array."
}

foreach ($weight in $manifest.weights) {
    $relativePath = Expand-ManifestValue ([string]$weight.path)
    $url = Expand-ManifestValue ([string]$weight.url)
    $sha256 = Expand-ManifestValue ([string]$weight.sha256)

    if ([string]::IsNullOrWhiteSpace($relativePath)) {
        throw "Each manifest item must include 'path'."
    }
    if ([string]::IsNullOrWhiteSpace($url)) {
        throw "Manifest item '$relativePath' must include 'url'."
    }

    $target = Join-Path $RepoPath $relativePath
    $targetDir = Split-Path -Parent $target
    if (-not (Test-Path -LiteralPath $targetDir)) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }

    $needsDownload = $true
    if (Test-Path -LiteralPath $target) {
        if (-not [string]::IsNullOrWhiteSpace($sha256)) {
            $currentHash = Get-FileSha256 -Path $target
            $needsDownload = $currentHash -ne $sha256.ToLowerInvariant()
        } else {
            $needsDownload = $false
        }
    }

    if (-not $needsDownload) {
        Write-Host "Weight is current: $relativePath"
        continue
    }

    $tmp = $target + ".download"
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Force
    }

    Write-Host "Downloading weight: $relativePath"
    Invoke-WebRequest -Uri $url -OutFile $tmp

    if (-not [string]::IsNullOrWhiteSpace($sha256)) {
        $downloadHash = Get-FileSha256 -Path $tmp
        if ($downloadHash -ne $sha256.ToLowerInvariant()) {
            Remove-Item -LiteralPath $tmp -Force
            throw "SHA256 mismatch for '$relativePath'. Expected $sha256, got $downloadHash."
        }
    }

    Move-Item -LiteralPath $tmp -Destination $target -Force
    Write-Host "Updated weight: $relativePath"
}
