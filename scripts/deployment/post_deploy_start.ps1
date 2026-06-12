[CmdletBinding()]
param(
    [string]$AppUrl = $(if ($env:TENNIS_APP_URL) { $env:TENNIS_APP_URL } else { "http://127.0.0.1:8000" }),
    [string]$ModelName = $(if ($env:TENNIS_AUTO_MODEL) { $env:TENNIS_AUTO_MODEL } else { "yolo_roadmap" }),
    [string]$Cameras = $(if ($env:TENNIS_AUTO_START_CAMERAS) { $env:TENNIS_AUTO_START_CAMERAS } else { "cam68" }),
    [string]$Enable3DPush = $(if ($env:TENNIS_ENABLE_3D_PUSH) { $env:TENNIS_ENABLE_3D_PUSH } else { "true" }),
    [string]$PushUrl = $env:TENNIS_3D_PUSH_URL,
    [int]$WaitSeconds = 120
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Test-Truthy {
    param([AllowNull()][string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    return $Value.Trim().ToLowerInvariant() -in @("1", "true", "yes", "y", "on")
}

function Join-ApiUrl {
    param([string]$BaseUrl, [string]$Path)
    return $BaseUrl.TrimEnd("/") + $Path
}

function Invoke-TennisApi {
    param(
        [ValidateSet("GET", "POST")]
        [string]$Method,
        [string]$Path
    )

    $uri = Join-ApiUrl -BaseUrl $AppUrl -Path $Path
    Write-Host "$Method $uri"
    try {
        if ($Method -eq "POST") {
            return Invoke-RestMethod -Method Post -Uri $uri -TimeoutSec 30
        }
        return Invoke-RestMethod -Method Get -Uri $uri -TimeoutSec 30
    } catch {
        throw "API call failed: $Method $uri - $($_.Exception.Message)"
    }
}

function Wait-TennisApp {
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    $lastError = $null
    do {
        try {
            $status = Invoke-TennisApi -Method GET -Path "/api/status"
            Write-Host "App is healthy."
            return $status
        } catch {
            $lastError = $_
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $deadline)

    throw "App did not become healthy within $WaitSeconds seconds. Last error: $lastError"
}

$null = Wait-TennisApp

if (-not [string]::IsNullOrWhiteSpace($ModelName)) {
    $safeModel = [uri]::EscapeDataString($ModelName.Trim())
    $null = Invoke-TennisApi -Method POST -Path "/api/model/switch/$safeModel"
}

$null = Invoke-TennisApi -Method POST -Path "/api/function/track/on"

$cameraNames = $Cameras.Split(",") |
    ForEach-Object { $_.Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

foreach ($camera in $cameraNames) {
    $safeCamera = [uri]::EscapeDataString($camera)
    $null = Invoke-TennisApi -Method POST -Path "/api/pipeline/$safeCamera/start"
}

if (Test-Truthy $Enable3DPush) {
    $path = "/api/3d-display/enable"
    if (-not [string]::IsNullOrWhiteSpace($PushUrl)) {
        $path = $path + "?url=" + [uri]::EscapeDataString($PushUrl)
    }
    $null = Invoke-TennisApi -Method POST -Path $path
}

$currentModel = Invoke-TennisApi -Method GET -Path "/api/model/current"
$toggles = Invoke-TennisApi -Method GET -Path "/api/function/toggles"

Write-Host "Post-deploy startup complete."
Write-Host ("Model: " + ($currentModel | ConvertTo-Json -Compress))
Write-Host ("Toggles: " + ($toggles | ConvertTo-Json -Compress))
