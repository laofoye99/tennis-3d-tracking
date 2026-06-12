[CmdletBinding()]
param(
    [string]$RepoPath = $(if ($env:TENNIS_DEPLOY_DIR) { $env:TENNIS_DEPLOY_DIR } else { "D:\tennis\tennis-3d-tracking" }),
    [string]$Branch = $(if ($env:TENNIS_DEPLOY_BRANCH) { $env:TENNIS_DEPLOY_BRANCH } else { "main" }),
    [string]$ServiceName = $(if ($env:TENNIS_SERVICE_NAME) { $env:TENNIS_SERVICE_NAME } else { "Tennis3DTracking" }),
    [string]$AppUrl = $(if ($env:TENNIS_APP_URL) { $env:TENNIS_APP_URL } else { "http://127.0.0.1:8000" }),
    [string]$Python = $(if ($env:TENNIS_PYTHON) { $env:TENNIS_PYTHON } else { "" }),
    [string]$InstallRequirements = $(if ($env:TENNIS_INSTALL_REQUIREMENTS) { $env:TENNIS_INSTALL_REQUIREMENTS } else { "true" }),
    [string]$ForceReset = $(if ($env:TENNIS_FORCE_RESET) { $env:TENNIS_FORCE_RESET } else { "false" }),
    [string]$RestartApp = $(if ($env:TENNIS_RESTART_APP) { $env:TENNIS_RESTART_APP } else { "true" }),
    [string]$RunPostDeploy = $(if ($env:TENNIS_RUN_POST_DEPLOY) { $env:TENNIS_RUN_POST_DEPLOY } else { "true" }),
    [string]$WeightsManifest = $(if ($env:TENNIS_WEIGHTS_MANIFEST) { $env:TENNIS_WEIGHTS_MANIFEST } else { "" }),
    [string]$WeightsSyncScript = $(if ($env:TENNIS_SYNC_WEIGHTS_SCRIPT) { $env:TENNIS_SYNC_WEIGHTS_SCRIPT } else { "" }),
    [int]$HealthWaitSeconds = 120
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

function Invoke-External {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$WorkingDirectory
    )
    Push-Location $WorkingDirectory
    try {
        Write-Host ("> " + $Exe + " " + ($Args -join " "))
        & $Exe @Args
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Args -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Resolve-Python {
    if (-not [string]::IsNullOrWhiteSpace($Python)) {
        return $Python
    }

    $venvPython = Join-Path $RepoPath ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }

    return "python"
}

function Update-Repository {
    Invoke-External -Exe "git" -Args @("rev-parse", "--is-inside-work-tree") -WorkingDirectory $RepoPath
    Invoke-External -Exe "git" -Args @("fetch", "origin", $Branch) -WorkingDirectory $RepoPath

    $currentBranch = (& git -C $RepoPath rev-parse --abbrev-ref HEAD).Trim()
    if ($currentBranch -ne $Branch) {
        Invoke-External -Exe "git" -Args @("checkout", $Branch) -WorkingDirectory $RepoPath
    }

    if (Test-Truthy $ForceReset) {
        Invoke-External -Exe "git" -Args @("reset", "--hard", "origin/$Branch") -WorkingDirectory $RepoPath
    } else {
        Invoke-External -Exe "git" -Args @("pull", "--ff-only", "origin", $Branch) -WorkingDirectory $RepoPath
    }

    $commit = (& git -C $RepoPath rev-parse --short HEAD).Trim()
    Write-Host "Repository is at $commit."
}

function Sync-Weights {
    $customSync = $WeightsSyncScript
    if (-not [string]::IsNullOrWhiteSpace($customSync)) {
        if (-not [System.IO.Path]::IsPathRooted($customSync)) {
            $customSync = Join-Path $RepoPath $customSync
        }
        if (-not (Test-Path -LiteralPath $customSync)) {
            throw "Weight sync script not found: $customSync"
        }
        Write-Host "Running custom weight sync script: $customSync"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $customSync -RepoPath $RepoPath
        if ($LASTEXITCODE -ne 0) {
            throw "Weight sync script failed with exit code $LASTEXITCODE."
        }
        return
    }

    $dvcDir = Join-Path $RepoPath ".dvc"
    if ((Test-Path -LiteralPath $dvcDir) -and (Get-Command dvc -ErrorAction SilentlyContinue)) {
        Invoke-External -Exe "dvc" -Args @("pull") -WorkingDirectory $RepoPath
        return
    }

    $manifest = $WeightsManifest
    if ([string]::IsNullOrWhiteSpace($manifest)) {
        $manifest = Join-Path $RepoPath "scripts\deployment\weights_manifest.json"
    } elseif (-not [System.IO.Path]::IsPathRooted($manifest)) {
        $manifest = Join-Path $RepoPath $manifest
    }

    if (Test-Path -LiteralPath $manifest) {
        $syncScript = Join-Path $RepoPath "scripts\deployment\sync_weights_from_manifest.ps1"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $syncScript -ManifestPath $manifest -RepoPath $RepoPath
        if ($LASTEXITCODE -ne 0) {
            throw "Manifest weight sync failed with exit code $LASTEXITCODE."
        }
        return
    }

    Write-Host "No remote weight sync configured; validating existing local weights."
}

function Get-RequiredWeights {
    if (-not [string]::IsNullOrWhiteSpace($env:TENNIS_REQUIRED_WEIGHTS)) {
        return $env:TENNIS_REQUIRED_WEIGHTS.Split(",") |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    }

    $configPath = Join-Path $RepoPath "config.yaml"
    $configText = Get-Content -LiteralPath $configPath -Raw
    return [regex]::Matches($configText, "model_weight/[^\s#]+") |
        ForEach-Object { $_.Value.Trim("`"", "'", ",") } |
        Sort-Object -Unique
}

function Assert-WeightsPresent {
    $missing = @()
    foreach ($relativePath in (Get-RequiredWeights)) {
        $path = Join-Path $RepoPath $relativePath
        if (-not (Test-Path -LiteralPath $path)) {
            $missing += $relativePath
            continue
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -le 0) {
            $missing += "$relativePath (empty)"
        }
    }

    if ($missing.Count -gt 0) {
        $message = "Missing required model weights:`n" + (($missing | ForEach-Object { "  - $_" }) -join "`n")
        throw $message
    }

    Write-Host "All required model weights are present."
}

function Install-Requirements {
    if (-not (Test-Truthy $InstallRequirements)) {
        Write-Host "Skipping requirements install."
        return
    }

    $requirements = Join-Path $RepoPath "requirements.txt"
    if (-not (Test-Path -LiteralPath $requirements)) {
        Write-Host "No requirements.txt found."
        return
    }

    $pythonExe = Resolve-Python
    Invoke-External -Exe $pythonExe -Args @("-m", "pip", "install", "-r", "requirements.txt") -WorkingDirectory $RepoPath
}

function Stop-FallbackProcesses {
    $escapedRepo = [regex]::Escape($RepoPath)
    $processes = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match $escapedRepo -and
            $_.CommandLine -match "main\.py"
        }

    foreach ($proc in $processes) {
        Write-Host "Stopping existing app process PID $($proc.ProcessId)."
        Stop-Process -Id $proc.ProcessId -Force
    }
}

function Start-FallbackProcess {
    $pythonExe = Resolve-Python
    $logsDir = Join-Path $RepoPath "logs"
    if (-not (Test-Path -LiteralPath $logsDir)) {
        New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logsDir "deploy_stdout_$stamp.log"
    $stderr = Join-Path $logsDir "deploy_stderr_$stamp.log"

    Write-Host "Starting app process with $pythonExe main.py"
    Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @("main.py") `
        -WorkingDirectory $RepoPath `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden
}

function Restart-TennisApp {
    if (-not (Test-Truthy $RestartApp)) {
        Write-Host "Skipping app restart."
        return
    }

    $service = $null
    if (-not [string]::IsNullOrWhiteSpace($ServiceName)) {
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    }

    if ($null -ne $service) {
        Write-Host "Restarting Windows service: $ServiceName"
        if ($service.Status -ne "Stopped") {
            Stop-Service -Name $ServiceName -Force
            $service.WaitForStatus("Stopped", "00:00:30")
        }
        Start-Service -Name $ServiceName
        return
    }

    Write-Host "Service '$ServiceName' was not found; using process fallback."
    Stop-FallbackProcesses
    Start-FallbackProcess
}

function Invoke-PostDeploy {
    if (-not (Test-Truthy $RunPostDeploy)) {
        Write-Host "Skipping post-deploy startup API calls."
        return
    }

    $script = Join-Path $RepoPath "scripts\deployment\post_deploy_start.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script -AppUrl $AppUrl -WaitSeconds $HealthWaitSeconds
    if ($LASTEXITCODE -ne 0) {
        throw "Post-deploy startup failed with exit code $LASTEXITCODE."
    }
}

$RepoPath = [System.IO.Path]::GetFullPath($RepoPath)
if (-not (Test-Path -LiteralPath $RepoPath)) {
    throw "Deployment repo path not found: $RepoPath"
}

$lockPath = Join-Path $RepoPath ".deploy.lock"
$lock = $null

try {
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    Write-Host "Deployment started for $RepoPath on branch $Branch."
    Update-Repository
    Sync-Weights
    Assert-WeightsPresent
    Install-Requirements
    Restart-TennisApp
    Invoke-PostDeploy
    Write-Host "Deployment finished successfully."
} finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
}
