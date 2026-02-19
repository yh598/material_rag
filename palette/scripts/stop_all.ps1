param(
    [switch]$Docker,
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"

$pidFiles = @(
    (Join-Path $PSScriptRoot "backend.pid"),
    (Join-Path $PSScriptRoot "ui.pid")
)

if ($Docker) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    Push-Location $repoRoot
    docker compose down
    Pop-Location
}

foreach ($file in $pidFiles) {
    if (-not (Test-Path $file)) {
        continue
    }

    try {
        $pidValue = [int](Get-Content -Path $file -ErrorAction Stop)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            if (-not $Quiet) {
                Write-Output "Stopped process $pidValue"
            }
        }
    } catch {
        if (-not $Quiet) {
            Write-Output "Could not stop process from $file"
        }
    }

    Remove-Item -Path $file -Force -ErrorAction SilentlyContinue
}

if (-not $Quiet) {
    Write-Output "Stop command finished."
}
