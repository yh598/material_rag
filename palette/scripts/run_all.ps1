param(
    [switch]$Docker,
    [int]$BackendPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

function Wait-Http {
    param(
        [string]$Url,
        [int]$Attempts = 40
    )
    for ($i = 0; $i -lt $Attempts; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 600
        }
    }
    return $false
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Docker) {
    docker compose up -d --build
    if (-not (Wait-Http -Url "http://127.0.0.1:$BackendPort/health")) {
        throw "Backend did not become healthy in time."
    }
    if (-not (Wait-Http -Url "http://127.0.0.1:$UiPort")) {
        throw "UI did not become available in time."
    }

    Write-Output "Stack is running with Docker."
    Write-Output "Backend: http://127.0.0.1:$BackendPort/health"
    Write-Output "UI:      http://127.0.0.1:$UiPort"
    exit 0
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    py -3 -m venv .venv
}

& $venvPython -m pip install --disable-pip-version-check -r requirements.txt | Out-Null

& (Join-Path $PSScriptRoot "stop_all.ps1") -Quiet

$backendProc = Start-Process -FilePath $venvPython `
    -ArgumentList "-m uvicorn app:app --host 127.0.0.1 --port $BackendPort" `
    -WorkingDirectory (Join-Path $repoRoot "backend") `
    -WindowStyle Hidden `
    -PassThru

$uiProc = Start-Process -FilePath $venvPython `
    -ArgumentList "-m streamlit run app.py --server.address 127.0.0.1 --server.port $UiPort --server.headless true" `
    -WorkingDirectory (Join-Path $repoRoot "ui") `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path (Join-Path $PSScriptRoot "backend.pid") -Value $backendProc.Id
Set-Content -Path (Join-Path $PSScriptRoot "ui.pid") -Value $uiProc.Id

if (-not (Wait-Http -Url "http://127.0.0.1:$BackendPort/health")) {
    throw "Backend did not become healthy in time."
}
if (-not (Wait-Http -Url "http://127.0.0.1:$UiPort")) {
    throw "UI did not become available in time."
}

Write-Output "Stack is running."
Write-Output "Backend: http://127.0.0.1:$BackendPort/health"
Write-Output "UI:      http://127.0.0.1:$UiPort"
