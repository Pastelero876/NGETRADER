param(
    [string]$WorkingDir = (Resolve-Path "..").Path,
    [switch]$InstallNSSM
)

Set-Location $WorkingDir

if ($InstallNSSM) {
  Write-Output "Instalar NSSM manualmente si no está presente y crear servicio luego."
}

# Lanzar API + UI en background (demo)
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command .\.venv\Scripts\Activate.ps1; uvicorn nge_trader.entrypoints.api:app --host 0.0.0.0 --port 8000" -WindowStyle Minimized

Write-Output "Servicio live iniciado (API en :8000)."

$ErrorActionPreference = "Stop"

# Activar venv y ejecutar live con watchdog simple
. .\.venv\Scripts\Activate.ps1

function Invoke-Healthcheck {
  try {
    python .\scripts\healthcheck.py | Out-Null
    return $true
  } catch {
    return $false
  }
}

$symbol = $env:LIVE_SYMBOL
if (-not $symbol) { $symbol = "BTCUSDT" }

while ($true) {
  if (Invoke-Healthcheck) {
    python -c "from nge_trader.services.live_engine import LiveConfig, LiveEngine; LiveEngine(LiveConfig(symbols=['$symbol'], capital_per_trade=1000.0, poll_seconds=30)).run()"
  } else {
    Start-Sleep -Seconds 10
  }
}


