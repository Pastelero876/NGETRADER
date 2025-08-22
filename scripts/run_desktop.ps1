param([switch]$NoInstall)

$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv/Scripts/pythonw.exe")) {
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
if (-not $NoInstall) {
  pip install -r requirements.txt | Out-Null
}
$pythonw = Join-Path (Resolve-Path ".\.venv\Scripts").Path "pythonw.exe"
Start-Process -FilePath $pythonw -ArgumentList "-m","nge_trader.entrypoints.desktop" -WorkingDirectory (Get-Location).Path

