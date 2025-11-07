Write-Host "Starting Bimify API and UI (dev)..." -ForegroundColor Cyan

# Ensure dependencies are installed (no-op if already present)
try {
  py -m poetry --version | Out-Null
} catch {
  py -m pip install --user poetry | Out-Null
}

# Start API (uvicorn) in background
# Exclude generated artifacts from reload to prevent mid-request restarts (e.g., writing to data/exports)
$api = Start-Process -FilePath "py" -ArgumentList "-m","poetry","run","uvicorn","services.api.main:app","--reload","--host","0.0.0.0","--port","8000","--reload-exclude","data","--reload-dir","services","--reload-dir","core" -PassThru
Set-Content -Path ".dev-api.pid" -Value $api.Id
Write-Host "API started (PID: $($api.Id)) on http://localhost:8000" -ForegroundColor Green

# Start UI (Next.js) in background from ./ui
$ui = Start-Process -FilePath "npm" -ArgumentList "run","dev","--prefix","ui" -PassThru
Set-Content -Path ".dev-ui.pid" -Value $ui.Id
Write-Host "UI started (PID: $($ui.Id)) on http://localhost:3000" -ForegroundColor Green

Write-Host "Use ./stop-dev.ps1 to stop both." -ForegroundColor Yellow


