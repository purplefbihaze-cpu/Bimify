Write-Host "Stopping Bimify dev processes..." -ForegroundColor Cyan

function Stop-IfExists($pidFile) {
  if (Test-Path $pidFile) {
    try {
      $pid = Get-Content $pidFile | Select-Object -First 1
      if ($pid) {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $pid" -ForegroundColor Green
      }
    } catch {}
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
}

Stop-IfExists ".dev-api.pid"
Stop-IfExists ".dev-ui.pid"

Write-Host "Done." -ForegroundColor Green


