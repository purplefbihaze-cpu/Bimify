# Bimify Development Shutdown Script
# Stops all running dev processes (API, UI, Worker)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stopping Bimify Development Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Stop-ProcessByPid {
    param(
        [string]$PidFile,
        [string]$ServiceName
    )
    
    if (Test-Path $PidFile) {
        try {
            $pid = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($pid) {
                $pid = [int]$pid
                $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                
                if ($process) {
                    Write-Host "Stopping $ServiceName (PID: $pid)..." -ForegroundColor Yellow
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    
                    # Wait for process to terminate
                    $timeout = 5
                    $elapsed = 0
                    while ($elapsed -lt $timeout) {
                        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
                        if (-not $proc) {
                            break
                        }
                        Start-Sleep -Milliseconds 200
                        $elapsed += 0.2
                    }
                    
                    $stillRunning = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($stillRunning) {
                        Write-Host "  Warning: Process still running, forcing termination..." -ForegroundColor Yellow
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    } else {
                        Write-Host "  OK: $ServiceName stopped" -ForegroundColor Green
                    }
                } else {
                    Write-Host "  Info: Process $pid not found (may have already stopped)" -ForegroundColor DarkGray
                }
            }
        } catch {
            Write-Host "  Warning: Error stopping $ServiceName : $($_.Exception.Message)" -ForegroundColor Yellow
        } finally {
            Remove-Item $PidFile -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "  Info: $ServiceName not running (no PID file)" -ForegroundColor DarkGray
    }
}

# Stop services in reverse order (UI -> Worker -> API)
Stop-ProcessByPid -PidFile ".dev-ui.pid" -ServiceName "UI"
Stop-ProcessByPid -PidFile ".dev-worker.pid" -ServiceName "Worker"
Stop-ProcessByPid -PidFile ".dev-api.pid" -ServiceName "API"

# Clean up any orphaned processes by name (fallback)
Write-Host ""
Write-Host "Checking for orphaned processes..." -ForegroundColor DarkGray

$orphaned = @()

# Check for uvicorn processes (by checking command line via WMI)
try {
    $uvicornProcs = Get-WmiObject Win32_Process | Where-Object { 
        $_.Name -eq "python.exe" -and 
        $_.CommandLine -like "*uvicorn*services.api.main*"
    }
    if ($uvicornProcs) {
        $orphaned += $uvicornProcs
    }
} catch {
    # WMI may not be available, skip orphaned process detection
}

# Check for Next.js processes
try {
    $nextProcs = Get-WmiObject Win32_Process | Where-Object { 
        $_.Name -eq "node.exe" -and 
        ($_.CommandLine -like "*next*dev*" -or $_.CommandLine -like "*next-server*")
    }
    if ($nextProcs) {
        $orphaned += $nextProcs
    }
} catch {
    # Ignore errors
}

# Check for Celery processes
try {
    $celeryProcs = Get-WmiObject Win32_Process | Where-Object { 
        $_.Name -eq "python.exe" -and 
        $_.CommandLine -like "*celery*worker*"
    }
    if ($celeryProcs) {
        $orphaned += $celeryProcs
    }
} catch {
    # Ignore errors
}

if ($orphaned.Count -gt 0) {
    Write-Host "  Found $($orphaned.Count) orphaned process(es)" -ForegroundColor Yellow
    $orphaned | ForEach-Object {
        $pid = $_.ProcessId
        $name = $_.Name
        Write-Host "    - $name (PID: $pid)" -ForegroundColor DarkGray
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "      OK: Stopped" -ForegroundColor Green
        } catch {
            Write-Host "      Warning: Could not stop" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  OK: No orphaned processes found" -ForegroundColor Green
}

# Clean up PID files
@(".dev-api.pid", ".dev-ui.pid", ".dev-worker.pid") | ForEach-Object {
    if (Test-Path $_) {
        Remove-Item $_ -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All services stopped" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
