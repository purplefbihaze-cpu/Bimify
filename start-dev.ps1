# Bimify Development Startup Script
# Starts API, UI, and optionally Celery Worker

param(
    [switch]$WithWorker = $false,
    [switch]$SkipUI = $false,
    [switch]$SkipAPI = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bimify Development Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if processes are already running
function Test-DevProcesses {
    $apiPid = if (Test-Path ".dev-api.pid") { Get-Content ".dev-api.pid" -ErrorAction SilentlyContinue } else { $null }
    $uiPid = if (Test-Path ".dev-ui.pid") { Get-Content ".dev-ui.pid" -ErrorAction SilentlyContinue } else { $null }
    $workerPid = if (Test-Path ".dev-worker.pid") { Get-Content ".dev-worker.pid" -ErrorAction SilentlyContinue } else { $null }
    
    $running = @()
    if ($apiPid) {
        try {
            $proc = Get-Process -Id $apiPid -ErrorAction SilentlyContinue
            if ($proc) { $running += "API (PID: $apiPid)" }
        } catch {
            # Ignore
        }
    }
    if ($uiPid) {
        try {
            $proc = Get-Process -Id $uiPid -ErrorAction SilentlyContinue
            if ($proc) { $running += "UI (PID: $uiPid)" }
        } catch {
            # Ignore
        }
    }
    if ($workerPid) {
        try {
            $proc = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
            if ($proc) { $running += "Worker (PID: $workerPid)" }
        } catch {
            # Ignore
        }
    }
    
    if ($running.Count -gt 0) {
        Write-Host "Warning: Found running processes:" -ForegroundColor Yellow
        $running | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
        Write-Host ""
        $response = Read-Host "Stop existing processes and continue? (y/N)"
        if ($response -ne "y" -and $response -ne "Y") {
            Write-Host "Aborted." -ForegroundColor Red
            exit 0
        }
        Write-Host "Stopping existing processes..." -ForegroundColor Yellow
        & "$PSScriptRoot\stop-dev.ps1"
        Start-Sleep -Seconds 2
    }
}

Test-DevProcesses

# Check prerequisites
function Test-Prerequisites {
    $errors = @()
    
    # Check Python
    try {
        $pythonVersion = py --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw }
        Write-Host "OK: Python: $pythonVersion" -ForegroundColor Green
    } catch {
        $errors += "Python not found. Install Python 3.11+ and ensure 'py' is in PATH."
    }
    
    # Check Poetry
    try {
        $poetryVersion = py -m poetry --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw }
        Write-Host "OK: Poetry: $poetryVersion" -ForegroundColor Green
    } catch {
        Write-Host "Installing Poetry..." -ForegroundColor Yellow
        py -m pip install --user poetry | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $errors += "Failed to install Poetry."
        } else {
            Write-Host "OK: Poetry installed" -ForegroundColor Green
        }
    }
    
    # Check Node.js (only if UI is needed)
    if (-not $SkipUI) {
        try {
            $nodeVersion = node --version 2>&1
            if ($LASTEXITCODE -ne 0) { throw }
            Write-Host "OK: Node.js: $nodeVersion" -ForegroundColor Green
        } catch {
            $errors += "Node.js not found. Install Node.js 18+ and ensure 'node' is in PATH."
        }
    }
    
    if ($errors.Count -gt 0) {
        Write-Host ""
        Write-Host "Prerequisites check failed:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host "  Error: $_" -ForegroundColor Red }
        exit 1
    }
    
    Write-Host ""
}

Test-Prerequisites

# Setup environment
$env:PYTHONUNBUFFERED = "1"
$env:NODE_ENV = "development"

# Load settings from data/settings.json
try {
    $settingsPath = Join-Path $PSScriptRoot "data\settings.json"
    if (Test-Path $settingsPath) {
        $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
        if ($settings.roboflow_api_key) {
            $env:ROBOFLOW_API_KEY = $settings.roboflow_api_key
            Write-Host "OK: Loaded ROBOFLOW_API_KEY from settings" -ForegroundColor DarkCyan
        }
        if ($null -ne $settings.matrix_enabled) { $env:MATRIX_ENABLED = "$($settings.matrix_enabled)" }
        if ($settings.matrix_speed) { $env:MATRIX_SPEED = "$($settings.matrix_speed)" }
        if ($settings.matrix_density) { $env:MATRIX_DENSITY = "$($settings.matrix_density)" }
        if ($settings.matrix_opacity) { $env:MATRIX_OPACITY = "$($settings.matrix_opacity)" }
        if ($settings.matrix_color) { $env:MATRIX_COLOR = "$($settings.matrix_color)" }
    }
} catch {
    Write-Host "Warning: Could not load settings: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Ensure logs directory
$logsDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Port detection
function Test-PortInUse {
    param([int]$Port)
    try {
        $conns = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        if ($conns.Count -gt 0) { return $true }
    } catch {
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse("127.0.0.1"), $Port)
            $listener.Start()
            $listener.Stop()
            return $false
        } catch {
            return $true
        }
    }
    return $false
}

function Find-FreePort {
    param([int[]]$Candidates)
    foreach ($port in $Candidates) {
        if (-not (Test-PortInUse -Port $port)) {
            return $port
        }
    }
    return $null
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan

# Python dependencies
try {
    Write-Host "  Installing Python dependencies..." -ForegroundColor DarkGray
    py -m poetry install --no-interaction 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: Python dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  Warning: Poetry install had warnings" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Warning: Poetry install failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# UI dependencies (if needed)
if (-not $SkipUI) {
    $uiDir = Join-Path $PSScriptRoot "ui"
    $nodeModules = Join-Path $uiDir "node_modules"
    if (-not (Test-Path $nodeModules)) {
        Write-Host "  Installing UI dependencies..." -ForegroundColor DarkGray
        $lockFile = Join-Path $uiDir "package-lock.json"
        if (Test-Path $lockFile) {
            Push-Location $uiDir
            npm ci 2>&1 | Out-Null
            Pop-Location
        } else {
            Push-Location $uiDir
            npm install 2>&1 | Out-Null
            Pop-Location
        }
        Write-Host "  OK: UI dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  OK: UI dependencies already installed" -ForegroundColor Green
    }
}

Write-Host ""

# Start API
$apiPort = $null
$apiProcess = $null

if (-not $SkipAPI) {
    Write-Host "Starting API server..." -ForegroundColor Cyan
    $apiCandidates = @(8000, 8001, 8080, 8081)
    $apiPort = Find-FreePort -Candidates $apiCandidates
    
    if (-not $apiPort) {
        Write-Host "Error: No free API port found in $($apiCandidates -join ', ')" -ForegroundColor Red
        exit 1
    }
    
    $apiLog = Join-Path $logsDir "api.log"
    $apiErr = Join-Path $logsDir "api.err.log"
    
    $env:NEXT_PUBLIC_API_BASE = "http://127.0.0.1:$apiPort"
    $env:UI_ORIGIN = "http://localhost:$apiPort"
    
    $apiProcess = Start-Process -FilePath "py" `
        -ArgumentList "-m", "poetry", "run", "uvicorn", "services.api.main:app", `
            "--reload", "--host", "127.0.0.1", "--port", "$apiPort", `
            "--reload-exclude", "data", "--reload-exclude", "*.pyc", "--reload-exclude", "__pycache__", `
            "--reload-dir", "services", "--reload-dir", "core", `
            "--reload-delay", "0.5" `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput $apiLog `
        -RedirectStandardError $apiErr `
        -WorkingDirectory $PSScriptRoot
    
    # Wait for API to be ready
    $maxAttempts = 15
    $attempt = 0
    $apiReady = $false
    
    while ($attempt -lt $maxAttempts -and -not $apiReady) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$apiPort/healthz" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $apiReady = $true
            }
        } catch {
            $attempt++
        }
    }
    
    if ($apiReady) {
        Set-Content -Path ".dev-api.pid" -Value $apiProcess.Id
        Write-Host "  OK: API started (PID: $($apiProcess.Id)) on http://127.0.0.1:$apiPort" -ForegroundColor Green
    } else {
        Write-Host "  Error: API failed to start (check $apiErr)" -ForegroundColor Red
        if (Test-Path $apiErr) {
            Get-Content $apiErr -Tail 10 | Write-Host
        }
        exit 1
    }
}

# Start UI
$uiPort = $null
$uiProcess = $null

if (-not $SkipUI) {
    Write-Host "Starting UI server..." -ForegroundColor Cyan
    $uiCandidates = @(3000, 3001, 3002, 3003)
    $uiPort = Find-FreePort -Candidates $uiCandidates
    
    if (-not $uiPort) {
        Write-Host "Error: No free UI port found in $($uiCandidates -join ', ')" -ForegroundColor Red
        if ($apiProcess) { Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue }
        exit 1
    }
    
    $uiLog = Join-Path $logsDir "ui.log"
    $uiErr = Join-Path $logsDir "ui.err.log"
    
    $uiProcess = Start-Process -FilePath "npx" `
        -ArgumentList "next", "dev", "-p", "$uiPort" `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput $uiLog `
        -RedirectStandardError $uiErr `
        -WorkingDirectory (Join-Path $PSScriptRoot "ui")
    
    # Wait for UI to be ready
    $maxAttempts = 20
    $attempt = 0
    $uiReady = $false
    
    while ($attempt -lt $maxAttempts -and -not $uiReady) {
        Start-Sleep -Seconds 2
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$uiPort" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $uiReady = $true
            }
        } catch {
            $attempt++
        }
    }
    
    if ($uiReady) {
        Set-Content -Path ".dev-ui.pid" -Value $uiProcess.Id
        Write-Host "  OK: UI started (PID: $($uiProcess.Id)) on http://localhost:$uiPort" -ForegroundColor Green
    } else {
        Write-Host "  Warning: UI may still be starting (check $uiErr)" -ForegroundColor Yellow
        Set-Content -Path ".dev-ui.pid" -Value $uiProcess.Id
    }
}

# Start Celery Worker (optional)
$workerProcess = $null

if ($WithWorker) {
    Write-Host "Starting Celery worker..." -ForegroundColor Cyan
    
    # Check if Redis is available
    $redisAvailable = $false
    try {
        $redisTest = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue
        if ($redisTest.TcpTestSucceeded) {
            $redisAvailable = $true
        }
    } catch {
        # Ignore
    }
    
    if (-not $redisAvailable) {
        Write-Host "  Warning: Redis not detected on localhost:6379" -ForegroundColor Yellow
        Write-Host "  Worker may not start properly. Start Redis or use Docker Compose." -ForegroundColor Yellow
    }
    
    $workerLog = Join-Path $logsDir "worker.log"
    $workerErr = Join-Path $logsDir "worker.err.log"
    
    $workerProcess = Start-Process -FilePath "py" `
        -ArgumentList "-m", "poetry", "run", "celery", "-A", "services.worker.app", "worker", "--loglevel=INFO" `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput $workerLog `
        -RedirectStandardError $workerErr `
        -WorkingDirectory $PSScriptRoot
    
    Start-Sleep -Seconds 2
    
    if (-not $workerProcess.HasExited) {
        Set-Content -Path ".dev-worker.pid" -Value $workerProcess.Id
        Write-Host "  OK: Worker started (PID: $($workerProcess.Id))" -ForegroundColor Green
    } else {
        Write-Host "  Error: Worker failed to start (check $workerErr)" -ForegroundColor Red
        if (Test-Path $workerErr) {
            Get-Content $workerErr -Tail 10 | Write-Host
        }
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Bimify Development Environment Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if (-not $SkipAPI) {
    Write-Host "  API:    http://127.0.0.1:$apiPort" -ForegroundColor Cyan
    Write-Host "          Logs: $logsDir\api.log" -ForegroundColor DarkGray
}
if (-not $SkipUI) {
    Write-Host "  UI:     http://localhost:$uiPort" -ForegroundColor Cyan
    Write-Host "          Logs: $logsDir\ui.log" -ForegroundColor DarkGray
}
if ($WithWorker) {
    Write-Host "  Worker: Running (PID: $($workerProcess.Id))" -ForegroundColor Cyan
    Write-Host "          Logs: $logsDir\worker.log" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Stop all services: .\stop-dev.ps1" -ForegroundColor Yellow
Write-Host ""
