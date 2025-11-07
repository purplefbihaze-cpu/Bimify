Param(
    [string]$Runtime = "win-x64",
    [switch]$SelfContained = $true
)

$ErrorActionPreference = "Stop"

Write-Host "=== Publish XbimPreprocess ($Runtime) ===" -ForegroundColor Cyan

$proj = Join-Path $PSScriptRoot "XbimPreprocess/XbimPreprocess.csproj"
if (-not (Test-Path $proj)) {
    Write-Error "Projektdatei nicht gefunden: $proj"
}

$publishDir = Join-Path $PSScriptRoot "XbimPreprocess/publish/$Runtime"
New-Item -ItemType Directory -Force -Path $publishDir | Out-Null

$sc = if ($SelfContained) { "true" } else { "false" }

dotnet publish `
  $proj `
  -c Release `
  -r $Runtime `
  --self-contained $sc `
  /p:PublishSingleFile=false `
  /p:PublishTrimmed=false `
  -o $publishDir

Write-Host "=== Fertig. Ausgabe: $publishDir ===" -ForegroundColor Green
Get-ChildItem $publishDir | Select-Object Name,Length,LastWriteTime | Format-Table


