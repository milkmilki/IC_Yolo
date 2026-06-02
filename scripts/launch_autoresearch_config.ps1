param(
    [string]$Config = "AutoResearch\configs\wm811k_autoresearch_topology_ctm.yaml",
    [string]$Python = "D:\anaconda3\envs\pcb_yolo\python.exe",
    [string]$ProjectRoot = "E:\Cjn\PCB_Yolo",
    [switch]$Background,
    [switch]$CheckConfig
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $ProjectRoot "scripts\run_wm811k_pipeline.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Pipeline script not found: $scriptPath"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = $Config
} else {
    $configPath = Join-Path $ProjectRoot $Config
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Config not found: $configPath"
}

$logDir = Join-Path $ProjectRoot "AutoResearch\launch_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ($Background) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir ("autoresearch_" + $stamp + ".out.log")
    $stderr = Join-Path $logDir ("autoresearch_" + $stamp + ".err.log")
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList (@($scriptPath, "--config", $configPath) + $(if ($CheckConfig) { @("--check-config") } else { @() })) `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Write-Host ("pid=" + $process.Id)
    Write-Host ("stdout=" + $stdout)
    Write-Host ("stderr=" + $stderr)
    exit 0
}

Set-Location $ProjectRoot
if ($CheckConfig) {
    & $Python $scriptPath --config $configPath --check-config
} else {
    & $Python $scriptPath --config $configPath
}
