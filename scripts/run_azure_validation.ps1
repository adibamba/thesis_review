# One-click Azure validation and local pipeline run
# - Checks the Azure health endpoint
# - Backs up previous metrics logs
# - Runs the surveillance pipeline
# - Launches Streamlit dashboard

param(
    [int]$Frames = 120,
    [int]$Seed = 1,
    [string]$Config = ""
)

Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:CLOUD_URL)) {
    Write-Error "CLOUD_URL is not set."
    exit 2
}
$healthUrl = "$($env:CLOUD_URL.TrimEnd('/'))/health"
$surveillanceScript = Join-Path $projectRoot "scripts\run_yolo_surveillance.py"
$dashboardScript = Join-Path $projectRoot "scripts\visualize_topology_dashboard.py"
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $projectRoot "config\yolo_controller_config.yaml"
}

Write-Host "Checking Azure health at $healthUrl..."
try {
    $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
        Write-Host "Azure health check passed (status $($resp.StatusCode))."
    } else {
        Write-Error "Azure health check failed with status $($resp.StatusCode). Aborting."
        exit 2
    }
} catch {
    Write-Error "Failed to reach Azure health endpoint: $_"
    exit 3
}

function Test-PythonCandidate([string]$Candidate) {
    $Candidate = $Candidate.Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($Candidate) -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    try {
        & $Candidate -c "import sys; sys.exit(0 if sys.executable else 1)" 2>$null | Out-Null
        $exitCode = $LASTEXITCODE
        return ($exitCode -eq 0)
    } catch {
        return $false
    }
}

$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe")
)
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $pythonCommand) {
    $pythonCandidates += $pythonCommand.Path
}
$pyCommand = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pyCommand) {
    try {
        $pyPython = & $pyCommand.Path -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($pyPython)) {
            $pythonCandidates += $pyPython.Trim()
        }
    } catch {
        # Continue with direct interpreter candidates.
    }
}
$pythonCandidates += @(
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe",
    "$env:USERPROFILE\anaconda3\python.exe"
)
$python = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    $normalizedCandidate = ([string]$candidate).Trim().Trim('"')
    if (Test-PythonCandidate $normalizedCandidate) {
        $python = $normalizedCandidate
        break
    }
}
if ([string]::IsNullOrWhiteSpace($python)) {
    Write-Error "No usable Python interpreter found."
    exit 4
}
Write-Host "Using Python: $python"

Write-Host "Running cloud smoke test..."
& $python (Join-Path $projectRoot "scripts\smoke_test_cloud.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Cloud smoke test failed. Aborting surveillance run."
    exit $LASTEXITCODE
}

# Backup previous metrics
$metricsPath = Join-Path -Path (Join-Path $projectRoot "logs") -ChildPath "yolo_surveillance_metrics.json"
$metricsPath = Resolve-Path -Path $metricsPath -ErrorAction SilentlyContinue
if ($null -ne $metricsPath) {
    $metricsFull = $metricsPath.Path
    if (Test-Path $metricsFull) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "$($metricsFull).bak.$timestamp"
        Copy-Item -Path $metricsFull -Destination $backup -Force
        Write-Host "Backed up existing metrics to $backup"
        Get-ChildItem -Path (Split-Path $metricsFull) -Filter "yolo_surveillance_metrics.json.bak.*" |
            Sort-Object LastWriteTime -Descending |
            Select-Object -Skip 5 |
            Remove-Item -Force
    }
} else {
    Write-Host "No existing metrics file found to back up."
}

# Run surveillance script
Write-Host "Running surveillance script (frames=$Frames)..."
$proc = Start-Process -FilePath $python -ArgumentList $surveillanceScript, "--frames", $Frames, "--seed", $Seed, "--config", $Config -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Error "Surveillance script exited with code $($proc.ExitCode). Aborting dashboard launch."
    exit $proc.ExitCode
}

# Generate final plots via Streamlit (run in browser)
Write-Host "Launching Streamlit dashboard..."
Start-Process -FilePath $python -ArgumentList "-m", "streamlit", "run", $dashboardScript
Write-Host "Streamlit has been launched. If a browser did not open automatically, visit the URL printed by Streamlit in the console."
