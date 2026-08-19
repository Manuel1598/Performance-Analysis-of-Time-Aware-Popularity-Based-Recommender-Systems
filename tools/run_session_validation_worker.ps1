param(
    [string]$WorkerName = "session_pc2",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $BundledPython = Join-Path $ProjectRoot ".venv-vsknn\Scripts\python.exe"
    if (Test-Path -LiteralPath $BundledPython) {
        $PythonExecutable = $BundledPython
    }
    else {
        $PythonExecutable = "python"
    }
}

$OutputDirectory = Join-Path $ProjectRoot "recbole_results\validation_first_workers\$WorkerName"
$Runner = Join-Path $ProjectRoot "tools\run_validation_first_experiments.py"
$RunnerArguments = @(
    "--phase", "tune",
    "--scenario", "session",
    "--device", $Device,
    "--output-dir", $OutputDirectory
)

Write-Host "Session validation worker: $WorkerName"
Write-Host "Device: $Device"
Write-Host "Results: $OutputDirectory"

& $PythonExecutable $Runner @RunnerArguments

if ($LASTEXITCODE -ne 0) {
    throw "Session validation worker failed with exit code $LASTEXITCODE"
}

Write-Host "Session validation worker completed successfully."
Write-Host "Copy this file back to the main computer:"
Write-Host (Join-Path $OutputDirectory "validation_trials.csv")
