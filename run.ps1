$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Host 'First follow the setup steps in README.md to create .venv and install requirements.'
    exit 1
}
& $projectPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1
