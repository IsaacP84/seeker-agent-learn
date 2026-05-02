# download-engine.ps1
# Downloads and extracts the prebuilt engine binaries from a GitHub release asset.
# Run this after cloning or when the engine/ folder is missing or outdated.
#
# Usage:
#   .\download-engine.ps1
#   .\download-engine.ps1 -Tag "v1.0" -Force

param(
    [string]$Tag   = "",
    [string]$Repo  = "IsaacP84/seeker-agent-learn",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Resolve tag from current git repo if not specified.
if (-not $Tag) {
    $gitTag = git -C $PSScriptRoot describe --tags --abbrev=0 2>$null
    if ($LASTEXITCODE -eq 0 -and $gitTag) {
        $Tag = $gitTag.Trim()
        Write-Host "[download-engine] Using current git tag: $Tag" -ForegroundColor Cyan
    } else {
        $Tag = "engine-latest"
        Write-Host "[download-engine] No git tag found; falling back to '$Tag'" -ForegroundColor Yellow
    }
}

$engineDir = Join-Path $PSScriptRoot "engine"
$zipPath   = Join-Path $PSScriptRoot "engine.zip"
$assetUrl  = "https://github.com/$Repo/releases/download/$Tag/engine.zip"

if ((Test-Path $engineDir) -and !$Force) {
    Write-Host "[download-engine] engine/ already exists. Use -Force to re-download." -ForegroundColor Cyan
    exit 0
}

Write-Host "[download-engine] Downloading $assetUrl ..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $assetUrl -OutFile $zipPath -UseBasicParsing

Write-Host "[download-engine] Extracting to $engineDir ..." -ForegroundColor Yellow
if (Test-Path $engineDir) {
    Remove-Item $engineDir -Recurse -Force
}
Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force
Remove-Item $zipPath -Force

Write-Host "[download-engine] Done." -ForegroundColor Green
