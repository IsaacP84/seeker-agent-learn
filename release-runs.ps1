# release-runs.ps1
# Zips checkpoint artifacts and stores the archive in assets/learning/runs.
# The archive is tracked with Git LFS and committed to the repository.
# Usage:
#   .\release-runs.ps1 -ZipName "custom-name.zip"

param(
    [string]$ZipName = ""
)

$ErrorActionPreference = "Stop"

if (-not $ZipName) {
    $ZipName = Read-Host "Enter the zip file name for the checkpoint archive (required)"
}

if ([string]::IsNullOrWhiteSpace($ZipName)) {
    Write-Host "[release-runs] ERROR: Zip name is required. Exiting." -ForegroundColor Red
    exit 1
}

if (-not $ZipName.ToLower().EndsWith('.zip')) {
    $ZipName += '.zip'
}

$repoRoot = $PSScriptRoot
$runFolder = Join-Path $repoRoot "seeker-agent-learn-install\bin\runs"
$outputFolder = Join-Path $repoRoot "assets\learning\runs"
$zipPath = Join-Path $outputFolder $ZipName

if (-not (Test-Path $runFolder)) {
    Write-Host "[release-runs] ERROR: Source folder not found: $runFolder" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $outputFolder)) {
    Write-Host "[release-runs] Creating output folder: $outputFolder" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $outputFolder -Force | Out-Null
}

if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "[release-runs] ERROR: Git LFS is required. Install it and run 'git lfs install'." -ForegroundColor Red
    exit 1
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Write-Host "[release-runs] Collecting checkpoint artifacts from runs folder..." -ForegroundColor Cyan
$allowedPattern = '\.pt(?:\.zip)?$|\.png$|\.log$'
$runFiles = Get-ChildItem -Path $runFolder -File | Where-Object { $_.Name -match $allowedPattern }

if (-not $runFiles) {
    Write-Host "[release-runs] ERROR: No checkpoint artifacts found in $runFolder. Nothing to package." -ForegroundColor Red
    exit 1
}

Write-Host "[release-runs] Zipping allowed run artifacts to $zipName ..." -ForegroundColor Cyan
Compress-Archive -Path $runFiles.FullName -DestinationPath $zipPath -Force

Write-Host "[release-runs] Tracking checkpoint archive with Git LFS..." -ForegroundColor Cyan
$trackSpec = "assets/learning/runs/*.zip"
git lfs track "$trackSpec" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[release-runs] ERROR: Failed to add Git LFS tracking for '$trackSpec'." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[release-runs] Staging archive and LFS attributes..." -ForegroundColor Cyan
git add .gitattributes "$zipPath"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[release-runs] ERROR: Failed to stage LFS tracking or archive file." -ForegroundColor Red
    exit $LASTEXITCODE
}

$commitMessage = "Add checkpoint run archive $ZipName via Git LFS"
Write-Host "[release-runs] Committing archive to Git..." -ForegroundColor Cyan
git commit -m "$commitMessage" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[release-runs] Notice: No commit was created. The staged changes may already be committed or there is nothing new to commit." -ForegroundColor Yellow
}

$currentBranch = git rev-parse --abbrev-ref HEAD
if ($LASTEXITCODE -eq 0) {
    Write-Host "[release-runs] Pushing branch '$currentBranch' to remote..." -ForegroundColor Cyan
git push origin "$currentBranch"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[release-runs] WARNING: Git push failed. The archive may still be staged locally." -ForegroundColor Yellow
    }
} else {
    Write-Host "[release-runs] WARNING: Could not determine current git branch; skipping push." -ForegroundColor Yellow
}

Write-Host "[release-runs] Archive stored at: $zipPath" -ForegroundColor Green
