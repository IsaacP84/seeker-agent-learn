<#
.SYNOPSIS
Configure, build, install, zip, and publish the release build of seeker-agent-learn.

.DESCRIPTION
This script runs CMake configure, build, and install steps using the same settings as
install-release.bat. After installing, it creates a zip archive of the install tree
and optionally publishes a GitHub release using the gh CLI.
#>

[CmdletBinding()]
param(
    [string]
    $RepoUrl = "https://github.com/IsaacP84/seeker-agent-learn.git",

    [string]
    $CloneDir = "$PSScriptRoot\temp-clone-release",

    [string]
    $BuildDir = "temp-build-release",

    [string]
    $InstallDir = "temp-install-release-windows",

    [string]
    $MagicDir = "engine\lib\cmake\Magic",

    [string]
    $PythonRoot = "C:/Github/magic-engine/python",

    [string]
    $PythonExecutable = "C:/Github/magic-engine/python/python.exe",

    [int]
    $Jobs = 6,

    [string]
    $ZipPath = "$PSScriptRoot\install-release-windows.zip",

    [string]
    $Repo = "IsaacP84/seeker-agent-learn",

    [string]
    $ReleaseName = "windows-latest",

    [string]
    $ReleaseTitle = "windows-latest",

    [string]
    $ReleaseNotes = "Updated install-release-windows folder",

    [string]
    $TargetBranch = "master",

    [Alias('y')]
    [switch]
    $AutoClean,

    [switch]
    $PublishRelease = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-TargetPath {
    param(
        [string]$Path,
        [string]$Base
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path -Path $Base -ChildPath $Path
}

$CloneDirFull = Resolve-TargetPath $CloneDir $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not available in PATH. Cannot clone repository."
}

if (Test-Path $CloneDirFull) {
    Write-Host "Removing existing clone directory: $CloneDirFull"
    Remove-Item -Recurse -Force $CloneDirFull
}

Write-Host "Cloning repository $RepoUrl into $CloneDirFull"
& git clone $RepoUrl $CloneDirFull
if ($LASTEXITCODE -ne 0) {
    throw "Git clone failed with exit code $LASTEXITCODE"
}

$RepoRoot = $CloneDirFull

$BuildDirFull = Resolve-TargetPath $BuildDir $RepoRoot
$InstallDirFull = Resolve-TargetPath $InstallDir $RepoRoot
$ZipPathFull = Resolve-TargetPath $ZipPath $PSScriptRoot

if (-not [System.IO.Path]::IsPathRooted($MagicDir)) {
    $MagicDir = Join-Path -Path $RepoRoot -ChildPath $MagicDir
}

Write-Host "Repo root: $RepoRoot"
Write-Host "Build directory: $BuildDirFull"
Write-Host "Install directory: $InstallDirFull"
Write-Host "Magic DIR: $MagicDir"
Write-Host "Python root: $PythonRoot"
Write-Host "Python executable: $PythonExecutable"
Write-Host "Zip output: $ZipPathFull"
if ($PublishRelease) {
    Write-Host "GitHub repo: $Repo"
    Write-Host "Release: $ReleaseName"
    Write-Host "Target branch: $TargetBranch"
}

New-Item -ItemType Directory -Force -Path $BuildDirFull | Out-Null
New-Item -ItemType Directory -Force -Path $InstallDirFull | Out-Null

Write-Host "Configuring release build..."
$configureArgs = @(
    '-S', $RepoRoot,
    '-B', $BuildDirFull,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_INSTALL_MESSAGE=LAZY',
    "-DMagic_DIR=$MagicDir",
    "-DPython3_ROOT_DIR=$PythonRoot",
    "-DPython3_EXECUTABLE=$PythonExecutable"
)

& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

Write-Host "Building release build..."
$buildArgs = @(
    '--build', $BuildDirFull,
    '--config', 'Release',
    '--', "-j$Jobs"
)

& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE"
}

Write-Host "Installing into $InstallDirFull..."
$installArgs = @(
    '--install', $BuildDirFull,
    '--config', 'Release',
    '--prefix', $InstallDirFull
)

& cmake @installArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake install failed with exit code $LASTEXITCODE"
}

Write-Host "Installation complete. Installed to: $InstallDirFull"

Write-Host "Creating zip archive: $ZipPathFull"
if (Test-Path $ZipPathFull) {
    Remove-Item $ZipPathFull -Force
}

$installFiles = Join-Path -Path $InstallDirFull -ChildPath '*'
Compress-Archive -Path $installFiles -DestinationPath $ZipPathFull -Force
if ($LASTEXITCODE -ne 0) {
    throw "Compress-Archive failed with exit code $LASTEXITCODE"
}

Write-Host "Zip archive created: $ZipPathFull"

if ($PublishRelease) {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw "GitHub CLI 'gh' is not installed or not available in PATH. Cannot publish release."
    }

    Write-Host "Publishing GitHub release: $ReleaseName"
    $releaseExists = $false
    & gh release view $ReleaseName --repo $Repo 2>$null
    if ($LASTEXITCODE -eq 0) {
        $releaseExists = $true
    }

    if ($releaseExists) {
        Write-Host "Deleting existing release: $ReleaseName"
        & gh release delete $ReleaseName --repo $Repo -y
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to delete existing release '$ReleaseName'."
        }
    }

    Write-Host "Creating release $ReleaseName from archive $ZipPathFull"
    & gh release create $ReleaseName $ZipPathFull --repo $Repo --title $ReleaseTitle --notes $ReleaseNotes --target $TargetBranch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create GitHub release '$ReleaseName'."
    }

    Write-Host "GitHub release published: $ReleaseName"
}

if ($AutoClean) {
    Write-Host "Auto-clean enabled. Removing generated artifacts..."
    if (Test-Path $ZipPathFull) {
        Remove-Item -Force $ZipPathFull
    }
    if (Test-Path $BuildDirFull) {
        Remove-Item -Recurse -Force $BuildDirFull
    }
    if (Test-Path $InstallDirFull) {
        Remove-Item -Recurse -Force $InstallDirFull
    }
    Write-Host "Cleanup complete. Removed generated artifacts."
} else {
    $cleanupPrompt = Read-Host "Delete the zip, build directory, and install directory? [y/N]"
    if ($cleanupPrompt -match '^[Yy](?:es)?$') {
        Write-Host "Removing zip archive..."
        if (Test-Path $ZipPathFull) {
            Remove-Item -Force $ZipPathFull
        }

        Write-Host "Removing build directory..."
        if (Test-Path $BuildDirFull) {
            Remove-Item -Recurse -Force $BuildDirFull
        }

        Write-Host "Removing install directory..."
        if (Test-Path $InstallDirFull) {
            Remove-Item -Recurse -Force $InstallDirFull
        }

        Write-Host "Cleanup complete. Removed generated artifacts."
    } else {
        Write-Host "Cleanup skipped. Generated artifacts were retained."
    }
}
