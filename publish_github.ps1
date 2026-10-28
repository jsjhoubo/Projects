param(
    [string]$Repository = 'distyl-lattice-systems-mock',
    [ValidateSet('private','public')][string]$Visibility = 'private'
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is required. Install via: winget install GitHub.cli; restart PowerShell.'
}
& gh auth status 2>$null
if ($LASTEXITCODE -ne 0) { & gh auth login }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required (winget install Git.Git).' }
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path .git)) { & git init -b main }
    & git add .
    & git -c user.name='Robert Hou' -c user.email='jsjhoubo@gmail.com' commit -m 'Add extensive LATTICE-inspired systems interview mock'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed; configure git identity if necessary.' }
    $full = "jsjhoubo/$Repository"
    & gh repo view $full *> $null
    if ($LASTEXITCODE -eq 0) {
        & git remote remove origin 2>$null
        & git remote add origin "https://github.com/$full.git"
        & git push -u origin main
    } else {
        & gh repo create $full --$Visibility --source . --remote origin --push
    }
    if ($LASTEXITCODE -ne 0) { throw 'GitHub publish failed. Check permissions and repository name.' }
    Write-Host "Published to https://github.com/$full"
} finally { Pop-Location }
