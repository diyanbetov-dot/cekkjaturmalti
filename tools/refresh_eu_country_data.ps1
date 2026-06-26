param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tempDir = Join-Path $env:TEMP "cekkjatur-eu-country-import"
$profileDir = Join-Path $tempDir "chrome-profile"
$mtHtml = Join-Path $tempDir "annex-a5-mt.html"
$enHtml = Join-Path $tempDir "annex-a5-en.html"

$chromeCandidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
)
$browser = $chromeCandidates |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1

if (-not $browser) {
    throw "Chrome or Edge is required to render the official EU tables."
}

New-Item -ItemType Directory -Force -Path $tempDir, $profileDir | Out-Null

function Save-RenderedTable([string]$Language, [string]$OutputPath) {
    $url = "https://style-guide.europa.eu/o/opportal-service/isg?resource=$Language/annex-a5-list-countries-territories-currencies.html"
    $errorPath = "$OutputPath.err"
    $arguments = @(
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--disable-default-apps",
        "--user-data-dir=$profileDir",
        "--virtual-time-budget=10000",
        "--dump-dom",
        $url
    )
    Start-Process -FilePath $browser -ArgumentList $arguments `
        -RedirectStandardOutput $OutputPath `
        -RedirectStandardError $errorPath `
        -WindowStyle Hidden -Wait

    if (-not (Select-String -Path $OutputPath -Pattern "annex-5-desktop-table" -Quiet)) {
        throw "The official $Language table did not render correctly."
    }
}

Save-RenderedTable "mt" $mtHtml
Save-RenderedTable "en" $enHtml

& $PythonExe (Join-Path $PSScriptRoot "import_eu_country_data.py") `
    --mt-html $mtHtml `
    --en-html $enHtml `
    --json-output (Join-Path $root "finaldics\eu_countries.json") `
    --dic-output (Join-Path $root "finaldics\eu_countries.dic")

if ($LASTEXITCODE -ne 0) {
    throw "Country-data generation failed with exit code $LASTEXITCODE."
}
