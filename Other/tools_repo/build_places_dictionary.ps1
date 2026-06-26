param(
    [Parameter(Mandatory = $true)]
    [string]$CitiesFile,

    [Parameter(Mandatory = $true)]
    [string]$MaltaFile,

    [Parameter(Mandatory = $true)]
    [string]$CountryInfoFile,

    [Parameter(Mandatory = $true)]
    [string]$TerritoriesFile,

    [string]$TimeZonesFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
$globalPlaces = [System.Collections.Generic.Dictionary[string, string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$malteseGlobalPlaces = [System.Collections.Generic.Dictionary[string, string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$maltesePlaces = [System.Collections.Generic.Dictionary[string, string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$existingDemonyms = [System.Collections.Generic.List[string]]::new()

if (Test-Path -LiteralPath $OutputFile) {
    foreach ($line in [IO.File]::ReadLines(
        (Resolve-Path -LiteralPath $OutputFile),
        [Text.Encoding]::UTF8
    )) {
        if ($line -match '^[^/]+/(?:MLT-)?(?:DNYMM|DYNMF|DYNMPL)-') {
            $taggedDemonym = $line -replace '/(?!MLT-)', '/MLT-'
            $existingDemonyms.Add($taggedDemonym)
        }
    }
}

function Test-LatinPlaceName {
    param([string]$Name)

    foreach ($character in $Name.ToCharArray()) {
        $codePoint = [int]$character
        if (
            [char]::IsLetterOrDigit($character) -and
            $codePoint -gt 0x024F
        ) {
            return $false
        }
    }
    return $true
}

function Add-Place {
    param(
        [string]$Name,
        [switch]$RequireLatin
    )

    if ([string]::IsNullOrWhiteSpace($Name)) {
        return
    }

    $clean = $Name.Normalize([Text.NormalizationForm]::FormC).Trim()
    $clean = [regex]::Replace($clean, '\s+', ' ')

    if (
        $clean.Length -lt 2 -or
        $clean.Length -gt 100 -or
        $clean -notmatch '\p{L}' -or
        $clean -notmatch "^[\p{L}\p{M}0-9 .'\u2019-]+$"
    ) {
        return
    }

    if ($RequireLatin -and -not (Test-LatinPlaceName $clean)) {
        return
    }

    if (-not $globalPlaces.ContainsKey($clean)) {
        $globalPlaces[$clean] = $clean
    }
}

function Add-MaltesePlace {
    param([string]$Name)

    if ([string]::IsNullOrWhiteSpace($Name)) {
        return
    }

    $clean = $Name.Normalize([Text.NormalizationForm]::FormC).Trim()
    $clean = $clean.Replace([char]0x2019, "'").Replace([char]0x2018, "'")
    $clean = [regex]::Replace($clean, '\s+', ' ')

    if (
        $clean.Length -ge 2 -and
        $clean.Length -le 100 -and
        $clean -match '\p{L}' -and
        $clean -match "^[\p{L}\p{M}0-9 .'-]+$" -and
        -not $maltesePlaces.ContainsKey($clean)
    ) {
        $maltesePlaces[$clean] = $clean
    }
}

function Add-MalteseGlobalPlace {
    param([string]$Name)

    Add-Place $Name
    if ([string]::IsNullOrWhiteSpace($Name)) {
        return
    }
    $clean = $Name.Normalize([Text.NormalizationForm]::FormC).Trim()
    $clean = [regex]::Replace($clean, '\s+', ' ')
    if ($clean -and -not $malteseGlobalPlaces.ContainsKey($clean)) {
        $malteseGlobalPlaces[$clean] = $clean
    }
}

function Test-MalteseArticleName {
    param([string]$Name)

    $hyphenIndex = $Name.IndexOf('-')
    if ($hyphenIndex -le 0) {
        return $false
    }

    $prefix = $Name.Substring(0, $hyphenIndex).ToLowerInvariant()
    return (
        $prefix -in @('il', 'l') -or
        ($prefix.Length -eq 2 -and $prefix.StartsWith('i'))
    )
}

function Get-PreferredMalteseName {
    param([string[]]$Fields)

    $canonical = $Fields[1]
    if ($Fields.Length -lt 4 -or [string]::IsNullOrWhiteSpace($Fields[3])) {
        return $canonical
    }

    $aliases = $Fields[3].Split(',')
    $hBarUpper = [char]294
    $zDotLower = [char]380
    $hApostrophe = "$hBarUpper'"
    $hCurlyApostrophe = "$hBarUpper$([char]0x2019)"
    $halPrefix = "$hBarUpper" + 'al '
    $halHyphenPrefix = "$hBarUpper" + 'al-'
    $hazPrefix = "$hBarUpper" + "a$zDotLower-"

    $unicodeMatch = $aliases |
        Where-Object {
            $_.StartsWith($hApostrophe) -or
            $_.StartsWith($hCurlyApostrophe)
        } |
        Sort-Object Length |
        Select-Object -First 1
    if ($unicodeMatch) {
        return $unicodeMatch
    }
    if ($aliases | Where-Object { $_ -match "^H['’]" }) {
        return "$hApostrophe$canonical"
    }

    $unicodeMatch = $aliases |
        Where-Object { $_.StartsWith($halPrefix) } |
        Sort-Object Length |
        Select-Object -First 1
    if ($unicodeMatch) {
        return $unicodeMatch
    }
    $unicodeMatch = $aliases |
        Where-Object { $_.StartsWith($halHyphenPrefix) } |
        Sort-Object Length |
        Select-Object -First 1
    if ($unicodeMatch) {
        return $unicodeMatch.Replace($halHyphenPrefix, $halPrefix)
    }
    if ($aliases | Where-Object { $_ -match '^Hal(?:[ -])' }) {
        return $halPrefix + ($canonical -replace '^Hal[- ]', '')
    }

    $unicodeMatch = $aliases |
        Where-Object { $_.StartsWith($hazPrefix) } |
        Sort-Object Length |
        Select-Object -First 1
    if ($unicodeMatch) {
        return $unicodeMatch
    }
    if ($aliases | Where-Object { $_ -match '^Haz-' }) {
        return $hazPrefix + ($canonical -replace '^Haz-', '')
    }

    $articleMatch = $aliases |
        Where-Object {
            Test-MalteseArticleName $_
        } |
        Sort-Object Length |
        Select-Object -First 1
    if ($articleMatch) {
        return $articleMatch
    }

    return $canonical
}

function Get-MalteseSortKey {
    param([string]$Name)

    $key = $Name.Normalize([Text.NormalizationForm]::FormC)
    $hBarUpper = [regex]::Escape([string][char]294)
    $zDotLower = [regex]::Escape([string][char]380)
    $key = $key -replace "^$hBarUpper['’]", ''
    $key = $key -replace "^${hBarUpper}al[ -]+", ''
    $key = $key -replace "^${hBarUpper}a${zDotLower}-", ''
    return $key.Normalize([Text.NormalizationForm]::FormD).ToLowerInvariant()
}

function Get-MalteseDuplicateKey {
    param([string]$Name)

    $key = $Name.Normalize([Text.NormalizationForm]::FormC).ToLowerInvariant()
    $key = $key.Replace([char]0x2019, "'").Replace([char]0x2018, "'")
    $key = $key.Replace(([char]295).ToString(), 'h')
    $key = $key.Replace(([char]294).ToString(), 'h')
    $key = $key.Replace(([char]289).ToString(), 'g')
    $key = $key.Replace(([char]288).ToString(), 'g')
    $key = $key.Replace(([char]267).ToString(), 'c')
    $key = $key.Replace(([char]266).ToString(), 'c')
    $key = $key.Replace(([char]380).ToString(), 'z')
    $key = $key.Replace(([char]379).ToString(), 'z')
    $key = $key.Replace('għ', 'gh')
    return $key.Normalize([Text.NormalizationForm]::FormD)
}

function Get-MalteseQualityScore {
    param([string]$Name)

    $score = 0
    foreach ($character in $Name.ToCharArray()) {
        if (
            $character -in @(
                [char]294, [char]295,
                [char]266, [char]267,
                [char]288, [char]289,
                [char]379, [char]380
            )
        ) {
            $score += 10
        }
    }
    $score -= ([regex]::Matches($Name, '(?i)gh').Count * 2)
    return $score
}

function Convert-DemonymCase {
    param([string]$Line)

    $slashIndex = $Line.IndexOf('/')
    if ($slashIndex -le 0) {
        return $Line
    }

    $surface = $Line.Substring(0, $slashIndex).ToLowerInvariant()
    $surface = (
        [char]::ToUpperInvariant($surface[0]) +
        $surface.Substring(1)
    )
    return $surface + $Line.Substring($slashIndex)
}

function Add-MalteseArticleVariants {
    param([string[]]$Fields)

    if ($Fields.Length -lt 4 -or [string]::IsNullOrWhiteSpace($Fields[3])) {
        return
    }

    foreach ($alias in $Fields[3].Split(',')) {
        $clean = $alias.Normalize([Text.NormalizationForm]::FormC).Trim()
        if (-not (Test-MalteseArticleName $clean)) {
            continue
        }

        Add-MaltesePlace $clean
        $articleTail = $clean.Substring($clean.IndexOf('-') + 1)
        $bareCandidates = @($Fields[1], $Fields[2]) + $Fields[3].Split(',')
        $bare = $bareCandidates |
            Where-Object {
                $_ -and
                -not (Test-MalteseArticleName $_) -and
                -not (
                    $articleTail -match '^I[mn][^aeiou]' -and
                    (Get-MalteseDuplicateKey $_) -eq
                    (Get-MalteseDuplicateKey $articleTail)
                ) -and
                (
                    (Get-MalteseDuplicateKey $_) -eq
                    (Get-MalteseDuplicateKey $articleTail) -or
                    (
                        $articleTail.StartsWith('I') -and
                        $articleTail.Length -gt 1 -and
                        (Get-MalteseDuplicateKey $_) -eq
                        (Get-MalteseDuplicateKey $articleTail.Substring(1))
                    )
                )
            } |
            Sort-Object `
                @{
                    Expression = {
                        if (
                            $articleTail.StartsWith('I') -and
                            $articleTail.Length -gt 1 -and
                            (Get-MalteseDuplicateKey $_) -eq
                            (Get-MalteseDuplicateKey $articleTail.Substring(1))
                        ) {
                            0
                        }
                        else {
                            1
                        }
                    }
                }, `
                @{ Expression = { -(Get-MalteseQualityScore $_) } }, `
                Length |
            Select-Object -First 1

        if ($bare) {
            Add-MaltesePlace $bare
        }
    }
}

# Unicode CLDR supplies the preferred Maltese country and territory names.
$territoriesJson = [IO.File]::ReadAllText(
    (Resolve-Path -LiteralPath $TerritoriesFile),
    [Text.Encoding]::UTF8
) | ConvertFrom-Json
$territories = $territoriesJson.main.mt.localeDisplayNames.territories
foreach ($property in $territories.PSObject.Properties) {
    if ($property.Name -notlike '*-alt-*') {
        $territoryName = [string]$property.Value
        Add-MalteseGlobalPlace $territoryName
        $hyphenIndex = $territoryName.IndexOf('-')
        if ($hyphenIndex -gt 0) {
            $article = $territoryName.Substring(
                0,
                $hyphenIndex
            ).ToLowerInvariant()
            if (
                $article -in @('il', 'l') -or
                ($article.Length -eq 2 -and $article.StartsWith('i'))
            ) {
                Add-MalteseGlobalPlace $territoryName.Substring($hyphenIndex + 1)
            }
        }
    }
}

function Add-ExemplarCities {
    param([object]$Node)

    if ($null -eq $Node) {
        return
    }

    foreach ($property in $Node.PSObject.Properties) {
        if ($property.Name -eq 'exemplarCity') {
            Add-MalteseGlobalPlace ([string]$property.Value)
        }
        elseif ($property.Value -is [System.Management.Automation.PSCustomObject]) {
            Add-ExemplarCities $property.Value
        }
    }
}

if ($TimeZonesFile) {
    $timeZonesJson = [IO.File]::ReadAllText(
        (Resolve-Path -LiteralPath $TimeZonesFile),
        [Text.Encoding]::UTF8
    ) | ConvertFrom-Json
    Add-ExemplarCities $timeZonesJson.main.mt.dates.timeZoneNames.zone
}

# GeoNames countryInfo adds internationally familiar country names and capitals.
foreach ($line in [IO.File]::ReadLines(
    (Resolve-Path -LiteralPath $CountryInfoFile),
    [Text.Encoding]::UTF8
)) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) {
        continue
    }
    $fields = $line.Split("`t")
    if ($fields.Length -ge 6) {
        Add-Place $fields[4]
        Add-Place $fields[5]
    }
}

# GeoNames cities15000 gives broad global city coverage without tiny features.
foreach ($line in [IO.File]::ReadLines(
    (Resolve-Path -LiteralPath $CitiesFile),
    [Text.Encoding]::UTF8
)) {
    $fields = $line.Split("`t")
    if (
        $fields.Length -ge 9 -and
        $fields[6] -eq 'P' -and
        $fields[8] -ne 'MT'
    ) {
        Add-Place $fields[1]
        Add-Place $fields[2] -RequireLatin
    }
}

# Include every Maltese populated place by its canonical and ASCII names.
# GeoNames' unlabelled alias field also contains transliterations and historical
# spellings, so it is deliberately not imported wholesale.
foreach ($line in [IO.File]::ReadLines(
    (Resolve-Path -LiteralPath $MaltaFile),
    [Text.Encoding]::UTF8
)) {
    $fields = $line.Split("`t")
    $isPopulatedPlace = $fields.Length -ge 9 -and $fields[6] -eq 'P'
    $isAdministrativeLocality = (
        $fields.Length -ge 9 -and
        $fields[6] -eq 'A' -and
        $fields[7] -eq 'ADM1'
    )
    if (-not ($isPopulatedPlace -or $isAdministrativeLocality)) {
        continue
    }

    Add-MaltesePlace (Get-PreferredMalteseName $fields)
    Add-MalteseArticleVariants $fields
}

# Established Maltese alternatives that are useful in real text.
foreach ($knownName in @(
    'Marsascala',
    'Marsa Scala',
    ('Wied il-G' + [char]295 + 'ajn'),
    'Wied il-Ghajn',
    'Il-Belt Valletta'
)) {
    Add-MaltesePlace $knownName
}

$orderedGlobal = $globalPlaces.Values |
    Where-Object {
        -not $maltesePlaces.ContainsKey($_) -and
        -not $malteseGlobalPlaces.ContainsKey($_)
    } |
    Sort-Object {
        $_.Normalize([Text.NormalizationForm]::FormD).ToLowerInvariant()
    }
$preferredMaltese = [System.Collections.Generic.Dictionary[string, string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($name in $maltesePlaces.Values) {
    $key = Get-MalteseDuplicateKey $name
    if (
        -not $preferredMaltese.ContainsKey($key) -or
        (Get-MalteseQualityScore $name) -gt
        (Get-MalteseQualityScore $preferredMaltese[$key])
    ) {
        $preferredMaltese[$key] = $name
    }
}

$orderedMaltese = $preferredMaltese.Values | Sort-Object `
    @{ Expression = { if ($_ -eq (([char]294).ToString() + "'Attard")) { 0 } else { 1 } } }, `
    @{ Expression = { Get-MalteseSortKey $_ } }, `
    @{ Expression = { $_.Normalize([Text.NormalizationForm]::FormD).ToLowerInvariant() } }
$taggedMaltese = $orderedMaltese | ForEach-Object {
    "$_/MLT-PLACE"
}
$orderedMalteseGlobal = $malteseGlobalPlaces.Values |
    Sort-Object {
        $_.Normalize([Text.NormalizationForm]::FormD).ToLowerInvariant()
    } |
    ForEach-Object { "$_/MLT-PLACE" }

$header = @(
    '# Generated by tools/build_places_dictionary.ps1',
    '# Sources: Unicode CLDR Maltese territories and time-zone cities; GeoNames countryInfo, cities15000, and MT',
    '# Global place names are followed by Maltese populated places.',
    '# GLOBAL PLACE NAMES'
)
$malteseHeader = @(
    '',
    '# MALTESE GLOBAL PLACE NAMES',
    '# Names sourced from Maltese Unicode CLDR.'
)
$malteseLocalHeader = @(
    '',
    '# MALTESE POPULATED PLACES',
    (
        '# Sorted by the locality name after ' +
        [char]294 + "', " + [char]294 + 'al, or ' +
        [char]294 + 'a' + [char]380 + '-.'
    )
)
$demonymSection = @()
if ($existingDemonyms.Count -gt 0) {
    $demonymSection = @(
        '',
        '# MALTESE LOCALITY DEMONYMS'
    ) + @(
        $existingDemonyms |
            ForEach-Object { Convert-DemonymCase $_ } |
            Sort-Object -Unique
    )
}
[IO.File]::WriteAllLines(
    [IO.Path]::GetFullPath($OutputFile),
    [string[]](
        $header +
        $orderedGlobal +
        $malteseHeader +
        $orderedMalteseGlobal +
        $malteseLocalHeader +
        $taggedMaltese +
        $demonymSection
    ),
    $utf8
)

Write-Output (
    "Wrote $($orderedGlobal.Count) global and " +
    "$($orderedMaltese.Count) Maltese place names, plus " +
    "$($existingDemonyms.Count) demonyms, to $OutputFile"
)
