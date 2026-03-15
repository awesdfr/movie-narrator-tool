param(
    [string]$ProjectId = "",
    [switch]$SkipFrontendBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RegexValue {
    param(
        [string]$Text,
        [string]$Pattern
    )
    $match = [regex]::Match($Text, $Pattern)
    if ($match.Success -and $match.Groups.Count -gt 1) {
        return $match.Groups[1].Value
    }
    return $null
}

function Get-RegexCount {
    param(
        [string]$Text,
        [string]$Pattern
    )
    return [regex]::Matches($Text, $Pattern).Count
}

function Get-ProjectMetrics {
    param(
        [string]$FilePath
    )

    $raw = Get-Content -LiteralPath $FilePath -Raw -Encoding UTF8
    $isJsonValid = $true
    try {
        $null = $raw | ConvertFrom-Json
    } catch {
        $isJsonValid = $false
    }

    $id = Get-RegexValue -Text $raw -Pattern '"id"\s*:\s*"([^"]+)"'
    if (-not $id) {
        $id = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
    }
    $name = Get-RegexValue -Text $raw -Pattern '"name"\s*:\s*"([^"]+)"'
    $status = Get-RegexValue -Text $raw -Pattern '"status"\s*:\s*"([^"]+)"'
    $progressStage = Get-RegexValue -Text $raw -Pattern '"stage"\s*:\s*"([^"]+)"'
    $progressMessage = Get-RegexValue -Text $raw -Pattern '"message"\s*:\s*"([^"]*)"'

    $segmentTotal = Get-RegexCount -Text $raw -Pattern '"segment_type"\s*:\s*"'
    $reviewTotal = Get-RegexCount -Text $raw -Pattern '"review_required"\s*:\s*true'
    $skippedTotal = Get-RegexCount -Text $raw -Pattern '"skip_matching"\s*:\s*true'
    $polishedEmpty = Get-RegexCount -Text $raw -Pattern '"polished_text"\s*:\s*""'
    $polishedAny = Get-RegexCount -Text $raw -Pattern '"polished_text"\s*:'

    $benchmarkRaw = Get-RegexValue -Text $raw -Pattern '"benchmark_accuracy"\s*:\s*(null|-?\d+(?:\.\d+)?)'
    $benchmarkPercent = $null
    if ($benchmarkRaw -and $benchmarkRaw -ne "null") {
        $benchmarkPercent = [double]$benchmarkRaw * 100.0
    }

    $matchedTotal = $null
    if ($progressMessage) {
        $matchedMatch = [regex]::Match($progressMessage, '(\d+)\s*/\s*(\d+)\s+matched')
        if ($matchedMatch.Success) {
            $matchedTotal = [int]$matchedMatch.Groups[1].Value
            if ($segmentTotal -eq 0) {
                $segmentTotal = [int]$matchedMatch.Groups[2].Value
            }
        }
    }
    if ($null -eq $matchedTotal -and $segmentTotal -gt 0) {
        $matchedTotal = $segmentTotal - $skippedTotal
    }

    $matchRate = $null
    if ($segmentTotal -gt 0 -and $null -ne $matchedTotal) {
        $matchRate = [math]::Round(($matchedTotal / $segmentTotal) * 100.0, 2)
    }

    [pscustomobject]@{
        FilePath = $FilePath
        Id = $id
        Name = $name
        Status = $status
        ProgressStage = $progressStage
        ProgressMessage = $progressMessage
        SegmentTotal = $segmentTotal
        MatchedTotal = $matchedTotal
        MatchRate = $matchRate
        ReviewTotal = $reviewTotal
        SkippedTotal = $skippedTotal
        BenchmarkPercent = $benchmarkPercent
        PolishedEmpty = $polishedEmpty
        PolishedAny = $polishedAny
        IsJsonValid = $isJsonValid
    }
}

function Get-FrontendBuildResult {
    param(
        [string]$RepoDir,
        [bool]$SkipBuild
    )

    if ($SkipBuild) {
        return [pscustomobject]@{
            Ok = $true
            Skipped = $true
            Tail = "frontend build skipped"
        }
    }

    $frontendDir = Join-Path $RepoDir "frontend"
    $logFile = [System.IO.Path]::GetTempFileName()
    & cmd /c "cd /d `"$frontendDir`" && npm run build > `"$logFile`" 2>&1"
    $exitCode = $LASTEXITCODE
    $output = @()
    if (Test-Path -LiteralPath $logFile) {
        $output = Get-Content -LiteralPath $logFile -Encoding UTF8
        Remove-Item -LiteralPath $logFile -Force -ErrorAction SilentlyContinue
    }
    $tail = ($output | Select-Object -Last 60) -join [Environment]::NewLine
    if (-not $tail) {
        $tail = "(no output)"
    }

    return [pscustomobject]@{
        Ok = ($exitCode -eq 0)
        Skipped = $false
        Tail = $tail
    }
}

$repoDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$projectsDir = Join-Path $HOME "MovieNarratorProjects"
$outputDir = Join-Path $repoDir "temp\claude_feedback"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

if (-not (Test-Path -LiteralPath $projectsDir)) {
    throw "Project directory not found: $projectsDir"
}

$projectFiles = Get-ChildItem -LiteralPath $projectsDir -File -Filter "proj_*.json" | Sort-Object LastWriteTime -Descending
if ($projectFiles.Count -eq 0) {
    throw "No project files found in: $projectsDir"
}

$currentFile = $null
if ($ProjectId) {
    $expectedName = "$ProjectId.json"
    $currentFile = $projectFiles | Where-Object { $_.Name -eq $expectedName } | Select-Object -First 1
    if (-not $currentFile) {
        throw "Requested project not found: $ProjectId"
    }
} else {
    $currentFile = $projectFiles[0]
}

$previousFile = $projectFiles | Where-Object { $_.FullName -ne $currentFile.FullName } | Select-Object -First 1
$current = Get-ProjectMetrics -FilePath $currentFile.FullName
$previous = $null
if ($previousFile) {
    $previous = Get-ProjectMetrics -FilePath $previousFile.FullName
}

$frontend = Get-FrontendBuildResult -RepoDir $repoDir -SkipBuild:$SkipFrontendBuild

$issues = @()
if (-not $current.IsJsonValid) { $issues += "Latest project JSON is malformed (cannot parse strictly)." }
if ($null -eq $current.MatchRate -or $current.MatchRate -lt 95.0) { $issues += "Match rate below target 95%." }
if ($current.ReviewTotal -gt 0) { $issues += "Segments requiring review remain: $($current.ReviewTotal)." }
if ($null -eq $current.BenchmarkPercent) { $issues += "benchmark_accuracy is missing/null." }
elseif ($current.BenchmarkPercent -lt 95.0) { $issues += "benchmark_accuracy below 95% ($($current.BenchmarkPercent)%)." }
if ($current.PolishedAny -gt 0 -and $current.PolishedEmpty -eq $current.PolishedAny) { $issues += "All polished_text fields are empty." }
if (-not $frontend.Ok) { $issues += "Frontend build failed." }

$regressions = @()
if ($previous) {
    if ($current.ReviewTotal -gt $previous.ReviewTotal) {
        $regressions += "review_required increased: $($previous.ReviewTotal) -> $($current.ReviewTotal)"
    }
    if ($current.MatchRate -lt $previous.MatchRate) {
        $regressions += "match rate dropped: $($previous.MatchRate)% -> $($current.MatchRate)%"
    }
    if (($null -eq $previous.BenchmarkPercent) -and ($null -ne $current.BenchmarkPercent)) {
        $regressions += "benchmark_accuracy became available: null -> $($current.BenchmarkPercent)%"
    }
}

$goalLines = @(
    "- Match quality >= 95%",
    "- No blocking bugs",
    "- Natural polished narration (no AI tone)"
) -join [Environment]::NewLine

$regressionBlock = if ($regressions.Count -gt 0) { ($regressions | ForEach-Object { "- $_" }) -join [Environment]::NewLine } else { "- No obvious regression vs previous project snapshot." }
$issueBlock = if ($issues.Count -gt 0) { ($issues | ForEach-Object { "- $_" }) -join [Environment]::NewLine } else { "- No blocking issue detected in fallback checks." }
$frontendStatus = if ($frontend.Ok) { "OK" } else { "FAIL" }
$frontendBlock = if ($frontend.Skipped) {
    "- frontend build skipped"
} else {
    "- $frontendStatus frontend build" + [Environment]::NewLine + [Environment]::NewLine + '```text' + [Environment]::NewLine + $frontend.Tail + [Environment]::NewLine + '```'
}

$benchmarkText = if ($null -eq $current.BenchmarkPercent) { "null" } else { "{0:N2}%" -f $current.BenchmarkPercent }
$matchText = if ($null -eq $current.MatchRate) { "unknown" } else { "{0:N2}%" -f $current.MatchRate }
$prevMatchText = if ($previous -and $null -ne $previous.MatchRate) { "{0:N2}%" -f $previous.MatchRate } else { "unknown" }
$prevText = if ($previous) { "$($previous.Id) (review=$($previous.ReviewTotal), match=$prevMatchText)" } else { "none" }

$report = @"
# Claude Handoff Report (Fallback)

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
Mode: PowerShell fallback (Python unavailable or failed before report generation)

## Current goals
$goalLines

## Current project snapshot
- Project: $($current.Name) ($($current.Id))
- Status: $($current.Status)
- Stage: $($current.ProgressStage)
- Segment total: $($current.SegmentTotal)
- Matched total: $($current.MatchedTotal)
- Match rate: $matchText
- Review required: $($current.ReviewTotal)
- Skipped matching: $($current.SkippedTotal)
- benchmark_accuracy: $benchmarkText
- polished_text empty/all: $($current.PolishedEmpty)/$($current.PolishedAny)
- JSON valid: $($current.IsJsonValid)
- Progress message: $($current.ProgressMessage)

## Regression check
- Current: $($current.Id)
- Previous: $prevText
$regressionBlock

## Checks
$frontendBlock

## Blocking issues
$issueBlock

## Request to Claude
Please prioritize fixes in this order:
1. Repair data integrity issues first (latest project JSON must be strictly valid).
2. Reduce review_required while preserving match correctness.
3. Produce non-null benchmark_accuracy and push it to >=95%.
4. Fill `polished_text` with natural, human-like narration (avoid AI tone).
After fixes, rerun run-claude-feedback.bat and compare against this report.
"@

$reportPath = Join-Path $outputDir "latest_report.md"
$promptPath = Join-Path $outputDir "latest_prompt.txt"
$report | Set-Content -LiteralPath $reportPath -Encoding UTF8
$report | Set-Content -LiteralPath $promptPath -Encoding UTF8

Write-Host "Report written: $reportPath"
Write-Host "Prompt written: $promptPath"
Write-Host "Current project: $($current.Id) match=$matchText review=$($current.ReviewTotal) benchmark=$benchmarkText json_valid=$($current.IsJsonValid)"
Write-Host "Frontend build: $frontendStatus"

if ($issues.Count -gt 0) {
    exit 1
}
exit 0
