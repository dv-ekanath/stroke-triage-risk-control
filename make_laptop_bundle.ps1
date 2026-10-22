# Builds a minimal copy of everything the demo needs, to carry to another machine.
# Disk-only: no GPU use, safe to run while training is going.
#
#   .\make_laptop_bundle.ps1 -Dest E:\bundle            # real copy
#   .\make_laptop_bundle.ps1 -Dest E:\bundle -DryRun    # just list + size, copy nothing
param(
    [Parameter(Mandatory)] [string] $Dest,
    [string] $Isles = "D:\ISLES-2024\train",
    [switch] $DryRun
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

# The 50 fold-0 validation patients are pinned in demo/server.py.
$text = Get-Content "$repo\demo\server.py" -Raw
$block = [regex]::Match($text, 'FOLD0_VAL_SUBJECTS\s*=\s*\[(.*?)\]', 'Singleline').Groups[1].Value
$subs = [regex]::Matches($block, 'sub-stroke\d+') | ForEach-Object Value | Sort-Object -Unique
Write-Host "$($subs.Count) subjects"

# Skip the big files the demo never reads: raw 4D CTP, follow-up DWI/ADC.
$skip = @("*_ctp*", "*dwi*", "*adc*")
$flags = @("/E", "/NFL", "/NDL", "/NJH", "/NP", "/R:1", "/W:1", "/XF") + $skip
if ($DryRun) { $flags += "/L" }

foreach ($s in $subs) {
    foreach ($part in "raw_data", "derivatives", "phenotype") {
        $src = Join-Path $Isles "$part\$s"
        if (Test-Path $src) {
            robocopy $src (Join-Path $Dest "ISLES-2024\train\$part\$s") @flags | Out-Null
        }
    }
}

# Repo: code, KG, tables, figures, paper. Checkpoints are gitignored, so copy them too.
# .venv (not portable) and the 100 GB caches are intentionally left out.
robocopy $repo (Join-Path $Dest "Stroke-code") /E /NFL /NDL /NJH /NP /XD .venv __pycache__ cache_1mm_wb64 /XF *.pyc @(if ($DryRun) { "/L" }) | Out-Null

if (-not $DryRun) {
    $gb = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("Bundle size: {0:N1} GB at {1}" -f $gb, $Dest)
}
