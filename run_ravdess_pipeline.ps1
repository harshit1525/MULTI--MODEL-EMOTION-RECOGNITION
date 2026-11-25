$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path "results" | Out-Null
"=== INIT $(Get-Date -Format o) ===" | Set-Content -LiteralPath "results\prepare.log" -Encoding utf8

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$HasGpu = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
$FreezeUntil = if ($HasGpu -eq "True") { "none" } else { "layer3" }

function Invoke-Step {
    param(
        [string]$LogPath,
        [string[]]$PyArgs
    )
    $Header = "=== START $(Get-Date -Format o) | GPU=$HasGpu | freeze=$FreezeUntil | $Python $($PyArgs -join ' ') ==="
    Add-Content -LiteralPath $LogPath -Value $Header -Encoding utf8
    & $Python -u @PyArgs 2>&1 | ForEach-Object {
        Add-Content -LiteralPath $LogPath -Value $_.ToString() -Encoding utf8
    }
    $Code = $LASTEXITCODE
    Add-Content -LiteralPath $LogPath -Value "=== END $(Get-Date -Format o) | exit=$Code ===" -Encoding utf8
    if ($Code -ne 0) {
        exit $Code
    }
}

Invoke-Step -LogPath "results\prepare.log" -PyArgs @(
    "prepare_ravdess.py",
    "--input-dir", "Video speech dataset",
    "--out-dir", "ravdess_crops"
)

Set-Content -LiteralPath "results\dryrun.log" -Value "" -Encoding utf8
Invoke-Step -LogPath "results\dryrun.log" -PyArgs @(
    "train_mixed.py",
    "--init", "external_models\resnet18_raf\best_resnet18.pt",
    "--raf-dir", "DATASET",
    "--extra-dir", "ravdess_crops",
    "--freeze-until", $FreezeUntil,
    "--dry-run"
)

Set-Content -LiteralPath "results\train_ravdess.log" -Value "" -Encoding utf8
Invoke-Step -LogPath "results\train_ravdess.log" -PyArgs @(
    "train_mixed.py",
    "--init", "external_models\resnet18_raf\best_resnet18.pt",
    "--raf-dir", "DATASET",
    "--extra-dir", "ravdess_crops",
    "--epochs", "8",
    "--out", "external_models\ravdess",
    "--freeze-until", $FreezeUntil
)
