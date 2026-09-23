# run_seeds.ps1 — Cross-seed training, sequential
# Run from your model_C directory:
# powershell -ExecutionPolicy Bypass -File run_seeds.ps1

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "train_c.py"
$seeds  = @(42, 123, 7)
$fusion = "cca"      # set to "concat" for the ablation run
$cache  = "cache_c"  # "cache_e" to train Model E
$prefix = "c"        # "e" for Model E

foreach ($seed in $seeds) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Starting training — seed $seed" -ForegroundColor Cyan
    Write-Host "  Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    & $python $script --seed $seed --fusion $fusion --cache $cache --prefix $prefix

    Write-Host ""
    Write-Host "  Seed $seed finished at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL 3 SEEDS COMPLETE" -ForegroundColor Green
Write-Host "  Finished: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green