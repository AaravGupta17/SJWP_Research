# run_seeds.ps1 — Cross-seed training, sequential
# Run from your model_C directory:
# powershell -ExecutionPolicy Bypass -File run_seeds.ps1

$python = "C:\Users\armaan\PycharmProjects\SJWP_Research\.venv\Scripts\python.exe"
$script = "C:\Users\armaan\PycharmProjects\SJWP_Research\model_C\train_c.py"
$seeds  = @(42, 123, 7)

foreach ($seed in $seeds) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Starting training — seed $seed" -ForegroundColor Cyan
    Write-Host "  Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    & $python $script --seed $seed

    Write-Host ""
    Write-Host "  Seed $seed finished at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL 3 SEEDS COMPLETE" -ForegroundColor Green
Write-Host "  Finished: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green