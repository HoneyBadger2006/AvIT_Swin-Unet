$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$resultsDir = "C:\Users\quanp\Downloads\ISIC 2017\results"
$driverLog = "$logDir\driver_clahe_ftl_kfold.log"

# Keep the machine awake for the duration of this unattended run.
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
"=== Sleep/display prevention set at $(Get-Date) ===" | Out-File -Append $driverLog

# Full 5-fold CLAHE + compound Focal Tversky Loss sweep: SwinUnet k0-k4 + AViT k0-k4,
# on CLAHE-preprocessed images (Image_clahe). Loss REPLACED (not added to) with
# 1*Dice + 2*FocalTversky(alpha=0.7, gamma=4/3) + 0.5*BCE via --use_focal_tversky_loss.
# k0 for both networks already completed during the pilot; RESUMABLE logic below
# skips them automatically.
$runs = @(
  @{exp="isic2017_swinunet_clahe_ftl_k0"; model="SwinUnet"; k=0},
  @{exp="isic2017_swinunet_clahe_ftl_k1"; model="SwinUnet"; k=1},
  @{exp="isic2017_swinunet_clahe_ftl_k2"; model="SwinUnet"; k=2},
  @{exp="isic2017_swinunet_clahe_ftl_k3"; model="SwinUnet"; k=3},
  @{exp="isic2017_swinunet_clahe_ftl_k4"; model="SwinUnet"; k=4},
  @{exp="isic2017_avit_clahe_ftl_k0"; model="SwinSeg_CNNprompt_adapt"; k=0},
  @{exp="isic2017_avit_clahe_ftl_k1"; model="SwinSeg_CNNprompt_adapt"; k=1},
  @{exp="isic2017_avit_clahe_ftl_k2"; model="SwinSeg_CNNprompt_adapt"; k=2},
  @{exp="isic2017_avit_clahe_ftl_k3"; model="SwinSeg_CNNprompt_adapt"; k=3},
  @{exp="isic2017_avit_clahe_ftl_k4"; model="SwinSeg_CNNprompt_adapt"; k=4}
)

"=== Driver started at $(Get-Date) ===" | Out-File -Append $driverLog

foreach ($r in $runs) {
  $existing = Get-ChildItem -Path $resultsDir -Directory -Filter "$($r.exp)_*" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "test_results.csv") }

  if ($existing) {
    "=== Skipping $($r.exp) (model=$($r.model) k_fold=$($r.k)) at $(Get-Date) - already complete: $($existing[0].FullName) ===" | Out-File -Append $driverLog
    continue
  }

  $logOut = "$logDir\$($r.exp).log"
  $logErr = "$logDir\$($r.exp)_err.log"
  "=== Starting $($r.exp) (model=$($r.model) k_fold=$($r.k)) at $(Get-Date) ===" | Out-File -Append $driverLog
  & $py -u multi_train_adapt.py `
    --exp_name $r.exp `
    --config_yml Configs/multi_train_local.yml `
    --model $r.model `
    --batch_size 16 `
    --dataset isic2017 `
    --k_fold $r.k `
    --num_epochs 30 `
    --meta_csv_name meta_isic2017_train2000.csv `
    --fixed_test_csv_name meta_isic2017_test600.csv `
    --image_subdir Image_clahe `
    --use_focal_tversky_loss `
    1> $logOut 2> $logErr
  "=== Finished $($r.exp) at $(Get-Date) with exit code $LASTEXITCODE ===" | Out-File -Append $driverLog
}

"=== ALL RUNS COMPLETE at $(Get-Date) ===" | Out-File -Append $driverLog
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 30
"=== Sleep timeout restored to 30 min ===" | Out-File -Append $driverLog
