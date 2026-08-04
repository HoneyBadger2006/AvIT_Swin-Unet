$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$resultsDir = "C:\Users\quanp\Downloads\ISIC 2017\results"
$driverLog = "$logDir\driver_clahe_focal_kfold.log"

# Keep the machine awake for the duration of this unattended run.
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
"=== Sleep prevention set (standby-timeout = 0) at $(Get-Date) ===" | Out-File -Append $driverLog

# CLAHE + Focal Loss stacking test: SwinUnet k0-k4 + AViT k0-k4, on
# CLAHE-preprocessed images (Image_clahe) with --use_focal_loss --focal_lambda 1.0
# on top. Same official train2000/test600 split as the baseline / CLAHE-alone /
# Focal-alone runs so all four conditions are directly comparable per fold.
#
# RESUMABLE: before launching each run, checks results/<exp_name>_* for a
# completed test_results.csv (only written after successful test evaluation).
# If found, the run is skipped. This lets the driver be re-run after an
# interruption without repeating already-finished folds.
$runs = @(
  @{exp="isic2017_swinunet_clahe_focal_k0"; model="SwinUnet"; k=0},
  @{exp="isic2017_swinunet_clahe_focal_k1"; model="SwinUnet"; k=1},
  @{exp="isic2017_swinunet_clahe_focal_k2"; model="SwinUnet"; k=2},
  @{exp="isic2017_swinunet_clahe_focal_k3"; model="SwinUnet"; k=3},
  @{exp="isic2017_swinunet_clahe_focal_k4"; model="SwinUnet"; k=4},
  @{exp="isic2017_avit_clahe_focal_k0"; model="SwinSeg_CNNprompt_adapt"; k=0},
  @{exp="isic2017_avit_clahe_focal_k1"; model="SwinSeg_CNNprompt_adapt"; k=1},
  @{exp="isic2017_avit_clahe_focal_k2"; model="SwinSeg_CNNprompt_adapt"; k=2},
  @{exp="isic2017_avit_clahe_focal_k3"; model="SwinSeg_CNNprompt_adapt"; k=3},
  @{exp="isic2017_avit_clahe_focal_k4"; model="SwinSeg_CNNprompt_adapt"; k=4}
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
    --use_focal_loss `
    --focal_lambda 1.0 `
    1> $logOut 2> $logErr
  "=== Finished $($r.exp) at $(Get-Date) with exit code $LASTEXITCODE ===" | Out-File -Append $driverLog
}

"=== ALL RUNS COMPLETE at $(Get-Date) ===" | Out-File -Append $driverLog
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 30
"=== Sleep timeout restored to 30 min ===" | Out-File -Append $driverLog
