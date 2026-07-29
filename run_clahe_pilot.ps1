$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$driverLog = "$logDir\driver_clahe_pilot.log"

# Keep the machine awake for the duration of this unattended run.
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
"=== Sleep prevention set (standby-timeout = 0) at $(Get-Date) ===" | Out-File -Append $driverLog

# CLAHE pilot: SwinUnet k0 + AViT k0, baseline loss (no focal), on
# CLAHE-preprocessed images (Image_clahe, clipLimit=2.0, tileGridSize=8x8,
# applied to LAB-L channel). Same official train2000/test600 split used for
# the existing baseline (0.8275 / 0.7594) so results are comparable.
$runs = @(
  @{exp="isic2017_swinunet_clahe_k0"; model="SwinUnet"; k=0},
  @{exp="isic2017_avit_clahe_k0"; model="SwinSeg_CNNprompt_adapt"; k=0}
)

"=== Driver started at $(Get-Date) ===" | Out-File -Append $driverLog

foreach ($r in $runs) {
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
    1> $logOut 2> $logErr
  "=== Finished $($r.exp) at $(Get-Date) with exit code $LASTEXITCODE ===" | Out-File -Append $driverLog
}

"=== ALL RUNS COMPLETE at $(Get-Date) ===" | Out-File -Append $driverLog
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 30
"=== Sleep timeout restored to 30 min ===" | Out-File -Append $driverLog
