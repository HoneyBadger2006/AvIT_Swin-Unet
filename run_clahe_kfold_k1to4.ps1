$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$driverLog = "$logDir\driver_clahe_kfold_k1to4.log"

# Keep the machine awake for the duration of this unattended run.
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
"=== Sleep prevention set (standby-timeout = 0) at $(Get-Date) ===" | Out-File -Append $driverLog

# CLAHE full 5-fold completion: SwinUnet k1-k4 + AViT k1-k4, baseline loss
# (no focal), on CLAHE-preprocessed images (Image_clahe, clipLimit=2.0,
# tileGridSize=8x8, applied to LAB-L channel). Same official train2000/test600
# split as k0 so results are comparable and poolable into a 5-fold mean+-std.
$runs = @(
  @{exp="isic2017_swinunet_clahe_k1"; model="SwinUnet"; k=1},
  @{exp="isic2017_swinunet_clahe_k2"; model="SwinUnet"; k=2},
  @{exp="isic2017_swinunet_clahe_k3"; model="SwinUnet"; k=3},
  @{exp="isic2017_swinunet_clahe_k4"; model="SwinUnet"; k=4},
  @{exp="isic2017_avit_clahe_k1"; model="SwinSeg_CNNprompt_adapt"; k=1},
  @{exp="isic2017_avit_clahe_k2"; model="SwinSeg_CNNprompt_adapt"; k=2},
  @{exp="isic2017_avit_clahe_k3"; model="SwinSeg_CNNprompt_adapt"; k=3},
  @{exp="isic2017_avit_clahe_k4"; model="SwinSeg_CNNprompt_adapt"; k=4}
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
