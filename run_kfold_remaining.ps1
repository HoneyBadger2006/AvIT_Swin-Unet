$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$driverLog = "$logDir\driver.log"

$runs = @(
  @{exp="isic2017_swinunet_k1"; model="SwinUnet"; k=1},
  @{exp="isic2017_swinunet_k2"; model="SwinUnet"; k=2},
  @{exp="isic2017_swinunet_k3"; model="SwinUnet"; k=3},
  @{exp="isic2017_swinunet_k4"; model="SwinUnet"; k=4},
  @{exp="isic2017_avit_k1"; model="SwinSeg_CNNprompt_adapt"; k=1},
  @{exp="isic2017_avit_k2"; model="SwinSeg_CNNprompt_adapt"; k=2},
  @{exp="isic2017_avit_k3"; model="SwinSeg_CNNprompt_adapt"; k=3},
  @{exp="isic2017_avit_k4"; model="SwinSeg_CNNprompt_adapt"; k=4}
)

"=== Driver started at $(Get-Date) ===" | Out-File -Append $driverLog

foreach ($r in $runs) {
  $logOut = "$logDir\$($r.exp).log"
  $logErr = "$logDir\$($r.exp)_err.log"
  "=== Starting $($r.exp) (model=$($r.model) k_fold=$($r.k)) at $(Get-Date) ===" | Out-File -Append $driverLog
  & $py -u multi_train_adapt.py --exp_name $r.exp --config_yml Configs/multi_train_local.yml --model $r.model --batch_size 16 --dataset isic2017 --k_fold $r.k --num_epochs 30 1> $logOut 2> $logErr
  "=== Finished $($r.exp) at $(Get-Date) with exit code $LASTEXITCODE ===" | Out-File -Append $driverLog
}

"=== ALL RUNS COMPLETE at $(Get-Date) ===" | Out-File -Append $driverLog
powercfg /change standby-timeout-ac 30
"=== Sleep timeout restored to 30 min ===" | Out-File -Append $driverLog
