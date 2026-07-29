$ErrorActionPreference = "Continue"
Set-Location "C:\Users\quanp\Downloads\ISIC 2017\AViT"
$py = "C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$driverLog = "$logDir\driver_swinunet_bceLogits_dice.log"

$runs = @(
  @{exp="isic2017_swinunet_bceLogits_dice_k0"; k="No"},
  @{exp="isic2017_swinunet_bceLogits_dice_k1"; k=1},
  @{exp="isic2017_swinunet_bceLogits_dice_k2"; k=2},
  @{exp="isic2017_swinunet_bceLogits_dice_k3"; k=3},
  @{exp="isic2017_swinunet_bceLogits_dice_k4"; k=4}
)

"=== Driver started at $(Get-Date) ===" | Out-File -Append $driverLog

foreach ($r in $runs) {
  $logOut = "$logDir\$($r.exp).log"
  $logErr = "$logDir\$($r.exp)_err.log"
  "=== Starting $($r.exp) (model=SwinUnet k_fold=$($r.k)) at $(Get-Date) ===" | Out-File -Append $driverLog
  & $py -u multi_train_adapt.py --exp_name $r.exp --config_yml Configs/multi_train_local.yml --model SwinUnet --batch_size 16 --dataset isic2017 --k_fold $r.k --num_epochs 30 1> $logOut 2> $logErr
  "=== Finished $($r.exp) at $(Get-Date) with exit code $LASTEXITCODE ===" | Out-File -Append $driverLog
}

"=== ALL RUNS COMPLETE at $(Get-Date) ===" | Out-File -Append $driverLog
