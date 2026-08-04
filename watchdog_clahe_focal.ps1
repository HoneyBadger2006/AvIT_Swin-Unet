$ErrorActionPreference = "Continue"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$resultsDir = "C:\Users\quanp\Downloads\ISIC 2017\results"
$watchdogLog = "$logDir\watchdog_clahe_focal.log"
$driverScript = "C:\Users\quanp\Downloads\ISIC 2017\run_clahe_focal_kfold_resumable.ps1"

$expNames = @(
  "isic2017_swinunet_clahe_focal_k0",
  "isic2017_swinunet_clahe_focal_k1",
  "isic2017_swinunet_clahe_focal_k2",
  "isic2017_swinunet_clahe_focal_k3",
  "isic2017_swinunet_clahe_focal_k4",
  "isic2017_avit_clahe_focal_k0",
  "isic2017_avit_clahe_focal_k1",
  "isic2017_avit_clahe_focal_k2",
  "isic2017_avit_clahe_focal_k3",
  "isic2017_avit_clahe_focal_k4"
)

# Re-assert power settings every tick in case something external resets them.
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match "multi_train_adapt\.py" }

if ($running) {
  "=== $(Get-Date): training process alive (PID $($running[0].ProcessId)), nothing to do ===" | Out-File -Append $watchdogLog
  exit 0
}

$allDone = $true
foreach ($exp in $expNames) {
  $done = Get-ChildItem -Path $resultsDir -Directory -Filter "$exp*" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "test_results.csv") }
  if (-not $done) { $allDone = $false; break }
}

if ($allDone) {
  "=== $(Get-Date): all 10 runs complete, watchdog has nothing left to do ===" | Out-File -Append $watchdogLog
  exit 0
}

"=== $(Get-Date): training process NOT found and sweep incomplete - relaunching driver ===" | Out-File -Append $watchdogLog
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$driverScript`"" -WindowStyle Hidden
"=== $(Get-Date): relaunch issued ===" | Out-File -Append $watchdogLog
