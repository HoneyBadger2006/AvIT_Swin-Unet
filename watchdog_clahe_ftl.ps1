$ErrorActionPreference = "Continue"
$logDir = "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs"
$resultsDir = "C:\Users\quanp\Downloads\ISIC 2017\results"
$watchdogLog = "$logDir\watchdog_clahe_ftl.log"
$driverScript = "C:\Users\quanp\Downloads\ISIC 2017\run_clahe_ftl_kfold_resumable.ps1"

$expNames = @(
  "isic2017_swinunet_clahe_ftl_k0",
  "isic2017_swinunet_clahe_ftl_k1",
  "isic2017_swinunet_clahe_ftl_k2",
  "isic2017_swinunet_clahe_ftl_k3",
  "isic2017_swinunet_clahe_ftl_k4",
  "isic2017_avit_clahe_ftl_k0",
  "isic2017_avit_clahe_ftl_k1",
  "isic2017_avit_clahe_ftl_k2",
  "isic2017_avit_clahe_ftl_k3",
  "isic2017_avit_clahe_ftl_k4"
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

# Flag GPU contention explicitly, since a game running here can both slow training
# to a crawl and eventually kill it silently (confirmed root cause of the k0 pilot's
# AViT death). Relaunching into a still-contended GPU would likely just fail again,
# so log free VRAM and any known game processes for visibility even though we relaunch
# regardless (the driver's own resumable skip-logic makes a relaunch attempt safe).
$fc26 = Get-Process -Name FC26 -ErrorAction SilentlyContinue
$roblox = Get-Process -Name RobloxPlayerBeta -ErrorAction SilentlyContinue
$freeMiB = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if ($fc26 -or $roblox -or [int]$freeMiB -lt 6000) {
  "=== $(Get-Date): WARNING - possible GPU contention at relaunch time (FC26=$([bool]$fc26) Roblox=$([bool]$roblox) freeMiB=$freeMiB) ===" | Out-File -Append $watchdogLog
}

"=== $(Get-Date): training process NOT found and sweep incomplete - relaunching driver ===" | Out-File -Append $watchdogLog
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$driverScript`"" -WindowStyle Hidden
"=== $(Get-Date): relaunch issued ===" | Out-File -Append $watchdogLog
