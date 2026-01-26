# PowerShell equivalent of 'watch nvidia-smi'
# Usage: .\watch-nvidia-smi.ps1 [interval_seconds]
# Default interval: 2 seconds

param(
    [int]$Interval = 2
)

# Clear screen and run nvidia-smi in a loop
while ($true) {
    Clear-Host
    Write-Host "=== nvidia-smi (refreshing every $Interval seconds) - Press Ctrl+C to exit ===" -ForegroundColor Cyan
    Write-Host ""
    nvidia-smi
    Start-Sleep -Seconds $Interval
}

