# 📌 Purpose – Enable WSL sparse VHD for automatic disk cleanup (per Microsoft blog)
# 🔄 Latest Changes – Initial creation
# ⚙️ Key Logic – Edits .wslconfig, sets sparseVhd=true, updates all distros, restarts WSL
# 📂 Expected File Path – ./enable_wsl_sparse_vhd.ps1
# 🧠 Reasoning – Automates disk space reclamation for Docker/WSL workflows, with backup safety

# Backup existing .wslconfig if present
$wslConfigPath = "$env:USERPROFILE\.wslconfig"
$backupPath = "$wslConfigPath.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path $wslConfigPath) {
    Copy-Item $wslConfigPath $backupPath
    Write-Host "Backed up existing .wslconfig to $backupPath"
}

# Read or create .wslconfig
$config = @()
if (Test-Path $wslConfigPath) {
    $config = Get-Content $wslConfigPath
}

# Ensure [experimental] section and sparseVhd=true
$expIndex = $config.IndexOf('[experimental]')
if ($expIndex -eq -1) {
    $config += '[experimental]'
    $config += 'sparseVhd=true'
} else {
    $found = $false
    for ($i = $expIndex + 1; $i -lt $config.Count; $i++) {
        if ($config[$i] -match '^\[') { break }
        if ($config[$i] -match '^sparseVhd=') {
            $config[$i] = 'sparseVhd=true'
            $found = $true
            break
        }
    }
    if (-not $found) {
        $config = $config[0..$expIndex] + @('sparseVhd=true') + $config[($expIndex+1)..($config.Count-1)]
    }
}

# Write updated config
Set-Content -Path $wslConfigPath -Value $config -Encoding UTF8
Write-Host ".wslconfig updated with [experimental] sparseVhd=true."

# Restart WSL
Write-Host "Shutting down WSL..."
wsl --shutdown
Start-Sleep -Seconds 2

# Set all existing distros to sparse
$distros = wsl --list --quiet
foreach ($distro in $distros) {
    Write-Host "Setting $distro to sparse VHD..."
    wsl --manage $distro --set-sparse true
}

Write-Host "WSL sparse VHD enabled. You may now start your WSL as usual." 