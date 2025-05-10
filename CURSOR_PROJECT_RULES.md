# 📌 Cursor Project Rules: WSL Disk Space Reclamation for Docker

## Rule: Enable Automatic Disk Space Cleanup for WSL2 (Sparse VHD)

**Purpose:**
Ensure that disk space used by Docker images/containers in WSL2 is automatically reclaimed, preventing storage bloat during scientific workflows.

**References:**
- [Microsoft Dev Blog: Automatic disk space clean up (Set sparse VHD)](https://devblogs.microsoft.com/commandline/windows-subsystem-for-linux-september-2023-update/#automatic-disk-space-clean-up-set-sparse-vhd)

**Instructions:**

### 1. Enable Sparse VHD for New and Existing WSL2 Distros

#### a. Update `.wslconfig` (Windows host)
1. Open (or create) `C:\Users\<yourusername>\.wslconfig`.
2. Add or update the following section:
   ```ini
   [experimental]
   sparseVhd=true
   ```
3. Save the file.

#### b. Restart WSL
- In PowerShell, run:
  ```powershell
  wsl --shutdown
  ```

#### c. Set Existing Distros to Sparse
- List your distros:
  ```powershell
  wsl --list --quiet
  ```
- For each user-facing distro (e.g., `Ubuntu`), run:
  ```powershell
  wsl --manage <distro> --set-sparse true
  ```
  (Do not run this for `docker-desktop` or blank lines.)

### 2. Automation
- You may use the provided `enable_wsl_sparse_vhd.ps1` script in the project root to automate all steps above. This script:
  - Backs up your `.wslconfig`.
  - Ensures `[experimental]` section with `sparseVhd=true`.
  - Restarts WSL.
  - Sets all valid user distros to sparse VHD.

**Reasoning:**
This ensures that disk space is automatically reclaimed as you delete Docker images/containers, keeping your WSL2 environment efficient and preventing storage issues during scientific computing workflows. 