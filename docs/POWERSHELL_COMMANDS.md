# PowerShell Commands Reference Guide

## File Operations

### Removing Files and Directories
```powershell
# Remove a single file
Remove-Item -Path "path/to/file" -Force

# Remove multiple files
Remove-Item -Path "file1", "file2" -Force

# Remove directory and contents
Remove-Item -Path "path/to/directory" -Recurse -Force

# Remove with error suppression
Remove-Item -Path "path/to/file" -ErrorAction SilentlyContinue
```

### Moving Files and Directories
```powershell
# Move a single file
Move-Item -Path "source/file" -Destination "dest/file" -Force

# Move multiple files
Move-Item -Path "source/*" -Destination "dest/" -Force

# Move with error handling
Move-Item -Path "source" -Destination "dest" -Force -ErrorAction Stop
```

### Creating Directories
```powershell
# Create a single directory
New-Item -ItemType Directory -Path "path/to/dir" -Force

# Create nested directories
New-Item -ItemType Directory -Path "path/to/nested/dir" -Force
```

### Copying Files and Directories
```powershell
# Copy a single file
Copy-Item -Path "source/file" -Destination "dest/file" -Force

# Copy directory and contents
Copy-Item -Path "source" -Destination "dest" -Recurse -Force
```

## Path Handling

### Best Practices
- Use double backslashes or single forward slashes for paths
  ```powershell
  "C:\\Users\\username\\file"  # Double backslash
  "C:/Users/username/file"     # Forward slash
  ```
- Always wrap paths containing spaces in quotes
  ```powershell
  "C:\Program Files\file name with spaces.txt"
  ```
- Use Join-Path for path concatenation
  ```powershell
  Join-Path -Path "C:\Users" -ChildPath "username"
  ```

## Error Handling

### Error Action Parameters
```powershell
# Continue on error (default)
-ErrorAction Continue

# Suppress errors
-ErrorAction SilentlyContinue

# Stop on error
-ErrorAction Stop

# Inquire on error
-ErrorAction Inquire
```

### Try-Catch Blocks
```powershell
try {
    Remove-Item -Path "file" -ErrorAction Stop
} catch {
    Write-Host "Error: $_"
}
```

## Common Patterns

### Check if File/Directory Exists
```powershell
# Check file
Test-Path -Path "path/to/file" -PathType Leaf

# Check directory
Test-Path -Path "path/to/directory" -PathType Container
```

### Get Directory Contents
```powershell
# List files
Get-ChildItem -Path "directory" -File

# List directories
Get-ChildItem -Path "directory" -Directory

# Recursive listing
Get-ChildItem -Path "directory" -Recurse
```

### Working with Multiple Items
```powershell
# Process multiple files
Get-ChildItem -Path "directory" -Filter "*.txt" | ForEach-Object {
    # Process each file
    $_ | Remove-Item -Force
}
```

## Tips and Tricks

1. Use `-WhatIf` parameter to preview operations:
   ```powershell
   Remove-Item -Path "file" -WhatIf
   ```

2. Use `-Verbose` for detailed operation information:
   ```powershell
   Move-Item -Path "source" -Destination "dest" -Verbose
   ```

3. Use wildcards carefully:
   ```powershell
   # Remove all .txt files
   Remove-Item -Path "*.txt" -Force
   ```

4. Always use absolute paths or verify current directory:
   ```powershell
   $currentDir = Get-Location
   Join-Path -Path $currentDir -ChildPath "file"
   ``` 