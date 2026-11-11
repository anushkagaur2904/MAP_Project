$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force "benchmarks\hardware" | Out-Null
$out = "benchmarks\hardware\hardware_info.txt"
"===== DATE =====" | Out-File $out
Get-Date | Out-File $out -Append
"===== OS =====" | Out-File $out -Append
Get-CimInstance Win32_OperatingSystem | Format-List * | Out-File $out -Append
"===== CPU =====" | Out-File $out -Append
Get-CimInstance Win32_Processor | Format-List Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | Out-File $out -Append
"===== MEMORY =====" | Out-File $out -Append
Get-CimInstance Win32_PhysicalMemory | Format-Table Manufacturer,Capacity,Speed -Auto | Out-File $out -Append
"===== NUMA =====" | Out-File $out -Append
if (Get-Command Get-NumaNode -ErrorAction SilentlyContinue) { Get-NumaNode | Format-List * | Out-File $out -Append } else { "Get-NumaNode cmdlet unavailable; use Coreinfo or Sysinternals for NUMA topology." | Out-File $out -Append }
"===== GCC =====" | Out-File $out -Append
gcc --version 2>&1 | Out-File $out -Append
"===== OPENMP MACRO =====" | Out-File $out -Append
cmd /c "echo. | gcc -fopenmp -dM -E -" 2>&1 | Select-String OPENMP | Out-File $out -Append
"Saved $out"
