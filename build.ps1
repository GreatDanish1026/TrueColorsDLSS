# Builds bin\TrueColorsDLSS.addon64 with MSVC (Visual Studio 2022+ with C++ tools).
# MSVC is required: ReShade is built with it, and MinGW returns API structs differently.
# Usage: .\build.ps1 [-Install] [-Package] [-GameBin <folder of Siren-Win64-Shipping.exe>]
#   -Install  copies the add-on, nvngx_dlss.dll, AMD's FSR runtime and the Streamline runtime (in a
#             streamline folder) next to the game's exe (game closed).
#   -Package  makes bin\release\TrueColorsDLSS-<version>.zip: the add-on and the NVIDIA and AMD runtimes
#             laid out as in the game folder, README, LICENSE, THIRD_PARTY_NOTICES and the licences.
# The NVIDIA runtime comes from third_party\streamline\bin\x64: bin\x64 of NVIDIA's Streamline
# SDK release 2.14.16 (github.com/NVIDIA-RTX/Streamline/releases), which also carries
# nvngx_dlss.dll 310.9.16. AMD's FSR runtime (amd_fidelityfx_loader_dx12.dll,
# amd_fidelityfx_upscaler_dx12.dll, amd_fidelityfx_framegeneration_dx12.dll) comes from
# Kits\FidelityFX\signedbin of AMD's FSR SDK 2.4.0
# (github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases), copied to third_party\fidelityfx\bin.
# Those DLLs aren't in this repository.
param(
    [switch]$Install,
    [switch]$Package,
    [string]$GameBin = "E:\SteamLibrary\steamapps\common\LifeIsStrange3\Siren\Binaries\Win64"
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$exe = "Siren-Win64-Shipping.exe"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"  # vcvars64.bat runs vswhere by name
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ x64 tools not found." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

$obj = "$root\bin\obj"
New-Item -ItemType Directory -Force $obj | Out-Null
$addon = "$root\bin\TrueColorsDLSS.addon64"

# ReShade add-on headers in third_party\reshade match ReShade 6.8.0.
$cmd = @(
    "call `"$vcvars`" >nul",
    "fxc /nologo /T cs_5_1 /E main /O3 /Vn kMotionCs /Fh `"$obj\motion_cs.h`" `"$root\src\motion.hlsl`" >nul",
    "fxc /nologo /T cs_5_1 /E main /O3 /Vn kTintCs /Fh `"$obj\tint_cs.h`" `"$root\src\tint.hlsl`" >nul",
    "fxc /nologo /T cs_5_1 /E main /O3 /Vn kComposeCs /Fh `"$obj\compose_cs.h`" `"$root\src\compose.hlsl`" >nul",
    "fxc /nologo /T cs_5_1 /E main /O3 /Vn kUiAlphaCs /Fh `"$obj\ui_alpha_cs.h`" `"$root\src\ui_alpha.hlsl`" >nul",
    "fxc /nologo /T cs_5_1 /E main /O3 /Vn kHudComposeCs /Fh `"$obj\hud_compose_cs.h`" `"$root\src\hud_compose.hlsl`" >nul",
    ("cl /nologo /std:c++17 /O2 /EHsc /MT /W3 /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX " +
     "/I `"$obj`" /I `"$root\third_party\imgui`" /I`"$root\third_party\reshade\include`" /I `"$root\third_party\dlss\include`" /I `"$root\third_party\streamline\include`" " +
     "/I `"$root\third_party\fidelityfx\api\include`" /I `"$root\third_party\fidelityfx\upscalers\include`" /I `"$root\third_party\fidelityfx\framegeneration\include`" " +
     "/Fo`"$obj\\`" `"$root\src\main.cpp`" `"$root\third_party\reshade\utils\descriptor_tracking.cpp`" `"$root\src\upscaler.cpp`" `"$root\src\dlss.cpp`" `"$root\src\fsr.cpp`" `"$root\src\fsrfg.cpp`" `"$root\src\framegen.cpp`" user32.lib advapi32.lib ole32.lib d3d12.lib dxgi.lib `"$root\third_party\dlss\lib\nvsdk_ngx_s.lib`" " +
     "/link /DLL /OUT:`"$addon`" /PDB:`"$obj\TrueColorsDLSS.pdb`" /DEBUG")
) -join " && "
cmd /c $cmd
if ($LASTEXITCODE) { throw "build failed" }
Write-Host "Built $addon"

# NVIDIA's runtime, all signed: DLSS next to the exe, Streamline (frame generation) in Win64\streamline.
$sl = "$root\third_party\streamline\bin\x64"
$slFiles = "sl.interposer.dll", "sl.common.dll", "sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "nvngx_dlssg.dll"
foreach ($f in @("nvngx_dlss.dll") + $slFiles) {
    if (($Install -or $Package) -and -not (Test-Path "$sl\$f")) { throw "$f not found in $sl (unzip NVIDIA's Streamline 2.14.16 release there)" }
}

# AMD's FSR runtime, signed, next to the exe.
$ffx = "$root\third_party\fidelityfx\bin"
$ffxFiles = "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll", "amd_fidelityfx_framegeneration_dx12.dll"
foreach ($f in $ffxFiles) {
    if (($Install -or $Package) -and -not (Test-Path "$ffx\$f")) { throw "$f not found in $ffx (copy Kits\FidelityFX\signedbin from AMD's FSR SDK 2.4.0)" }
}

# Lays out the add-on and the NVIDIA and AMD runtimes as they go in the game's Win64 folder.
function Copy-Runtime([string]$dest) {
    Copy-Item $addon $dest -Force
    Copy-Item "$sl\nvngx_dlss.dll" $dest -Force
    foreach ($f in $ffxFiles) { Copy-Item "$ffx\$f" $dest -Force }
    New-Item -ItemType Directory -Force "$dest\streamline" | Out-Null
    foreach ($f in $slFiles) { Copy-Item "$sl\$f" "$dest\streamline" -Force }
}

if ($Install) {
    if (-not (Test-Path "$GameBin\$exe")) { throw "$exe not found in $GameBin (pass -GameBin)" }
    if (Get-Process ([IO.Path]::GetFileNameWithoutExtension($exe)) -ErrorAction SilentlyContinue) { throw "Close the game first." }
    Copy-Runtime $GameBin
    Copy-Item "$root\third_party\streamline\license.txt" "$GameBin\streamline\streamline.license.txt" -Force
    Copy-Item "$root\third_party\streamline\licenses\reflex.license.txt" "$GameBin\streamline" -Force
    Write-Host "Installed to $GameBin"
}

if ($Package) {
    $version = [regex]::Match((Get-Content "$root\src\version.h" -Raw), 'RD_VERSION "([^"]+)"').Groups[1].Value
    $name = "TrueColorsDLSS-$version"
    $stage = "$root\bin\release\$name"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force "$stage\licenses" | Out-Null
    # Everything at the top level (and the streamline folder) goes next to the game's shipping exe.
    Copy-Runtime $stage
    Copy-Item "$root\README.md", "$root\LICENSE", "$root\THIRD_PARTY_NOTICES.txt" $stage
    Copy-Item "$root\third_party\streamline\license.txt" "$stage\licenses\Streamline-LICENSE.txt"
    Copy-Item "$root\third_party\streamline\licenses\nvngx_dlss.license.txt" "$stage\licenses\NVIDIA-DLSS-LICENSE.txt"
    Copy-Item "$root\third_party\streamline\licenses\reflex.license.txt" "$stage\licenses\NVIDIA-Reflex-LICENSE.txt"
    Copy-Item "$root\third_party\fidelityfx\license.md" "$stage\licenses\AMD-FSR-LICENSE.txt"
    $zip = "$root\bin\release\$name.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path "$stage\*" -DestinationPath $zip
    Write-Host "Packaged $zip"
    Get-ChildItem $stage -Recurse -File | ForEach-Object {
        "{0,-40} {1,-14} {2,12:N0}" -f $_.FullName.Substring($stage.Length + 1), $_.VersionInfo.FileVersion, $_.Length
    }
}
