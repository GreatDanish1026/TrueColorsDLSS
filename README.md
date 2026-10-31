# TrueColorsDLSS 1.1.0

ReShade add-on for **Life is Strange: True Colors** (Unreal Engine 4.25, DirectX 12) that replaces
the game's TAA anti-aliasing with **NVIDIA DLSS** or **AMD FSR**, and adds **DLSS Frame
Generation** or **FSR Frame Generation**. The game has none of these. It's separate from
TrueColorsTweaks (graphics settings, ray tracing, HDR on DirectX 12), and the two can be installed
side by side.

Needs the game's **DirectX 12** launch option (Steam: Play > "Life is Strange: True Colors
(DirectX 12)") and ReShade 6.8 with add-on support.

| | Graphics card |
|---|---|
| DLSS upscaling | NVIDIA RTX |
| FSR upscaling | any DirectX 12 card: FSR 4 on Radeon RX 6000 and newer, FSR 3.1 elsewhere |
| DLSS Frame Generation | NVIDIA RTX 40 or 50 (3x, 4x and Dynamic: RTX 50), hardware-accelerated GPU scheduling |
| FSR Frame Generation | any DirectX 12 card: FSR 4 on Radeon RX 9000 under Windows 11, FSR 3.1 elsewhere |

Upscaler and frame generation can be mixed (for example DLSS upscaling with FSR Frame Generation on
an RTX 30 card).

## What it does

- **Upscaler** (F9 to switch on and off): Auto (DLSS on NVIDIA RTX, FSR elsewhere), DLSS or FSR in
  place of the game's TAA. DLSS Super Resolution 310.9.16; FSR through AMD's FSR SDK 2.4.0, which
  picks FSR 4 or FSR 3.1 for the card. Motion vectors are computed by the add-on from the depth
  buffer and the camera, plus the engine's own motion of moving objects.
- **Render resolution**: Native/DLAA, Quality, Balanced, Performance or Ultra Performance. The game
  renders at 100% by itself, so "Game's" is Native. The lower modes set the engine's screen
  percentage and upsampling TAA while the upscaler is on and give the game's values back when it's
  off.
- **DLSS model**: K (transformer, default), J, L, M or NVIDIA's default. **FSR sharpness**: 0-1.
- **Frame generation**: Off, DLSS 2x / 3x / 4x / Dynamic (NVIDIA Streamline 2.14.16, with Reflex),
  or FSR 2x (AMD's frame generation swapchain; no Reflex, so a little more latency). Runs in game,
  not in menus. Switching it on, or between DLSS and FSR, takes a game restart; DLSS's multipliers
  switch right away. The game draws its HUD and subtitles straight onto the screen, so frame
  generation has no HUD-free picture to work from: **subtitles can shimmer with frame generation**
  (DLSS or FSR). Upscaling alone doesn't affect them.

Camera cuts keep the game's own TAA for that frame (the engine has no motion data then).

The overlay (ReShade's Add-ons tab) shows a summary of the upscaler and frame generation at the top,
with **Settings**, **Status** and **Advanced** (testing tools including FSR's, frame capture) tabs.
Settings are saved in `ReShade.ini` under `[TrueColorsDLSS]`.

## Install

1. Install ReShade 6.8 or later with full add-on support for `Siren-Win64-Shipping.exe`, in
   `LifeIsStrange3\Siren\Binaries\Win64`.
2. Copy everything from the release zip into that folder: `TrueColorsDLSS.addon64`,
   `nvngx_dlss.dll`, the three `amd_fidelityfx_*_dx12.dll` files and the `streamline` folder (the
   licence files can go along).
3. Start the game with the DirectX 12 launch option, open ReShade (Home key), Add-ons tab,
   TrueColorsDLSS: switch the upscaler on (or press F9), pick DLSS or FSR and a render resolution
   and, if you like, frame generation (takes effect after a restart).
4. A DLSS model override for the game in the NVIDIA App takes precedence over DLSS model.

The zip includes NVIDIA's DLSS 310.9.16 and Streamline 2.14.16 runtime files, signed by NVIDIA
(the add-on checks Streamline's signature before loading it), and AMD's FSR SDK 2.4.0 runtime
files, signed by AMD. Their licences are in `licenses`, and attribution is in
`THIRD_PARTY_NOTICES.txt`.

Tested with Steam build 8050014 at 3840x2160 (HDR via TrueColorsTweaks) on an NVIDIA RTX card:
DLSS and FSR 3.1 upscaling, DLSS and FSR Frame Generation. To uninstall, delete those files. Logs:
`TrueColorsDLSS.log` next to the game's exe, Streamline's in `streamline\sl.log`.

## Build

`.\build.ps1` (Visual Studio 2022 C++ tools, Windows SDK for `fxc`). The runtimes aren't in the
repository: unzip `bin\x64` of NVIDIA's Streamline 2.14.16 release into
`third_party\streamline\bin\x64`, and copy `amd_fidelityfx_loader_dx12.dll`,
`amd_fidelityfx_upscaler_dx12.dll` and `amd_fidelityfx_framegeneration_dx12.dll` from
`Kits\FidelityFX\signedbin` of AMD's FSR SDK 2.4.0 into `third_party\fidelityfx\bin`.
`.\build.ps1 -Install` copies the add-on and the runtimes into the game folder (game closed);
`.\build.ps1 -Package` makes the release zip in `bin\release`.

## How it works

Built from DoubleExposureDLSS (UE 5.2, TSR). `docs/FINDINGS.md` has the UE 4.25 details: the
single TAA pass and its bindings, the TAAU permutation, the View uniform buffer offsets, why the
frame capture only reads declared registers, and how the render resolution is set on the game
thread. FSR and the frame generation code are shared with
[DoubleExposureDLSS](https://github.com/GreatDanish1026/DoubleExposureDLSS) (its `docs/FSR.md`
covers FSR) and [ReunionDLSS](https://github.com/GreatDanish1026/ReunionDLSS). FSR Frame Generation
tells HDR from SDR by the swapchain's colour space. The F10 frame capture (Advanced tab) writes two
frames of GPU commands to `TrueColorsDLSS-frame-N.txt`, for finding the TAA pass again after a
game update.
