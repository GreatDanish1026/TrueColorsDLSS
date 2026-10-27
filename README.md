# DoubleExposureDLSS 1.0.2

ReShade add-on for **Life is Strange: Double Exposure** (Unreal Engine 5.2, DirectX 12) that
replaces the game's TSR upscaler with **NVIDIA DLSS** and adds **DLSS Frame Generation**. The game
has neither (its menu's FSR switch does nothing). It's separate from
[DoubleExposureTweaks](https://github.com/GreatDanish1026/DoubleExposureTweaks) (graphics
settings), and the two can be installed side by side.

Needs an NVIDIA RTX GPU; frame generation needs an RTX 40 or 50 series card and
hardware-accelerated GPU scheduling (3x, 4x and Dynamic need an RTX 50 series card). ReShade 6.8
with add-on support.

## What it does

- **DLSS** (F9 to switch on and off): DLSS Super Resolution 310.9.16 upscales in place of TSR.
  Motion vectors are computed by the add-on from the depth buffer and the camera, plus the
  engine's own motion of moving objects (characters, foliage in the wind, cloth).
  The game draws its translucent effects (particles, light beams, light shafts, outlines) on a
  separate layer that its own upscaler blends in; the add-on blends that layer in before DLSS, and
  "Translucent effects" (Advanced) keeps DLSS from smearing them.
- **Render resolution**: DLAA, Quality, Balanced, Performance or Ultra Performance set the render
  resolution while DLSS is on; "Game's Resolution Scale" leaves it to the game's menu. Keep the
  game's Secondary Resolution Scale at 100%.
- **Model**: K (transformer, default), J, L, M or NVIDIA's default.
- **Frame generation**: Off, 2x, 3x, 4x or Dynamic, through NVIDIA Streamline 2.14.16 with Reflex.
  It runs in game, not in menus (the menus don't use the game's upscaler). The HUD and subtitles
  are kept from warping with the game's own HUD layer (HDR output). Switching it on from Off takes
  a game restart; the multipliers switch right away.

The overlay (ReShade's Add-ons tab) shows a one-line summary of DLSS and frame generation at the
top, with **Settings**, **Status** and **Advanced** (testing tools, frame capture) tabs. Settings
are saved in `ReShade.ini` under `[DoubleExposureDLSS]`.

## Install

1. Install ReShade 6.8 or later with full add-on support for `Chronos-Win64-Shipping.exe`, in
   `LifeIsStrangeDoubleExposure\Chronos\Binaries\Win64`.
2. Copy everything from the release zip into that `Win64` folder: `DoubleExposureDLSS.addon64`,
   `nvngx_dlss.dll` and the `streamline` folder (the licence files can go along).
3. Start the game, open ReShade (Home key), Add-ons tab, DoubleExposureDLSS: switch DLSS on (or
   press F9), pick a render resolution and, if you like, frame generation (takes effect after a
   restart).
4. In the NVIDIA App, a DLSS model override for the game takes precedence over Model.

The zip includes NVIDIA's DLSS 310.9.16 and Streamline 2.14.16 runtime files, all signed by
NVIDIA (the add-on checks Streamline's signature before loading it). Their licences are in
`licenses`, and attribution is in `THIRD_PARTY_NOTICES.txt`.

Tested with Steam build 16505703 at 3840x2160 with HDR on (play at your screen's resolution: a higher one left in GameUserSettings.ini makes the game render an extra view that frame generation can't keep up with). With HDR off, frame generation may run
without the HUD protection (the game then draws its HUD straight onto the screen, as Reunion does).

To uninstall, delete those files. Logs: `DoubleExposureDLSS.log` next to the game's exe,
Streamline's in `streamline\sl.log`.

## Build

`.\build.ps1` (Visual Studio 2022 C++ tools, Windows SDK for `fxc`). The NVIDIA runtime isn't in
the repository: unzip `bin\x64` of NVIDIA's Streamline 2.14.16 release into
`third_party\streamline\bin\x64` first. `.\build.ps1 -Install` copies the add-on and the runtime
into the game folder (game closed); `.\build.ps1 -Package` makes the release zip in
`bin\release`.

## How it works

Built from ReunionDLSS (Life is Strange: Reunion, UE 5.6). `docs/FINDINGS.md` has what differs in
UE 5.2: TSR's nine passes and their shader ids, the chain linked at DecimateHistory, the
square-root encoding of object velocity, the View uniform buffer offsets, and frame generation's
inputs handed over when the game submits each frame. The F10 frame capture (Advanced tab) writes
two frames of GPU commands to `DoubleExposureDLSS-frame-N.txt`, for finding TSR's passes again
after a game update.
