# TrueColorsDLSS 0.3.0

ReShade add-on for **Life is Strange: True Colors** (Unreal Engine 4.25, DirectX 12) that replaces
the game's TAA anti-aliasing with **NVIDIA DLSS** and adds **DLSS Frame Generation**. The game has
neither. It's separate from TrueColorsTweaks (graphics settings, ray tracing, HDR on DirectX 12),
and the two can be installed side by side.

Needs the game's **DirectX 12** launch option (Steam: Play > "Life is Strange: True Colors
(DirectX 12)"), an NVIDIA RTX GPU, and ReShade 6.8 with add-on support. Frame generation needs an
RTX 40 or 50 series card and hardware-accelerated GPU scheduling (3x, 4x and Dynamic need an RTX
50 series card).

## What it does

- **DLSS** (F9 to switch on and off): DLSS Super Resolution 310.9.16 in place of the game's TAA.
  Motion vectors are computed by the add-on from the depth buffer and the camera, plus the
  engine's own motion of moving objects.
- **Render resolution**: DLAA, Quality, Balanced, Performance or Ultra Performance. The game
  renders at 100% by itself, so "Game's" is DLAA. The lower modes set the engine's screen
  percentage and upsampling TAA while DLSS is on and give the game's values back when it's off.
- **Model**: K (transformer, default), J, L, M or NVIDIA's default.
- **Frame generation**: Off, 2x, 3x, 4x or Dynamic, through NVIDIA Streamline 2.14.16 with Reflex.
  Runs in game, not in menus. Switching it on from Off takes a game restart; the multipliers
  switch right away. The game draws its HUD straight onto the screen, so frame generation has no
  HUD-free picture to work from.

Camera cuts keep the game's own TAA for that frame (the engine has no motion data then).

The overlay (ReShade's Add-ons tab) shows a summary of DLSS and frame generation at the top, with
**Settings**, **Status** and **Advanced** (testing tools, frame capture) tabs. Settings are saved in
`ReShade.ini` under `[TrueColorsDLSS]`.

## Install

1. Install ReShade 6.8 or later with full add-on support for `Siren-Win64-Shipping.exe`, in
   `LifeIsStrange3\Siren\Binaries\Win64`.
2. Copy `TrueColorsDLSS.addon64`, `nvngx_dlss.dll` and the `streamline` folder into that folder.
3. Start the game with the DirectX 12 launch option, open ReShade (Home key), Add-ons tab,
   TrueColorsDLSS: switch DLSS on (or press F9), pick a render resolution and, if you like, frame
   generation (takes effect after a restart).
4. A DLSS model override for the game in the NVIDIA App takes precedence over Model.

Tested with Steam build 8050014 at 3840x2160 (HDR via TrueColorsTweaks). To uninstall, delete
those files. Logs: `TrueColorsDLSS.log` next to the game's exe, Streamline's in
`streamline\sl.log`.

## Build

`.\build.ps1` (Visual Studio 2022 C++ tools, Windows SDK for `fxc`). The NVIDIA runtime isn't in
the repository: unzip `bin\x64` of NVIDIA's Streamline 2.14.16 release into
`third_party\streamline\bin\x64` first. `.\build.ps1 -Install` copies the add-on and the runtime
into the game folder (game closed).

## How it works

Built from DoubleExposureDLSS 1.0.2 (UE 5.2, TSR). `docs/FINDINGS.md` has the UE 4.25 details: the
single TAA pass and its bindings, the TAAU permutation, the View uniform buffer offsets, why the
frame capture only reads declared registers, and how the render resolution is set on the game
thread. The F10 frame capture (Advanced tab) writes two frames of GPU commands to
`TrueColorsDLSS-frame-N.txt`, for finding the TAA pass again after a game update.
