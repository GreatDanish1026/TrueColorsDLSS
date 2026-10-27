# DoubleExposureDLSS: findings

Game facts (engine, paths, cvar layout): see DoubleExposureTweaks' `docs/FINDINGS.md`. Built from
ReunionDLSS 1.0.1 (Life is Strange: Reunion, UE 5.6); its `docs/FINDINGS.md` and `docs/FRAMEGEN.md`
explain the method (TSR hook, own motion vectors, NGX state restore, frame generation).

## Upscaler today

Only TSR (`r.AntiAliasingMethod=4`); no DLSS, NGX, Streamline or XeSS in the exe. The menu's FSR
switch names `r.FidelityFX.FSR3.Enabled`, which isn't a registered cvar (does nothing).

TSR settings (probe, 2026-10-26): `r.TSR.AsyncCompute=0` (graphics queue), `r.TSR.History.ScreenPercentage=200`,
`r.TSR.History.UpdateQuality=3`, `r.TSR.History.R11G11B10=1`, `r.TSR.RejectionAntiAliasingQuality=2`,
`r.TSR.ShadingRejection.Flickering=1`, `r.TemporalAASamples=8`. The user plays at 3840x2160, 70%
resolution scale, HDR10.

UE 5.2's TSR shader files (from the exe): ClearPrevTextures, ForwardScatterDepth, DilateVelocity,
DecimateHistory, ComputeMoireLuma, RejectShading, SpatialAntiAliasing, FilterAntiAliasing,
UpdateHistory, ResolveHistory (plus CompareTranslucency, DebugHistory). Reunion's 5.6 build ran six
passes. Named TSR textures include `TSR.Velocity.Dilated`, `TSR.ClosestDepthTexture`,
`TSR.History.ColorArray`, `TSR.Output`, `TSR.HalfResOutput`, `TSR.QuarterResOutput`.

## Differences from Reunion that the code already handles

- cvars: UE 5.2 layout (DoubleExposureTweaks' `cvars.hpp`); the DLSS mode's `r.ScreenPercentage`
  write is copied to the render thread by the add-on.

## TSR map (capture 2026-10-26, gameplay, 3840x2160 at 100% resolution scale)

`DoubleExposureDLSS-frame-1.txt`: two frames, 518 dispatches. TSR is nine compute passes in one
command list on the graphics queue, found by TSR's common parameter block (root CBV param 2, history
size 7680x4320 at byte 112). Render textures are allocated at the full 3840x2160 whatever the
resolution scale; the view size comes from the parameters.

| shader id | groups (at 100%) | pass | key bindings |
|---|---|---|---|
| `42e09aac` | 240x135 | ClearPrevTextures | u0-u2 r32uint |
| `02016352` | 480x270 | ForwardScatterDepth | t0 r32uint -> u0 r32uint |
| `cf6ec20a` | 480x270 | **DilateVelocity** | t0 depth `r32g8x24_typeless`, t1 velocity `rgba16unorm` -> u0 dilated velocity `rg16unorm`, u1 closest depth r16f |
| `67aa2e5d` | 480x270 | **DecimateHistory** | t0 = DilateVelocity's u0, t1 closest depth |
| `bd30efd9` | 320x180 | RejectShading | t0 scene colour rgba16f, t2 rgba16f |
| `d9c8c910` | 480x270 | SpatialAntiAliasing | |
| `d77fb051` | 480x270 | FilterAntiAliasing | |
| `23742aef` | 960x540 | **UpdateHistory** | t0 scene colour rgba16f, t1 rgba16f -> u0 history 7680x4320 rgba16f |
| `d14da7eb` | 480x270 | **ResolveHistory** | t0 history -> u0 output 3840x2160 r11g11b10f, u1 1920x1080 half-res copy |

ComputeMoireLuma didn't run (no pass between DilateVelocity and DecimateHistory).

TSR's common parameters (param 2) have the same layout as UE 5.6's: 0 extent, 32 ViewportMin/Max,
48 ViewportSize, 112 history size, 256 jitter in render pixels (`-0.375, -0.0556` in the capture),
272 ScreenVelocityToInputPixelVelocity (1920, -1080 at 100%). The View uniform buffer is root CBV
param 3 (as in 5.6); ClipToPrevClip's offset is searched at run time (Reunion's 2176 is skipped).

Differences from 5.6 handled in 0.1.0: the chain is linked at DecimateHistory's t0 (5.6:
RejectShading's t8); the scene colour feed is UpdateHistory's t0 as in 5.6 (5.2 has no separate
RejectShading feed). The output is r11g11b10f even with HDR output (DLSS accepts it).

## First DLSS test (0.1.0, 2026-10-26)

Log: View.ClipToPrevClip found at **byte 1872** of the View uniform buffer; NGX 310.9.16 (NVIDIA App
override); DLAA 3840x2160, Quality 2561x1441 and Balanced 2228x1253 all evaluated; presets J/K/L/M
switched; F9 off/on restored the game's 100% scale. No errors.

User saw **noise in the trees and smearing on the character's satchel**. Likely cause: meshes animated
in their vertex shader (wind on foliage, cloth) write no velocity, so our motion vectors give them
the camera's motion only; TSR's rejection hides that, DLSS doesn't.

0.1.1 set `r.Velocity.EnableVertexDeformation=1` while DLSS was on: **the game hung** right after
(log stops inside the Set). Changing it recreates every primitive's render state, which has to run
on the game thread; ReShade's present thread deadlocked. It was pointless anyway: the game's
values (logged) are `r.Velocity.EnableVertexDeformation=2` (auto: on while TSR is the AA method,
which it stays under DLSS), `r.VertexDeformationOutputsVelocity=-1`, `r.BasePassOutputsVelocity=-1`,
`r.Velocity.ForceOutput=0`, `r.VelocityOutputPass=1`, `r.SkinCache.Mode=2`, `r.HairStrands.VelocityType=1`.
0.1.2 only logs them. Lesson: never set cvars with change delegates that touch render state from
ReShade's thread.

So wind-animated foliage should already write velocity. Remaining suspects for the tree noise and
satchel smear: (1) UE 5.2's velocity flag for pixel/vertex animation (TSR treats those pixels
specially; DLSS has a "bias current colour" mask for exactly this), (2) the satchel being
CPU-simulated cloth without previous positions, (3) the object velocity decode in motion.hlsl.

## Animation mask (0.2.0)

motion.hlsl also writes an r8 mask at render resolution, passed to DLSS as the "bias current colour"
mask (trust this frame over history there). Source: UE 5.1+ packs a "pixel animation" flag into the
lowest bit of the encoded velocity's w (rgba16unorm velocity: z and the rest of w hold depth); TSR
uses it to distrust history. Overlay "Animation mask": Off / Animated materials (flag; default) /
Animated materials and moving objects (also every pixel with object velocity). "Show mask (debug)"
tints masked pixels red in the output (tint.hlsl, after DLSS, before post-processing). Untested:
whether Deck Nine's foliage and cloth materials set the flag is what the debug view shows.

### Result (2026-10-26)

"Animated materials": nothing turned red, so the game's materials don't set the pixel-animation
flag. "Animated materials and moving objects": the trees and the satchel turned red, so they write
object velocity (WPO velocity is on). The mask over them changed nothing: no less noise in the trees,
no less smearing on the satchel. So the history isn't the problem; the object velocity DLSS gets for
them likely is (the decode in motion.hlsl came from 5.6). 0.2.1 adds a diagnosis switch, "Object
motion": linear decode (as before), square-root decode (VELOCITY_ENCODE_GAMMA), or ignore object
velocity; and preset E (CNN) to check whether the transformer presets ignore the mask.

## Two TSR views (0.2.2)

The game runs TSR twice a frame: the 3840x2160 screen and a 4096x2160 view (2868x1512 render at
70%; possibly the other timeline). Each output gets its own DLSS feature. A DLSS mode change gives
TSR new output textures, so 0.2.1 reached its four-feature limit and refused the fifth ("too many
TSR views"; that view stayed on TSR). 0.2.2 releases features unused for 3 frames before creating
a new one.

## Object velocity encoding: square-root (0.3.0)

User's comparison (0.2.2, 2026-10-26) of the three Object motion choices on the trees and the satchel:
**square-root decode was best**. UE 5.2 writes object velocity with VELOCITY_ENCODE_GAMMA:
encoded = sign(V) * sqrt(2|V|), then the usual `* 0.499 * 0.5 + 32767/65535`; decode
V = sign(v) * v^2 / 2. Reunion's UE 5.6 build stores it linearly (DLSS was clean there with the
linear decode). 0.3.0 makes square-root the default (the switch stays under Diagnosis, not saved).
The game's materials never set the pixel-animation flag; the animation mask stays available
(default "Animated materials", which marks nothing in the scenes seen so far).

**Confirmed by the user (2026-10-26): with the square-root decode the trees and the satchel look
clean.** DLSS Super Resolution works in this game.

## Frame generation (0.4.0, untested)

ReunionDLSS's Streamline DLSS-G code (docs/FRAMEGEN.md there) is game-agnostic except:
- **Two TSR views**: tags and camera came from whichever view was evaluated first in a frame. 0.4.0
  keeps each view's View uniform buffer with its motion vectors (Link/Chain) and tags only the view
  whose output is the back buffer's size (3840x2160), never the 4096x2160 one.
- **Camera offsets** in the View UB (TranslatedWorldToClip 0, RelativeWorldToClip 64,
  ViewToTranslatedWorld 256, ViewToClip 448, ViewToClipNoAA 512) are the same in 5.2 (first members
  unchanged); ClipToPrevClip is the found 1872. CheckCamera verifies at run time.
- **Pre-HUD picture**: the capture has the same HDR UI-composite copy as Reunion (back buffer
  3840x2160 rgb10a2unorm copied to a same-format texture), so the hudless tag should work. The HUD
  layer's alpha orientation is checked at run time (readback), as in Reunion.

## Correction: one view, at 4096x2160 (0.4.1)

The "second TSR view" was the screen itself. The game's resolution is set to **4096x2160** (DCI 4K,
`ResolutionSizeX=4096` in GameUserSettings.ini, fullscreen): its swapchain is created at 3840x2160
and resized to 4096x2160 a second later (ReShade.log), so a 3840x2160 TSR output exists only for
the first frames. Releasing unused DLSS features (0.2.2) is still right; tagging only the
back-buffer-sized view (0.4.0) is harmless.

First frame generation test (0.4.0): Streamline started, the pre-HUD copy and the inverted HUD
layer were found, but **no frames were generated** ("waiting for DLSS"): `View buffer check FAILED
(projection 0.7031 1.3333 near 10.000)`. The View uniform buffer offsets differ from 5.6. From the
capture (View UB, 3840x2160 at 100%):

| byte | UE 5.2 |
|---|---|
| 0 | TranslatedWorldToClip |
| 64 | (extra matrix, large world coordinates) |
| 128 | RelativeWorldToClip (translation row = camera offset) |
| 192 | ClipToRelativeWorld |
| 256 | TranslatedWorldToView |
| 320 | ViewToTranslatedWorld (camera axes) |
| 384 / 448 | TranslatedWorldToCameraView / CameraViewToTranslatedWorld |
| 512 | ViewToClip (jitter in row 2: -0.000195, 5.14e-05) |
| 576 | ViewToClipNoAA |
| 1872 | ClipToPrevClip (found at run time) |

0.4.1 uses 0 / 128 / 320 / 512 / 576 (5.6: 0 / 64 / 256 / 448 / 512).

## Frame generation works (0.4.1, 2026-10-26)

User: frames are generated in game (log: "2 frame(s) presented per game frame"; 2x and Dynamic
tried). Not on the main menu: no TSR view to tag there, so frames are presented without generated
ones (by design, as in Reunion; generated frames would warp flat UI). Leftovers: TSR's output
texture comes from UE's render target pool and is sometimes swapped (features released "unused for
2021 frames"), which recreates the DLSS feature (history reset) and leaves a frame untagged for
Streamline. **0.4.2 keys DLSS features and the reset/size bookkeeping by output size**, so a
swapped texture of the same size keeps DLSS's history (untested).

## 0.5.0 check (2026-10-26)

User: the new overlay looks good. Log (about a minute of play, DLSS Balanced, frame generation
Dynamic): only three DLSS features created, all at startup (3840x2160, then the Balanced render
size, then 4096x2160), none released or recreated during play: keying by output size works.
Dynamic moved between 2, 3 and 6 frames per rendered frame. Still three single frames without tags
(Streamline reuses the last; frame generation pauses briefly) in about 55 s; to look into.

## Tags at submission (0.6.0, untested)

The few frames a minute without tags came from tagging at record time: the add-on had to guess the
frame (the current one, or the next if the current one already had inputs), and UE records frame
N+1, during hitches N+2, while N is still presenting. 0.6.0 prepares the inputs at record time and
hands them to Streamline in `execute_command_list` (ReShade calls it for each list the game
submits): lists are submitted in the order frames are presented, so the frame is the current token,
exactly. All tagged resources are ours and eValidUntilPresent, so no command list is needed at
submission: UE's depth is copied into one of two alternating textures of its own format
(r32g8x24_typeless, D3D12 CopyResource on the game's list), plus our motion vectors, our pre-HUD
copy and the HUD mask. The log warns if two sets of inputs are submitted for one frame.

### Result and exit crash (0.6.1)

0.6.0 log (about a minute of play): untagged frames only at the two moments frame generation
switched on (start, and resuming after loading at 22 s: once two input sets landed in one frame,
"Setting different 'common' constants multiple times within the same frame"; the game submits from
more than one thread); then 35 s of play to exit with **no errors** (0.5.0: about three a minute).

**The game aborted at exit** (UE crash report "Abort signal received"): our OnDestroyDevice held
g_mutex while calling slShutdown; DLSS-G submitted a last command list, ReShade's
execute_command_list event called OnExecute on the same thread, and std::mutex threw "resource
deadlock would occur". Stack: game -> ReShade dxgi -> our add-on -> sl.interposer -> DLSS-G plugin
-> d3d12 -> ReShade dxgi -> our add-on -> abort. 0.6.1 makes framegen.cpp's mutex recursive
(OnExecute then sees g_ready false and returns). **User confirmed 2026-10-26: no crash at exit with 0.6.1.**

## Separate translucency (1.0.1)

A user reported that with DLSS on, UI outlines, Safi's flashlight beam, volumetric lighting and
the floating particles in Out of the Gutter disappeared. Cause: UE 5.2 draws translucency into its
own texture and TSR blends it in itself. In the capture, 16 draws go to a 3840x2160 rgba16f render
target with the scene depth just before TSR. Only TSR reads that texture: UpdateHistory t1 and
RejectShading t2. It never reaches the scene colour (UpdateHistory t0), the only colour DLSS got.
UE 5.6 (Reunion) composes translucency before TSR, so ReunionDLSS never needed this.

1.0.1 (`compose.hlsl`, run at UpdateHistory):
- Writes `scene * t.a + t.rgb` (UE's ComposeSeparateTranslucency: premultiplied colour, alpha =
  how much of the scene shows through) into the add-on's own rgba16f texture, which DLSS gets as
  its colour.
- Adds translucent coverage to a copy of the bias mask: `1 - a`, or for additive effects
  translucency luminance / (scene luminance * 0.5).
- The overlay's "Translucent effects" setting (Advanced) controls the mask strength: Off 0,
  Partial 0.5 (default), Full 1.
- If t1 isn't rgba16f, or is under a quarter of the scene's size, the colour is copied without
  translucency, as in 1.0.0.

## Open

- View.ClipToPrevClip offset (logged on the first DLSS frame).
- Frame generation: the HUD-less copy and HUD alpha were found for 5.6; untested here.

## Mapping notes (before the capture)

- TSR's shader ids (common.hpp; placeholders and `kTsrMapped = false` until then).
- Which pass binds depth and velocity (DilateVelocity in 5.6: t0, t1), the scene colour feed, and
  the output; their registers.
- TSR's common parameters: render size, jitter (5.6: offset 256), motion vector scale (272).
- View.ClipToPrevClip's offset in the View uniform buffer (5.6: 2176; found at run time anyway).
- Velocity buffer format (5.6: rgba16unorm; motion.hlsl decodes it).
- HUD composite for frame generation (5.6: the HUD alpha was inverted).

0.0.1's capture writes every compute pass in full (bound textures by register, the first 1 KB of
each constant buffer, the View buffer once per frame) because TSR's passes aren't known yet.

## Frame generation after 1.0.1 (1.0.2)

- After 1.0.1, frame generation stopped at Balanced, but only while the game ran at 4096x2160, a
  resolution left in GameUserSettings that the menu no longer offers (the screen is 3840x2160).
  At that size the game renders a second TSR view (3840x2160 output beside the 4096x2160 screen),
  and the screen view was rarely upscaled. At 3840x2160 there is one view and every frame is tagged
  (1.0.2 diagnostics, 2026-10-27, blend on and off). The 4096 case wasn't reproduced.
- About 1 frame in 60 was left to TSR (no frame generation inputs): the 16 KB View buffer read
  failed when the View uniform buffer sat near the end of its upload buffer. 1.0.2 reads what the
  buffer holds and needs only the first 2.5 KB.
- At a hitch, two frames' lists could be submitted before one Present; Streamline refused the
  second set of constants ("Setting different 'common' constants multiple times within the same
  frame"). 1.0.2 keeps the first set and skips the second.
- Diagnostics: when a view is left to TSR or a frame goes without inputs, a "diag" line every
  2 s (at most 30) gives each view's outcome by output size and the reasons.
