# TrueColorsDLSS: findings

Game: Life is Strange: True Colors, Steam build 8050014, `Siren-Win64-Shipping.exe` PE timestamp
`0x61e748ff`, UE **4.25Plus**, DirectX 12 (Steam's "DirectX 12" launch entry, `-dx12`). Engine
facts (paths, cvar layout, HDR, ray tracing) are in TrueColorsTweaks' `docs/FINDINGS.md`.

Started from DoubleExposureDLSS 1.0.2 (UE 5.2, TSR). True Colors has no TSR: it anti-aliases with
UE 4's TAA (`r.DefaultFeature.AntiAliasing=2`), at 100% screen percentage with
`r.TemporalAA.Upsampling=0` (probe 2026-10-27).

## Frame capture (0.0.2, 2026-10-27, 3840x2160, gameplay, camera moving)

0.0.1's F10 capture crashed the game (access violation in `GetDesc` under `ResolveTables`, read
from the UE4 minidump with `llvm-symbolizer` on a line-info rebuild with identical code). UE 4's
D3D12 root signatures use fixed-size descriptor tables, and the slots a shader doesn't use keep old
descriptors of resources that may have been freed. 0.0.2 parses each compute shader's DXBC
declarations (UE 4.25 ships SM5 DXBC on D3D12; parser from QuarryDLSS) and resolves only declared
registers. Capture: 482 dispatches over two frames, 31 MB.

Most compute work is 1D (skinning, particles, light grid). Full-screen passes (480x270 groups of
8x8) are the ray-traced shadow denoiser (they read velocity too) and, last, the **main TAA,
`cs4a48869b`**:

| Binding | Resource | DLSS input |
|---|---|---|
| table 0 t0 | 1x1 rgba32f | exposure (DLSS uses auto exposure) |
| table 0 t1 | 3840x2160 rgba16f | **colour** (translucency already composited) |
| table 0 t2 | 3840x2160 r32g8x24_typeless | **depth** (reversed Z) |
| table 0 t3 | 3840x2160 rg16unorm | **velocity** (moving objects only) |
| table 0 t4 | depth again | stencil |
| table 0 t5 | 3840x2160 rgba16f | history |
| table 2 u0 | 3840x2160 rgba16f | **output** (also the next frame's history) |
| root CBV 3 | TAA parameters | sub-allocated from an 8 MB upload buffer (16 KB dump runs into other data) |
| root CBV 4 | View uniform buffer | camera, jitter, sizes |

Same shape as UE 4.26's TAA in The Quarry (QuarryDLSS FINDINGS), with rg16 velocity instead of
rgba16.

## View uniform buffer (UE 4.25, byte offsets)

| Offset | Field | Value in the capture |
|---|---|---|
| 0 | TranslatedWorldToClip | |
| 64 | WorldToClip | translation row -4210, -675 |
| 192 / 256 | TranslatedWorldToView / ViewToTranslatedWorld | rotations |
| 448 | ViewToClip (jittered) | 2.2387, 3.9799; jitter in row 480; near 10 at 496 |
| 512 | ViewToClipNoAA | |
| 576 | ClipToView | |
| 1952 | **ClipToPrevClip** (unjittered) | near identity, translation -2.1e-6, -8.0e-6 |
| 2016 | **TemporalAAJitter** | -8.54e-5, -2.63e-4 (this frame), 1.91e-4, 2.85e-4 (last) |
| 2064 | ViewRectMin | 0, 0 |
| 2080 | **ViewSizeAndInvSize** | 3840, 2160 |
| 2096 | BufferSizeAndInvSize | 3840, 2160 |

UE 5.x put ClipToPrevClipWithAA between ClipToPrevClip and the jitter; 4.25 has no such matrix,
so the jitter follows ClipToPrevClip directly. Jitter in pixels (UE's TemporalJitterPixels) =
(jitter.x * width / 2, -jitter.y * height / 2): -0.164, 0.284 px in the capture.

## 0.1.0: DLSS at the TAA (untested)

At `cs4a48869b`'s dispatch: motion.hlsl computes motion vectors (camera from depth and
ClipToPrevClip, plus the engine's velocity, linear decode as in UE 4.26), DLSS reads colour, depth
and motion and writes u0, and the game's TAA dispatch is skipped. Layout checked each frame (the
jitter in ViewToClip's row 480 must equal TemporalAAJitter, and ClipToPrevClip must look like one).
Frames whose velocity isn't rg16 at full size (UE binds a 1x1 placeholder on camera-cut frames)
stay with the game's TAA and reset DLSS on the next frame. Frame generation's View offsets moved to
UE 4.25's (0 / 64 / 256 / 448 / 512).

Render resolution: only the game's (100%, so DLSS runs as DLAA). Lower DLSS modes need
`r.ScreenPercentage` < 100 with `r.TemporalAA.Upsampling=1` (a different TAA permutation, so a
different shader id) and setting cvars on the game thread (UE 4).
