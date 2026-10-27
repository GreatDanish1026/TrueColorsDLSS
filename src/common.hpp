// Shared between the capture tool (main.cpp), the TSR hook (upscaler.cpp) and the NGX wrapper
// (dlss.cpp).
#pragma once

#ifndef ImTextureID
#define ImTextureID ImU64
#endif
#include <imgui.h>
#include <reshade.hpp>

#include <cstdint>
#include <string>

struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

namespace rd {

// TrueColorsDLSS.log next to the game exe.
void Log(const char* fmt, ...);

// The folder of the game exe, with a trailing backslash.
const std::wstring& ExeDir();

// FNV-1a of a compute pipeline's shader bytecode (0 if unknown): stable across runs.
uint32_t PipeId(reshade::api::pipeline pipe);

// The upscaler's compute shaders. True Colors (UE 4.25) upscales with TAA/TAAU, not TSR, so
// these are placeholders until a frame capture maps the passes (docs/FINDINGS.md); while
// kTsrMapped is false, DLSS and frame generation stay off and F10 captures every compute pass
// in full. The names are kept from the TSR add-ons for now. Placeholders are never 0 (unknown
// pipelines hash to 0).
constexpr bool kTsrMapped = true;
// UE 4.25's main TAA (TemporalAA.usf compute, the full-screen permutation at 100% screen
// percentage; frame capture 2026-10-27, docs/FINDINGS.md): t0 exposure (1x1), t1 scene colour
// (rgba16f, translucency already composited), t2 depth, t3 velocity (rg16unorm), t4 stencil,
// t5 history; u0 output (= next frame's history); root CBV 3 its parameters, 4 the View buffer.
constexpr uint32_t kTaaMain = 0x4a48869b;
constexpr uint32_t kTsrClearPrevTextures = 0xdead0001;
constexpr uint32_t kTsrForwardScatterDepth = 0xdead0002;
constexpr uint32_t kTsrDilateVelocity = 0xdead0003;
constexpr uint32_t kTsrDecimateHistory = 0xdead0004;
constexpr uint32_t kTsrRejectShading = 0xdead0005;
constexpr uint32_t kTsrSpatialAntiAliasing = 0xdead0006;
constexpr uint32_t kTsrFilterAntiAliasing = 0xdead0007;
constexpr uint32_t kTsrUpdateHistory = 0xdead0008;
constexpr uint32_t kTsrResolveHistory = 0xdead0009;

// upscaler.cpp
void RegisterUpscaler();
void UnregisterUpscaler();
// Overlay: the Settings tab and the Status tab.
void DrawUpscalerSettings();
void DrawUpscalerStatus();
void DrawUpscalerAdvanced();
// One line for the top of the overlay; *warn when it's a problem.
std::string UpscalerHeadline(bool* warn);

// Overlay helpers (main.cpp), in the Tweaks add-ons' style.
float ControlWidth();
void Help(const char* text);  // a "(?)" after the last item, with <text> as its tooltip
extern const ImVec4 kWarn;
// Puts the game's pipeline, root signatures, descriptor heaps and bindings back on <cl> after
// the add-on recorded work of its own on it.
void RestoreListState(reshade::api::command_list* cl);

// framegen.cpp: DLSS Frame Generation through Streamline.
void RegisterFrameGen();
void UnregisterFrameGen();
void DrawFrameGenSettings();
void DrawFrameGenStatus();
std::string FrameGenHeadline(bool* warn);
// The View uniform buffer (first 16 KB) of this frame's main view, with View.ClipToPrevClip's offset.
void FrameGenSetCamera(const float* view, int clipToPrevClipOffset);
// True if a TSR output of this size is the picture on screen (the back buffer's size).
bool FrameGenIsScreen(unsigned width, unsigned height);
std::string FrameGenDiag();  // diagnostics for the upscaler's diag line
// This frame's depth (UE's, state depth read + shader resource) and our motion vectors (render
// pixels, previous minus current; state non-pixel shader resource), tagged on <list>.
void FrameGenTag(ID3D12GraphicsCommandList* list, ID3D12Resource* depth, ID3D12Resource* motion, unsigned renderWidth,
                 unsigned renderHeight, float jitterX, float jitterY, bool reset);

}  // namespace rd
