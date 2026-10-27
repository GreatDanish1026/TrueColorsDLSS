// DLSS Frame Generation through NVIDIA Streamline 2.14.16 (manual hooking).
//
// Streamline's frame generation lives in a proxy swapchain. When a D3D12 device is created, the
// real DXGI factory's CreateSwapChain / CreateSwapChainForHwnd are hooked. When the game creates
// its swapchain, Streamline is loaded from Win64\streamline\ (signature checked) for the device
// of that swapchain's queue, and the swapchain the game gets is Streamline's proxy.
//
// ReShade loads add-ons per device and unloads them when the last one is destroyed: UE creates
// and destroys a test device at startup, so this add-on is loaded twice. Every hook is removed
// when it unloads, and Streamline is only started for the swapchain (the real device). Its Present is hooked too: that's where each game frame ends (Reflex
// latency markers, the frame token, frame generation on or off).
//
// Per frame, at the point DLSS upscales (upscaler.cpp), the depth (copied), our motion vectors and
// the camera constants (from the View uniform buffer) are prepared, and in HDR the picture before
// the HUD (see OnCopyResource); they are handed to Streamline when the command list holding them is
// submitted (OnExecute), which ties them to the right frame. Frames without them
// (menus, loading, DLSS off) are presented without generated frames.
//
// Loading Streamline only happens if frame generation was on at launch: switching it on later
// asks for a restart, switching it off is live.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <intrin.h>

#include "common.hpp"

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#include <sl_security.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "ui_alpha_cs.h"  // kUiAlphaCs: ui_alpha.hlsl compiled by build.ps1
#include "version.h"

using namespace reshade::api;

namespace rd {
namespace {

constexpr const char* kProjectId = "6f1d6e2a-5b8c-4f2e-9a43-7c2b8e1d4a90";

// Byte offsets in UE 5.2's View uniform buffer (FViewUniformShaderParameters), read from a capture
// (docs/FINDINGS.md): 5.2 has one more matrix after TranslatedWorldToClip than 5.6 (Reunion: 64,
// 256, 448, 512), so the rest sit 64 bytes later. Checked against the buffer at run time
// (CheckCamera).
constexpr uint32_t kTranslatedWorldToClip = 0, kRelativeWorldToClip = 128, kViewToTranslatedWorld = 320,
                   kViewToClip = 512, kViewToClipNoAA = 576;

// Frame generation modes, as in The Quarry's add-on: Off, 2x, 3x, 4x (1-3 generated frames per
// rendered frame) and Dynamic (Streamline picks the multiplier to reach the display's refresh rate).
enum Mode : int { kOff, k2x, k3x, k4x, kDynamic, kModeCount };
const char* const kModeNames[kModeCount] = {"Off", "2x", "3x", "4x", "Dynamic"};

struct Settings {
    int mode = kOff;
    bool enabled = false;  // mode != kOff
    // sl::Constants::minRelativeLinearDepthObjectSeparation: how far apart in depth two surfaces
    // must be for DLSS-G to treat them as separate objects, in units of 1 / device depth. UE's
    // near plane is 10 cm, so NVIDIA's default 40 is 4 m.
    float separation = 40;
} g_set;

uint32_t g_maxFrames = 1;  // DLSSGState::numFramesToGenerateMax
bool g_dynamicSupported = false, g_capsKnown = false;
int g_appliedMode = -1;  // what Streamline was last told (-1: nothing yet)

struct SlApi {
    HMODULE mod = nullptr;
    PFun_slInit* init = nullptr;
    PFun_slShutdown* shutdown = nullptr;
    PFun_slIsFeatureSupported* isFeatureSupported = nullptr;
    PFun_slGetFeatureRequirements* getFeatureRequirements = nullptr;
    PFun_slSetTagForFrame* setTagForFrame = nullptr;
    PFun_slSetConstants* setConstants = nullptr;
    PFun_slGetFeatureFunction* getFeatureFunction = nullptr;
    PFun_slGetNewFrameToken* getNewFrameToken = nullptr;
    PFun_slSetD3DDevice* setD3DDevice = nullptr;
    PFun_slUpgradeInterface* upgradeInterface = nullptr;
    PFun_slDLSSGSetOptions* dlssgSetOptions = nullptr;
    PFun_slDLSSGGetState* dlssgGetState = nullptr;
    PFun_slReflexSetOptions* reflexSetOptions = nullptr;
    PFun_slReflexGetState* reflexGetState = nullptr;
    PFun_slReflexSleep* reflexSleep = nullptr;
    PFun_slPCLSetMarker* pclSetMarker = nullptr;
} g_sl;

// Recursive: Streamline calls made under it (its shutdown at exit, which submits a last command
// list) come back into this add-on on the same thread through ReShade's events (OnExecute). A plain
// std::mutex throws there ("resource deadlock would occur") and the game aborted at exit (0.6.0).
std::recursive_mutex g_mutex;
bool g_initTried = false, g_ready = false;
std::atomic<bool> g_upgraded{false};  // the game's swapchain is Streamline's proxy
std::string g_status = "off";

// Frames: the token of the frame being submitted; it advances after each game present.
sl::FrameToken* g_token = nullptr;
uint32_t g_frame = 0;
// The last two frames given each kind of input.
struct Tagged {
    uint32_t last = UINT32_MAX, previous = UINT32_MAX;
    bool Has(uint32_t frame) const { return last == frame || previous == frame; }
    void Set(uint32_t frame) { previous = last, last = frame; }
};
Tagged g_inputs, g_hudless;  // depth + motion + constants; the picture before the HUD

// Inputs are prepared while the game records a frame's command list and handed to Streamline when
// that list is submitted (execute_command_list). Lists are submitted in the order frames are
// presented, so the frame being submitted is the one the next Present shows: g_frame, exactly.
// (0.4.x tagged at record time and had to guess: UE records frame N+1, during hitches N+2, while
// N is still presenting, and the guess left a frame without inputs about three times a minute.)
// Everything tagged is ours and stays valid until the Present: our motion vectors, our copy of the
// depth, our copy of the picture before the HUD and the HUD mask.
struct PendingInputs {
    sl::Constants constants{};
    ID3D12Resource* depth = nullptr;  // our copy
    ID3D12Resource* motion = nullptr;
    uint32_t rw = 0, rh = 0;
};
struct PendingHud {
    bool mask = false;  // the HUD mask was recorded too
};
std::map<uint64_t, PendingInputs> g_pendingInputs;  // by native command list
std::map<uint64_t, PendingHud> g_pendingHud;
// Our copies of UE's depth (two, alternating), the depth's size and format.
ID3D12Resource* g_depthCopy[2] = {};
int g_depthNext = 0;
D3D12_RESOURCE_DESC g_depthDesc{};
constexpr D3D12_RESOURCE_STATES kDepthReadState =
    D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

void ReleaseDepthCopies() {
    for (ID3D12Resource*& r : g_depthCopy)
        if (r) r->Release(), r = nullptr;
    g_depthDesc = {};
}

// A texture like <depth> to copy it into (state kDepthReadState), or null.
ID3D12Resource* DepthCopy(ID3D12Resource* depth) {
    const D3D12_RESOURCE_DESC d = depth->GetDesc();
    if (d.Width != g_depthDesc.Width || d.Height != g_depthDesc.Height || d.Format != g_depthDesc.Format ||
        d.Flags != g_depthDesc.Flags)
        ReleaseDepthCopies();
    ID3D12Resource*& slot = g_depthCopy[g_depthNext];
    if (!slot) {
        ID3D12Device* dev = nullptr;
        if (FAILED(depth->GetDevice(IID_PPV_ARGS(&dev)))) return nullptr;
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        const HRESULT hr = dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, kDepthReadState, nullptr,
                                                        IID_PPV_ARGS(&slot));
        dev->Release();
        if (FAILED(hr)) {
            slot = nullptr;
            Log("frame generation: couldn't create a %llux%u depth copy (0x%08x)", d.Width, d.Height, static_cast<unsigned>(hr));
            return nullptr;
        }
        g_depthDesc = d;
        Log("frame generation: created %llux%u depth copy (format %u)", d.Width, d.Height, static_cast<unsigned>(d.Format));
    }
    ID3D12Resource* r = slot;
    g_depthNext ^= 1;
    return r;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}

// Back buffer size and format (ReShade's swapchain), and our copy of the picture before the HUD.
uint32_t g_bbWidth = 0, g_bbHeight = 0;
format g_bbFormat = format::unknown;
resource g_hudlessTex = {0};
resource g_uiReadback = {0};  // one row of the HUD layer, for its check
device* g_hudlessDevice = nullptr;
std::atomic<uint64_t> g_hudlessCount{0};

// The HUD mask: ui_alpha.hlsl writes the layer's alpha (flipped if inverted) to an r8 texture.
struct MaskState {
    pipeline_layout layout = {0};
    pipeline pipe = {0};
    bool failed = false;
    resource tex = {0};
    resource_view uav = {0};
    std::vector<std::pair<uint64_t, resource_view>> srvs;  // per HUD layer texture
} g_mask;

void DestroyMask(device* dev) {
    for (auto& [h, v] : g_mask.srvs) dev->destroy_resource_view(v);
    if (g_mask.uav.handle) dev->destroy_resource_view(g_mask.uav);
    if (g_mask.tex.handle) dev->destroy_resource(g_mask.tex);
    if (g_mask.pipe.handle) dev->destroy_pipeline(g_mask.pipe);
    if (g_mask.layout.handle) dev->destroy_pipeline_layout(g_mask.layout);
    g_mask = MaskState();
}

bool g_fgOn = false;                  // what Streamline was last told
std::atomic<uint64_t> g_presents{0}, g_tagged{0};
uint32_t g_untagged = 0;  // game frames in a row without inputs
// Diagnostics (upscaler.cpp's diag line): tag calls that returned early, tags set, frames untagged.
std::map<std::string, uint32_t> g_tagWhy;
uint32_t g_tagsSet = 0, g_framesUntagged = 0;
sl::DLSSGState g_state{};  // last read on the presenting thread

// The camera, copied from the View uniform buffer when our motion vectors are computed.
struct Camera {
    float m[5][16];  // TranslatedWorldToClip, RelativeWorldToClip, ViewToTranslatedWorld, ViewToClipNoAA, ClipToPrevClip
    bool valid = false;
} g_camera;
bool g_cameraChecked = false, g_cameraOk = false;

// ---------------------------------------------------------------- matrices (row-major, row vector * matrix)

bool Invert(const float* m, float* out) {
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = static_cast<double>(m[0]) * inv[0] + static_cast<double>(m[1]) * inv[4] +
                       static_cast<double>(m[2]) * inv[8] + static_cast<double>(m[3]) * inv[12];
    if (det == 0 || !std::isfinite(det)) return false;
    for (int i = 0; i < 16; ++i) out[i] = static_cast<float>(inv[i] / det);
    return true;
}

void Multiply(const float* a, const float* b, float* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += static_cast<double>(a[r * 4 + k]) * b[k * 4 + c];
            out[r * 4 + c] = static_cast<float>(s);
        }
}

void ToSl(sl::float4x4& d, const float* s) {
    for (uint32_t r = 0; r < 4; ++r) d.setRow(r, sl::float4(s[r * 4], s[r * 4 + 1], s[r * 4 + 2], s[r * 4 + 3]));
}

// The View buffer layout this code assumes, checked once: the unjittered projection is a
// reversed-Z perspective matrix (w = view z) matching the jittered one apart from the jitter.
bool CheckCamera(const float* view) {
    const float* p = view + kViewToClipNoAA / 4;
    const float* pj = view + kViewToClip / 4;
    const float* axes = view + kViewToTranslatedWorld / 4;
    const auto len = [](const float* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    const bool ok = p[0] > 0.1f && p[5] > 0.1f && std::fabs(p[11] - 1) < 1e-4f && std::fabs(p[15]) < 1e-4f &&
                    std::fabs(p[0] - pj[0]) < 1e-4f && std::fabs(p[5] - pj[5]) < 1e-4f && p[14] > 0 &&
                    std::fabs(len(axes) - 1) < 1e-3f && std::fabs(len(axes + 4) - 1) < 1e-3f &&
                    std::fabs(len(axes + 8) - 1) < 1e-3f;
    Log("frame generation: View buffer check %s (projection %.4f %.4f near %.3f, axes %.3f %.3f %.3f)",
        ok ? "passed" : "FAILED", p[0], p[5], p[14], len(axes), len(axes + 4), len(axes + 8));
    return ok;
}

// ---------------------------------------------------------------- frames, Reflex and DLSS-G options

void Marker(sl::PCLMarker m) {
    if (g_sl.pclSetMarker && g_token) g_sl.pclSetMarker(m, *g_token);
}

// The mode actually used: 3x/4x need multi-frame generation, Dynamic a recent driver.
int EffectiveMode(int mode) {
    if (mode == kDynamic && !g_dynamicSupported) return k2x;
    if (mode >= k2x && mode <= k4x && static_cast<uint32_t>(mode) > g_maxFrames) return static_cast<int>(g_maxFrames);
    return mode;
}

void SetFrameGeneration(bool on) {
    if (!g_sl.dlssgSetOptions) return;
    if (!g_capsKnown && g_sl.dlssgGetState) {
        sl::DLSSGState st{};
        if (g_sl.dlssgGetState(sl::ViewportHandle(0), st, nullptr) == sl::Result::eOk) {
            g_capsKnown = true;
            g_maxFrames = st.numFramesToGenerateMax ? st.numFramesToGenerateMax : 1;
            g_dynamicSupported = st.bIsDynamicMFGSupported == sl::Boolean::eTrue;
            Log("frame generation: up to %u generated frames per rendered frame, Dynamic %s", g_maxFrames,
                g_dynamicSupported ? "supported" : "not supported");
        }
    }
    const int mode = on ? EffectiveMode(g_set.mode) : kOff;
    if (mode == g_appliedMode) return;
    sl::DLSSGOptions o{};
    o.mode = mode == kOff ? sl::DLSSGMode::eOff : mode == kDynamic ? sl::DLSSGMode::eDynamic : sl::DLSSGMode::eOn;
    o.numFramesToGenerate = mode >= k2x && mode <= k4x ? static_cast<uint32_t>(mode) : 1;  // ignored by Dynamic
    o.dynamicTargetFrameRate = 0;  // Dynamic: the display's refresh rate
    o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    const sl::Result r = g_sl.dlssgSetOptions(sl::ViewportHandle(0), o);
    if (r != sl::Result::eOk) {
        Log("frame generation: slDLSSGSetOptions(%s) failed: %d", kModeNames[mode], static_cast<int>(r));
    } else if (on != g_fgOn || g_appliedMode != kOff) {
        Log("frame generation %s", mode == kOff ? "paused" : kModeNames[mode]);
    }
    g_appliedMode = mode;
    g_fgOn = mode != kOff;
}

void CheckUiRow();

void BeforePresent() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_ready) return;
    ++g_presents;
    Marker(sl::PCLMarker::eRenderSubmitEnd);
    // A frame now and then misses its inputs (another view, a hitch): frame generation only
    // pauses after several in a row (menus, loading, DLSS off), since each switch can stutter.
    g_untagged = g_inputs.Has(g_frame) ? 0 : g_untagged + 1;
    if (g_untagged) ++g_framesUntagged;
    SetFrameGeneration(g_set.enabled && g_tagged > 0 && g_untagged < 8);
    // The overlay's status, read here: Streamline wants it read on the presenting thread.
    if (g_sl.dlssgGetState && g_presents % 30 == 0) {
        sl::DLSSGState st{};
        if (g_sl.dlssgGetState(sl::ViewportHandle(0), st, nullptr) == sl::Result::eOk) {
            if (st.status != g_state.status || st.numFramesActuallyPresented != g_state.numFramesActuallyPresented ||
                g_presents == 30) {
                sl::ReflexState rs{};
                const bool reflex = g_sl.reflexGetState && g_sl.reflexGetState(rs) == sl::Result::eOk;
                Log("frame generation state: %s, %u frame(s) presented per game frame, status 0x%x%s%s%s%s%s, min size %u, "
                    "max generated %u; Reflex low latency %s",
                    g_fgOn ? "on" : "off", st.numFramesActuallyPresented, static_cast<unsigned>(st.status),
                    (static_cast<unsigned>(st.status) & 1) ? " resolution-too-low" : "",
                    (static_cast<unsigned>(st.status) & 2) ? " reflex-not-detected" : "",
                    (static_cast<unsigned>(st.status) & 4) ? " hdr-format-not-supported" : "",
                    (static_cast<unsigned>(st.status) & 8) ? " constants-invalid" : "",
                    (static_cast<unsigned>(st.status) & 16) ? " GetCurrentBackBufferIndex-not-called" : "",
                    st.minWidthOrHeight, st.numFramesToGenerateMax,
                    !reflex ? "unknown" : rs.lowLatencyAvailable ? "available" : "NOT available");
            }
            g_state = st;
        }
    }
    CheckUiRow();
    Marker(sl::PCLMarker::ePresentStart);
}

void AfterPresent() {
    sl::FrameToken* token = nullptr;
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_ready) return;
        Marker(sl::PCLMarker::ePresentEnd);
        const uint32_t next = g_frame + 1;
        if (g_sl.getNewFrameToken(token, &next) != sl::Result::eOk || !token) return;
        g_token = token, g_frame = next;
    }
    // The add-on only runs on the render thread: the next frame starts here. Reflex's sleep
    // runs without the lock (the overlay and the TSR hook take it).
    if (g_sl.reflexSleep) g_sl.reflexSleep(*token);
    g_sl.pclSetMarker(sl::PCLMarker::eSimulationStart, *token);
    g_sl.pclSetMarker(sl::PCLMarker::eSimulationEnd, *token);
    g_sl.pclSetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
}

// ---------------------------------------------------------------- vtable hooks

struct PatchRecord {
    void** entry;
    void* hook;
    void* original;
};
PatchRecord g_patches[8];
int g_patchCount = 0;

void Write(void** entry, void* value) {
    DWORD old = 0;
    if (!VirtualProtect(entry, sizeof(void*), PAGE_READWRITE, &old)) return;
    *entry = value;
    VirtualProtect(entry, sizeof(void*), old, &old);
}

bool Patch(void* object, int slot, void* hook, void** original) {
    void** entry = &(*static_cast<void***>(object))[slot];
    for (int i = 0; i < g_patchCount; ++i)
        if (g_patches[i].entry == entry && *entry == hook) return true;  // already ours
    if (g_patchCount == 8) return false;
    *original = *entry;
    g_patches[g_patchCount++] = {entry, hook, *original};
    Write(entry, hook);
    return true;
}

// Puts the vtables back (only entries that still point at our hooks).
void Unpatch() {
    for (int i = g_patchCount - 1; i >= 0; --i)
        if (*g_patches[i].entry == g_patches[i].hook) Write(g_patches[i].entry, g_patches[i].original);
    g_patchCount = 0;
}

using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using Present1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
using CreateSwapChainFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
using CreateForHwndFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
PresentFn g_present = nullptr;
Present1Fn g_present1 = nullptr;
CreateSwapChainFn g_createSwapChain = nullptr;
CreateForHwndFn g_createForHwnd = nullptr;

HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain* sc, UINT sync, UINT flags) {
    const bool frame = (flags & DXGI_PRESENT_TEST) == 0;
    if (frame) BeforePresent();
    const HRESULT hr = g_present(sc, sync, flags);
    if (frame) AfterPresent();
    return hr;
}

HRESULT STDMETHODCALLTYPE HookPresent1(IDXGISwapChain1* sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) {
    const bool frame = (flags & DXGI_PRESENT_TEST) == 0;
    if (frame) BeforePresent();
    const HRESULT hr = g_present1(sc, sync, flags, p);
    if (frame) AfterPresent();
    return hr;
}

// Streamline creates swapchains of its own through the same factory: those stay as they are.
bool CalledFromStreamline(void* returnAddress) {
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(returnAddress), &mod))
        return false;
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(mod, path, MAX_PATH);
    const wchar_t* name = wcsrchr(path, L'\\');
    name = name ? name + 1 : path;
    return _wcsnicmp(name, L"sl.", 3) == 0 || _wcsnicmp(name, L"nvngx", 5) == 0;
}

thread_local bool t_creating = false;

bool InitStreamline(ID3D12Device* device, std::string& error);

// Starts Streamline (once) for the device of the queue the game creates its swapchain on.
bool EnsureStreamline(IUnknown* queueOrDevice) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_initTried) return g_ready;
    ID3D12CommandQueue* queue = nullptr;
    if (!queueOrDevice || FAILED(queueOrDevice->QueryInterface(IID_PPV_ARGS(&queue)))) return false;  // not D3D12
    g_initTried = true;
    ID3D12Device* device = nullptr;
    const HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
    queue->Release();
    std::string err;
    if (FAILED(hr) || !device) err = "couldn't get the swapchain's device";
    else if (!InitStreamline(device, err)) device->Release(), device = nullptr;
    if (!device) {
        g_status = err;
        Log("frame generation: %s", err.c_str());
        return false;
    }
    device->Release();  // the game keeps it alive
    g_ready = true;
    g_status = "ready";
    Log("frame generation: Streamline 2.14 started for the game's swapchain");
    return true;
}

// DLSS-G sets itself up while the swapchain is created (its "before CreateSwapChain" hooks), so
// the game's swapchain is created through Streamline's proxy of the factory the game called,
// which returns Streamline's proxy swapchain. The proxy factory creates the real swapchain
// through the same vtable entry: t_creating lets that call through untouched.
void* UpgradeFactory(IUnknown* factory) {
    factory->AddRef();  // the proxy takes this reference
    void* p = factory;
    const sl::Result r = g_sl.upgradeInterface(&p);
    if (r != sl::Result::eOk || !p || p == factory) {
        factory->Release();
        Log("frame generation: slUpgradeInterface(factory) failed: %d", static_cast<int>(r));
        return nullptr;
    }
    return p;
}

void HookProxyPresent(IDXGISwapChain* sc) {
    const bool hooked = Patch(sc, 8, reinterpret_cast<void*>(&HookPresent), reinterpret_cast<void**>(&g_present)) &&
                        Patch(sc, 22, reinterpret_cast<void*>(&HookPresent1), reinterpret_cast<void**>(&g_present1));
    g_upgraded = hooked;
    Log("frame generation: swapchain created through Streamline%s", hooked ? "" : " (present hook FAILED)");
}

HRESULT STDMETHODCALLTYPE HookCreateSwapChain(IDXGIFactory* f, IUnknown* dev, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** out) {
    void* caller = _ReturnAddress();
    if (t_creating || CalledFromStreamline(caller) || !EnsureStreamline(dev)) return g_createSwapChain(f, dev, desc, out);
    void* proxy = UpgradeFactory(f);
    if (!proxy) return g_createSwapChain(f, dev, desc, out);
    t_creating = true;
    const HRESULT hr = static_cast<IDXGIFactory*>(proxy)->CreateSwapChain(dev, desc, out);
    t_creating = false;
    static_cast<IUnknown*>(proxy)->Release();
    if (SUCCEEDED(hr) && out && *out) HookProxyPresent(*out);
    else Log("frame generation: CreateSwapChain through Streamline failed: 0x%08lx", static_cast<unsigned long>(hr));
    return hr;
}

HRESULT STDMETHODCALLTYPE HookCreateForHwnd(IDXGIFactory2* f, IUnknown* dev, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                                            const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* output, IDXGISwapChain1** out) {
    void* caller = _ReturnAddress();
    if (t_creating || CalledFromStreamline(caller) || !EnsureStreamline(dev))
        return g_createForHwnd(f, dev, hwnd, desc, fs, output, out);
    void* proxy = UpgradeFactory(f);
    if (!proxy) return g_createForHwnd(f, dev, hwnd, desc, fs, output, out);
    t_creating = true;
    const HRESULT hr = static_cast<IDXGIFactory2*>(proxy)->CreateSwapChainForHwnd(dev, hwnd, desc, fs, output, out);
    t_creating = false;
    static_cast<IUnknown*>(proxy)->Release();
    if (SUCCEEDED(hr) && out && *out) HookProxyPresent(*out);
    else Log("frame generation: CreateSwapChainForHwnd through Streamline failed: 0x%08lx", static_cast<unsigned long>(hr));
    return hr;
}

// Hooks the system DXGI factory's vtable (shared by every factory, ReShade's hooks included).
bool HookFactory(std::string& error) {
    wchar_t sys[MAX_PATH] = {};
    GetSystemDirectoryW(sys, MAX_PATH);
    HMODULE dxgi = LoadLibraryW((std::wstring(sys) + L"\\dxgi.dll").c_str());
    using CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
    auto create = dxgi ? reinterpret_cast<CreateFactory2>(GetProcAddress(dxgi, "CreateDXGIFactory2")) : nullptr;
    IDXGIFactory2* factory = nullptr;
    if (!create || FAILED(create(0, IID_PPV_ARGS(&factory)))) {
        error = "couldn't create a DXGI factory";
        return false;
    }
    const bool ok = Patch(factory, 10, reinterpret_cast<void*>(&HookCreateSwapChain), reinterpret_cast<void**>(&g_createSwapChain)) &&
                    Patch(factory, 15, reinterpret_cast<void*>(&HookCreateForHwnd), reinterpret_cast<void**>(&g_createForHwnd));
    factory->Release();
    if (!ok) error = "couldn't hook swapchain creation";
    return ok;
}

// ---------------------------------------------------------------- Streamline

void OnSlLog(sl::LogType type, const char* msg) {
    if (type == sl::LogType::eInfo) return;  // Streamline's own log file has them
    std::string s = msg ? msg : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    Log("streamline %s: %s", type == sl::LogType::eError ? "error" : "warning", s.c_str());
}

template <typename T>
bool Proc(T*& fn, const char* name) {
    fn = reinterpret_cast<T*>(GetProcAddress(g_sl.mod, name));
    return fn != nullptr;
}

template <typename T>
void Feature(sl::Feature f, T*& fn, const char* name) {
    void* p = nullptr;
    fn = g_sl.getFeatureFunction(f, name, p) == sl::Result::eOk ? reinterpret_cast<T*>(p) : nullptr;
    if (!fn) Log("frame generation: %s not available", name);
}

bool InitStreamline(ID3D12Device* device, std::string& error) {
    static std::wstring dir;
    dir = ExeDir() + L"streamline\\";
    const std::wstring interposer = dir + L"sl.interposer.dll";
    if (GetFileAttributesW(interposer.c_str()) == INVALID_FILE_ATTRIBUTES) {
        error = "streamline\\sl.interposer.dll missing next to the game";
        return false;
    }
    if (!sl::security::verifyEmbeddedSignature(interposer.c_str())) {
        error = "sl.interposer.dll isn't signed by NVIDIA";
        return false;
    }
    g_sl.mod = LoadLibraryW(interposer.c_str());
    if (!g_sl.mod || !Proc(g_sl.init, "slInit") || !Proc(g_sl.shutdown, "slShutdown") ||
        !Proc(g_sl.isFeatureSupported, "slIsFeatureSupported") || !Proc(g_sl.getFeatureRequirements, "slGetFeatureRequirements") ||
        !Proc(g_sl.setTagForFrame, "slSetTagForFrame") || !Proc(g_sl.setConstants, "slSetConstants") ||
        !Proc(g_sl.getFeatureFunction, "slGetFeatureFunction") || !Proc(g_sl.getNewFrameToken, "slGetNewFrameToken") ||
        !Proc(g_sl.setD3DDevice, "slSetD3DDevice") || !Proc(g_sl.upgradeInterface, "slUpgradeInterface")) {
        error = "couldn't load sl.interposer.dll";
        return false;
    }

    sl::Preferences pref{};
    const wchar_t* paths[] = {dir.c_str()};
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = dir.c_str();  // sl.log
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = &OnSlLog;
    // No over-the-air updates: only the DLLs that ship with the add-on are used.
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 3;
    pref.engine = sl::EngineType::eUnreal;
    pref.engineVersion = "5.6";
    pref.projectId = kProjectId;
    pref.renderAPI = sl::RenderAPI::eD3D12;
    sl::Result r = g_sl.init(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) {
        error = "slInit failed (" + std::to_string(static_cast<int>(r)) + ")";
        return false;
    }
    r = g_sl.setD3DDevice(device);
    if (r != sl::Result::eOk) {
        error = "slSetD3DDevice failed (" + std::to_string(static_cast<int>(r)) + ")";
        return false;
    }
    LUID luid = device->GetAdapterLuid();
    sl::AdapterInfo adapter{};
    adapter.deviceLUID = reinterpret_cast<uint8_t*>(&luid);
    adapter.deviceLUIDSizeInBytes = sizeof(luid);
    r = g_sl.isFeatureSupported(sl::kFeatureDLSS_G, adapter);
    if (r != sl::Result::eOk) {
        error = "DLSS Frame Generation not supported here (" + std::to_string(static_cast<int>(r)) +
                "): needs an RTX 40/50, a recent driver and hardware-accelerated GPU scheduling";
        return false;
    }
    sl::FeatureRequirements req{};
    if (g_sl.getFeatureRequirements(sl::kFeatureDLSS_G, req) == sl::Result::eOk)
        Log("frame generation: requirements flags 0x%x", static_cast<unsigned>(req.flags));

    Feature(sl::kFeatureDLSS_G, g_sl.dlssgSetOptions, "slDLSSGSetOptions");
    Feature(sl::kFeatureDLSS_G, g_sl.dlssgGetState, "slDLSSGGetState");
    Feature(sl::kFeatureReflex, g_sl.reflexSetOptions, "slReflexSetOptions");
    Feature(sl::kFeatureReflex, g_sl.reflexSleep, "slReflexSleep");
    Feature(sl::kFeatureReflex, g_sl.reflexGetState, "slReflexGetState");
    Feature(sl::kFeaturePCL, g_sl.pclSetMarker, "slPCLSetMarker");
    if (!g_sl.dlssgSetOptions || !g_sl.reflexSetOptions || !g_sl.pclSetMarker) {
        error = "Streamline features missing (see the log)";
        return false;
    }
    sl::ReflexOptions reflex{};
    reflex.mode = sl::ReflexMode::eLowLatency;  // DLSS-G requires Reflex
    r = g_sl.reflexSetOptions(reflex);
    if (r != sl::Result::eOk) Log("frame generation: slReflexSetOptions failed: %d", static_cast<int>(r));
    const uint32_t first = 0;
    if (g_sl.getNewFrameToken(g_token, &first) != sl::Result::eOk) {
        error = "slGetNewFrameToken failed";
        return false;
    }
    g_frame = 0;
    return true;
}

bool g_factoryHooked = false;

void OnInitDevice(device* dev) {
    if (dev->get_api() != device_api::d3d12) return;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_factoryHooked || !g_set.enabled) return;
    std::string err;
    if (!HookFactory(err)) {
        g_status = err;
        Log("frame generation: %s", err.c_str());
        return;
    }
    g_factoryHooked = true;
    g_status = "waiting for the game's swapchain";
    Log("frame generation: swapchain creation hooked");
}

void OnDestroyDevice(device* dev) {
    if (dev->get_api() != device_api::d3d12) return;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_hudlessTex.handle && g_hudlessDevice == dev) dev->destroy_resource(g_hudlessTex), g_hudlessTex = {0};
    if (g_uiReadback.handle && g_hudlessDevice == dev) dev->destroy_resource(g_uiReadback), g_uiReadback = {0};
    if (g_hudlessDevice == dev) DestroyMask(dev);
    ReleaseDepthCopies();
    g_pendingInputs.clear(), g_pendingHud.clear();
    if (!g_ready) return;
    g_ready = false;
    g_sl.shutdown();
    g_status = "shut down";
    Log("frame generation: Streamline shut down");
}

// The frame index inputs are meant for: this frame, or the next one if this one has them already.
// UINT32_MAX if both have them.


// ---------------------------------------------------------------- the HUD layer
//
// The HUD is drawn into its own back-buffer-sized 8-bit layer, the last one bound before the
// scene copy (below): its alpha says where the HUD is, which frame generation needs to keep the
// HUD from warping. UE may store that alpha inverted, so a row of it is read back once (top
// middle of the screen, normally free of HUD): alpha 0 there means it can be used as is.

struct __declspec(uuid("9a4e0c2b-6d1f-4b83-a5e7-2c8f1b0d3e74")) FgList {
    resource uiLayer = {0};  // last back-buffer-sized 8-bit render target bound on this list
};

enum class UiCheck { pending, reading, usable, inverted, unusable };
UiCheck g_uiCheck = UiCheck::pending;
uint64_t g_uiReadAt = 0;  // present count when the readback copy was recorded
int g_uiTries = 0;
std::atomic<uint64_t> g_uiCount{0};

void OnInitList(command_list* cl) { cl->create_private_data<FgList>(); }

void OnDestroyResource(device* dev, resource r) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    for (size_t i = 0; i < g_mask.srvs.size(); ++i)
        if (g_mask.srvs[i].first == r.handle) {
            dev->destroy_resource_view(g_mask.srvs[i].second);
            g_mask.srvs.erase(g_mask.srvs.begin() + i);
            return;
        }
}
void OnDestroyList(command_list* cl) { cl->destroy_private_data<FgList>(); }
void OnResetList(command_list* cl) {
    if (FgList* l = cl->get_private_data<FgList>()) l->uiLayer = {0};
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    g_pendingInputs.erase(cl->get_native());
    g_pendingHud.erase(cl->get_native());
}

bool IsUiFormat(format f) {
    return f == format::b8g8r8a8_typeless || f == format::b8g8r8a8_unorm || f == format::r8g8b8a8_typeless ||
           f == format::r8g8b8a8_unorm;
}

void OnBindRenderTargets(command_list* cl, uint32_t count, const resource_view* rtvs, resource_view) {
    if (!g_ready || !g_set.enabled || !g_bbWidth || count == 0 || !rtvs[0].handle) return;
    FgList* l = cl->get_private_data<FgList>();
    if (!l) return;
    device* dev = cl->get_device();
    const resource r = dev->get_resource_from_view(rtvs[0]);
    if (!r.handle) return;
    const resource_desc d = dev->get_resource_desc(r);
    if (d.type == resource_type::texture_2d && d.texture.width == g_bbWidth && d.texture.height == g_bbHeight &&
        IsUiFormat(d.texture.format))
        l->uiLayer = r;
}

// Records a copy of 64 pixels of the HUD layer's top-middle row (state render target).
void ReadUiRow(command_list* cl, resource ui) {
    device* dev = cl->get_device();
    if (!g_uiReadback.handle &&
        !dev->create_resource(resource_desc(256, memory_heap::readback, resource_usage::copy_dest), nullptr,
                              resource_usage::copy_dest, &g_uiReadback)) {
        g_uiCheck = UiCheck::unusable;
        return;
    }
    const subresource_box box = {g_bbWidth / 2 - 32, g_bbHeight / 4, 0, g_bbWidth / 2 + 32, g_bbHeight / 4 + 1, 1};
    cl->barrier(ui, resource_usage::render_target, resource_usage::copy_source);
    cl->copy_texture_to_buffer(ui, 0, &box, g_uiReadback, 0, 64, 1);
    cl->barrier(ui, resource_usage::copy_source, resource_usage::render_target);
    g_uiCheck = UiCheck::reading;
    g_uiReadAt = g_presents;
}

// A few frames later (the GPU is done): alpha 0 everywhere in the row means usable.
void CheckUiRow() {
    if (g_uiCheck != UiCheck::reading || g_presents < g_uiReadAt + 4 || !g_hudlessDevice) return;
    void* data = nullptr;
    if (!g_hudlessDevice->map_buffer_region(g_uiReadback, 0, 256, map_access::read_only, &data) || !data) {
        g_uiCheck = UiCheck::unusable;
        return;
    }
    const uint8_t* px = static_cast<const uint8_t*>(data);
    int zero = 0, full = 0;
    for (int i = 0; i < 64; ++i) zero += px[i * 4 + 3] == 0, full += px[i * 4 + 3] == 255;
    g_hudlessDevice->unmap_buffer_region(g_uiReadback);
    if (zero == 64) {
        g_uiCheck = UiCheck::usable;
        Log("frame generation: HUD layer alpha is 0 where there's no HUD: used as the HUD mask");
    } else if (full == 64) {
        g_uiCheck = UiCheck::inverted;
        Log("frame generation: HUD layer alpha is 1 where there's no HUD (inverted): used as the HUD mask, flipped");
    } else if (++g_uiTries < 5) {
        g_uiCheck = UiCheck::pending;  // maybe HUD there right now: look again later
        Log("frame generation: HUD layer check %d: %d of 64 pixels transparent, %d opaque", g_uiTries, zero, full);
    } else {
        g_uiCheck = UiCheck::unusable;
        Log("frame generation: HUD layer alpha doesn't look like a UI mask (%d of 64 transparent): not used", zero);
    }
}

bool CreateMaskPipeline(device* dev) {
    if (g_mask.pipe.handle) return true;
    if (g_mask.failed) return false;
    constant_range c = {};
    c.count = 4;
    c.visibility = shader_stage::all_compute;
    descriptor_range srv = {};
    srv.count = 1;
    srv.visibility = shader_stage::all_compute;
    srv.type = descriptor_type::shader_resource_view;
    descriptor_range uav = {};
    uav.count = 1;
    uav.visibility = shader_stage::all_compute;
    uav.type = descriptor_type::unordered_access_view;
    const pipeline_layout_param params[3] = {pipeline_layout_param(c), pipeline_layout_param(srv), pipeline_layout_param(uav)};
    shader_desc cs = {};
    cs.code = kUiAlphaCs;
    cs.code_size = sizeof(kUiAlphaCs);
    pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
    const resource_desc d(g_bbWidth, g_bbHeight, 1, 1, format::r8_unorm, 1, memory_heap::default_,
                          resource_usage::shader_resource | resource_usage::unordered_access);
    if (!dev->create_pipeline_layout(3, params, &g_mask.layout) || !dev->create_pipeline(g_mask.layout, 1, &sub, &g_mask.pipe) ||
        !dev->create_resource(d, nullptr, resource_usage::shader_resource, &g_mask.tex) ||
        !dev->create_resource_view(g_mask.tex, resource_usage::unordered_access, resource_view_desc(format::r8_unorm), &g_mask.uav)) {
        g_mask.failed = true;
        Log("frame generation: couldn't create the HUD mask shader");
        return false;
    }
    return true;
}

format UnormOf(format f) {
    return f == format::b8g8r8a8_typeless ? format::b8g8r8a8_unorm : f == format::r8g8b8a8_typeless ? format::r8g8b8a8_unorm : f;
}

// Records the mask from the HUD layer (state render target) and puts the game's state back.
bool RecordMask(command_list* cl, resource ui) {
    device* dev = cl->get_device();
    if (!CreateMaskPipeline(dev)) return false;
    resource_view srv = {0};
    for (const auto& [h, v] : g_mask.srvs)
        if (h == ui.handle) srv = v;
    if (!srv.handle) {
        const format f = UnormOf(dev->get_resource_desc(ui).texture.format);
        if (g_mask.srvs.size() >= 16 || !dev->create_resource_view(ui, resource_usage::shader_resource, resource_view_desc(f), &srv))
            return false;
        g_mask.srvs.push_back({ui.handle, srv});
    }
    const uint32_t params[4] = {g_bbWidth, g_bbHeight, g_uiCheck == UiCheck::inverted ? 1u : 0u, 0};
    cl->barrier(ui, resource_usage::render_target, resource_usage::shader_resource);
    cl->barrier(g_mask.tex, resource_usage::shader_resource, resource_usage::unordered_access);
    cl->bind_pipeline(pipeline_stage::all_compute, g_mask.pipe);
    cl->push_constants(shader_stage::all_compute, g_mask.layout, 0, 0, 4, params);
    descriptor_table_update u = {};
    u.count = 1;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = &srv;
    cl->push_descriptors(shader_stage::all_compute, g_mask.layout, 1, u);
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = &g_mask.uav;
    cl->push_descriptors(shader_stage::all_compute, g_mask.layout, 2, u);
    cl->dispatch((g_bbWidth + 7) / 8, (g_bbHeight + 7) / 8, 1);
    cl->barrier(g_mask.tex, resource_usage::unordered_access, resource_usage::shader_resource);
    cl->barrier(ui, resource_usage::shader_resource, resource_usage::render_target);
    RestoreListState(cl);
    return true;
}

// UE's HDR output draws the HUD into separate layers, copies the finished scene out of the back
// buffer and then blends the HUD over that copy. The copy's source is the picture before the
// HUD: it's copied into our own texture and handed to frame generation, so the HUD isn't warped
// with the scene. (Without that copy, as in SDR, frame generation runs without it.)
bool OnCopyResource(command_list* cl, resource src, resource dst) {
    if (!g_ready || !g_set.enabled || !g_bbWidth) return false;
    device* dev = cl->get_device();
    const resource_desc sd = dev->get_resource_desc(src);
    if (sd.type != resource_type::texture_2d || sd.texture.width != g_bbWidth || sd.texture.height != g_bbHeight ||
        sd.texture.format != g_bbFormat)
        return false;
    const resource_desc dd = dev->get_resource_desc(dst);
    if (dd.texture.width != g_bbWidth || dd.texture.height != g_bbHeight || dd.texture.format != g_bbFormat) return false;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_token) return false;
    if (!g_hudlessTex.handle) {
        const resource_desc d(g_bbWidth, g_bbHeight, 1, 1, g_bbFormat, 1, memory_heap::default_,
                              resource_usage::shader_resource | resource_usage::copy_dest);
        if (!dev->create_resource(d, nullptr, resource_usage::shader_resource, &g_hudlessTex)) {
            Log("frame generation: couldn't create the %ux%u pre-HUD texture", g_bbWidth, g_bbHeight);
            g_bbWidth = 0;  // don't try again
            return false;
        }
        g_hudlessDevice = dev;
        Log("frame generation: pre-HUD picture found (%ux%u, format %u)", g_bbWidth, g_bbHeight,
            static_cast<uint32_t>(g_bbFormat));
    }
    // The game has the back buffer in the copy-source state for its own copy, recorded after ours.
    cl->barrier(g_hudlessTex, resource_usage::shader_resource, resource_usage::copy_dest);
    cl->copy_resource(src, g_hudlessTex);
    cl->barrier(g_hudlessTex, resource_usage::copy_dest, resource_usage::shader_resource);
    PendingHud& pending = g_pendingHud[cl->get_native()];  // tagged when the list is submitted
    pending.mask = false;
    FgList* l = cl->get_private_data<FgList>();
    const resource ui = l ? l->uiLayer : resource{0};
    if (ui.handle) {
        if (g_uiCheck == UiCheck::pending && g_presents > 120) ReadUiRow(cl, ui);  // past the first frames
        if ((g_uiCheck == UiCheck::usable || g_uiCheck == UiCheck::inverted) && RecordMask(cl, ui)) pending.mask = true;
    }
    return false;  // the game's copy goes ahead
}

void OnInitSwapchain(swapchain* sc, bool) {
    const resource bb = sc->get_back_buffer(0);
    if (!bb.handle) return;
    const resource_desc d = sc->get_device()->get_resource_desc(bb);
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_hudlessTex.handle && (d.texture.width != g_bbWidth || d.texture.height != g_bbHeight || d.texture.format != g_bbFormat)) {
        g_hudlessDevice->destroy_resource(g_hudlessTex);  // resize: the game waits for the GPU first
        g_hudlessTex = {0};
        DestroyMask(g_hudlessDevice);
    }
    g_bbWidth = d.texture.width, g_bbHeight = d.texture.height, g_bbFormat = d.texture.format;
}

void OnDestroySwapchain(swapchain*, bool) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    g_bbWidth = 0;  // no pre-HUD copies until the swapchain is back
}

}  // namespace

std::string FrameGenDiag() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_ready) return "";
    char buf[160];
    snprintf(buf, sizeof(buf), " FG: screen %ux%u, tags set %u, frames untagged %u, pending %zu, %s", g_bbWidth, g_bbHeight,
             g_tagsSet, g_framesUntagged, g_pendingInputs.size(), g_fgOn ? "on" : "paused");
    std::string s = buf;
    for (const auto& [why, n] : g_tagWhy) {
        snprintf(buf, sizeof(buf), " {tag: %s %u}", why.c_str(), n);
        s += buf;
    }
    g_tagWhy.clear(), g_tagsSet = 0, g_framesUntagged = 0;
    return s;
}

bool FrameGenIsScreen(unsigned width, unsigned height) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_bbWidth && width == g_bbWidth && height == g_bbHeight;
}

void FrameGenSetCamera(const float* view, int clipToPrevClipOffset) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_ready || clipToPrevClipOffset < 0) return;
    if (!g_cameraChecked) g_cameraChecked = true, g_cameraOk = CheckCamera(view);
    if (!g_cameraOk) return;
    memcpy(g_camera.m[0], view + kTranslatedWorldToClip / 4, 64);
    memcpy(g_camera.m[1], view + kRelativeWorldToClip / 4, 64);
    memcpy(g_camera.m[2], view + kViewToTranslatedWorld / 4, 64);
    memcpy(g_camera.m[3], view + kViewToClipNoAA / 4, 64);
    memcpy(g_camera.m[4], view + clipToPrevClipOffset / 4, 64);
    g_camera.valid = true;
}

void FrameGenTag(ID3D12GraphicsCommandList* list, ID3D12Resource* depth, ID3D12Resource* motion, unsigned rw,
                 unsigned rh, float jitterX, float jitterY, bool reset) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_ready || !g_set.enabled || !g_token) return;
    if (!g_camera.valid) return void(++g_tagWhy["no camera"]);
    const Camera& cam = g_camera;
    sl::Constants c{};
    float inv[16], prevToCur[16], m[16];
    ToSl(c.cameraViewToClip, cam.m[3]);
    if (!Invert(cam.m[3], inv) || !Invert(cam.m[4], prevToCur)) return void(++g_tagWhy["matrix"]);
    ToSl(c.clipToCameraView, inv);
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    ToSl(c.clipToLensClip, identity);
    ToSl(c.clipToPrevClip, cam.m[4]);
    ToSl(c.prevClipToClip, prevToCur);
    c.jitterOffset = sl::float2(jitterX, jitterY);
    c.mvecScale = sl::float2(1.0f / rw, 1.0f / rh);  // our motion vectors are in render pixels
    c.cameraPinholeOffset = sl::float2(0, 0);
    // Camera position in UE's relative world (the world tile): RelativeWorldToClip =
    // T(tile - camera) * TranslatedWorldToClip.
    float invTw[16];
    if (Invert(cam.m[0], invTw)) {
        Multiply(cam.m[1], invTw, m);
        c.cameraPos = sl::float3(-m[12], -m[13], -m[14]);
    } else {
        c.cameraPos = sl::float3(0, 0, 0);
    }
    const float* axes = cam.m[2];  // view x right, y up, z forward
    c.cameraRight = sl::float3(axes[0], axes[1], axes[2]);
    c.cameraUp = sl::float3(axes[4], axes[5], axes[6]);
    c.cameraFwd = sl::float3(axes[8], axes[9], axes[10]);
    const float* p = cam.m[3];
    c.cameraNear = p[14];
    c.cameraFar = 1e6f;  // UE's projection has no far plane
    c.cameraFOV = 2 * std::atan(1 / p[5]);
    c.cameraAspectRatio = p[5] / p[0];
    c.depthInverted = sl::Boolean::eTrue;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    c.minRelativeLinearDepthObjectSeparation = g_set.separation;
    // UE reuses its depth buffer later in the frame: copy it (our copy stays valid until the Present).
    ID3D12Resource* copy = DepthCopy(depth);
    if (!copy) return void(++g_tagWhy["depth copy"]);
    if (g_pendingInputs.count(reinterpret_cast<uint64_t>(list))) ++g_tagWhy["two in one list"];
    const D3D12_RESOURCE_BARRIER before[2] = {Transition(depth, kDepthReadState, D3D12_RESOURCE_STATE_COPY_SOURCE),
                                              Transition(copy, kDepthReadState, D3D12_RESOURCE_STATE_COPY_DEST)};
    const D3D12_RESOURCE_BARRIER after[2] = {Transition(depth, D3D12_RESOURCE_STATE_COPY_SOURCE, kDepthReadState),
                                             Transition(copy, D3D12_RESOURCE_STATE_COPY_DEST, kDepthReadState)};
    list->ResourceBarrier(2, before);
    list->CopyResource(copy, depth);
    list->ResourceBarrier(2, after);
    PendingInputs& pending = g_pendingInputs[reinterpret_cast<uint64_t>(list)];  // tagged when the list is submitted
    pending.constants = c;
    pending.depth = copy, pending.motion = motion, pending.rw = rw, pending.rh = rh;
}

namespace {

// A command list was submitted: what it prepared belongs to the frame the next Present shows.
void OnExecute(command_queue*, command_list* cl) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_pendingInputs.empty() && g_pendingHud.empty()) return;
    const uint64_t key = cl->get_native();
    auto in = g_pendingInputs.find(key);
    auto hud = g_pendingHud.find(key);
    if (in == g_pendingInputs.end() && hud == g_pendingHud.end()) return;
    if (!g_ready || !g_token) {
        if (in != g_pendingInputs.end()) g_pendingInputs.erase(in);
        if (hud != g_pendingHud.end()) g_pendingHud.erase(hud);
        return;
    }
    if (in != g_pendingInputs.end()) {
        const PendingInputs p = in->second;
        g_pendingInputs.erase(in);
        // Two frames' lists before one Present (a hitch): Streamline takes one set per frame and
        // refuses different constants for the same frame, so the first one stays.
        if (g_inputs.Has(g_frame)) {
            ++g_tagWhy["second set in a frame, skipped"];
        } else {
            sl::Result r = g_sl.setConstants(p.constants, *g_token, sl::ViewportHandle(0));
            if (r != sl::Result::eOk) {
                static int logged = 0;
                if (logged++ < 10) Log("frame generation: slSetConstants failed: %d", static_cast<int>(r));
            } else {
                sl::Resource depthRes(sl::ResourceType::eTex2d, p.depth, nullptr, nullptr, kDepthReadState);
                sl::Resource motionRes(sl::ResourceType::eTex2d, p.motion, nullptr, nullptr,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                const sl::Extent extent{0, 0, p.rw, p.rh};
                // The back buffer itself comes from the swapchain; only its extent (all of it) is passed.
                const sl::Extent bbExtent{0, 0, g_bbWidth, g_bbHeight};
                const sl::ResourceTag tags[3] = {
                    sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &extent),
                    sl::ResourceTag(&motionRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &extent),
                    sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent, &bbExtent)};
                r = g_sl.setTagForFrame(*g_token, sl::ViewportHandle(0), tags, g_bbWidth ? 3 : 2, nullptr);
                if (r != sl::Result::eOk) {
                    static int logged = 0;
                    if (logged++ < 10) Log("frame generation: slSetTagForFrame failed: %d", static_cast<int>(r));
                } else {
                    g_inputs.Set(g_frame);
                    ++g_tagsSet;
                    if (++g_tagged == 1) Log("frame generation: first frame tagged at submission (%ux%u)", p.rw, p.rh);
                }
            }
        }
    }
    if (hud != g_pendingHud.end()) {
        const PendingHud h = hud->second;
        g_pendingHud.erase(hud);
        const sl::Extent extent{0, 0, g_bbWidth, g_bbHeight};
        sl::Resource res(sl::ResourceType::eTex2d, reinterpret_cast<void*>(g_hudlessTex.handle), nullptr, nullptr,
                         D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        sl::Resource maskRes(sl::ResourceType::eTex2d, reinterpret_cast<void*>(g_mask.tex.handle), nullptr, nullptr,
                             D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const sl::ResourceTag tags[2] = {
            sl::ResourceTag(&res, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent, &extent),
            sl::ResourceTag(&maskRes, sl::kBufferTypeUIAlpha, sl::ResourceLifecycle::eValidUntilPresent, &extent)};
        const uint32_t n = h.mask && g_mask.tex.handle ? 2 : 1;
        if (g_hudlessTex.handle &&
            g_sl.setTagForFrame(*g_token, sl::ViewportHandle(0), tags, n, nullptr) == sl::Result::eOk) {
            g_hudless.Set(g_frame);
            ++g_hudlessCount;
            if (n == 2) ++g_uiCount;
        } else {
            static int logged = 0;
            if (logged++ < 10) Log("frame generation: tagging the pre-HUD picture failed");
        }
    }
}

}  // namespace

void RegisterFrameGen() {
    if (!reshade::get_config_value(nullptr, RD_PRODUCT, "FrameGenMode", g_set.mode)) {
        bool old = false;  // 0.3.x: an on/off switch
        reshade::get_config_value(nullptr, RD_PRODUCT, "FrameGeneration", old);
        g_set.mode = old ? k2x : kOff;
    }
    if (g_set.mode < kOff || g_set.mode >= kModeCount) g_set.mode = kOff;
    reshade::get_config_value(nullptr, RD_PRODUCT, "ObjectSeparation", g_set.separation);
    if (!(g_set.separation >= 1 && g_set.separation <= 100)) g_set.separation = 40;
    g_set.enabled = g_set.mode != kOff && kTsrMapped;  // needs DLSS, which needs TSR mapped
    reshade::register_event<reshade::addon_event::init_device>(&OnInitDevice);
    reshade::register_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::register_event<reshade::addon_event::init_swapchain>(&OnInitSwapchain);
    reshade::register_event<reshade::addon_event::destroy_swapchain>(&OnDestroySwapchain);
    reshade::register_event<reshade::addon_event::copy_resource>(&OnCopyResource);
    reshade::register_event<reshade::addon_event::execute_command_list>(&OnExecute);
    reshade::register_event<reshade::addon_event::init_command_list>(&OnInitList);
    reshade::register_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::register_event<reshade::addon_event::destroy_command_list>(&OnDestroyList);
    reshade::register_event<reshade::addon_event::reset_command_list>(&OnResetList);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&OnBindRenderTargets);
}

void UnregisterFrameGen() {
    reshade::unregister_event<reshade::addon_event::init_device>(&OnInitDevice);
    reshade::unregister_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(&OnInitSwapchain);
    reshade::unregister_event<reshade::addon_event::destroy_swapchain>(&OnDestroySwapchain);
    reshade::unregister_event<reshade::addon_event::copy_resource>(&OnCopyResource);
    reshade::unregister_event<reshade::addon_event::execute_command_list>(&OnExecute);
    reshade::unregister_event<reshade::addon_event::init_command_list>(&OnInitList);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::unregister_event<reshade::addon_event::destroy_command_list>(&OnDestroyList);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(&OnResetList);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&OnBindRenderTargets);
    Unpatch();  // this add-on's code is about to go away
}

void DrawFrameGenSettings() {
    ImGui::SeparatorText("Frame generation");
    int mode = g_set.mode;
    ImGui::SetNextItemWidth(ControlWidth());
    ImGui::BeginDisabled(!kTsrMapped);
    const bool picked = ImGui::Combo("Mode##framegen", &mode, kModeNames, kModeCount);
    ImGui::EndDisabled();
    if (picked) {
        g_set.mode = mode;
        g_set.enabled = mode != kOff;
        reshade::set_config_value(nullptr, RD_PRODUCT, "FrameGenMode", mode);
        Log("settings: frame generation %s", kModeNames[mode]);
    }
    Help("NVIDIA DLSS Frame Generation: 2x, 3x and 4x add 1, 2 or 3 generated frames per rendered frame. Dynamic "
         "picks the multiplier to reach the display's refresh rate.\n\nNeeds DLSS on, and runs in game only (not "
         "in menus). Switching on from Off takes a game restart; the multipliers switch right away.");

    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (g_set.enabled && !g_ready && !g_initTried && !g_factoryHooked)
            ImGui::TextColored(kWarn, "Restart the game to start frame generation.");
        if (g_capsKnown && EffectiveMode(g_set.mode) != g_set.mode)
            ImGui::TextColored(kWarn, "%s isn't supported here, so %s.", kModeNames[g_set.mode],
                               kModeNames[EffectiveMode(g_set.mode)]);
    }

    float sep = g_set.separation;
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::SliderFloat("Object separation", &sep, 1, 40, "%.0f")) g_set.separation = sep;
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        reshade::set_config_value(nullptr, RD_PRODUCT, "ObjectSeparation", g_set.separation);
        Log("settings: object separation %.0f", g_set.separation);
    }
    Help("How far apart in depth two things must be for frame generation to move them separately. 40 (NVIDIA's "
         "default) is about 4 m, 10 about 1 m. Lower it if people moving in front of walls ghost.");
}

std::string FrameGenHeadline(bool* warn) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    *warn = false;
    if (!g_set.enabled) return "Frame generation off";
    if (!g_ready) {
        *warn = true;
        return g_initTried || g_factoryHooked ? "Frame generation: " + g_status
                                              : std::string("Frame generation: restart the game to start it");
    }
    if (!g_upgraded) return (*warn = true), "Frame generation not running: the game's swapchain wasn't handed over";
    if (!g_fgOn) return "Frame generation paused (DLSS off, menu or loading)";
    char buf[96];
    snprintf(buf, sizeof(buf), "Frame generation %s: %u frames per rendered frame",
             g_appliedMode >= 0 && g_appliedMode < kModeCount ? kModeNames[g_appliedMode] : "on",
             g_state.numFramesActuallyPresented);
    return buf;
}

void DrawFrameGenStatus() {
    ImGui::SeparatorText("Frame generation");
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_ready) {
        ImGui::TextDisabled("Streamline: %s", g_status.c_str());
        return;
    }
    const unsigned st = static_cast<unsigned>(g_state.status);
    ImGui::TextDisabled("Streamline 2.14: %s; up to %u generated frames, Dynamic %s", g_fgOn ? kModeNames[g_appliedMode] : "paused",
                        g_maxFrames, g_dynamicSupported ? "supported" : "not supported");
    if (st)
        ImGui::TextColored(kWarn, "DLSS-G status 0x%x%s%s%s%s%s", st, (st & 1) ? " resolution too low" : "",
                           (st & 2) ? " Reflex not detected" : "", (st & 4) ? " HDR format not supported" : "",
                           (st & 8) ? " constants invalid" : "", (st & 16) ? " back buffer index not queried" : "");
    ImGui::TextDisabled("Game frames %llu: with depth and motion %llu, with the picture before the HUD %llu",
                        static_cast<unsigned long long>(g_presents.load()), static_cast<unsigned long long>(g_tagged.load()),
                        static_cast<unsigned long long>(g_hudlessCount.load()));
    ImGui::TextDisabled("HUD mask: %s (%llu frames)",
                        g_uiCheck == UiCheck::usable     ? "used"
                        : g_uiCheck == UiCheck::inverted ? "used (the game's alpha, flipped)"
                        : g_uiCheck == UiCheck::unusable ? "not usable"
                                                         : "checking",
                        static_cast<unsigned long long>(g_uiCount.load()));
}

}  // namespace rd
