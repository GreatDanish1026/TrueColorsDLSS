// AMD FSR upscaling through the FidelityFX API. The signed runtime (amd_fidelityfx_loader_dx12.dll
// and amd_fidelityfx_upscaler_dx12.dll, FSR SDK 2.4.0) sits next to the game exe. The runtime picks
// the upscaler: FSR 4 (machine learning) on Radeon GPUs that support it, FSR 3.1 on the others.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.hpp"
#include "fsr.hpp"

#ifndef _WINDOWS
#define _WINDOWS
#endif
#include <ffx_api.h>
#include <ffx_api_loader.h>
#include <dx12/ffx_api_dx12.h>
#include <ffx_upscale.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace fsr {
namespace {

std::mutex g_mutex;
ID3D12Device* g_device = nullptr;
HMODULE g_module = nullptr;
ffxFunctions g_ffx = {};
bool g_initTried = false, g_ready = false;
std::string g_initError, g_status = "not started", g_version;

// One context (and so one history) per output size, like the DLSS features.
struct Context {
    ffxContext ctx = nullptr;
    unsigned maxRenderW = 0, maxRenderH = 0, outW = 0, outH = 0;
    unsigned renderW = 0, renderH = 0;  // last dispatch's
    uint64_t lastFrame = 0;
    LARGE_INTEGER lastEval = {};  // for the frame time
    float tuning[3] = {-99, -99, -99};  // last applied velocity factor, min disocclusion, shading change
    bool debug = false;  // created with FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION
};

std::string Code(ffxReturnCode_t r);

// Applies the tuning values to the context when they changed (FSR 3.1; FSR 4 doesn't take them).
void ApplyTuning(Context& c, const Inputs& in) {
    const float want[3] = {in.velocityFactor, in.minDisocclusion, in.shadingChangeScale};
    const uint64_t keys[3] = {FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR,
                              FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION,
                              FFX_API_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE};
    if (!g_ffx.Configure) return;
    for (int i = 0; i < 3; ++i) {
        if (c.tuning[i] == want[i]) continue;
        float value = want[i];
        ffxConfigureDescUpscaleKeyValue kv = {};
        kv.header.type = FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE;
        kv.key = keys[i];
        kv.ptr = &value;  // read during the call
        const ffxReturnCode_t r = g_ffx.Configure(&c.ctx, &kv.header);
        c.tuning[i] = want[i];
        static int logged = 0;
        if (logged < 12) {
            ++logged;
            rd::Log("FSR tuning: key %llu = %.3f (%s)", static_cast<unsigned long long>(keys[i]), value, Code(r).c_str());
        }
    }
}
constexpr size_t kMaxContexts = 4;
constexpr uint64_t kStaleFrames = 3;
uint64_t SizeKey(unsigned w, unsigned h) { return (static_cast<uint64_t>(w) << 32) | h; }
std::map<uint64_t, Context> g_contexts;

// A replaced context may still be used by command lists the GPU hasn't run yet.
struct Retired {
    ffxContext ctx;
    uint64_t frame;
};
std::vector<Retired> g_retired;
uint64_t g_frame = 0;
constexpr uint64_t kRetireFrames = 8;

// Frame time for FSR, measured between a view's evaluations (not presents: with frame generation
// every rendered frame is presented twice or more).
LARGE_INTEGER g_freq = {};

float FrameMs(LARGE_INTEGER& last) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    float ms = 16.7f;
    if (g_freq.QuadPart && last.QuadPart) {
        const float d = static_cast<float>(now.QuadPart - last.QuadPart) * 1000.0f / g_freq.QuadPart;
        if (d > 0.5f && d < 200) ms = d;
    }
    last = now;
    return ms;
}

void Message(uint32_t type, const wchar_t* message) {
    static int count = 0;
    if (count >= 50) return;
    ++count;
    std::wstring m = message ? message : L"";
    while (!m.empty() && (m.back() == L'\n' || m.back() == L'\r')) m.pop_back();
    rd::Log("FSR %s: %ls%s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", m.c_str(),
            count == 50 ? " (last FSR message logged)" : "");
}

std::string Code(ffxReturnCode_t r) {
    switch (r) {
        case FFX_API_RETURN_OK: return "ok";
        case FFX_API_RETURN_ERROR: return "error";
        case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "unknown description type";
        case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "runtime error";
        case FFX_API_RETURN_NO_PROVIDER: return "no provider (GPU or driver not supported)";
        case FFX_API_RETURN_ERROR_MEMORY: return "out of memory";
        case FFX_API_RETURN_ERROR_PARAMETER: return "bad parameter";
        case FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "runtime too old";
        default: return "code " + std::to_string(r);
    }
}

void Destroy(ffxContext ctx) {
    if (ctx) g_ffx.DestroyContext(&ctx, nullptr);
}

bool CreateContext(const Inputs& in, Context& c, std::string& error) {
    if (c.ctx) g_retired.push_back({c.ctx, g_frame});
    c.ctx = nullptr;
    // Room for the largest render size: mode changes then keep the context (and its history start).
    const unsigned maxW = in.renderWidth > in.outputWidth ? in.renderWidth : in.outputWidth;
    const unsigned maxH = in.renderHeight > in.outputHeight ? in.renderHeight : in.outputHeight;

    ffxCreateContextDescUpscale desc = {};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                 FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
                 (in.debugView ? FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION : 0u);
    desc.maxRenderSize = {maxW, maxH};
    desc.maxUpscaleSize = {in.outputWidth, in.outputHeight};
    desc.fpMessage = &Message;
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = g_device;
    ffxCreateContextDescUpscaleVersion version = {};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    version.version = FFX_UPSCALER_VERSION;
    desc.header.pNext = &backend.header;
    backend.header.pNext = &version.header;

    const ffxReturnCode_t r = g_ffx.CreateContext(&c.ctx, &desc.header, nullptr);
    if (r != FFX_API_RETURN_OK) {
        c.ctx = nullptr;
        error = "FSR context creation failed (" + Code(r) + ")";
        return false;
    }
    c.maxRenderW = maxW, c.maxRenderH = maxH, c.outW = in.outputWidth, c.outH = in.outputHeight;
    c.tuning[0] = c.tuning[1] = c.tuning[2] = -99;  // a new context has AMD's defaults
    c.debug = in.debugView;

    ffxQueryGetProviderVersion pv = {};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (g_ffx.Query(&c.ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName) {
        g_version = pv.versionName;
        if (g_version.rfind("FSR", 0) != 0) g_version = "FSR " + g_version;  // "FSR 4.1.2"
    }
    rd::Log("FSR context created: up to %ux%u -> %ux%u, upscaler %s (%zu output sizes)", maxW, maxH, in.outputWidth,
            in.outputHeight, g_version.empty() ? "?" : g_version.c_str(), g_contexts.size());
    return true;
}

FfxApiResource Res(ID3D12Resource* r, uint32_t state) { return ffxApiGetResourceDX12(r, state); }

}  // namespace

bool Init(ID3D12Device* device, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initTried) {
        error = g_initError;
        return g_ready;
    }
    g_initTried = true;
    g_device = device;
    QueryPerformanceFrequency(&g_freq);

    const std::wstring path = rd::ExeDir() + L"amd_fidelityfx_loader_dx12.dll";
    g_module = LoadLibraryW(path.c_str());
    if (!g_module) {
        g_initError = "FSR not available: is amd_fidelityfx_loader_dx12.dll next to the game exe?";
        g_status = g_initError;
        rd::Log("%s (error %lu)", g_initError.c_str(), GetLastError());
        error = g_initError;
        return false;
    }
    ffxLoadFunctions(&g_ffx, g_module);
    if (!g_ffx.CreateContext || !g_ffx.DestroyContext || !g_ffx.Query || !g_ffx.Dispatch) {
        g_initError = "FSR runtime (amd_fidelityfx_loader_dx12.dll) is incomplete";
        g_status = g_initError;
        rd::Log("%s", g_initError.c_str());
        FreeLibrary(g_module);
        g_module = nullptr;
        error = g_initError;
        return false;
    }

    // The upscaler versions this GPU and driver offer (newest first), for the log.
    uint64_t count = 8;
    uint64_t ids[8] = {};
    const char* names[8] = {};
    ffxQueryDescGetVersions v = {};
    v.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    v.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    v.device = device;
    v.outputCount = &count;
    v.versionIds = ids;
    v.versionNames = names;
    const ffxReturnCode_t r = g_ffx.Query(nullptr, &v.header);
    if (r != FFX_API_RETURN_OK || count == 0) {
        g_initError = "FSR upscaler not available (" + Code(r) + "): is amd_fidelityfx_upscaler_dx12.dll next to the game exe?";
        g_status = g_initError;
        rd::Log("%s", g_initError.c_str());
        error = g_initError;
        return false;
    }
    std::string list;
    for (uint64_t i = 0; i < count && i < 8; ++i) list += std::string(i ? ", " : "") + (names[i] ? names[i] : "?");
    rd::Log("FSR runtime loaded, upscalers offered: %s", list.c_str());
    g_ready = true;
    g_status = "ready";
    return true;
}

bool Evaluate(ID3D12GraphicsCommandList* list, const Inputs& in, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready) {
        error = g_initError.empty() ? "FSR not started" : g_initError;
        return false;
    }
    const uint64_t key = SizeKey(in.outputWidth, in.outputHeight);
    auto it = g_contexts.find(key);
    if (it == g_contexts.end()) {
        for (auto s = g_contexts.begin(); s != g_contexts.end();) {  // outputs the game stopped using
            if (s->second.lastFrame + kStaleFrames < g_frame) {
                if (s->second.ctx) g_retired.push_back({s->second.ctx, g_frame});
                rd::Log("FSR context for %ux%u output released", s->second.outW, s->second.outH);
                s = g_contexts.erase(s);
            } else {
                ++s;
            }
        }
        if (g_contexts.size() >= kMaxContexts) {
            error = "too many TSR views";
            return false;
        }
        it = g_contexts.emplace(key, Context()).first;
    }
    Context& c = it->second;
    c.lastFrame = g_frame;
    const bool fresh = !c.ctx || in.renderWidth > c.maxRenderW || in.renderHeight > c.maxRenderH || in.debugView != c.debug;
    if (fresh && !CreateContext(in, c, error)) {
        g_status = error;
        return false;
    }
    c.renderW = in.renderWidth, c.renderH = in.renderHeight;
    ApplyTuning(c, in);

    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = list;
    // States: FFX adds a barrier only when the given state lacks what it needs, and restores the
    // given state afterwards. UE's depth is in depth read + shader resource, which already reads.
    d.color = Res(in.color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Res(in.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = Res(in.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.exposure = Res(nullptr, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.reactive = Res(in.reactive, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.transparencyAndComposition = Res(nullptr, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Res(in.output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {in.jitterX, in.jitterY};
    d.motionVectorScale = {in.motionScaleX, in.motionScaleY};
    d.renderSize = {in.renderWidth, in.renderHeight};
    d.upscaleSize = {in.outputWidth, in.outputHeight};
    d.enableSharpening = in.sharpness > 0.001f;
    d.sharpness = in.sharpness;
    d.frameTimeDelta = FrameMs(c.lastEval);
    d.preExposure = 1.0f;
    d.reset = in.reset || fresh;
    d.cameraNear = in.cameraNear;
    d.cameraFar = 1e8f;  // UE's far plane is at infinity (FFX_UPSCALE_ENABLE_DEPTH_INFINITE)
    d.cameraFovAngleVertical = in.cameraFovY;
    d.viewSpaceToMetersFactor = 0.01f;  // UE units are centimetres
    d.flags = in.debugView ? FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW : 0u;
    const ffxReturnCode_t r = g_ffx.Dispatch(&c.ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        error = "FSR dispatch failed (" + Code(r) + ")";
        g_status = error;
        return false;
    }
    char buf[192];
    snprintf(buf, sizeof(buf), "running: %s, %ux%u -> %ux%u, %zu view%s", g_version.empty() ? "?" : g_version.c_str(),
             in.renderWidth, in.renderHeight, in.outputWidth, in.outputHeight, g_contexts.size(),
             g_contexts.size() == 1 ? "" : "s");
    g_status = buf;
    return true;
}

void EndFrame() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_frame;
    for (size_t i = 0; i < g_retired.size();) {
        if (g_frame - g_retired[i].frame < kRetireFrames) {
            ++i;
            continue;
        }
        Destroy(g_retired[i].ctx);
        g_retired.erase(g_retired.begin() + i);
    }
}

void ReleaseContexts() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& [key, c] : g_contexts)
        if (c.ctx) g_retired.push_back({c.ctx, g_frame});
    g_contexts.clear();
}

void Shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ready) {
        for (auto& [key, c] : g_contexts) Destroy(c.ctx);
        for (Retired& r : g_retired) Destroy(r.ctx);
    }
    g_contexts.clear();
    g_retired.clear();
    if (g_module) FreeLibrary(g_module);
    g_module = nullptr;
    g_ffx = {};
    g_ready = false;
    g_initTried = false;
    g_device = nullptr;
}

std::string Status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_status;
}

std::string VersionName() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_version.empty() ? "FSR" : g_version;
}

}  // namespace fsr
