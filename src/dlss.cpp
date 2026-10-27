// NVIDIA DLSS Super Resolution through NGX. This software contains source code provided by
// NVIDIA Corporation (the NGX helper headers in third_party/dlss).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.hpp"
#include "dlss.hpp"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_d3d.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace dlss {
namespace {

// Project id for NVSDK_NGX_ENGINE_TYPE_CUSTOM (any GUID identifying this application).
constexpr const char* kProjectId = "6f1d6e2a-5b8c-4f2e-9a43-7c2b8e1d4a90";

std::mutex g_mutex;
ID3D12Device* g_device = nullptr;
bool g_initTried = false, g_ready = false;
std::string g_initError;
NVSDK_NGX_Parameter* g_params = nullptr;
std::string g_status = "not started";

// One DLSS feature (and so one history) per output: the game renders more than one TSR view.
struct Feature {
    NVSDK_NGX_Handle* handle = nullptr;
    unsigned renderW = 0, renderH = 0, outW = 0, outH = 0;
    int preset = -1;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    uint64_t lastFrame = 0;  // last frame it was evaluated in
};
// One feature per output size. Double Exposure starts at 3840x2160 and switches to its 4096x2160
// setting a second later, so features of sizes no longer used are released (after kStaleFrames).
constexpr size_t kMaxFeatures = 4;
constexpr uint64_t kStaleFrames = 3;
// Keyed by output size, not by the output texture: TSR's output comes from UE's render target pool
// and the game swaps it now and then; a feature per texture restarted DLSS (history) each time.
uint64_t SizeKey(unsigned w, unsigned h) { return (static_cast<uint64_t>(w) << 32) | h; }
std::map<uint64_t, Feature> g_features;

// A replaced feature may still be used by command lists the GPU hasn't run yet, so it's released
// only some frames later.
struct Retired {
    NVSDK_NGX_Handle* handle;
    uint64_t frame;
};
std::vector<Retired> g_retired;
uint64_t g_frame = 0;
constexpr uint64_t kRetireFrames = 8;

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string m = message ? message : "";
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    rd::Log("NGX: %s", m.c_str());
}

std::string ResultText(NVSDK_NGX_Result r) {
    char buf[64];
    snprintf(buf, sizeof(buf), "0x%08x", static_cast<unsigned>(r));
    return buf;
}

const char* QualityName(NVSDK_NGX_PerfQuality_Value q) {
    switch (q) {
        case NVSDK_NGX_PerfQuality_Value_DLAA: return "DLAA";
        case NVSDK_NGX_PerfQuality_Value_UltraQuality: return "Ultra Quality";
        case NVSDK_NGX_PerfQuality_Value_MaxQuality: return "Quality";
        case NVSDK_NGX_PerfQuality_Value_Balanced: return "Balanced";
        case NVSDK_NGX_PerfQuality_Value_MaxPerf: return "Performance";
        case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return "Ultra Performance";
        default: return "?";
    }
}

// The DLSS quality mode whose supported render size range holds the game's render size.
NVSDK_NGX_PerfQuality_Value PickQuality(unsigned renderW, unsigned renderH, unsigned outW, unsigned outH) {
    if (renderW >= outW && renderH >= outH) return NVSDK_NGX_PerfQuality_Value_DLAA;
    const NVSDK_NGX_PerfQuality_Value order[] = {
        NVSDK_NGX_PerfQuality_Value_DLAA,    NVSDK_NGX_PerfQuality_Value_UltraQuality, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,      NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    for (NVSDK_NGX_PerfQuality_Value q : order) {
        unsigned optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
        float sharp = 0;
        if (NVSDK_NGX_FAILED(NGX_DLSS_GET_OPTIMAL_SETTINGS(g_params, outW, outH, q, &optW, &optH, &maxW, &maxH, &minW, &minH,
                                                           &sharp)) ||
            !optW)
            continue;
        if (renderW >= minW && renderW <= maxW && renderH >= minH && renderH <= maxH) return q;
    }
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

bool CreateFeature(ID3D12GraphicsCommandList* list, const Inputs& in, Feature& f, std::string& error) {
    if (f.handle) g_retired.push_back({f.handle, g_frame});
    f.handle = nullptr;
    const NVSDK_NGX_PerfQuality_Value q = PickQuality(in.renderWidth, in.renderHeight, in.outputWidth, in.outputHeight);
    const char* presetKeys[] = {
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality};
    for (const char* k : presetKeys) NVSDK_NGX_Parameter_SetUI(g_params, k, static_cast<unsigned>(in.preset));

    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = in.renderWidth;
    cp.Feature.InHeight = in.renderHeight;
    cp.Feature.InTargetWidth = in.outputWidth;
    cp.Feature.InTargetHeight = in.outputHeight;
    cp.Feature.InPerfQualityValue = q;
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                              NVSDK_NGX_DLSS_Feature_Flags_DepthInverted | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(list, 1, 1, &f.handle, g_params, &cp);
    if (NVSDK_NGX_FAILED(r)) {
        f.handle = nullptr;
        error = "DLSS feature creation failed (" + ResultText(r) + ")";
        return false;
    }
    f.renderW = in.renderWidth, f.renderH = in.renderHeight, f.outW = in.outputWidth, f.outH = in.outputHeight;
    f.preset = in.preset, f.quality = q;
    rd::Log("DLSS feature created: %ux%u -> %ux%u, %s mode, preset %d (%zu output sizes)",
            in.renderWidth, in.renderHeight, in.outputWidth, in.outputHeight, QualityName(q),
            in.preset, g_features.size());
    return true;
}

}  // namespace

bool Init(ID3D12Device* device, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initTried) {
        error = g_initError;
        return g_ready;
    }
    g_initTried = true;
    g_device = device;

    static const std::wstring dir = rd::ExeDir();
    static const wchar_t* paths[] = {dir.c_str()};
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    info.LoggingInfo.LoggingCallback = &NgxLog;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    info.LoggingInfo.DisableOtherLoggingSinks = true;

    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", dir.c_str(),
                                                             device, &info);
    if (NVSDK_NGX_FAILED(r)) {
        g_initError = "NGX didn't start (" + ResultText(r) + "): NVIDIA GPU and driver needed";
        g_status = g_initError;
        rd::Log("%s", g_initError.c_str());
        error = g_initError;
        return false;
    }
    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params);
    int available = 0;
    if (NVSDK_NGX_SUCCEED(r)) NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (!available) {
        int needsUpdate = 0;
        unsigned minMajor = 0, minMinor = 0;
        if (g_params) {
            NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdate);
            NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minMajor);
            NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minMinor);
        }
        g_initError = needsUpdate ? "DLSS needs a newer NVIDIA driver (" + std::to_string(minMajor) + "." +
                                        std::to_string(minMinor) + " or later)"
                                  : "DLSS not available: is nvngx_dlss.dll next to the game exe?";
        g_status = g_initError;
        rd::Log("%s", g_initError.c_str());
        error = g_initError;
        return false;
    }
    g_ready = true;
    g_status = "ready";
    rd::Log("NGX ready, DLSS Super Resolution available");
    return true;
}

bool Evaluate(ID3D12GraphicsCommandList* list, const Inputs& in, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready) {
        error = g_initError.empty() ? "DLSS not started" : g_initError;
        return false;
    }
    const uint64_t key = SizeKey(in.outputWidth, in.outputHeight);
    auto it = g_features.find(key);
    if (it == g_features.end()) {
        for (auto s = g_features.begin(); s != g_features.end();) {  // outputs the game stopped using
            if (s->second.lastFrame + kStaleFrames < g_frame) {
                if (s->second.handle) g_retired.push_back({s->second.handle, g_frame});
                rd::Log("DLSS feature for %ux%u output released (unused for %llu frames)", s->second.outW, s->second.outH,
                        static_cast<unsigned long long>(g_frame - s->second.lastFrame));
                s = g_features.erase(s);
            } else {
                ++s;
            }
        }
        if (g_features.size() >= kMaxFeatures) {
            error = "too many TSR views";
            return false;
        }
        it = g_features.emplace(key, Feature()).first;
    }
    Feature& f = it->second;
    f.lastFrame = g_frame;
    const bool fresh = !f.handle || in.renderWidth != f.renderW || in.renderHeight != f.renderH ||
                       in.outputWidth != f.outW || in.outputHeight != f.outH || in.preset != f.preset;
    if (fresh && !CreateFeature(list, in, f, error)) {
        g_status = error;
        return false;
    }
    NVSDK_NGX_D3D12_DLSS_Eval_Params ep = {};
    ep.Feature.pInColor = in.color;
    ep.Feature.pInOutput = in.output;
    ep.pInDepth = in.depth;
    ep.pInMotionVectors = in.motion;
    ep.pInBiasCurrentColorMask = in.biasMask;  // null: none
    ep.InJitterOffsetX = in.jitterX;
    ep.InJitterOffsetY = in.jitterY;
    ep.InRenderSubrectDimensions = {in.renderWidth, in.renderHeight};
    ep.InReset = in.reset || fresh ? 1 : 0;
    ep.InMVScaleX = in.motionScaleX;
    ep.InMVScaleY = in.motionScaleY;
    ep.InPreExposure = 1.0f;
    ep.InExposureScale = 1.0f;
    const NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSS_EXT(list, f.handle, g_params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        error = "DLSS evaluation failed (" + ResultText(r) + ")";
        g_status = error;
        return false;
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "running: %ux%u -> %ux%u, %s mode, preset %c, %zu view%s", f.renderW, f.renderH, f.outW,
             f.outH, QualityName(f.quality), f.preset ? static_cast<char>('A' + f.preset - 1) : 'D', g_features.size(),
             g_features.size() == 1 ? "" : "s");
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
        NVSDK_NGX_D3D12_ReleaseFeature(g_retired[i].handle);
        g_retired.erase(g_retired.begin() + i);
    }
}

void Shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& [out, f] : g_features)
        if (f.handle) NVSDK_NGX_D3D12_ReleaseFeature(f.handle);
    g_features.clear();
    for (Retired& r : g_retired) NVSDK_NGX_D3D12_ReleaseFeature(r.handle);
    g_retired.clear();
    if (g_params) NVSDK_NGX_D3D12_DestroyParameters(g_params);
    g_params = nullptr;
    if (g_ready && g_device) NVSDK_NGX_D3D12_Shutdown1(g_device);
    g_ready = false;
    g_initTried = false;
    g_device = nullptr;
}

std::string Status() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_status;
}

}  // namespace dlss
