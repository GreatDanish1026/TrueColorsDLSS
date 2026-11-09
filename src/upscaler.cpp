// Replaces the game's TAA with DLSS, DLSS Ray Reconstruction or FSR (Life is Strange: True Colors, UE 4.25).
//
// UE 4's TAA is one compute pass with every input bound (common.hpp, kTaaMain): scene colour t1
// (translucency already composited), depth t2, engine velocity t3, output u0; root CBV 4 is the
// View uniform buffer. At that dispatch the add-on records:
//   motion.hlsl: our own motion vectors. The engine's velocity only covers moving objects; the
//     camera's motion comes from the depth and View.ClipToPrevClip (byte 1952 in UE 4.25).
//   DLSS: colour, depth and our motion in, the TAA's output written, render size from
//     View.ViewSizeAndInvSize (byte 2080) and jitter from View.TemporalAAJitter (byte 2016).
// The game's TAA dispatch is then skipped. Its output is also its history for the next frame
// (and the post-processing reads it), so DLSS's result serves both.
//
// Render resolution: a DLSS mode sets r.ScreenPercentage and r.TemporalAA.Upsampling=1 (UE 4's
// TAAU: the TAA then takes the render-size image and writes the full-size output) on the game
// thread, and gives both back when DLSS goes off. TAAU is another permutation of the TAA shader
// (another id): shaders declaring exactly the TAA's registers are checked once by their bindings
// and taken as TAA if they match (IsTaa).
//
// The TSR code from DoubleExposureDLSS (DilateVelocity chain, translucency blend) is still here
// but nothing calls it in this game.
//
// NGX records its own compute state and descriptor heaps on the game's command list, and the
// engine assumes its state is unchanged afterwards. So every command list's compute and
// graphics bindings are tracked, and put back after DLSS through ReShade's API (which also
// knows the game's descriptor heaps and root signatures).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d12.h>

#include "common.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../third_party/reshade/utils/descriptor_tracking.hpp"
#include "cvars.hpp"
#include "dlss.hpp"
#include "fsr.hpp"
#include "motion_cs.h"  // kMotionCs: motion.hlsl compiled by build.ps1
#include "tint_cs.h"    // kTintCs: tint.hlsl (mask debug view)
#include "compose_cs.h"  // kComposeCs: compose.hlsl (translucency)
#include "guide_cs.h"       // kGuideCs: guide.hlsl (Ray Reconstruction guides from the G-buffer)
#include "guide_view_cs.h"  // kGuideViewCs: guide_view.hlsl (guide debug view)
#include "version.h"

using namespace reshade::api;

namespace rd {
namespace {

// ---------------------------------------------------------------- settings

constexpr int kToggleHotkey = VK_F9;

// Which upscaler replaces the TAA. Auto: DLSS if NGX starts (NVIDIA RTX), else FSR.
enum Backend { kAuto, kDlss, kFsr, kBackendCount };
const char* const kBackendNames[kBackendCount] = {"Auto (DLSS on NVIDIA RTX, else FSR)", "DLSS", "FSR"};

struct Settings {
    bool enabled = false;  // remembered across launches
    int backend = kAuto;   // Backend; ini key Upscaler
    float sharpness = 0.3f;  // FSR's extra sharpening (RCAS), 0..1; ini key FsrSharpness
    // FSR testing tools (Advanced): jitter sign and debug view (not saved), AMD's FSR 3.1 tuning
    // (saved: FsrVelocityFactor, FsrSwaying, FsrShadingChange).
    bool fsrJitterFlip = false, fsrDebug = false;
    float fsrVelocity = 1.0f, fsrSwaying = -1.0f / 3, fsrShading = 1.0f;
    int preset = dlss::kPresetAuto;  // DLSS model: Automatic, or NVSDK_NGX_DLSS_Hint_Render_Preset_* (0 = NVIDIA's)
    int mode = 0;          // DlssMode: 0 leaves the render resolution to the game's Resolution Scale
    int maskMode = 0;      // motion.hlsl MaskMode: 0 off, 2 moving objects (UE 4 has no animation flag)
    bool showMask = false;  // debug tint, not saved
    // motion.hlsl ObjectMode: 0 linear decode, UE 4.2x's encoding (QuarryDLSS, UE 4.26: checked against
    // camera reprojection); 1 square-root (UE 5.2) and 2 ignored stay for diagnosis. Not saved.
    int objectMode = 0;
    int transMask = 1;  // translucency in the bias mask: 0 off, 1 partial, 2 full
    // DLSS Ray Reconstruction in place of Super Resolution (ini RayReconstruction): the game's
    // ray tracing denoisers go off while it runs. rrPreset: NVSDK_NGX_RayReconstruction_Hint_Render_Preset_*
    // (ini RrModel, 0 = NVIDIA's default). guideView: debug view of one guide (not saved).
    bool rr = false;
    int rrPreset = 0;
    int guideView = 0;
} g_set;

const char* const kGuideViewNames[] = {"Off", "Diffuse albedo", "Specular albedo", "Normals", "Roughness"};
constexpr int kGuideViewCount = 5;

const char* const kTransMaskNames[] = {"Off", "Partial (recommended)", "Full"};
constexpr int kTransMaskCount = 3;
const float kTransMaskStrength[kTransMaskCount] = {0, 0.5f, 1};

const char* const kMaskModeNames[] = {"Off", "Off", "Moving objects"};
constexpr int kMaskModeCount = 3;
const char* const kObjectModeNames[] = {"Engine, linear (this game)", "Engine, square-root (UE 5.2 games)", "Ignore (camera motion only)"};
constexpr int kObjectModeCount = 3;

void LoadSettings() {
    reshade::get_config_value(nullptr, RD_PRODUCT, "Enabled", g_set.enabled);
    reshade::get_config_value(nullptr, RD_PRODUCT, "Upscaler", g_set.backend);
    if (g_set.backend < 0 || g_set.backend >= kBackendCount) g_set.backend = kAuto;
    reshade::get_config_value(nullptr, RD_PRODUCT, "FsrSharpness", g_set.sharpness);
    if (!(g_set.sharpness >= 0 && g_set.sharpness <= 1)) g_set.sharpness = 0.3f;
    reshade::get_config_value(nullptr, RD_PRODUCT, "FsrVelocityFactor", g_set.fsrVelocity);
    if (!(g_set.fsrVelocity >= 0 && g_set.fsrVelocity <= 1)) g_set.fsrVelocity = 1.0f;
    reshade::get_config_value(nullptr, RD_PRODUCT, "FsrSwaying", g_set.fsrSwaying);
    if (!(g_set.fsrSwaying >= -1 && g_set.fsrSwaying <= 1)) g_set.fsrSwaying = -1.0f / 3;
    reshade::get_config_value(nullptr, RD_PRODUCT, "FsrShadingChange", g_set.fsrShading);
    if (!(g_set.fsrShading >= 0 && g_set.fsrShading <= 2)) g_set.fsrShading = 1.0f;
    reshade::get_config_value(nullptr, RD_PRODUCT, "DlssModel", g_set.preset);
    reshade::get_config_value(nullptr, RD_PRODUCT, "DlssMode", g_set.mode);
    reshade::get_config_value(nullptr, RD_PRODUCT, "AnimationMask", g_set.maskMode);
    if (g_set.maskMode != 0 && g_set.maskMode != 2) g_set.maskMode = 0;
    reshade::get_config_value(nullptr, RD_PRODUCT, "TranslucencyMask", g_set.transMask);
    if (g_set.transMask < 0 || g_set.transMask >= kTransMaskCount) g_set.transMask = 1;
    reshade::get_config_value(nullptr, RD_PRODUCT, "RayReconstruction", g_set.rr);
    reshade::get_config_value(nullptr, RD_PRODUCT, "RrModel", g_set.rrPreset);
    if (g_set.rrPreset != 0 && (g_set.rrPreset < 4 || g_set.rrPreset > 6)) g_set.rrPreset = 0;
    if (!kTsrMapped) g_set.enabled = false;  // the upscaler's passes aren't known in this game yet
    if (g_set.mode < 0 || g_set.mode >= 6) g_set.mode = 0;
}

void SaveSettings() {
    reshade::set_config_value(nullptr, RD_PRODUCT, "Enabled", g_set.enabled);
    reshade::set_config_value(nullptr, RD_PRODUCT, "Upscaler", g_set.backend);
    reshade::set_config_value(nullptr, RD_PRODUCT, "FsrSharpness", g_set.sharpness);
    reshade::set_config_value(nullptr, RD_PRODUCT, "FsrVelocityFactor", g_set.fsrVelocity);
    reshade::set_config_value(nullptr, RD_PRODUCT, "FsrSwaying", g_set.fsrSwaying);
    reshade::set_config_value(nullptr, RD_PRODUCT, "FsrShadingChange", g_set.fsrShading);
    reshade::set_config_value(nullptr, RD_PRODUCT, "DlssModel", g_set.preset);
    reshade::set_config_value(nullptr, RD_PRODUCT, "DlssMode", g_set.mode);
    reshade::set_config_value(nullptr, RD_PRODUCT, "AnimationMask", g_set.maskMode);
    reshade::set_config_value(nullptr, RD_PRODUCT, "TranslucencyMask", g_set.transMask);
    reshade::set_config_value(nullptr, RD_PRODUCT, "RayReconstruction", g_set.rr);
    reshade::set_config_value(nullptr, RD_PRODUCT, "RrModel", g_set.rrPreset);
}

// ---------------------------------------------------------------- per command list bindings

struct StageState {
    pipeline_layout layout = {0};
    std::vector<descriptor_table> tables;
    struct Root {
        shader_stage stages;
        descriptor_type type;
        buffer_range range;
    };
    std::map<uint32_t, Root> roots;  // root descriptors by parameter
    struct Constants {
        shader_stage stages;
        std::vector<uint32_t> values;
    };
    std::map<uint32_t, Constants> constants;  // root constants by parameter

    void Layout(pipeline_layout l) {
        if (l == layout) return;
        layout = l;  // a new root signature resets every binding
        tables.clear();
        roots.clear();
        constants.clear();
    }
};

// One TSR run's inputs, gathered in the command list that records DecimateHistory, UpdateHistory
// and ResolveHistory (in that order).
struct Chain {
    resource depth = {0}, motion = {0}, color = {0}, mask = {0};
    std::vector<float> view;  // this view's View uniform buffer (camera for frame generation)
};

struct __declspec(uuid("3d0f6b1e-8c42-4a57-b1e9-5a7c0d2f9e61")) ListState {
    pipeline pipe = {0};
    pipeline_stage pipeStages = pipeline_stage::all;
    uint32_t computeId = 0;  // shader id of the bound pipeline (0 if graphics or unknown)
    StageState compute, graphics;
    Chain chain;
};

ListState* State(command_list* cl) { return cl->get_private_data<ListState>(); }

StageState* Stage(ListState* s, shader_stage stages) {
    if ((stages & shader_stage::all_compute) != 0) return &s->compute;
    if ((stages & shader_stage::all_graphics) != 0) return &s->graphics;
    return nullptr;
}

void OnInitList(command_list* cl) { cl->create_private_data<ListState>(); }
void OnDestroyList(command_list* cl) { cl->destroy_private_data<ListState>(); }

void OnResetList(command_list* cl) {
    if (ListState* s = State(cl)) *s = ListState();
}

void OnBindPipeline(command_list* cl, pipeline_stage stages, pipeline pipe) {
    ListState* s = State(cl);
    if (!s) return;
    s->pipe = pipe;
    s->pipeStages = stages;
    s->computeId = PipeId(pipe);
}

void OnBindTables(command_list* cl, shader_stage stages, pipeline_layout layout, uint32_t first, uint32_t count,
                  const descriptor_table* tables, uint32_t, const uint32_t*) {
    ListState* s = State(cl);
    if (!s) return;
    StageState* st = Stage(s, stages);
    if (!st) return;
    st->Layout(layout);
    if (st->tables.size() < first + count) st->tables.resize(first + count, descriptor_table{0});
    for (uint32_t i = 0; i < count; ++i) st->tables[first + i] = tables[i];
}

void OnPushDescriptors(command_list* cl, shader_stage stages, pipeline_layout layout, uint32_t param,
                       const descriptor_table_update& u) {
    ListState* s = State(cl);
    if (!s) return;
    StageState* st = Stage(s, stages);
    if (!st) return;
    st->Layout(layout);
    if (u.count != 1 || u.table.handle) return;
    if (u.type == descriptor_type::constant_buffer || u.type == descriptor_type::buffer_shader_resource_view ||
        u.type == descriptor_type::buffer_unordered_access_view)
        st->roots[param] = {stages, u.type, *static_cast<const buffer_range*>(u.descriptors)};
}

void OnPushConstants(command_list* cl, shader_stage stages, pipeline_layout layout, uint32_t param, uint32_t first,
                     uint32_t count, const void* values) {
    ListState* s = State(cl);
    if (!s) return;
    StageState* st = Stage(s, stages);
    if (!st) return;
    st->Layout(layout);
    StageState::Constants& c = st->constants[param];
    c.stages = stages;
    if (c.values.size() < first + count) c.values.resize(first + count);
    memcpy(c.values.data() + first, values, count * 4);
}

// Puts the game's bindings back after DLSS. A bind with no tables makes ReShade set the game's
// descriptor heaps and root signature again (NGX changed both behind its back).
void RestoreStage(command_list* cl, const StageState& st, shader_stage stages) {
    if (!st.layout.handle) return;
    cl->bind_descriptor_tables(stages, st.layout, 0, 0, nullptr);
    for (uint32_t i = 0; i < st.tables.size(); ++i)
        if (st.tables[i].handle) cl->bind_descriptor_tables(stages, st.layout, i, 1, &st.tables[i]);
    for (const auto& [param, r] : st.roots) {
        descriptor_table_update u = {};
        u.count = 1;
        u.type = r.type;
        u.descriptors = &r.range;
        cl->push_descriptors(r.stages, st.layout, param, u);
    }
    for (const auto& [param, c] : st.constants)
        cl->push_constants(c.stages, st.layout, param, 0, static_cast<uint32_t>(c.values.size()), c.values.data());
}

void Restore(command_list* cl, const ListState& s) {
    RestoreStage(cl, s.graphics, shader_stage::all_graphics);
    RestoreStage(cl, s.compute, shader_stage::all_compute);  // last: the game's next pass is compute or graphics
    if (s.pipe.handle) cl->bind_pipeline(s.pipeStages, s.pipe);
}

// ---------------------------------------------------------------- TSR inputs

// The resource at shader register <reg> of the bound table <param> (SRVs t#, UAVs u#).
resource Bound(command_list* cl, const StageState& st, uint32_t param, uint32_t reg) {
    if (param >= st.tables.size() || !st.tables[param].handle) return {0};
    device* dev = cl->get_device();
    descriptor_tracking* dt = dev->get_private_data<descriptor_tracking>();
    if (!dt) return {0};
    const pipeline_layout_param p = dt->get_pipeline_layout_param(st.layout, param);
    if (p.type != pipeline_layout_param_type::descriptor_table || !p.descriptor_table.ranges) return {0};
    for (uint32_t r = 0; r < p.descriptor_table.count; ++r) {
        const descriptor_range& range = p.descriptor_table.ranges[r];
        if (reg < range.dx_register_index || reg - range.dx_register_index >= range.count) continue;
        descriptor_heap heap = {0};
        uint32_t offset = 0;
        dev->get_descriptor_heap_offset(st.tables[param], range.binding + (reg - range.dx_register_index), 0, &heap, &offset);
        const resource_view v = dt->get_resource_view(heap, offset);
        return v.handle ? dev->get_resource_from_view(v) : resource{0};
    }
    return {0};
}

// <count> floats at byte <offset> of root constant buffer <param>.
bool ReadConstants(command_list* cl, const StageState& st, uint32_t param, uint32_t offset, float* out, uint32_t count) {
    auto it = st.roots.find(param);
    if (it == st.roots.end() || it->second.type != descriptor_type::constant_buffer) return false;
    device* dev = cl->get_device();
    const buffer_range& b = it->second.range;
    const resource_desc bd = dev->get_resource_desc(b.buffer);
    if (bd.type != resource_type::buffer || b.offset + offset + count * 4ull > bd.buffer.size) return false;
    void* data = nullptr;
    if (!dev->map_buffer_region(b.buffer, b.offset + offset, count * 4, map_access::read_only, &data) || !data) return false;
    memcpy(out, data, count * 4);
    dev->unmap_buffer_region(b.buffer);
    return true;
}

// Up to <maxCount> floats from the start of root constant buffer <param>, and zeros after the end of
// the buffer (UE's View uniform buffer may sit near the end of its upload buffer, too close for the
// whole scan). Returns how many floats were read.
uint32_t ReadConstantsUpTo(command_list* cl, const StageState& st, uint32_t param, float* out, uint32_t maxCount) {
    auto it = st.roots.find(param);
    if (it == st.roots.end() || it->second.type != descriptor_type::constant_buffer) return 0;
    device* dev = cl->get_device();
    const buffer_range& b = it->second.range;
    const resource_desc bd = dev->get_resource_desc(b.buffer);
    if (bd.type != resource_type::buffer || b.offset >= bd.buffer.size) return 0;
    const uint32_t count = static_cast<uint32_t>((bd.buffer.size - b.offset) / 4 < maxCount ? (bd.buffer.size - b.offset) / 4 : maxCount);
    if (!count || !ReadConstants(cl, st, param, 0, out, count)) return 0;
    if (count < maxCount) memset(out + count, 0, (maxCount - count) * 4);
    return count;
}

// The render size from TSR's parameters (root CBV 2, the same in every TSR pass): InputInfo is
// UE's FScreenPassTextureViewportParameters, with the texture extent at 0, ViewportMin (uint2)
// at 32 and ViewportSize at 48. The extent is rounded up (Balanced's 2227x1253 view sits in a
// larger texture), so the view's size is used. False if the view doesn't start at 0, 0.
bool RenderSize(command_list* cl, const StageState& st, float out[2]) {
    float extent[2] = {0, 0}, size[2] = {0, 0};
    uint32_t min[2] = {1, 1};
    if (!ReadConstants(cl, st, 2, 0, extent, 2) || !ReadConstants(cl, st, 2, 48, size, 2) ||
        !ReadConstants(cl, st, 2, 32, reinterpret_cast<float*>(min), 2))
        return false;
    if (min[0] || min[1] || !(size[0] >= 16 && size[1] >= 16 && size[0] <= extent[0] && size[1] <= extent[1])) {
        static int logged = 0;
        if (logged++ < 5)
            Log("TSR view %.0fx%.0f at %u,%u in %.0fx%.0f: not upscaled", size[0], size[1], min[0], min[1], extent[0], extent[1]);
        return false;
    }
    out[0] = size[0], out[1] = size[1];
    return true;
}

// Textures of our own that keep copies of transient TSR inputs until DLSS runs. Kept in the
// non-pixel shader resource state between uses. Never destroyed while the game runs (a texture
// may still be used by command lists the GPU hasn't run yet): one per size and format.
struct CopyPool {
    struct Entry {
        resource_desc desc;
        resource tex = {0};
    };
    std::vector<Entry> entries;

    resource Get(device* dev, const resource_desc& src) {
        for (const Entry& e : entries)
            if (e.desc.texture.width == src.texture.width && e.desc.texture.height == src.texture.height &&
                e.desc.texture.format == src.texture.format)
                return e.tex;
        if (entries.size() >= 8) return {0};
        Entry e;
        e.desc = resource_desc(src.texture.width, src.texture.height, 1, 1, src.texture.format, 1, memory_heap::default_,
                               resource_usage::shader_resource | resource_usage::copy_dest);
        if (!dev->create_resource(e.desc, nullptr, resource_usage::shader_resource_non_pixel, &e.tex)) {
            Log("couldn't create a %ux%u copy texture", src.texture.width, src.texture.height);
            return {0};
        }
        Log("created %ux%u copy texture (format %u)", src.texture.width, src.texture.height,
            static_cast<uint32_t>(src.texture.format));
        entries.push_back(e);
        return e.tex;
    }

    void Destroy(device* dev) {
        for (Entry& e : entries) dev->destroy_resource(e.tex);
        entries.clear();
    }
};

std::recursive_mutex g_mutex;  // destroying resources can call back into OnDestroyResource
CopyPool g_pool;

// Linear colour formats DLSS takes. The game's scene colour is rgba16f with HDR output and
// r11g11b10f with SDR output.
bool IsHdrColor(format f) { return f == format::r16g16b16a16_float || f == format::r11g11b10_float; }

// Copies <src> (in state <srcState>, format <want>) into a pool texture; {0} if the format isn't
// the expected one (menus and other views bind other textures in these slots). Wanting rgba16f
// also takes r11g11b10f.
resource CopyInput(command_list* cl, resource src, resource_usage srcState, format want, const char* what) {
    if (!src.handle) return {0};
    device* dev = cl->get_device();
    const resource_desc d = dev->get_resource_desc(src);
    const bool ok = d.texture.format == want || (want == format::r16g16b16a16_float && IsHdrColor(d.texture.format));
    if (d.type != resource_type::texture_2d || !ok) {
        static int logged = 0;
        if (logged < 20) {
            ++logged;
            Log("%s: unexpected %ux%u format %u: TSR kept for this view", what, d.texture.width, d.texture.height,
                static_cast<uint32_t>(d.texture.format));
        }
        return {0};
    }
    const resource tex = g_pool.Get(dev, d);
    if (!tex.handle) return {0};
    cl->barrier(src, srcState, resource_usage::copy_source);
    cl->barrier(tex, resource_usage::shader_resource_non_pixel, resource_usage::copy_dest);
    cl->copy_resource(src, tex);
    cl->barrier(src, resource_usage::copy_source, srcState);
    cl->barrier(tex, resource_usage::copy_dest, resource_usage::shader_resource_non_pixel);
    return tex;
}

// DilateVelocity writes a texture (u0) that DecimateHistory reads (t0), and may be recorded in another
// command list, so its depth and our motion vectors are found through that texture.
struct Link {
    resource depth = {0}, motion = {0}, mask = {0};
    std::vector<float> view;
};
std::map<uint64_t, Link> g_links;

std::atomic<uint64_t> g_evaluated{0}, g_skipped{0};

// Counts a view left to TSR, and logs each distinct reason the first few times.
void Skip(const char* reason, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0) {
    ++g_skipped;
    static std::map<std::string, int> logged;
    int& n = logged[reason];
    if (n++ < 3) Log("TSR kept for this view: %s (%u %u %u)", reason, a, b, c);
}
std::string g_error;

// Diagnostics, logged every couple of seconds while DLSS is on when a view was left to TSR or a frame
// went without frame generation inputs (at most 30 lines): what happened to each view (by output
// size) and why inputs were missing.
struct ViewStats {
    uint32_t evaluated = 0, tagged = 0, noOutput = 0, noDepth = 0, noMotion = 0, noColor = 0, other = 0;
};
std::map<uint64_t, ViewStats> g_stats;
std::map<std::string, uint32_t> g_why;
int g_statLines = 0;
void Why(const char* reason) { ++g_why[reason]; }

// Per output size (not texture: the game swaps TSR's output texture from UE's pool).
std::map<uint64_t, bool> g_resetByOutput;
// Render size per view: a new size (DLSS mode or menu change) starts DLSS and frame generation
// over, instead of blending in the previous size's frame (a one-frame streak at the bottom edge).
std::map<uint64_t, uint64_t> g_sizeByOutput;
bool g_resetAll = true;
float g_lastJitter[2] = {0, 0};
unsigned g_lastRender[2] = {0, 0}, g_lastOutput[2] = {0, 0};  // sizes of the last view DLSS upscaled

constexpr uint32_t kViewScanFloats = 4096;  // the first 16 KB of the View uniform buffer
// What's needed from it: ClipToPrevClip (byte 1872 here) and the jitter row after it, and the
// camera matrices frame generation uses (up to byte 640).
constexpr uint32_t kViewNeededFloats = 640;

struct MotionState {
    pipeline_layout layout = {0};
    pipeline pipe = {0};
    bool failed = false;
    int offset = -1;  // byte offset of View.ClipToPrevClip, -1 until found
    int candidate = -1, hits = 0, misses = 0;
    std::map<std::pair<uint64_t, int>, resource_view> views;  // by resource and kind (0 SRV, 1 UAV)
    struct Texture {
        uint32_t width = 0, height = 0;
        format fmt = format::unknown;
        int slot = 0;
        resource tex = {0};
    };
    // rg16f motion, r8 masks (slot 0 from the motion pass, 1 with translucency) and rgba16f composed
    // colour, one per size, kept until shutdown
    std::vector<Texture> textures;
    // Debug tint of the masked pixels (tint.hlsl).
    pipeline_layout tintLayout = {0};
    pipeline tintPipe = {0};
    bool tintFailed = false;
    // Translucency composite (compose.hlsl).
    pipeline_layout composeLayout = {0};
    pipeline composePipe = {0};
    bool composeFailed = false;
    uint64_t composed = 0, copied = 0;
    // Ray Reconstruction guides (guide.hlsl) and their debug view (guide_view.hlsl).
    pipeline_layout guideLayout = {0};
    pipeline guidePipe = {0};
    bool guideFailed = false;
    pipeline_layout guideViewLayout = {0};
    pipeline guideViewPipe = {0};
    bool guideViewFailed = false;
} g_motion;

// The G-buffer render targets of a base pass (Ray Reconstruction guides), by texture size.
struct GBufferSet {
    resource a = {0}, b = {0}, c = {0};
    format fa = format::unknown, fb = format::unknown, fc = format::unknown;  // the render target views' formats
};
std::map<uint64_t, GBufferSet> g_gbuffers;

// ---------------------------------------------------------------- motion vectors

bool Near(float a, float b) { return std::fabs(a - b) <= b * 2e-3f + 1e-9f; }

// UE's View uniform buffer has ClipToPrevClip, ClipToPrevClipWithAA, TemporalAAJitter in that
// order. TemporalAAJitter is (this frame's jitter, last frame's) in clip units (2 / size, either
// sign). The jittered projection matrices hold this frame's jitter too, but followed by 0 and 1.
// ClipToPrevClip maps a clip position to almost the same one (the camera moves little per frame),
// so its diagonal is near 1. A match at the same place several frames in a row is it.
bool LooksLikeClipToPrevClip(const float* m) {
    for (int i = 0; i < 16; ++i)
        if (!std::isfinite(m[i])) return false;
    return std::fabs(m[0] - 1) < 0.2f && std::fabs(m[5] - 1) < 0.2f;
}

// TemporalAAJitter (this frame's jitter, last frame's, in clip units): small, and this frame's
// x:y proportion matches TSR's jitter in pixels (scaled by the aspect ratio). Matching the
// proportion rather than the values keeps it independent of how UE scales the view (the DLSS
// modes change the render size, and an exact match on the size failed there).
bool LooksLikeJitterRow(const float* r, const float jitter[2], const float size[2]) {
    for (int i = 0; i < 4; ++i)
        if (!std::isfinite(r[i]) || std::fabs(r[i]) >= 0.01f) return false;
    if (r[0] == 0 || r[1] == 0 || (r[2] == 0 && r[3] == 0)) return false;
    if (std::fabs(jitter[0]) < 0.02f || std::fabs(jitter[1]) < 0.02f) return true;  // can't compare: accept
    const float want = std::fabs(jitter[0] / jitter[1]) * (size[1] / size[0]);
    return std::fabs(std::fabs(r[0] / r[1]) / want - 1) < 0.05f;
}

constexpr int kKnownClipToPrevClip = -1;  // not known for this game yet: searched (and logged)

void FindClipToPrevClip(const float* v, const float jitter[2], const float size[2]) {
    const auto valid = [&](int byteOffset) {
        const float* m = v + byteOffset / 4;
        return LooksLikeClipToPrevClip(m) && LooksLikeJitterRow(m + 32, jitter, size);
    };
    int found = -1;
    if (kKnownClipToPrevClip >= 0 && valid(kKnownClipToPrevClip)) {
        found = kKnownClipToPrevClip;
    } else {
        for (uint32_t i = 32; i + 4 <= kViewScanFloats; i += 4)
            if (valid(static_cast<int>(i - 32) * 4)) {
                found = static_cast<int>(i - 32) * 4;
                break;
            }
    }
    if (found < 0) {
        g_motion.candidate = -1, g_motion.hits = 0;
        if (++g_motion.misses == 600)
            Log("motion: the camera matrix wasn't found in the View uniform buffer (600 tries): DLSS can't run "
                "(render size %.0fx%.0f, jitter %.3f %.3f)", size[0], size[1], jitter[0], jitter[1]);
        return;
    }
    if (found == g_motion.candidate) ++g_motion.hits;
    else g_motion.candidate = found, g_motion.hits = 1;
    if (g_motion.hits < 4) return;
    g_motion.offset = found;
    Log("View.ClipToPrevClip at byte %d of the View uniform buffer", found);
}

bool CreateMotionPipeline(device* dev) {
    if (g_motion.pipe.handle) return true;
    if (g_motion.failed) return false;
    constant_range c = {};
    c.count = 24;
    c.visibility = shader_stage::all_compute;
    descriptor_range srv = {};
    srv.count = 2;
    srv.visibility = shader_stage::all_compute;
    srv.type = descriptor_type::shader_resource_view;
    descriptor_range uav = {};
    uav.count = 2;
    uav.visibility = shader_stage::all_compute;
    uav.type = descriptor_type::unordered_access_view;
    const pipeline_layout_param params[3] = {pipeline_layout_param(c), pipeline_layout_param(srv), pipeline_layout_param(uav)};
    shader_desc cs = {};
    cs.code = kMotionCs;
    cs.code_size = sizeof(kMotionCs);
    pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
    if (!dev->create_pipeline_layout(3, params, &g_motion.layout) ||
        !dev->create_pipeline(g_motion.layout, 1, &sub, &g_motion.pipe)) {
        g_motion.failed = true;
        g_error = "couldn't create the motion vector shader";
        Log("%s", g_error.c_str());
        return false;
    }
    Log("motion vector shader created");
    return true;
}

// MotionTexture slots of the Ray Reconstruction guides.
constexpr int kSlotDiffuse = 10, kSlotSpecular = 11, kSlotNormal = 12;

// Our own render-resolution texture of <fmt> (rg16f motion vectors, r8 mask, rgba16f colour).
resource MotionTexture(device* dev, uint32_t w, uint32_t h, format fmt = format::r16g16_float, int slot = 0) {
    for (const MotionState::Texture& t : g_motion.textures)
        if (t.width == w && t.height == h && t.fmt == fmt && t.slot == slot) return t.tex;
    if (g_motion.textures.size() >= 16) return {0};
    MotionState::Texture t;
    t.width = w, t.height = h, t.fmt = fmt, t.slot = slot;
    const resource_desc d(w, h, 1, 1, fmt, 1, memory_heap::default_,
                          resource_usage::shader_resource | resource_usage::unordered_access);
    const char* what = fmt == format::r8_unorm             ? (slot ? "translucency mask" : "animation mask")
                       : slot == kSlotDiffuse              ? "diffuse albedo"
                       : slot == kSlotSpecular             ? "specular albedo"
                       : slot == kSlotNormal               ? "normal + roughness"
                       : fmt == format::r16g16b16a16_float ? "composed colour"
                                                           : "motion vector";
    if (!dev->create_resource(d, nullptr, resource_usage::shader_resource_non_pixel, &t.tex)) {
        Log("couldn't create a %ux%u %s texture", w, h, what);
        return {0};
    }
    Log("created %ux%u %s texture", w, h, what);
    g_motion.textures.push_back(t);
    return t.tex;
}

resource_view View(device* dev, resource r, format f, bool uav) {
    const auto key = std::make_pair(r.handle, uav ? 1 : 0);
    auto it = g_motion.views.find(key);
    if (it != g_motion.views.end()) return it->second;
    resource_view v = {0};
    if (!dev->create_resource_view(r, uav ? resource_usage::unordered_access : resource_usage::shader_resource,
                                   resource_view_desc(f), &v))
        return {0};
    g_motion.views[key] = v;
    return v;
}

void LogOnce(int& count, const char* fmt, uint32_t a, uint32_t b) {
    if (count >= 10) return;
    ++count;
    Log(fmt, a, b);
}

// UE's velocity buffer: rg16unorm in True Colors (4.25), rgba16unorm in The Quarry (4.26) when
// the engine writes velocity in the base pass. motion.hlsl reads only .xy, so both work.
bool IsVelocityFormat(format f) { return f == format::r16g16_unorm || f == format::r16g16b16a16_unorm; }

// UE 4.25's View uniform buffer (FViewUniformShaderParameters), byte offsets from the capture:
// ClipToPrevClip 1952 (unjittered), TemporalAAJitter 2016 (this frame's, last frame's; clip
// units), ViewRectMin 2064, ViewSizeAndInvSize 2080. ViewToClip (448) holds the jitter in its
// third row, which checks the layout each frame.
constexpr int kViewClipToPrevClip = 1952, kViewJitter = 2016, kViewRectMin = 2064, kViewSize = 2080,
              kViewToClipJitterRow = 480;

// Records motion.hlsl before DLSS (in the TAA's command list) and puts the game's state back.
resource ComputeMotion(command_list* cl, ListState& s, resource depth, resource velocity, const float size[2],
                       const std::vector<float>& view, resource* maskOut) {
    *maskOut = {0};
    static int badFormat = 0;
    device* dev = cl->get_device();
    if (!velocity.handle) return Why("motion: no velocity"), resource{0};
    const resource_desc dd = dev->get_resource_desc(depth), vd = dev->get_resource_desc(velocity);
    if (dd.texture.format != format::r32_g8_typeless || !IsVelocityFormat(vd.texture.format) ||
        vd.texture.width < size[0] || vd.texture.height < size[1]) {
        LogOnce(badFormat, "motion: depth format %u, velocity format %u: TAA kept for this view",
                static_cast<uint32_t>(dd.texture.format), static_cast<uint32_t>(vd.texture.format));
        return Why("motion: velocity is a placeholder or another format (camera cut)"), resource{0};
    }
    g_motion.offset = kViewClipToPrevClip;
    if (!CreateMotionPipeline(dev)) return {0};
    const resource tex = MotionTexture(dev, dd.texture.width, dd.texture.height);
    const resource mask = MotionTexture(dev, dd.texture.width, dd.texture.height, format::r8_unorm);
    if (!tex.handle || !mask.handle) return Why("motion: no texture (limit)"), resource{0};
    const resource_view srvs[2] = {View(dev, depth, format::r32_float_x8_uint, false),
                                   View(dev, velocity, vd.texture.format, false)};
    const resource_view uavs[2] = {View(dev, tex, format::r16g16_float, true), View(dev, mask, format::r8_unorm, true)};
    if (!srvs[0].handle || !srvs[1].handle || !uavs[0].handle || !uavs[1].handle) return Why("motion: views"), resource{0};

    float params[24];
    const float* m = view.data() + g_motion.offset / 4;
    memcpy(params, m, 16 * sizeof(float));
    params[16] = size[0], params[17] = size[1], params[18] = 1 / size[0], params[19] = 1 / size[1];
    const uint32_t maskMode = static_cast<uint32_t>(g_set.maskMode);
    memcpy(&params[20], &maskMode, 4);
    params[21] = 1.0f;  // mask strength
    const uint32_t objectMode = static_cast<uint32_t>(g_set.objectMode), pad = 0;
    memcpy(&params[22], &objectMode, 4);
    memcpy(&params[23], &pad, 4);

    const resource both[2] = {tex, mask};
    const resource_usage from[2] = {resource_usage::shader_resource_non_pixel, resource_usage::shader_resource_non_pixel};
    const resource_usage to[2] = {resource_usage::unordered_access, resource_usage::unordered_access};
    cl->barrier(2, both, from, to);
    cl->bind_pipeline(pipeline_stage::all_compute, g_motion.pipe);
    cl->push_constants(shader_stage::all_compute, g_motion.layout, 0, 0, 24, params);
    descriptor_table_update u = {};
    u.count = 2;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = srvs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.layout, 1, u);
    u.count = 2;
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = uavs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.layout, 2, u);
    cl->dispatch((static_cast<uint32_t>(size[0]) + 7) / 8, (static_cast<uint32_t>(size[1]) + 7) / 8, 1);
    cl->barrier(2, both, to, from);
    Restore(cl, s);
    *maskOut = mask;
    return tex;
}

// ---------------------------------------------------------------- mask debug view

bool CreateTintPipeline(device* dev) {
    if (g_motion.tintPipe.handle) return true;
    if (g_motion.tintFailed) return false;
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
    cs.code = kTintCs;
    cs.code_size = sizeof(kTintCs);
    pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
    if (!dev->create_pipeline_layout(3, params, &g_motion.tintLayout) ||
        !dev->create_pipeline(g_motion.tintLayout, 1, &sub, &g_motion.tintPipe)) {
        g_motion.tintFailed = true;
        Log("couldn't create the mask debug shader");
        return false;
    }
    return true;
}

// Tints <output> (in the unordered access state) red where <mask> is set.
void TintMask(command_list* cl, resource output, resource mask, unsigned renderW, unsigned renderH) {
    device* dev = cl->get_device();
    if (!mask.handle || !CreateTintPipeline(dev)) return;
    const resource_desc od = dev->get_resource_desc(output);
    const resource_view srv = View(dev, mask, format::r8_unorm, false);
    const resource_view uav = View(dev, output, od.texture.format, true);
    if (!srv.handle || !uav.handle) return;
    const float params[4] = {static_cast<float>(od.texture.width), static_cast<float>(od.texture.height),
                             static_cast<float>(renderW) / od.texture.width, static_cast<float>(renderH) / od.texture.height};
    cl->bind_pipeline(pipeline_stage::all_compute, g_motion.tintPipe);
    cl->push_constants(shader_stage::all_compute, g_motion.tintLayout, 0, 0, 4, params);
    descriptor_table_update u = {};
    u.count = 1;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = &srv;
    cl->push_descriptors(shader_stage::all_compute, g_motion.tintLayout, 1, u);
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = &uav;
    cl->push_descriptors(shader_stage::all_compute, g_motion.tintLayout, 2, u);
    cl->dispatch((od.texture.width + 7) / 8, (od.texture.height + 7) / 8, 1);
    cl->barrier(output, resource_usage::unordered_access, resource_usage::unordered_access);
}

void OnDestroyResource(device* dev, resource r) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    for (auto g = g_gbuffers.begin(); g != g_gbuffers.end();)  // a G-buffer texture the game let go
        g = g->second.a == r || g->second.b == r || g->second.c == r ? g_gbuffers.erase(g) : std::next(g);
    for (int kind = 0; kind < 2; ++kind) {
        auto it = g_motion.views.find(std::make_pair(r.handle, kind));
        if (it == g_motion.views.end()) continue;
        dev->destroy_resource_view(it->second);
        g_motion.views.erase(it);
    }
}

// ---------------------------------------------------------------- translucency

bool CreateComposePipeline(device* dev) {
    if (g_motion.composePipe.handle) return true;
    if (g_motion.composeFailed) return false;
    constant_range c = {};
    c.count = 8;
    c.visibility = shader_stage::all_compute;
    descriptor_range srv = {};
    srv.count = 3;
    srv.visibility = shader_stage::all_compute;
    srv.type = descriptor_type::shader_resource_view;
    descriptor_range uav = {};
    uav.count = 2;
    uav.visibility = shader_stage::all_compute;
    uav.type = descriptor_type::unordered_access_view;
    const pipeline_layout_param params[3] = {pipeline_layout_param(c), pipeline_layout_param(srv), pipeline_layout_param(uav)};
    shader_desc cs = {};
    cs.code = kComposeCs;
    cs.code_size = sizeof(kComposeCs);
    pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
    if (!dev->create_pipeline_layout(3, params, &g_motion.composeLayout) ||
        !dev->create_pipeline(g_motion.composeLayout, 1, &sub, &g_motion.composePipe)) {
        g_motion.composeFailed = true;
        Log("couldn't create the translucency shader: colour copied without translucency");
        return false;
    }
    Log("translucency shader created");
    return true;
}

// Blends UpdateHistory's separate translucency (t1) over its scene colour (t0) into our own
// texture, and the translucency into a copy of the bias mask. False if the inputs aren't the
// expected ones (then the colour is copied as before, without translucency).
bool Compose(command_list* cl, ListState& s) {
    static int logged = 0;
    device* dev = cl->get_device();
    const resource scene = Bound(cl, s.compute, 0, 0), trans = Bound(cl, s.compute, 0, 1);
    if (!scene.handle || !trans.handle || !s.chain.mask.handle) return Why("translucency: input missing"), false;
    const resource_desc sd = dev->get_resource_desc(scene), td = dev->get_resource_desc(trans);
    if (sd.type != resource_type::texture_2d || td.type != resource_type::texture_2d || !IsHdrColor(sd.texture.format) ||
        td.texture.format != format::r16g16b16a16_float || td.texture.width * 4 < sd.texture.width ||
        td.texture.height * 4 < sd.texture.height) {
        LogOnce(logged, "translucency: scene format %u, translucency format %u: colour copied without translucency",
                static_cast<uint32_t>(sd.texture.format), static_cast<uint32_t>(td.texture.format));
        return false;
    }
    float size[2] = {0, 0};
    if (!RenderSize(cl, s.compute, size) || size[0] > sd.texture.width || size[1] > sd.texture.height)
        return Why("translucency: render size"), false;
    if (!CreateComposePipeline(dev)) return false;
    const resource color = MotionTexture(dev, sd.texture.width, sd.texture.height, format::r16g16b16a16_float);
    const resource mask = MotionTexture(dev, sd.texture.width, sd.texture.height, format::r8_unorm, 1);
    if (!color.handle || !mask.handle) return Why("translucency: no texture (limit)"), false;
    const resource_view srvs[3] = {View(dev, scene, sd.texture.format, false),
                                   View(dev, trans, format::r16g16b16a16_float, false),
                                   View(dev, s.chain.mask, format::r8_unorm, false)};
    const resource_view uavs[2] = {View(dev, color, format::r16g16b16a16_float, true), View(dev, mask, format::r8_unorm, true)};
    if (!srvs[0].handle || !srvs[1].handle || !srvs[2].handle || !uavs[0].handle || !uavs[1].handle)
        return Why("translucency: views"), false;

    const float params[8] = {size[0],
                             size[1],
                             static_cast<float>(td.texture.width) / sd.texture.width,
                             static_cast<float>(td.texture.height) / sd.texture.height,
                             kTransMaskStrength[g_set.transMask],
                             0,
                             0,
                             0};
    const resource both[2] = {color, mask};
    const resource_usage from[2] = {resource_usage::shader_resource_non_pixel, resource_usage::shader_resource_non_pixel};
    const resource_usage to[2] = {resource_usage::unordered_access, resource_usage::unordered_access};
    cl->barrier(2, both, from, to);
    cl->bind_pipeline(pipeline_stage::all_compute, g_motion.composePipe);
    cl->push_constants(shader_stage::all_compute, g_motion.composeLayout, 0, 0, 8, params);
    descriptor_table_update u = {};
    u.count = 3;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = srvs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.composeLayout, 1, u);
    u.count = 2;
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = uavs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.composeLayout, 2, u);
    cl->dispatch((static_cast<uint32_t>(size[0]) + 7) / 8, (static_cast<uint32_t>(size[1]) + 7) / 8, 1);
    cl->barrier(2, both, to, from);
    Restore(cl, s);
    s.chain.color = color;
    s.chain.mask = mask;
    if (++g_motion.composed == 1) Log("translucency: blended into DLSS's input");
    return true;
}

// ---------------------------------------------------------------- Ray Reconstruction guides
//
// UE 4.25's base pass draws into scene colour + GBufferA (rgb10a2: world normal) + GBufferB (8-bit:
// metallic, specular, roughness, shading model) + GBufferC (8-bit: base colour) [+ GBufferD/E,
// velocity]. That render target set is remembered per size when it's bound; at the TAA, the
// G-buffer of the TAA's size becomes Ray Reconstruction's guides (guide.hlsl). The G-buffer is taken
// to be in a shader resource state by then (the lighting passes read it).

std::string g_rrError;                     // why Ray Reconstruction isn't running (overlay)
std::atomic<uint64_t> g_rrEvaluated{0}, g_rrLastPresent{0}, g_presents{0};
bool g_lastWasRr = false;

bool Is8BitColor(format f) {
    return f == format::b8g8r8a8_unorm || f == format::b8g8r8a8_unorm_srgb || f == format::b8g8r8a8_typeless ||
           f == format::r8g8b8a8_unorm || f == format::r8g8b8a8_unorm_srgb || f == format::r8g8b8a8_typeless;
}
bool IsNormalFormat(format f) { return f == format::r10g10b10a2_unorm || f == format::r10g10b10a2_typeless; }

void OnBindRenderTargets(command_list* cl, uint32_t count, const resource_view* rtvs, resource_view) {
    if (count < 4 || !g_set.enabled || !g_set.rr) return;
    device* dev = cl->get_device();
    format f[4];
    for (uint32_t i = 0; i < 4; ++i) {
        if (!rtvs[i].handle) return;
        f[i] = dev->get_resource_view_desc(rtvs[i]).format;
    }
    if (!IsHdrColor(f[0]) || !IsNormalFormat(f[1]) || !Is8BitColor(f[2]) || !Is8BitColor(f[3])) {
        static std::atomic<int> logged{0};  // what else binds 4+ targets (a base pass we don't recognise?)
        if (logged++ < 6) {
            const resource_desc d = dev->get_resource_desc(dev->get_resource_from_view(rtvs[0]));
            Log("render targets not taken as a G-buffer: %u targets %ux%u, formats %u %u %u %u", count, d.texture.width,
                d.texture.height, static_cast<uint32_t>(f[0]), static_cast<uint32_t>(f[1]), static_cast<uint32_t>(f[2]),
                static_cast<uint32_t>(f[3]));
        }
        return;
    }
    const resource a = dev->get_resource_from_view(rtvs[1]), b = dev->get_resource_from_view(rtvs[2]),
                   c = dev->get_resource_from_view(rtvs[3]);
    if (!a.handle || !b.handle || !c.handle) return;
    const resource_desc ad = dev->get_resource_desc(a);
    const uint64_t key = (static_cast<uint64_t>(ad.texture.width) << 32) | ad.texture.height;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    GBufferSet& g = g_gbuffers[key];
    if (g.a == a && g.b == b && g.c == c) return;
    g.a = a, g.b = b, g.c = c, g.fa = f[1], g.fb = f[2], g.fc = f[3];
    static int logged = 0;
    if (logged++ < 8)
        Log("G-buffer %ux%u found (%u render targets): A format %u, B %u, C %u", ad.texture.width, ad.texture.height,
            count, static_cast<uint32_t>(f[1]), static_cast<uint32_t>(f[2]), static_cast<uint32_t>(f[3]));
}

bool CreateGuidePipelines(device* dev) {
    if (g_motion.guidePipe.handle && g_motion.guideViewPipe.handle) return true;
    if (g_motion.guideFailed) return false;
    constant_range c = {};
    c.count = 20;
    c.visibility = shader_stage::all_compute;
    descriptor_range srv = {};
    srv.count = 3;
    srv.visibility = shader_stage::all_compute;
    srv.type = descriptor_type::shader_resource_view;
    descriptor_range uav = {};
    uav.count = 3;
    uav.visibility = shader_stage::all_compute;
    uav.type = descriptor_type::unordered_access_view;
    const pipeline_layout_param params[3] = {pipeline_layout_param(c), pipeline_layout_param(srv), pipeline_layout_param(uav)};
    shader_desc cs = {};
    cs.code = kGuideCs;
    cs.code_size = sizeof(kGuideCs);
    pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};

    constant_range vc = {};
    vc.count = 8;
    vc.visibility = shader_stage::all_compute;
    descriptor_range vsrv = srv, vuav = uav;
    vsrv.count = 1, vuav.count = 1;
    const pipeline_layout_param vparams[3] = {pipeline_layout_param(vc), pipeline_layout_param(vsrv), pipeline_layout_param(vuav)};
    shader_desc vcs = {};
    vcs.code = kGuideViewCs;
    vcs.code_size = sizeof(kGuideViewCs);
    pipeline_subobject vsub = {pipeline_subobject_type::compute_shader, 1, &vcs};

    if (!dev->create_pipeline_layout(3, params, &g_motion.guideLayout) ||
        !dev->create_pipeline(g_motion.guideLayout, 1, &sub, &g_motion.guidePipe) ||
        !dev->create_pipeline_layout(3, vparams, &g_motion.guideViewLayout) ||
        !dev->create_pipeline(g_motion.guideViewLayout, 1, &vsub, &g_motion.guideViewPipe)) {
        g_motion.guideFailed = true;
        g_rrError = "couldn't create the Ray Reconstruction guide shader";
        Log("%s", g_rrError.c_str());
        return false;
    }
    Log("Ray Reconstruction guide shader created");
    return true;
}

// UE 4.25's View buffer: TranslatedWorldToView at byte 192 (camera
// at the origin, so no translation), ViewToClip 448 (jittered), ViewToClipNoAA 512.
constexpr int kViewTranslatedWorldToView = 192, kViewToClip = 448, kViewToClipNoAA = 512;

// Checks those matrices are what they should be: a rotation without translation, and the same
// projection with and without jitter. Logs the first failures.
bool ViewMatricesOk(const float* v) {
    const float* w = v + kViewTranslatedWorldToView / 4;
    const float* p = v + kViewToClip / 4;
    const float* n = v + kViewToClipNoAA / 4;
    bool ok = w[3] == 0 && w[7] == 0 && w[11] == 0 && std::fabs(w[15] - 1) < 1e-4f;
    for (int i = 0; i < 3 && ok; ++i) {
        for (int j = 0; j < 3; ++j) {
            const float d = w[i * 4] * w[j * 4] + w[i * 4 + 1] * w[j * 4 + 1] + w[i * 4 + 2] * w[j * 4 + 2];
            if (std::fabs(d - (i == j ? 1.0f : 0.0f)) > 2e-3f) ok = false;
        }
        if (std::fabs(w[12 + i]) > 10) ok = false;  // translated world: the camera is at the origin
    }
    ok = ok && Near(n[0], p[0]) && Near(n[5], p[5]) && n[8] == 0 && n[9] == 0 && n[11] == 1 && n[14] == p[14];
    static int logged = 0;
    if (logged < (ok ? 1 : 4)) {
        ++logged;
        Log("View matrices for Ray Reconstruction %s: TranslatedWorldToView rows (%.3f %.3f %.3f %.3f) (%.3f %.3f %.3f %.3f) "
            "(%.3f %.3f %.3f %.3f) (%.3f %.3f %.3f %.3f); ViewToClipNoAA x %.4f y %.4f jitter row %.5f %.5f %.3f %.3f, "
            "near %.3f (jittered: %.5f %.5f)",
            ok ? "ok" : "NOT as expected", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11],
            w[12], w[13], w[14], w[15], n[0], n[5], n[8], n[9], n[10], n[11], n[14], p[8], p[9]);
    }
    return ok;
}

// The G-buffer's shader resource format: the render target's own (sRGB decodes), typeless as unorm.
format SrvFormat(format f) { return format_to_default_typed(f, 0); }

// Records guide.hlsl (in the TAA's command list) and puts the game's state back. The guides are
// textures of the G-buffer's size; <size> is the view's.
bool BuildGuides(command_list* cl, ListState& s, resource depth, const float size[2], const float* v, resource out[3]) {
    device* dev = cl->get_device();
    const resource_desc dd = dev->get_resource_desc(depth);
    const uint64_t key = (static_cast<uint64_t>(dd.texture.width) << 32) | dd.texture.height;
    auto it = g_gbuffers.find(key);
    if (it == g_gbuffers.end()) {
        g_rrError = "G-buffer not found yet (no base pass of the render size seen)";
        return false;
    }
    const GBufferSet& g = it->second;
    if (!CreateGuidePipelines(dev)) return false;
    const uint32_t w = dd.texture.width, h = dd.texture.height;
    out[0] = MotionTexture(dev, w, h, format::r8g8b8a8_unorm, kSlotDiffuse);
    out[1] = MotionTexture(dev, w, h, format::r8g8b8a8_unorm, kSlotSpecular);
    out[2] = MotionTexture(dev, w, h, format::r16g16b16a16_float, kSlotNormal);
    if (!out[0].handle || !out[1].handle || !out[2].handle) {
        g_rrError = "couldn't create the guide textures";
        return false;
    }
    const resource_view srvs[3] = {View(dev, g.a, SrvFormat(g.fa), false), View(dev, g.b, SrvFormat(g.fb), false),
                                   View(dev, g.c, SrvFormat(g.fc), false)};
    const resource_view uavs[3] = {View(dev, out[0], format::r8g8b8a8_unorm, true),
                                   View(dev, out[1], format::r8g8b8a8_unorm, true),
                                   View(dev, out[2], format::r16g16b16a16_float, true)};
    for (const resource_view& x : srvs)
        if (!x.handle) return g_rrError = "couldn't read the G-buffer (views)", false;
    for (const resource_view& x : uavs)
        if (!x.handle) return g_rrError = "guide texture views failed", false;

    float params[20] = {};
    const float* m = v + kViewTranslatedWorldToView / 4;
    for (int r = 0; r < 3; ++r) memcpy(&params[r * 4], m + r * 4, 3 * sizeof(float));
    params[12] = size[0], params[13] = size[1], params[14] = 1 / size[0], params[15] = 1 / size[1];
    params[16] = v[kViewToClipNoAA / 4], params[17] = v[kViewToClipNoAA / 4 + 5];

    const resource_usage from[3] = {resource_usage::shader_resource_non_pixel, resource_usage::shader_resource_non_pixel,
                                    resource_usage::shader_resource_non_pixel};
    const resource_usage to[3] = {resource_usage::unordered_access, resource_usage::unordered_access,
                                  resource_usage::unordered_access};
    cl->barrier(3, out, from, to);
    cl->bind_pipeline(pipeline_stage::all_compute, g_motion.guidePipe);
    cl->push_constants(shader_stage::all_compute, g_motion.guideLayout, 0, 0, 20, params);
    descriptor_table_update u = {};
    u.count = 3;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = srvs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.guideLayout, 1, u);
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = uavs;
    cl->push_descriptors(shader_stage::all_compute, g_motion.guideLayout, 2, u);
    cl->dispatch((static_cast<uint32_t>(size[0]) + 7) / 8, (static_cast<uint32_t>(size[1]) + 7) / 8, 1);
    cl->barrier(3, out, to, from);
    Restore(cl, s);
    return true;
}

// Debug view: draws guide <which> (1 diffuse, 2 specular, 3 normals, 4 roughness) over <output>
// (unordered access state).
void ShowGuide(command_list* cl, resource output, const resource guides[3], int which, unsigned renderW, unsigned renderH) {
    device* dev = cl->get_device();
    if (!g_motion.guideViewPipe.handle || which < 1 || which > 4) return;
    const resource g = guides[which == 1 ? 0 : which == 2 ? 1 : 2];
    const format gf = which <= 2 ? format::r8g8b8a8_unorm : format::r16g16b16a16_float;
    const resource_desc od = dev->get_resource_desc(output);
    const resource_view srv = View(dev, g, gf, false);
    const resource_view uav = View(dev, output, od.texture.format, true);
    if (!srv.handle || !uav.handle) return;
    float params[8] = {static_cast<float>(od.texture.width), static_cast<float>(od.texture.height),
                       static_cast<float>(renderW) / od.texture.width, static_cast<float>(renderH) / od.texture.height};
    const uint32_t mode = which <= 2 ? 1 : which == 3 ? 2 : 3;
    memcpy(&params[4], &mode, 4);
    cl->bind_pipeline(pipeline_stage::all_compute, g_motion.guideViewPipe);
    cl->push_constants(shader_stage::all_compute, g_motion.guideViewLayout, 0, 0, 8, params);
    descriptor_table_update u = {};
    u.count = 1;
    u.type = descriptor_type::shader_resource_view;
    u.descriptors = &srv;
    cl->push_descriptors(shader_stage::all_compute, g_motion.guideViewLayout, 1, u);
    u.type = descriptor_type::unordered_access_view;
    u.descriptors = &uav;
    cl->push_descriptors(shader_stage::all_compute, g_motion.guideViewLayout, 2, u);
    cl->dispatch((od.texture.width + 7) / 8, (od.texture.height + 7) / 8, 1);
    cl->barrier(output, resource_usage::unordered_access, resource_usage::unordered_access);
}

// ---------------------------------------------------------------- DLSS

int g_lastBackend = 0;  // kDlss or kFsr once one ran (0: none yet)
int g_autoBackend = 0;  // what Auto resolved to (0: not tried yet)
const char* BackendName(int b) { return b == kDlss ? "DLSS" : b == kFsr ? "FSR" : "Upscaler"; }

// FSR wants the camera's near plane and vertical field of view: UE 4.25's View.ViewToClip (byte
// 448, jittered; reversed-Z, row-major: [1][1] = 1 / tan(fovY / 2), [3][2] = near plane).
void CameraFromView(const float* v, float* nearPlane, float* fovY) {
    *nearPlane = 10, *fovY = 1.0f;
    const float* m = v + 448 / 4;
    const float yScale = m[5], n = m[14];
    if (std::isfinite(yScale) && yScale > 0.1f && yScale < 20) *fovY = 2 * std::atan(1 / yScale);
    if (std::isfinite(n) && n > 0.01f && n < 1000) *nearPlane = n;
    static int logged = 0;
    if (logged < 1) {
        ++logged;
        Log("FSR camera: near %.3f, vertical FOV %.1f degrees (ViewToClip [1][1] %.4f, [2][2] %.6f (0 = infinite far), "
            "[3][2] %.4f)", *nearPlane, *fovY * 57.29578f, yScale, m[10], n);
    }
}

// The game's TAA (kTaaMain): DLSS or FSR in its place. True if the game's dispatch is to be skipped.
bool OnTaa(command_list* cl, ListState& s) {
    device* dev = cl->get_device();
    const resource color = Bound(cl, s.compute, 0, 1), depth = Bound(cl, s.compute, 0, 2),
                   velocity = Bound(cl, s.compute, 0, 3), output = Bound(cl, s.compute, 2, 0);
    uint64_t outKey = 0;
    if (output.handle) {
        const resource_desc od = dev->get_resource_desc(output);
        outKey = (static_cast<uint64_t>(od.texture.width) << 32) | od.texture.height;
    }
    ViewStats& vs = g_stats[outKey];
    if (!output.handle || !color.handle || !depth.handle) {
        if (!output.handle) ++vs.noOutput;
        else if (!depth.handle) ++vs.noDepth;
        else ++vs.noColor;
        Skip("an input is missing (output, colour, depth)", output.handle != 0, color.handle != 0, depth.handle != 0);
        return false;
    }
    const resource_desc out = dev->get_resource_desc(output), cd = dev->get_resource_desc(color);
    if (!IsHdrColor(out.texture.format) || !IsHdrColor(cd.texture.format)) {
        Skip("colour or output has an unexpected format (output, colour)", static_cast<uint32_t>(out.texture.format),
             static_cast<uint32_t>(cd.texture.format));
        return false;
    }
    static std::vector<float> view(kViewScanFloats);
    const uint32_t viewFloats = ReadConstantsUpTo(cl, s.compute, 4, view.data(), kViewScanFloats);
    if (viewFloats * 4u < kViewSize + 16) {
        Why("View buffer not readable");
        Skip("the View buffer isn't readable", viewFloats);
        return false;
    }
    const float* v = view.data();
    const float size[2] = {v[kViewSize / 4], v[kViewSize / 4 + 1]};
    const float jitterClip[2] = {v[kViewJitter / 4], v[kViewJitter / 4 + 1]};
    uint32_t rectMin[2] = {1, 1};
    memcpy(rectMin, v + kViewRectMin / 4, 8);
    const bool layoutOk = v[kViewToClipJitterRow / 4] == jitterClip[0] && v[kViewToClipJitterRow / 4 + 1] == jitterClip[1] &&
                          LooksLikeClipToPrevClip(v + kViewClipToPrevClip / 4);
    if (!layoutOk || rectMin[0] || rectMin[1] || !(size[0] >= 16 && size[1] >= 16) || size[0] > cd.texture.width ||
        size[1] > cd.texture.height || size[0] > out.texture.width || size[1] > out.texture.height) {
        static int logged = 0;
        if (logged++ < 5)
            Log("TAA view: layout %s, rect min %u,%u, size %.0fx%.0f, colour %ux%u, output %ux%u: TAA kept", layoutOk ? "ok" : "NOT as expected",
                rectMin[0], rectMin[1], size[0], size[1], cd.texture.width, cd.texture.height, out.texture.width,
                out.texture.height);
        Why("TAA view: layout or size");
        ++g_skipped;
        return false;
    }
    // UE's TemporalJitterPixels: clip units to pixels, y down.
    const float jitter[2] = {jitterClip[0] * size[0] * 0.5f, -jitterClip[1] * size[1] * 0.5f};
    if (std::fabs(jitter[0]) > 0.5f || std::fabs(jitter[1]) > 0.5f) {
        Skip("jitter out of range");
        return false;
    }
    resource mask = {0};
    const resource motion = ComputeMotion(cl, s, depth, velocity, size, view, &mask);
    if (!motion.handle) {
        ++vs.noMotion;
        ++g_skipped;  // camera-cut frames bind a 1x1 velocity placeholder: the game's TAA handles them
        g_resetByOutput[outKey] = true;
        return false;
    }
    std::string err;
    ID3D12Device* native = reinterpret_cast<ID3D12Device*>(dev->get_native());
    int backend = g_set.backend;
    if (backend == kAuto) {
        if (!g_autoBackend) {
            std::string e;
            g_autoBackend = dlss::Init(native, e) ? kDlss : kFsr;
            Log("upscaler Auto: %s%s%s", BackendName(g_autoBackend), e.empty() ? "" : " (DLSS: ", e.empty() ? "" : (e + ")").c_str());
        }
        backend = g_autoBackend;
    }
    if (backend != g_lastBackend) {  // a different upscaler starts its own history
        if (g_lastBackend == kDlss) dlss::ReleaseFeatures();
        if (g_lastBackend == kFsr) fsr::ReleaseContexts();
        if (g_lastBackend) Log("upscaler: %s", BackendName(backend));
        g_lastBackend = backend;
    }
    if (!(backend == kDlss ? dlss::Init(native, err) : fsr::Init(native, err))) {
        g_error = err;
        ++g_skipped;
        return false;
    }
    // Ray Reconstruction: guides from the G-buffer; Super Resolution this frame if they can't be made.
    bool useRr = false;
    resource guides[3] = {{0}, {0}, {0}};
    if (backend == kDlss && g_set.rr) {
        std::string why;
        if (!dlss::RrAvailable(&why)) g_rrError = why;
        else if (!ViewMatricesOk(v)) g_rrError = "the View buffer's camera matrices aren't as expected (see the log)";
        else if (BuildGuides(cl, s, depth, size, v, guides)) useRr = true, g_rrError.clear();
        if (!useRr) Why("ray reconstruction: not running (super resolution instead)");
    }

    const unsigned rw = static_cast<unsigned>(size[0]), rh = static_cast<unsigned>(size[1]);
    g_lastJitter[0] = jitter[0], g_lastJitter[1] = jitter[1];
    if (g_resetAll) g_resetByOutput.clear(), g_resetAll = false;
    auto rs = g_resetByOutput.find(outKey);
    const uint64_t sizeKey = (static_cast<uint64_t>(rw) << 32) | rh;
    auto ss = g_sizeByOutput.find(outKey);
    const bool resized = ss != g_sizeByOutput.end() && ss->second != sizeKey;
    g_sizeByOutput[outKey] = sizeKey;
    if (g_sizeByOutput.size() > 64) g_sizeByOutput.clear();
    const bool reset = rs == g_resetByOutput.end() || rs->second || resized;

    dlss::Inputs in;
    in.color = reinterpret_cast<ID3D12Resource*>(color.handle);
    in.motion = reinterpret_cast<ID3D12Resource*>(motion.handle);
    in.depth = reinterpret_cast<ID3D12Resource*>(depth.handle);
    in.output = reinterpret_cast<ID3D12Resource*>(output.handle);
    if (g_set.maskMode > 0) in.biasMask = reinterpret_cast<ID3D12Resource*>(mask.handle);
    in.renderWidth = rw, in.renderHeight = rh;
    g_lastRender[0] = rw, g_lastRender[1] = rh;
    g_lastOutput[0] = out.texture.width, g_lastOutput[1] = out.texture.height;
    in.outputWidth = out.texture.width, in.outputHeight = out.texture.height;
    in.jitterX = jitter[0], in.jitterY = jitter[1];
    in.motionScaleX = 1, in.motionScaleY = 1;  // motion.hlsl writes previous minus current in pixels
    in.reset = reset;
    in.preset = g_set.preset;
    if (useRr) {
        in.rr = true;
        in.rrPreset = g_set.rrPreset;
        in.diffuseAlbedo = reinterpret_cast<ID3D12Resource*>(guides[0].handle);
        in.specularAlbedo = reinterpret_cast<ID3D12Resource*>(guides[1].handle);
        in.normalRoughness = reinterpret_cast<ID3D12Resource*>(guides[2].handle);
        in.worldToView = v + kViewTranslatedWorldToView / 4;
        in.viewToClip = v + kViewToClipNoAA / 4;
    }
    bool ok = false;
    if (backend == kDlss) {
        ok = dlss::Evaluate(reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()), in, err);
    } else {
        fsr::Inputs fi;
        fi.color = in.color, fi.depth = in.depth, fi.motion = in.motion, fi.output = in.output;
        fi.reactive = in.biasMask;  // the moving-objects mask, if switched on (off by default)
        fi.renderWidth = rw, fi.renderHeight = rh;
        fi.outputWidth = in.outputWidth, fi.outputHeight = in.outputHeight;
        const float sign = g_set.fsrJitterFlip ? -1.0f : 1.0f;
        fi.jitterX = sign * jitter[0], fi.jitterY = sign * jitter[1];
        fi.motionScaleX = 1, fi.motionScaleY = 1;
        CameraFromView(view.data(), &fi.cameraNear, &fi.cameraFovY);
        fi.sharpness = g_set.sharpness;
        fi.reset = reset;
        fi.velocityFactor = g_set.fsrVelocity;
        fi.minDisocclusion = g_set.fsrSwaying;
        fi.shadingChangeScale = g_set.fsrShading;
        fi.debugView = g_set.fsrDebug;
        ok = fsr::Evaluate(reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()), fi, err);
    }
    cl->barrier(output, resource_usage::unordered_access, resource_usage::unordered_access);
    if (ok && g_set.showMask) {
        cl->bind_descriptor_tables(shader_stage::all_compute, s.compute.layout, 0, 0, nullptr);  // ReShade's heaps back
        TintMask(cl, output, mask, rw, rh);
    }
    if (ok && useRr && g_set.guideView > 0) {
        cl->bind_descriptor_tables(shader_stage::all_compute, s.compute.layout, 0, 0, nullptr);
        ShowGuide(cl, output, guides, g_set.guideView, rw, rh);
    }
    if (ok) {
        if (useRr) {
            if (++g_rrEvaluated == 1) Log("DLSS Ray Reconstruction: first frame evaluated (%ux%u -> %ux%u)", rw, rh, out.texture.width, out.texture.height);
            g_rrLastPresent = g_presents.load();
        }
        g_lastWasRr = useRr;
    }
    Restore(cl, s);
    if (!ok) {
        ++vs.other;
        ++g_skipped;
        if (g_error != err) {
            g_error = err;
            Log("%s: %s", BackendName(backend), err.c_str());
        }
        return false;  // the game's TAA still runs
    }
    ++vs.evaluated;
    if (FrameGenIsScreen(out.texture.width, out.texture.height)) {
        ++vs.tagged;
        FrameGenSetCamera(view.data(), kViewClipToPrevClip);
        FrameGenTag(reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()), in.depth, in.motion, rw, rh,
                    in.jitterX, in.jitterY, reset);
    }
    g_resetByOutput[outKey] = false;
    g_error.clear();
    if (++g_evaluated == 1)
        Log("%s: first frame evaluated (%ux%u -> %ux%u, jitter %.3f %.3f px)", BackendName(backend), rw, rh,
            out.texture.width, out.texture.height, jitter[0], jitter[1]);
    return true;  // the upscaler wrote the TAA's output: skip the game's dispatch
}

// Whether the bound compute shader is a TAA permutation: the known one, or one declaring the TAA's
// registers with its binding shapes (1x1 exposure, HDR colour, depth, rg16 velocity, HDR history
// and output). Checked once per shader id; logged.
bool IsTaa(command_list* cl, ListState& s) {
    if (s.computeId == kTaaMain) return true;
    if (!s.computeId) return false;
    static std::map<uint32_t, bool> verdict;
    auto it = verdict.find(s.computeId);
    if (it != verdict.end()) return it->second;
    if (!PipeHasTaaRegisters(s.pipe)) return verdict[s.computeId] = false;
    device* dev = cl->get_device();
    resource r[6];
    for (uint32_t i = 0; i < 6; ++i) r[i] = Bound(cl, s.compute, 0, i);
    const resource out = Bound(cl, s.compute, 2, 0);
    for (const resource& x : r)
        if (!x.handle) return false;  // not decided: try again at its next dispatch
    if (!out.handle) return false;
    resource_desc d[6];
    for (uint32_t i = 0; i < 6; ++i) d[i] = dev->get_resource_desc(r[i]);
    const resource_desc od = dev->get_resource_desc(out);
    const bool match = d[0].texture.width == 1 && d[0].texture.height == 1 && IsHdrColor(d[1].texture.format) &&
                       d[2].texture.format == format::r32_g8_typeless && IsVelocityFormat(d[3].texture.format) &&
                       IsHdrColor(d[5].texture.format) && IsHdrColor(od.texture.format) &&
                       od.texture.width >= d[1].texture.width && od.texture.height >= d[1].texture.height;
    if (verdict.size() < 256) verdict[s.computeId] = match;
    if (match)
        Log("TAA permutation cs%08x taken (colour %ux%u -> output %ux%u)", s.computeId, d[1].texture.width,
            d[1].texture.height, od.texture.width, od.texture.height);
    return match;
}

bool OnDispatch(command_list* cl, uint32_t x, uint32_t y, uint32_t z) {
    if (!g_set.enabled) return false;
    ListState* s = State(cl);
    if (!s) return false;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    (void)x, (void)y, (void)z;
    return IsTaa(cl, *s) && OnTaa(cl, *s);
}

void OnDestroyDevice(device* dev) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    dlss::Shutdown();
    fsr::Shutdown();
    g_lastBackend = 0, g_autoBackend = 0;
    g_pool.Destroy(dev);
    for (auto& [key, v] : g_motion.views) dev->destroy_resource_view(v);
    g_motion.views.clear();
    for (MotionState::Texture& t : g_motion.textures) dev->destroy_resource(t.tex);
    if (g_motion.pipe.handle) dev->destroy_pipeline(g_motion.pipe);
    if (g_motion.tintPipe.handle) dev->destroy_pipeline(g_motion.tintPipe);
    if (g_motion.composePipe.handle) dev->destroy_pipeline(g_motion.composePipe);
    if (g_motion.guidePipe.handle) dev->destroy_pipeline(g_motion.guidePipe);
    if (g_motion.guideLayout.handle) dev->destroy_pipeline_layout(g_motion.guideLayout);
    if (g_motion.guideViewPipe.handle) dev->destroy_pipeline(g_motion.guideViewPipe);
    if (g_motion.guideViewLayout.handle) dev->destroy_pipeline_layout(g_motion.guideViewLayout);
    g_gbuffers.clear();
    if (g_motion.composeLayout.handle) dev->destroy_pipeline_layout(g_motion.composeLayout);
    if (g_motion.tintLayout.handle) dev->destroy_pipeline_layout(g_motion.tintLayout);
    if (g_motion.layout.handle) dev->destroy_pipeline_layout(g_motion.layout);
    g_motion = MotionState();
    g_links.clear();
}

bool g_keyWasDown = false;

// ---------------------------------------------------------------- render resolution (DLSS mode)
//
// DLSS renders at r.ScreenPercentage. A DLSS mode sets it to NVIDIA's ratio for that mode, with
// r.TemporalAA.Upsampling=1 so UE's TAA (which DLSS replaces) takes the render-size image and
// writes the full-size output, while DLSS is on; both go back to the game's values and priorities
// when DLSS is off. In UE 4 a cvar change only reaches the renderer when it's set on the game
// thread, so the setting happens in the console manager's CallAllConsoleVariableSinks, called
// once per frame by FEngineLoop::Tick (cvars.hpp HookSinks, as in TrueColorsTweaks; the two
// add-ons' hooks chain). The present thread only says what it wants (g_res.want).

enum DlssMode { kModeGame, kModeDlaa, kModeQuality, kModeBalanced, kModePerformance, kModeUltraPerformance, kDlssModeCount };
const char* const kDlssModeNames[kDlssModeCount] = {"Game's (100%)", "Native / DLAA (100%)", "Quality (67%)",
                                                    "Balanced (58%)", "Performance (50%)", "Ultra Performance (33%)"};
const float kDlssModePercent[kDlssModeCount] = {0, 100, 66.667f, 58, 50, 33.333f};

struct Resolution {
    void* manager = nullptr;
    void* screen = nullptr;  // r.ScreenPercentage
    void* upsample = nullptr;  // r.TemporalAA.Upsampling
    bool failed = false, hooked = false;
    std::atomic<float> want{0};  // render percentage wanted (0: the game's)
    // game thread:
    bool overriding = false;
    float gameScreen = 100, ourScreen = 0;
    int gameUpsample = 0;
    uint32_t gameScreenPriority = 0, gameUpsamplePriority = 0;
    uint64_t ticks = 0;
    std::string error;
} g_res;

cvars::SinksFn g_origSinks = nullptr;

// UE 4.25's ray tracing denoisers (names from the game's exe). Ray Reconstruction needs the raw,
// noisy ray-traced lighting, so they go to 0 while it runs and back to the game's values (and
// priorities) after. Set on the game thread like the render resolution.
const wchar_t* const kDenoiserNames[] = {L"r.Shadow.Denoiser",          L"r.Reflections.Denoiser",
                                         L"r.AmbientOcclusion.Denoiser", L"r.DiffuseIndirect.Denoiser",
                                         L"r.RayTracing.GlobalIllumination.Denoiser", L"r.RayTracing.SkyLight.Denoiser"};
constexpr int kDenoiserCount = 6;
struct Denoisers {
    void* var[kDenoiserCount] = {};
    int32_t game[kDenoiserCount] = {};
    uint32_t priority[kDenoiserCount] = {};
    bool overriding = false;  // game thread
    std::atomic<bool> wantOff{false};
    std::atomic<int> off{0};  // how many are off (overlay)
} g_dn;

void FindDenoisers() {
    for (int i = 0; i < kDenoiserCount; ++i) {
        g_dn.var[i] = cvars::FindVariable(g_res.manager, kDenoiserNames[i]);
        int32_t value = -1;
        if (g_dn.var[i] && !cvars::GetInt(g_dn.var[i], value)) g_dn.var[i] = nullptr;
        Log("Ray Reconstruction: %ls %s (%d)", kDenoiserNames[i], g_dn.var[i] ? "found" : "not found", value);
    }
}

// Game thread, about every 15 frames.
void DenoiserTick() {
    if (g_dn.wantOff) {
        int n = 0;
        for (int i = 0; i < kDenoiserCount; ++i) {
            int32_t now = 0;
            if (!g_dn.var[i] || !cvars::GetInt(g_dn.var[i], now)) continue;
            if (!g_dn.overriding) {
                g_dn.game[i] = now;
                g_dn.priority[i] = cvars::GetFlags(g_dn.var[i]) & cvars::kPriorityMask;
            } else if (now != 0) {
                g_dn.game[i] = now;  // the game set it again
                Log("Ray Reconstruction: the game set %ls to %d again", kDenoiserNames[i], now);
            }
            if (now != 0) cvars::Set(g_dn.var[i], L"0");
            ++n;
        }
        if (!g_dn.overriding)
            Log("Ray Reconstruction: %d ray tracing denoisers off (game's: shadows %d, reflections %d, AO %d, diffuse "
                "indirect %d, GI %d, sky light %d)", n, g_dn.game[0], g_dn.game[1], g_dn.game[2], g_dn.game[3],
                g_dn.game[4], g_dn.game[5]);
        g_dn.overriding = true;
        g_dn.off = n;
    } else if (g_dn.overriding) {
        for (int i = 0; i < kDenoiserCount; ++i) {
            if (!g_dn.var[i]) continue;
            wchar_t text[16];
            swprintf(text, 16, L"%d", g_dn.game[i]);
            cvars::Set(g_dn.var[i], text);
            cvars::RestorePriority(g_dn.var[i], g_dn.priority[i]);
        }
        g_dn.overriding = false;
        g_dn.off = 0;
        Log("Ray Reconstruction off: ray tracing denoisers back to the game's values");
    }
}

bool SetFloat(void* var, float v) {
    wchar_t text[32];
    swprintf(text, 32, L"%.3f", v);
    return cvars::Set(var, text);
}

// Game thread, about every 15 frames.
void ResolutionTick() {
    if (++g_res.ticks % 15 != 0) return;
    DenoiserTick();
    const float want = g_res.want.load();
    float now = 0;
    int32_t up = 0;
    if (!cvars::GetFloat(g_res.screen, now) || !cvars::GetInt(g_res.upsample, up)) return;
    if (want > 0) {
        if (!g_res.overriding) {
            g_res.gameScreen = now, g_res.gameUpsample = up;
            g_res.gameScreenPriority = cvars::GetFlags(g_res.screen) & cvars::kPriorityMask;
            g_res.gameUpsamplePriority = cvars::GetFlags(g_res.upsample) & cvars::kPriorityMask;
            g_res.overriding = true;
        } else if (std::fabs(now - g_res.ourScreen) > 0.01f && std::fabs(now - want) > 0.01f) {
            g_res.gameScreen = now;  // the game set it again (menu, resolution change)
            Log("DLSS mode: the game set its screen percentage to %g%%", now);
        }
        if (std::fabs(now - want) > 0.01f) {
            const bool ok = SetFloat(g_res.screen, want);
            g_res.ourScreen = want;
            Log("DLSS mode: render resolution %g%% %s (the game's: %g%%)", want, ok ? "set" : "NOT set", g_res.gameScreen);
        }
        if (up != 1) cvars::Set(g_res.upsample, L"1");
    } else if (g_res.overriding) {
        g_res.overriding = false;
        if (std::fabs(now - g_res.ourScreen) < 0.01f) {
            SetFloat(g_res.screen, g_res.gameScreen);
            cvars::RestorePriority(g_res.screen, g_res.gameScreenPriority);
        }
        wchar_t text[16];
        swprintf(text, 16, L"%d", g_res.gameUpsample);
        cvars::Set(g_res.upsample, text);
        cvars::RestorePriority(g_res.upsample, g_res.gameUpsamplePriority);
        Log("DLSS mode: back to the game's screen percentage (%g%%) and TAA upsampling (%d)", g_res.gameScreen,
            g_res.gameUpsample);
    }
}

void __fastcall HookedSinks(void* self) {
    ResolutionTick();
    g_origSinks(self);
}

bool FindResolutionCvars() {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
    void** singleton = cvars::FindConsoleManagerSingleton(base);
    void* manager = nullptr;
    if (singleton) cvars::SafeCopy(&manager, singleton, 8);
    if (!manager || !cvars::IsConsoleManager(base, manager)) {
        g_res.error = "the game's console variables weren't found";
        return false;
    }
    void* aniso = cvars::FindVariable(manager, L"r.MaxAnisotropy");
    void* screen = cvars::FindVariable(manager, L"r.ScreenPercentage");
    void* upsample = cvars::FindVariable(manager, L"r.TemporalAA.Upsampling");
    float value = -1;
    if (!aniso || !screen || !upsample || !cvars::HasModuleVtable(base, screen) || !cvars::FindTypeSlots(aniso, screen) ||
        !cvars::GetFloat(screen, value) || !(value >= 10 && value <= 400)) {
        g_res.error = "r.ScreenPercentage didn't look right (see the log)";
        Log("DLSS mode: r.ScreenPercentage %p reads %g, r.TemporalAA.Upsampling %p: not used", screen, value, upsample);
        return false;
    }
    g_res.manager = manager, g_res.screen = screen, g_res.upsample = upsample;
    if (!cvars::HookSinks(base, manager, &HookedSinks, &g_origSinks)) {
        g_res.error = "couldn't reach the game thread";
        Log("DLSS mode: game-thread hook failed");
        return false;
    }
    g_res.hooked = true;
    Log("DLSS mode: r.ScreenPercentage found (now %g); game-thread hook in place", value);
    FindDenoisers();
    return true;
}

// ---------------------------------------------------------------- velocity settings (logged)

const wchar_t* const kVelocityLogged[] = {L"r.BasePassOutputsVelocity", L"r.SkinCache.Mode",
                                          L"r.TemporalAA.Upsampling", L"r.ScreenPercentage",
                                          L"r.TemporalAASamples", L"r.TemporalAACurrentFrameWeight"};
bool g_velocityLogged = false;

void LogVelocitySettings() {
    if (!g_res.manager || g_velocityLogged) return;
    g_velocityLogged = true;
    for (const wchar_t* name : kVelocityLogged) {
        void* var = cvars::FindVariable(g_res.manager, name);
        if (!var) {
            Log("cvar: %ls not found", name);
            continue;
        }
        wchar_t buf[64] = L"?";
        cvars::GetText(var, cvars::GetType(var), buf, 64);
        Log("cvar: %ls = %ls (%s, set by 0x%02x)", name, buf, cvars::TypeName(cvars::GetType(var)), cvars::Priority(var));
    }
}

// About twice a second, on ReShade's present: what the game thread should hold.
void UpdateResolution() {
    static uint64_t presents = 0;
    if (++presents % 30 != 0) return;
    const bool want = g_set.enabled && g_set.mode > kModeGame && g_set.mode < kDlssModeCount;
    if (!g_res.manager && !g_res.failed && (want || g_set.enabled) && !FindResolutionCvars()) g_res.failed = true;
    LogVelocitySettings();
    g_res.want = want ? kDlssModePercent[g_set.mode] : 0.0f;
    // Denoisers off while Ray Reconstruction ran in the last couple of seconds.
    g_dn.wantOff = g_set.enabled && g_set.rr && g_rrEvaluated.load() > 0 && g_presents.load() - g_rrLastPresent.load() < 120;
}

void LogStats() {
    static uint64_t presents = 0;
    if (++presents % 120 != 0) return;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const std::string fg = FrameGenDiag();  // also resets frame generation's counters
    bool problem = !g_why.empty() || fg.find("untagged 0,") == std::string::npos;
    for (const auto& [key, v] : g_stats) problem |= v.noOutput || v.noDepth || v.noMotion || v.noColor || v.other;
    if (!g_set.enabled || g_statLines >= 30 || !problem) {
        g_stats.clear(), g_why.clear();
        return;
    }
    std::string line;
    char buf[200];
    for (const auto& [key, v] : g_stats) {
        snprintf(buf, sizeof(buf), " [%ux%u: upscaled %u, tagged %u, no output %u, depth %u, motion %u, colour %u, failed %u]",
                 static_cast<unsigned>(key >> 32), static_cast<unsigned>(key & 0xffffffff), v.evaluated, v.tagged,
                 v.noOutput, v.noDepth, v.noMotion, v.noColor, v.other);
        line += buf;
    }
    for (const auto& [why, n] : g_why) {
        snprintf(buf, sizeof(buf), " {%s: %u}", why.c_str(), n);
        line += buf;
    }
    line += fg;
    if (++g_statLines == 30) line += " (last diag line)";
    Log("diag (render %ux%u):%s", g_lastRender[0], g_lastRender[1], line.c_str());
    g_stats.clear(), g_why.clear();
}

void OnPresent(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*) {
    ++g_presents;
    LogStats();
    UpdateResolution();
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    const bool down = pid == GetCurrentProcessId() && (GetAsyncKeyState(kToggleHotkey) & 0x8000);
    dlss::EndFrame();
    fsr::EndFrame();
    if (down && !g_keyWasDown && kTsrMapped) {
        g_set.enabled = !g_set.enabled;
        g_resetAll = true;
        SaveSettings();
        Log("%s %s (F9)", BackendName(g_lastBackend), g_set.enabled ? "on" : "off");
    }
    g_keyWasDown = down;
}

}  // namespace

void RestoreListState(command_list* cl) {
    if (ListState* s = State(cl)) Restore(cl, *s);
}

void RegisterUpscaler() {
    LoadSettings();
    reshade::register_event<reshade::addon_event::init_command_list>(&OnInitList);
    reshade::register_event<reshade::addon_event::destroy_command_list>(&OnDestroyList);
    reshade::register_event<reshade::addon_event::reset_command_list>(&OnResetList);
    reshade::register_event<reshade::addon_event::bind_pipeline>(&OnBindPipeline);
    reshade::register_event<reshade::addon_event::bind_descriptor_tables>(&OnBindTables);
    reshade::register_event<reshade::addon_event::push_descriptors>(&OnPushDescriptors);
    reshade::register_event<reshade::addon_event::push_constants>(&OnPushConstants);
    reshade::register_event<reshade::addon_event::dispatch>(&OnDispatch);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&OnBindRenderTargets);
    reshade::register_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::register_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::register_event<reshade::addon_event::present>(&OnPresent);
}

void UnregisterUpscaler() {
    if (g_res.hooked) cvars::UnhookSinks(g_res.manager), g_res.hooked = false;
    reshade::unregister_event<reshade::addon_event::init_command_list>(&OnInitList);
    reshade::unregister_event<reshade::addon_event::destroy_command_list>(&OnDestroyList);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(&OnResetList);
    reshade::unregister_event<reshade::addon_event::bind_pipeline>(&OnBindPipeline);
    reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(&OnBindTables);
    reshade::unregister_event<reshade::addon_event::push_descriptors>(&OnPushDescriptors);
    reshade::unregister_event<reshade::addon_event::push_constants>(&OnPushConstants);
    reshade::unregister_event<reshade::addon_event::dispatch>(&OnDispatch);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&OnBindRenderTargets);
    reshade::unregister_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::unregister_event<reshade::addon_event::present>(&OnPresent);
}

void SaveAndLog() {
    g_resetAll = true;
    SaveSettings();
    Log("settings: upscaler %s (%s), DLSS preset %d, FSR sharpness %.2f, mode %s, Ray Reconstruction %s (preset %d)",
        g_set.enabled ? "on" : "off", kBackendNames[g_set.backend], g_set.preset, g_set.sharpness,
        kDlssModeNames[g_set.mode >= 0 && g_set.mode < kDlssModeCount ? g_set.mode : 0], g_set.rr ? "on" : "off",
        g_set.rrPreset);
}

// DLSS models offered (NVSDK_NGX_DLSS_Hint_Render_Preset_*; dlss::kPresetAuto = per render scale).
const char* const kModelNames[] = {"Automatic (recommended)", "K (transformer)", "J (transformer)",
                                   "L (transformer 2, DLSS 4.5)", "M (transformer 2, DLSS 4.5)", "NVIDIA's default"};
const int kModelIds[] = {dlss::kPresetAuto, 11, 10, 12, 13, 0};
constexpr int kModelCount = 6;
int ModelIndex(int id) {
    for (int i = 0; i < kModelCount; ++i)
        if (kModelIds[i] == id) return i;
    return 0;
}

const char* PresetName(int id) {
    if (id == dlss::kPresetAuto) return "Automatic";
    if (id == 0) return "NVIDIA default";
    static char buf[8];
    snprintf(buf, sizeof(buf), "%c", id > 0 && id < 26 ? static_cast<char>('A' + id - 1) : '?');
    return buf;
}

void DrawUpscalerSettings() {
    ImGui::SeparatorText("Upscaler");
    ImGui::BeginDisabled(!kTsrMapped);
    if (ImGui::Checkbox("On (F9)", &g_set.enabled)) SaveAndLog();
    ImGui::EndDisabled();
    Help("DLSS or FSR in place of the game's TAA anti-aliasing. F9 switches it on and off while playing, to compare.");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Upscaler", &g_set.backend, kBackendNames, kBackendCount)) SaveAndLog();
    Help("DLSS: NVIDIA RTX graphics cards.\nFSR: any DirectX 12 graphics card. AMD's runtime picks FSR 4 on Radeon "
         "cards that support it (RX 6000 and newer) and FSR 3.1 on the others.\nAuto: DLSS where it runs, FSR "
         "elsewhere.\n\nBoth need the game's DirectX 12 mode.");
    const int shown = g_set.backend != kAuto ? g_set.backend : g_autoBackend;  // 0: Auto not resolved yet
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Render resolution", &g_set.mode, kDlssModeNames, kDlssModeCount)) SaveAndLog();
    Help("The resolution the upscaler renders at, as a share of the screen's. The game itself renders at 100%, so\n"
         "\"Game's\" and Native are the same (best image, no speed-up). Quality, Balanced and Performance render\n"
         "fewer pixels and let the upscaler rebuild the full image: faster. The game's values come back when the "
         "upscaler is off.");
    if (g_res.failed) ImGui::TextColored(kWarn, "Render resolution: %s", g_res.error.c_str());

    if (shown != kFsr) {
        int cur = ModelIndex(g_set.preset);
        ImGui::SetNextItemWidth(ControlWidth());
        if (ImGui::Combo("DLSS model", &cur, kModelNames, kModelCount)) g_set.preset = kModelIds[cur], SaveAndLog();
        Help("Which DLSS model upscales. Automatic follows NVIDIA's recommendation for the render "
             "resolution: K from 100% down to Balanced, M at Performance (50%), L at Ultra Performance "
             "(33%). K, J, L or M use that model at every resolution; L and M (DLSS 4.5's "
             "second-generation transformer) cost more, especially on RTX 20 and 30 cards. NVIDIA's "
             "default leaves it to the driver. A DLSS override set in the NVIDIA App takes precedence.");

        if (ImGui::Checkbox("Ray Reconstruction", &g_set.rr)) SaveAndLog();
        Help("NVIDIA's DLSS Ray Reconstruction: one AI model cleans up the ray-traced effects AND upscales, in place\n"
             "of the game's ray tracing denoisers and DLSS Super Resolution. Sharper, steadier ray-traced\n"
             "reflections, shadows and ambient occlusion, with less smearing in motion; costs a bit more than Super\n"
             "Resolution. The game's denoisers are switched off while it runs and come back when it's off.\n"
             "Only makes a difference with ray-traced effects on (TrueColorsTweaks' ray tracing options).");
        if (g_set.rr) {
            static const char* const kRrNames[] = {"NVIDIA's default", "D (transformer)", "E (latest transformer)",
                                                   "F (RR 2)"};
            static const int kRrIds[] = {0, 4, 5, 6};
            int cur = g_set.rrPreset == 4 ? 1 : g_set.rrPreset == 5 ? 2 : g_set.rrPreset == 6 ? 3 : 0;
            ImGui::SetNextItemWidth(ControlWidth());
            if (ImGui::Combo("Ray Reconstruction model", &cur, kRrNames, 4)) g_set.rrPreset = kRrIds[cur], SaveAndLog();
            Help("Which Ray Reconstruction model runs: D (transformer), E (newer transformer) or F (NVIDIA's\n"
                 "second-generation Ray Reconstruction model). NVIDIA's default lets the driver choose. The DLSS\n"
                 "model choice above doesn't apply while Ray Reconstruction is on.\n\n"
                 "A Ray Reconstruction model override set in the NVIDIA App (DLSS Override - Model Presets) takes\n"
                 "precedence over this choice: set it to \"Use app default\" there for this choice to apply.");
            std::string err;
            {
                std::lock_guard<std::recursive_mutex> lock(g_mutex);
                err = g_rrError;
            }
            if (!err.empty() && g_set.enabled) ImGui::TextColored(kWarn, "Ray Reconstruction: %s", err.c_str());
        }
    }
    if (shown != kDlss) {
        ImGui::SetNextItemWidth(ControlWidth());
        if (ImGui::SliderFloat("FSR sharpness", &g_set.sharpness, 0, 1, "%.2f")) SaveSettings();
        if (ImGui::IsItemDeactivatedAfterEdit()) Log("settings: FSR sharpness %.2f", g_set.sharpness);
        Help("Extra sharpening after FSR (AMD's RCAS). 0 is off.");
    }
}

void DrawUpscalerAdvanced() {
    ImGui::SeparatorText("Motion of moving objects");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Object motion", &g_set.objectMode, kObjectModeNames, kObjectModeCount)) {
        g_resetAll = true;
        Log("diagnosis: object motion %s", kObjectModeNames[g_set.objectMode]);
    }
    Help("How the motion of things that move by themselves (characters, trees in the wind, cloth) is read from the "
         "engine. Linear is UE 4's encoding; the others are for testing and go back to it at each launch.");

    ImGui::SeparatorText("Moving objects mask");
    bool movingMask = g_set.maskMode == 2;
    if (ImGui::Checkbox("Mask moving objects", &movingMask)) g_set.maskMode = movingMask ? 2 : 0, SaveAndLog();
    Help("Tells the upscaler to trust the current frame over its history on moving objects: less smearing behind\n"
         "characters, slightly less stable fine detail on them. FSR takes it much more strongly than DLSS (noise).");
    ImGui::Checkbox("Show the mask in red", &g_set.showMask);
    Help("Tints the masked pixels red. For testing; off at each launch.");

    ImGui::SeparatorText("Ray Reconstruction guides");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Show guide", &g_set.guideView, kGuideViewNames, kGuideViewCount))
        Log("diagnosis: show guide %s", kGuideViewNames[g_set.guideView]);
    Help("Shows one of the surface pictures Ray Reconstruction gets from the game in place of the image: surface\n"
         "colour (diffuse albedo), reflectivity (specular albedo), normals or roughness. For checking they are\n"
         "read right; only while Ray Reconstruction runs. Off at each launch.");

    ImGui::SeparatorText("History");
    if (ImGui::Button("Reset upscaler history")) g_resetAll = true;
    Help("Starts the upscaler's accumulation over, as after a camera cut.");

    ImGui::SeparatorText("FSR testing");
    if (ImGui::Checkbox("Flip FSR's jitter", &g_set.fsrJitterFlip)) {
        g_resetAll = true;
        Log("diagnosis: FSR jitter %s", g_set.fsrJitterFlip ? "flipped" : "as DLSS's");
    }
    Help("Passes the camera jitter to FSR with the opposite sign. The right sign gives a steady, sharp picture "
         "when standing still; the wrong one shimmers. Off at each launch.");
    const auto tuned = [](const char* what, float v) {
        SaveSettings();
        Log("settings: FSR %s %.3f", what, v);
    };
    ImGui::SetNextItemWidth(ControlWidth());
    ImGui::SliderFloat("Swaying detail", &g_set.fsrSwaying, -1, 1, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) tuned("swaying detail", g_set.fsrSwaying);
    Help("FSR 3.1: how much history FSR keeps where things uncover each other (character edges, hair, leaves). "
         "Higher: less noise there, possibly a little ghosting. AMD's default -0.33.");
    ImGui::SetNextItemWidth(ControlWidth());
    ImGui::SliderFloat("Velocity factor", &g_set.fsrVelocity, 0, 1, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) tuned("velocity factor", g_set.fsrVelocity);
    Help("FSR 3.1: how much motion lowers trust in history. Lower: steadier moving detail, possibly more "
         "smearing. AMD's default 1.");
    ImGui::SetNextItemWidth(ControlWidth());
    ImGui::SliderFloat("Shading change", &g_set.fsrShading, 0, 2, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) tuned("shading change", g_set.fsrShading);
    Help("FSR 3.1: how strongly FSR reacts to lighting changing on a surface. Lower: less noise; too low can "
         "leave trails. AMD's default 1.");
    if (ImGui::Button("FSR defaults")) {
        g_set.fsrSwaying = -1.0f / 3, g_set.fsrVelocity = 1, g_set.fsrShading = 1;
        tuned("tuning reset, swaying", g_set.fsrSwaying);
    }
    if (ImGui::Checkbox("FSR debug view", &g_set.fsrDebug)) Log("diagnosis: FSR debug view %s", g_set.fsrDebug ? "on" : "off");
    Help("AMD's debug overlay in place of the picture: FSR's motion vectors, depth, disocclusion and how much "
         "history each pixel keeps. Off at each launch.");
}

std::string UpscalerHeadline(bool* warn) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    *warn = false;
    if (!kTsrMapped) return (*warn = true), "Upscaler unavailable: TAA not mapped for this game build";
    if (!g_set.enabled) return "Upscaler off: the game's TAA";
    const std::string name = g_lastBackend == kFsr ? fsr::VersionName() : BackendName(g_lastBackend);
    if (!g_error.empty()) return (*warn = true), name + ": " + g_error;
    if (!g_lastRender[0] || !g_lastBackend) return "Upscaler on: starts with the next game frame";
    char buf[160];
    if (g_lastBackend == kDlss && g_lastWasRr)
        snprintf(buf, sizeof(buf), "DLSS Ray Reconstruction on: %ux%u -> %ux%u", g_lastRender[0], g_lastRender[1],
                 g_lastOutput[0], g_lastOutput[1]);
    else if (g_lastBackend == kDlss && g_set.rr && !g_rrError.empty())
        return (*warn = true), "DLSS on, Ray Reconstruction not running: " + g_rrError;
    else if (g_lastBackend == kDlss)
        snprintf(buf, sizeof(buf), "DLSS on: %ux%u -> %ux%u, model %s", g_lastRender[0], g_lastRender[1],
                 g_lastOutput[0], g_lastOutput[1], PresetName(g_set.preset));
    else
        snprintf(buf, sizeof(buf), "%s on: %ux%u -> %ux%u, sharpness %.2f", name.c_str(), g_lastRender[0],
                 g_lastRender[1], g_lastOutput[0], g_lastOutput[1], g_set.sharpness);
    return buf;
}

void DrawUpscalerStatus() {
    ImGui::SeparatorText("Upscaler");
    const int b = g_lastBackend;
    ImGui::TextDisabled("%s", !g_set.enabled ? "off" : b == kFsr ? fsr::Status().c_str() : b == kDlss ? dlss::Status().c_str() : "waiting for a game frame");
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        ImGui::TextDisabled("Frames upscaled %llu, left to the game's TAA %llu (menus, loading, camera cuts)",
                            static_cast<unsigned long long>(g_evaluated.load()),
                            static_cast<unsigned long long>(g_skipped.load()));
        if (g_motion.composed || g_motion.copied)
            ImGui::TextDisabled("Translucency: %s", g_motion.composed ? "blended into the upscaler's input"
                                                                      : "not found (colour copied without it)");
        if (g_res.hooked)
            ImGui::TextDisabled("Screen percentage: %s", g_res.want.load() > 0 ? "set by the Render resolution choice" : "the game's");
        if (g_set.rr)
            ImGui::TextDisabled("Ray Reconstruction frames %llu; ray tracing denoisers: %s",
                                static_cast<unsigned long long>(g_rrEvaluated.load()),
                                g_dn.off.load() ? "off (Ray Reconstruction denoises)" : "the game's");
    }
}

}  // namespace rd
