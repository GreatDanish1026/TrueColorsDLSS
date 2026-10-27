// Replaces TSR's output with DLSS.
//
// TSR's intermediates are transient (their memory is reused within the frame), so the inputs are
// taken while they exist:
//   DilateVelocity (t0 depth, t1 engine velocity): our own motion vectors (motion.hlsl) are
//     computed there into our own texture. The engine's velocity only covers moving objects; the
//     camera's motion comes from the depth and View.ClipToPrevClip (found in the View uniform
//     buffer from TemporalAAJitter). DilateVelocity is recorded in a different command list, so
//     the result is found later through its output u0, which DecimateHistory reads (t0; in UE 5.6
//     it was RejectShading's t8).
//   UpdateHistory (t0 scene color, t1 separate translucency): UE 5.2's TSR blends translucency
//     (particles, light beams, light shafts, outline materials) in itself, so compose.hlsl blends
//     it into our own copy of the scene color; it also adds the translucency to the bias mask
//   ResolveHistory (u0): the 3840x2160 output, and from its parameters the render size, jitter and
//     motion vector scale (offsets 0, 256 and 272, docs/FINDINGS.md). The game's pass runs first
//     (it also writes a half-resolution copy the post-processing uses), then DLSS overwrites the
//     full output.
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
#include "motion_cs.h"  // kMotionCs: motion.hlsl compiled by build.ps1
#include "tint_cs.h"    // kTintCs: tint.hlsl (mask debug view)
#include "compose_cs.h"  // kComposeCs: compose.hlsl (translucency)
#include "version.h"

using namespace reshade::api;

namespace rd {
namespace {

// ---------------------------------------------------------------- settings

constexpr int kToggleHotkey = VK_F9;

struct Settings {
    bool enabled = false;  // remembered across launches
    int preset = 11;       // NVSDK_NGX_DLSS_Hint_Render_Preset_K (transformer)
    int mode = 0;          // DlssMode: 0 leaves the render resolution to the game's Resolution Scale
    int maskMode = 1;      // motion.hlsl MaskMode: 0 off, 1 animated materials, 2 also moving objects
    bool showMask = false;  // debug tint, not saved
    // motion.hlsl ObjectMode: 1 square-root decode, right for UE 5.2 (user's comparison, 2026-10-26: least
    // tree noise and satchel smearing); 0 linear (UE 5.6, Reunion) and 2 ignored stay for diagnosis. Not saved.
    int objectMode = 1;
    int transMask = 1;  // translucency in the bias mask: 0 off, 1 partial, 2 full
} g_set;

const char* const kTransMaskNames[] = {"Off", "Partial (recommended)", "Full"};
constexpr int kTransMaskCount = 3;
const float kTransMaskStrength[kTransMaskCount] = {0, 0.5f, 1};

const char* const kMaskModeNames[] = {"Off", "Animated materials", "Animated materials and moving objects"};
constexpr int kMaskModeCount = 3;
const char* const kObjectModeNames[] = {"Engine, linear (UE 5.6 games)", "Engine, square-root (this game)", "Ignore (camera motion only)"};
constexpr int kObjectModeCount = 3;

void LoadSettings() {
    reshade::get_config_value(nullptr, RD_PRODUCT, "Enabled", g_set.enabled);
    reshade::get_config_value(nullptr, RD_PRODUCT, "Preset", g_set.preset);
    reshade::get_config_value(nullptr, RD_PRODUCT, "DlssMode", g_set.mode);
    reshade::get_config_value(nullptr, RD_PRODUCT, "AnimationMask", g_set.maskMode);
    if (g_set.maskMode < 0 || g_set.maskMode >= kMaskModeCount) g_set.maskMode = 1;
    reshade::get_config_value(nullptr, RD_PRODUCT, "TranslucencyMask", g_set.transMask);
    if (g_set.transMask < 0 || g_set.transMask >= kTransMaskCount) g_set.transMask = 1;
    if (!kTsrMapped) g_set.enabled = false;  // TSR's passes aren't known in this game yet
}

void SaveSettings() {
    reshade::set_config_value(nullptr, RD_PRODUCT, "Enabled", g_set.enabled);
    reshade::set_config_value(nullptr, RD_PRODUCT, "Preset", g_set.preset);
    reshade::set_config_value(nullptr, RD_PRODUCT, "DlssMode", g_set.mode);
    reshade::set_config_value(nullptr, RD_PRODUCT, "AnimationMask", g_set.maskMode);
    reshade::set_config_value(nullptr, RD_PRODUCT, "TranslucencyMask", g_set.transMask);
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
} g_motion;

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

// Records motion.hlsl before the game's DilateVelocity (same inputs) and puts its state back.
resource ComputeMotion(command_list* cl, ListState& s, resource depth, resource velocity, resource* maskOut,
                       std::vector<float>* viewOut) {
    *maskOut = {0};
    static int badFormat = 0, badParams = 0;
    device* dev = cl->get_device();
    if (!velocity.handle) return Why("motion: no velocity"), resource{0};
    const resource_desc dd = dev->get_resource_desc(depth), vd = dev->get_resource_desc(velocity);
    if (dd.texture.format != format::r32_g8_typeless || vd.texture.format != format::r16g16b16a16_unorm) {
        LogOnce(badFormat, "motion: depth format %u, velocity format %u: TSR kept for this view",
                static_cast<uint32_t>(dd.texture.format), static_cast<uint32_t>(vd.texture.format));
        return {0};
    }
    float size[2] = {0, 0}, jitter[2] = {0, 0};
    if (!RenderSize(cl, s.compute, size) || !ReadConstants(cl, s.compute, 2, 256, jitter, 2) ||
        size[0] < 16 || size[1] < 16 || size[0] > dd.texture.width || size[1] > dd.texture.height ||
        std::fabs(jitter[0]) > 0.5f || std::fabs(jitter[1]) > 0.5f) {
        LogOnce(badParams, "motion: DilateVelocity parameters not usable (size %u x %u)", static_cast<uint32_t>(size[0]),
                static_cast<uint32_t>(size[1]));
        return {0};
    }
    static std::vector<float> view(kViewScanFloats);
    const uint32_t viewFloats = ReadConstantsUpTo(cl, s.compute, 3, view.data(), kViewScanFloats);
    if (viewFloats < kViewNeededFloats || (g_motion.offset >= 0 && viewFloats * 4u < static_cast<uint32_t>(g_motion.offset) + 144))
        return Why("motion: View buffer"), resource{0};
    if (g_motion.offset < 0) {
        FindClipToPrevClip(view.data(), jitter, size);
        if (g_motion.offset < 0) return Why("motion: camera not found yet"), resource{0};
    }
    if (!CreateMotionPipeline(dev)) return {0};
    const resource tex = MotionTexture(dev, dd.texture.width, dd.texture.height);
    const resource mask = MotionTexture(dev, dd.texture.width, dd.texture.height, format::r8_unorm);
    if (!tex.handle || !mask.handle) return Why("motion: no texture (limit)"), resource{0};
    const resource_view srvs[2] = {View(dev, depth, format::r32_float_x8_uint, false),
                                   View(dev, velocity, format::r16g16b16a16_unorm, false)};
    const resource_view uavs[2] = {View(dev, tex, format::r16g16_float, true), View(dev, mask, format::r8_unorm, true)};
    if (!srvs[0].handle || !srvs[1].handle || !uavs[0].handle || !uavs[1].handle) return Why("motion: views"), resource{0};

    float params[24];
    const float* m = view.data() + g_motion.offset / 4;
    memcpy(params, m, 16 * sizeof(float));
    // Frame generation takes the camera at ResolveHistory, and only for the view on screen (the game
    // renders a second TSR view): kept with this view's motion vectors until then.
    *viewOut = view;
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
    for (int kind = 0; kind < 2; ++kind) {
        auto it = g_motion.views.find(std::make_pair(r.handle, kind));
        if (it == g_motion.views.end()) continue;
        dev->destroy_resource_view(it->second);
        g_motion.views.erase(it);
    }
}

void OnDilateVelocity(command_list* cl, ListState& s) {
    const resource depth = Bound(cl, s.compute, 0, 0);
    const resource velocity = Bound(cl, s.compute, 0, 1);
    const resource out = Bound(cl, s.compute, 1, 0);
    if (!depth.handle || !out.handle) return;
    if (g_links.size() > 64) g_links.clear();
    Link& link = g_links[out.handle];
    link.depth = depth;
    link.motion = ComputeMotion(cl, s, depth, velocity, &link.mask, &link.view);
}

// UE 5.2: DecimateHistory is the first pass after DilateVelocity that reads its output (t0);
// RejectShading doesn't (5.6's read it at t8).
void OnChainStart(command_list* cl, ListState& s) {
    s.chain = Chain();
    const resource dilated = Bound(cl, s.compute, 0, 0);
    auto it = g_links.find(dilated.handle);
    if (it == g_links.end()) return Why("chain: DilateVelocity not seen");
    s.chain.depth = it->second.depth;
    s.chain.motion = it->second.motion;  // none: TSR kept for this view
    s.chain.mask = it->second.mask;
    s.chain.view = it->second.view;
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

void OnUpdateHistory(command_list* cl, ListState& s) {
    if (!s.chain.motion.handle) return Why("update: no motion in this list");
    if (Compose(cl, s)) return;
    ++g_motion.copied;
    s.chain.color = CopyInput(cl, Bound(cl, s.compute, 0, 0), resource_usage::shader_resource_non_pixel,
                              format::r16g16b16a16_float, "color");
    if (!s.chain.color.handle) Why("update: colour copy failed");
}

// ---------------------------------------------------------------- DLSS

bool Evaluate(command_list* cl, ListState& s, uint32_t x, uint32_t y, uint32_t z) {
    device* dev = cl->get_device();
    const Chain chain = s.chain;
    s.chain = Chain();  // used once
    const resource output = Bound(cl, s.compute, 1, 0);
    {
        uint64_t key = 0;
        if (output.handle) {
            const resource_desc od = dev->get_resource_desc(output);
            key = (static_cast<uint64_t>(od.texture.width) << 32) | od.texture.height;
        }
        ViewStats& vs = g_stats[key];
        if (!output.handle) ++vs.noOutput;
        else if (!chain.depth.handle) ++vs.noDepth;
        else if (!chain.motion.handle) ++vs.noMotion;
        else if (!chain.color.handle) ++vs.noColor;
    }
    if (!output.handle || !chain.depth.handle || !chain.motion.handle || !chain.color.handle) {
        Skip("an input is missing (output, depth, motion, colour)", output.handle != 0, chain.depth.handle != 0,
             (chain.motion.handle != 0) * 10 + (chain.color.handle != 0));
        return false;
    }
    const resource_desc out = dev->get_resource_desc(output);
    if (!IsHdrColor(out.texture.format)) {
        Skip("TSR's output has an unexpected format (width, height, format)", out.texture.width, out.texture.height,
             static_cast<uint32_t>(out.texture.format));
        return false;
    }
    float common[2] = {0, 0}, jitter[2] = {0, 0};
    if (!RenderSize(cl, s.compute, common) || !ReadConstants(cl, s.compute, 2, 256, jitter, 2)) {
        g_error = "TSR parameters not readable";
        Skip("TSR's parameters aren't readable");
        return false;
    }
    const resource_desc cd = dev->get_resource_desc(chain.color);
    const unsigned rw = static_cast<unsigned>(common[0]), rh = static_cast<unsigned>(common[1]);
    if (!rw || !rh || rw > cd.texture.width || rh > cd.texture.height || std::fabs(jitter[0]) > 0.5f ||
        std::fabs(jitter[1]) > 0.5f) {
        g_error = "TSR parameters out of range";
        Skip("TSR's parameters are out of range (render width, height, colour width)", rw, rh, cd.texture.width);
        return false;
    }
    std::string err;
    if (!dlss::Init(reinterpret_cast<ID3D12Device*>(dev->get_native()), err)) {
        g_error = err;
        ++g_skipped;
        return false;
    }
    g_lastJitter[0] = jitter[0], g_lastJitter[1] = jitter[1];
    if (g_resetAll) g_resetByOutput.clear(), g_resetAll = false;
    const uint64_t outKey = (static_cast<uint64_t>(out.texture.width) << 32) | out.texture.height;
    auto rs = g_resetByOutput.find(outKey);
    const uint64_t size = (static_cast<uint64_t>(rw) << 32) | rh;
    auto ss = g_sizeByOutput.find(outKey);
    const bool resized = ss != g_sizeByOutput.end() && ss->second != size;
    g_sizeByOutput[outKey] = size;
    if (g_sizeByOutput.size() > 64) g_sizeByOutput.clear();
    const bool reset = rs == g_resetByOutput.end() || rs->second || resized;

    cl->dispatch(x, y, z);  // the game's ResolveHistory, with its own bindings
    cl->barrier(output, resource_usage::unordered_access, resource_usage::unordered_access);

    dlss::Inputs in;
    in.color = reinterpret_cast<ID3D12Resource*>(chain.color.handle);
    in.motion = reinterpret_cast<ID3D12Resource*>(chain.motion.handle);
    in.depth = reinterpret_cast<ID3D12Resource*>(chain.depth.handle);
    in.output = reinterpret_cast<ID3D12Resource*>(output.handle);
    if (g_set.maskMode > 0 || g_set.transMask > 0) in.biasMask = reinterpret_cast<ID3D12Resource*>(chain.mask.handle);
    in.renderWidth = rw, in.renderHeight = rh;
    g_lastRender[0] = rw, g_lastRender[1] = rh;
    g_lastOutput[0] = out.texture.width, g_lastOutput[1] = out.texture.height;
    in.outputWidth = out.texture.width, in.outputHeight = out.texture.height;
    in.jitterX = jitter[0], in.jitterY = jitter[1];
    in.motionScaleX = 1, in.motionScaleY = 1;  // motion.hlsl writes previous minus current in pixels
    in.reset = reset;
    in.preset = g_set.preset;
    const bool ok = dlss::Evaluate(reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()), in, err);
    cl->barrier(output, resource_usage::unordered_access, resource_usage::unordered_access);
    if (ok && g_set.showMask) {
        cl->bind_descriptor_tables(shader_stage::all_compute, s.compute.layout, 0, 0, nullptr);  // ReShade's heaps back
        TintMask(cl, output, chain.mask, rw, rh);
    }
    Restore(cl, s);
    {
        ViewStats& vs = g_stats[outKey];
        if (!ok) ++vs.other;
        else ++vs.evaluated, vs.tagged += FrameGenIsScreen(out.texture.width, out.texture.height) && !chain.view.empty();
    }
    if (ok) {
        // Only the view on screen: Double Exposure also runs TSR for a 4096x2160 view.
        if (FrameGenIsScreen(out.texture.width, out.texture.height) && !chain.view.empty()) {
            FrameGenSetCamera(chain.view.data(), g_motion.offset);
            FrameGenTag(reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()), in.depth, in.motion, rw, rh,
                        in.jitterX, in.jitterY, reset);
        }
        g_resetByOutput[outKey] = false;
        g_error.clear();
        if (++g_evaluated == 1) Log("DLSS: first frame evaluated");
    } else {
        ++g_skipped;
        if (g_error != err) {
            g_error = err;
            Log("DLSS: %s", err.c_str());
        }
    }
    return true;  // the game's dispatch was recorded above
}

bool OnDispatch(command_list* cl, uint32_t x, uint32_t y, uint32_t z) {
    if (!g_set.enabled) return false;
    ListState* s = State(cl);
    if (!s) return false;
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    switch (s->computeId) {
        case kTsrDilateVelocity:
            OnDilateVelocity(cl, *s);
            return false;
        case kTsrDecimateHistory:
            OnChainStart(cl, *s);
            return false;
        case kTsrUpdateHistory:
            OnUpdateHistory(cl, *s);
            return false;
        case kTsrResolveHistory:
            return Evaluate(cl, *s, x, y, z);
        default:
            return false;
    }
}

void OnDestroyDevice(device* dev) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    dlss::Shutdown();
    g_pool.Destroy(dev);
    for (auto& [key, v] : g_motion.views) dev->destroy_resource_view(v);
    g_motion.views.clear();
    for (MotionState::Texture& t : g_motion.textures) dev->destroy_resource(t.tex);
    if (g_motion.pipe.handle) dev->destroy_pipeline(g_motion.pipe);
    if (g_motion.tintPipe.handle) dev->destroy_pipeline(g_motion.tintPipe);
    if (g_motion.composePipe.handle) dev->destroy_pipeline(g_motion.composePipe);
    if (g_motion.composeLayout.handle) dev->destroy_pipeline_layout(g_motion.composeLayout);
    if (g_motion.tintLayout.handle) dev->destroy_pipeline_layout(g_motion.tintLayout);
    if (g_motion.layout.handle) dev->destroy_pipeline_layout(g_motion.layout);
    g_motion = MotionState();
    g_links.clear();
}

bool g_keyWasDown = false;

// ---------------------------------------------------------------- render resolution (DLSS mode)
//
// DLSS renders at the game's r.ScreenPercentage (the menu's Resolution Scale). A DLSS mode sets
// it to NVIDIA's ratio for that mode while DLSS is on, and puts the game's value back when DLSS is
// off (so TSR gets the menu's). The game sets it again when its menu is applied: that value is
// remembered as the game's and the mode's put back.

enum DlssMode { kModeGame, kModeDlaa, kModeQuality, kModeBalanced, kModePerformance, kModeUltraPerformance, kDlssModeCount };
const char* const kDlssModeNames[kDlssModeCount] = {"Game's Resolution Scale", "DLAA (100%)", "Quality (67%)",
                                                    "Balanced (58%)", "Performance (50%)", "Ultra Performance (33%)"};
const float kDlssModePercent[kDlssModeCount] = {0, 100, 66.667f, 58, 50, 33.333f};

struct Resolution {
    void* manager = nullptr;
    void* var = nullptr;  // r.ScreenPercentage
    int sinkFlag = -1;
    bool failed = false;
    bool overriding = false;
    float gameValue = 0, ourValue = 0;
    uint64_t presents = 0;
    std::string error;
} g_res;

bool FindScreenPercentage() {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
    void** singleton = cvars::FindConsoleManagerSingleton(base);
    void* manager = nullptr;
    if (singleton) cvars::SafeCopy(&manager, singleton, 8);
    if (!manager || !cvars::HasModuleVtable(base, manager)) {
        g_res.error = "the game's console variables weren't found";
        return false;
    }
    void* aniso = cvars::FindVariable(manager, L"r.MaxAnisotropy");
    void* screen = cvars::FindVariable(manager, L"r.ScreenPercentage");
    float value = -1;
    if (!aniso || !screen || !cvars::HasModuleVtable(base, screen) || !cvars::FindTypeSlots(aniso, screen) ||
        !cvars::GetFloat(screen, value) || !(value >= 10 && value <= 400)) {
        g_res.error = "r.ScreenPercentage didn't look right (see the log)";
        Log("DLSS mode: r.ScreenPercentage %p reads %g: not used", screen, value);
        return false;
    }
    g_res.manager = manager, g_res.var = screen;
    g_res.sinkFlag = cvars::FindSinkFlagOffset(manager);
    Log("DLSS mode: r.ScreenPercentage found (now %g)", value);
    return true;
}

bool SetScreenPercentage(float v) {
    wchar_t text[32];
    swprintf(text, 32, L"%.3f", v);
    if (!cvars::Set(g_res.var, text)) return false;
    // UE 5.2 only copies a value to the render thread on the game thread (cvars.hpp).
    cvars::MirrorToRenderer(reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr)), g_res.var,
                            cvars::GetType(g_res.var));
    if (g_res.sinkFlag >= 0) cvars::RequestSinks(g_res.manager, g_res.sinkFlag);
    g_res.ourValue = v;
    return true;
}

// ---------------------------------------------------------------- velocity settings (logged)
//
// Logged once for diagnosis. 0.1.1 also set r.Velocity.EnableVertexDeformation=1 while DLSS was
// on: changing it makes the engine recreate every primitive's render state, which must happen on
// the game thread, and doing it from ReShade's thread hung the game (2026-10-26). It was already
// 2 (auto: on while TSR is the anti-aliasing method, which it stays under DLSS), so it isn't
// changed any more.

const wchar_t* const kVelocityLogged[] = {L"r.Velocity.EnableVertexDeformation", L"r.VertexDeformationOutputsVelocity",
                                          L"r.BasePassOutputsVelocity",          L"r.Velocity.ForceOutput",
                                          L"r.VelocityOutputPass",               L"r.SkinCache.Mode",
                                          L"r.SkinCache.Allow",                  L"r.HairStrands.VelocityType"};
bool g_velocityLogged = false;

void LogVelocitySettings() {
    if (!g_res.manager || g_velocityLogged) return;
    g_velocityLogged = true;
    for (const wchar_t* name : kVelocityLogged) {
        void* var = cvars::FindVariable(g_res.manager, name);
        if (!var) {
            Log("velocity: %ls not found", name);
            continue;
        }
        wchar_t buf[64] = L"?";
        cvars::GetText(var, cvars::GetType(var), buf, 64);
        Log("velocity: %ls = %ls (%s, set by 0x%02x)", name, buf, cvars::TypeName(cvars::GetType(var)), cvars::Priority(var));
    }
}

// About twice a second, on ReShade's present.
void UpdateResolution() {
    if (++g_res.presents % 30 != 0) return;
    if (g_set.enabled && !g_res.manager && !g_res.failed && !FindScreenPercentage()) g_res.failed = true;
    LogVelocitySettings();
    const bool want = g_set.enabled && g_set.mode > kModeGame && g_set.mode < kDlssModeCount;
    if (!g_res.var) {
        if (!want || g_res.failed) return;
        if (!FindScreenPercentage()) {
            g_res.failed = true;
            return;
        }
    }
    float now = 0;
    if (!cvars::GetFloat(g_res.var, now)) return;
    if (want) {
        const float target = kDlssModePercent[g_set.mode];
        if (!g_res.overriding) {
            g_res.gameValue = now, g_res.overriding = true;
        } else if (std::fabs(now - g_res.ourValue) > 0.01f) {
            g_res.gameValue = now;  // the game's menu set it again
            Log("DLSS mode: the game set its resolution scale to %g%%", now);
        }
        if (std::fabs(now - target) > 0.01f && SetScreenPercentage(target))
            Log("DLSS mode %s: render resolution %g%% (the game's: %g%%)", kDlssModeNames[g_set.mode], target, g_res.gameValue);
    } else if (g_res.overriding) {
        g_res.overriding = false;
        if (std::fabs(now - g_res.ourValue) < 0.01f && SetScreenPercentage(g_res.gameValue))
            Log("DLSS mode: back to the game's resolution scale (%g%%)", g_res.gameValue);
    }
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
    LogStats();
    UpdateResolution();
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    const bool down = pid == GetCurrentProcessId() && (GetAsyncKeyState(kToggleHotkey) & 0x8000);
    dlss::EndFrame();
    if (down && !g_keyWasDown && kTsrMapped) {
        g_set.enabled = !g_set.enabled;
        g_resetAll = true;
        SaveSettings();
        Log("DLSS %s (F9)", g_set.enabled ? "on" : "off");
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
    reshade::register_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::register_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::register_event<reshade::addon_event::present>(&OnPresent);
}

void UnregisterUpscaler() {
    reshade::unregister_event<reshade::addon_event::init_command_list>(&OnInitList);
    reshade::unregister_event<reshade::addon_event::destroy_command_list>(&OnDestroyList);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(&OnResetList);
    reshade::unregister_event<reshade::addon_event::bind_pipeline>(&OnBindPipeline);
    reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(&OnBindTables);
    reshade::unregister_event<reshade::addon_event::push_descriptors>(&OnPushDescriptors);
    reshade::unregister_event<reshade::addon_event::push_constants>(&OnPushConstants);
    reshade::unregister_event<reshade::addon_event::dispatch>(&OnDispatch);
    reshade::unregister_event<reshade::addon_event::destroy_device>(&OnDestroyDevice);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(&OnDestroyResource);
    reshade::unregister_event<reshade::addon_event::present>(&OnPresent);
}

void SaveAndLog() {
    g_resetAll = true;
    SaveSettings();
    Log("settings: DLSS %s, preset %d, mode %s", g_set.enabled ? "on" : "off", g_set.preset,
        kDlssModeNames[g_set.mode >= 0 && g_set.mode < kDlssModeCount ? g_set.mode : 0]);
}

// Preset ids (NVSDK_NGX_DLSS_Hint_Render_Preset_*) and their names in the overlay.
const char* const kPresetNames[] = {"NVIDIA default", "J", "K (recommended)", "L", "M"};
const int kPresetIds[] = {0, 10, 11, 12, 13};
constexpr int kPresetCount = 5;

const char* PresetName(int id) {
    if (id == 0) return "NVIDIA default";
    static char buf[8];
    snprintf(buf, sizeof(buf), "%c", id > 0 && id < 26 ? static_cast<char>('A' + id - 1) : '?');
    return buf;
}

void DrawUpscalerSettings() {
    ImGui::SeparatorText("DLSS");
    ImGui::BeginDisabled(!kTsrMapped);
    if (ImGui::Checkbox("DLSS (F9)", &g_set.enabled)) SaveAndLog();
    ImGui::EndDisabled();
    Help("NVIDIA DLSS in place of the game's TSR upscaler. F9 switches it on and off while playing, to compare.");

    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Render resolution", &g_set.mode, kDlssModeNames, kDlssModeCount)) SaveAndLog();
    Help("The resolution DLSS renders at, as a share of the screen's. \"Game's Resolution Scale\" keeps the game's "
         "menu setting; the others override it while DLSS is on and give it back when DLSS is off.\n\nKeep the "
         "game's Secondary Resolution Scale at 100%, so DLSS outputs the full screen resolution.");
    if (g_res.failed) ImGui::TextColored(kWarn, "Render resolution: %s", g_res.error.c_str());

    int cur = 2;
    for (int i = 0; i < kPresetCount; ++i)
        if (kPresetIds[i] == g_set.preset) cur = i;
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Model", &cur, kPresetNames, kPresetCount)) g_set.preset = kPresetIds[cur], SaveAndLog();
    Help("DLSS's model (preset). K is NVIDIA's transformer model, sharpest in motion. A DLSS override set in the "
         "NVIDIA App takes precedence over this.");
}

void DrawUpscalerAdvanced() {
    ImGui::SeparatorText("Motion of moving objects");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Object motion", &g_set.objectMode, kObjectModeNames, kObjectModeCount)) {
        g_resetAll = true;
        Log("diagnosis: object motion %s", kObjectModeNames[g_set.objectMode]);
    }
    Help("How the motion of things that move by themselves (characters, trees in the wind, cloth) is read from the "
         "engine. Square-root is right for this game (UE 5.2); the others are for testing and go back to it at "
         "each launch.");

    ImGui::SeparatorText("Animation mask");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Mask", &g_set.maskMode, kMaskModeNames, kMaskModeCount)) SaveAndLog();
    Help("Tells DLSS where to trust the current frame over its history.\n\nAnimated materials: what the game's "
         "materials flag as animated (none in the scenes tested).\nAnimated materials and moving objects: every "
         "moving object as well; less smearing, slightly less stable fine detail on them.\nOff: no mask.");
    ImGui::SetNextItemWidth(ControlWidth());
    if (ImGui::Combo("Translucent effects", &g_set.transMask, kTransMaskNames, kTransMaskCount)) SaveAndLog();
    Help("Adds translucent effects (particles, light beams, light shafts) to the mask. They have no motion of "
         "their own, so without it DLSS can smear or ghost them over a moving background.\n\nPartial: less "
         "ghosting, the background behind them stays smooth.\nFull: least ghosting, a little more shimmer "
         "under large effects.\nOff: not masked (they're still drawn).");
    ImGui::Checkbox("Show the mask in red", &g_set.showMask);
    Help("Tints the masked pixels red. For testing; off at each launch.");

    ImGui::SeparatorText("History");
    if (ImGui::Button("Reset DLSS history")) g_resetAll = true;
    Help("Starts DLSS's accumulation over, as after a camera cut.");
}

std::string UpscalerHeadline(bool* warn) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    *warn = false;
    if (!kTsrMapped) return (*warn = true), "DLSS unavailable: TSR not mapped for this game build";
    if (!g_set.enabled) return "DLSS off: the game's TSR";
    if (!g_error.empty()) return (*warn = true), "DLSS: " + g_error;
    if (!g_lastRender[0]) return "DLSS on: starts with the next game frame";
    char buf[160];
    snprintf(buf, sizeof(buf), "DLSS on: %ux%u -> %ux%u, model %s", g_lastRender[0], g_lastRender[1], g_lastOutput[0],
             g_lastOutput[1], PresetName(g_set.preset));
    return buf;
}

void DrawUpscalerStatus() {
    ImGui::SeparatorText("DLSS");
    ImGui::TextDisabled("%s", g_set.enabled ? dlss::Status().c_str() : "off");
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        ImGui::TextDisabled("Frames upscaled %llu, left to TSR %llu (menus, loading, first frames)",
                            static_cast<unsigned long long>(g_evaluated.load()),
                            static_cast<unsigned long long>(g_skipped.load()));
        if (g_motion.composed || g_motion.copied)
            ImGui::TextDisabled("Translucency: %s", g_motion.composed ? "blended into DLSS's input"
                                                                      : "not found (colour copied without it)");
        if (g_res.var)
            ImGui::TextDisabled("Resolution scale: %s", g_res.overriding ? "set by the Render resolution choice" : "the game's");
    }
}

}  // namespace rd
