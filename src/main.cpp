// TrueColorsDLSS: a ReShade add-on for Life is Strange: True Colors (UE 4.25Plus, DirectX 12)
// that will replace the game's TAA/TAAU with DLSS (and add frame generation). Built from
// DoubleExposureDLSS 1.0.2 (UE 5.2, TSR). Separate from TrueColorsTweaks.
//
// This file is the frame capture tool (F10): it records two frames of GPU work (every compute
// dispatch with a stable id of its shader, the textures bound to its declared registers, its
// constant buffers' first bytes, resource state transitions, copies, render target binds, and which
// queue each command list ran on) to TrueColorsDLSS-frame-N.txt next to the game exe, for finding
// the TAA pass again after a game update. DLSS is in upscaler.cpp, frame generation in framegen.cpp.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "dxbc.hpp"
#include "version.h"
#include "../third_party/reshade/utils/descriptor_tracking.hpp"

extern "C" __declspec(dllexport) const char* NAME = RD_PRODUCT;
extern "C" __declspec(dllexport) const char* DESCRIPTION = "DLSS for " RD_GAME " in place of TSR (in development).";

namespace {

using namespace reshade::api;

constexpr int kCaptureHotkey = VK_F10;
constexpr int kCaptureFrames = 2;

// ---------------------------------------------------------------- logging

std::mutex g_logMutex;
FILE* g_log = nullptr;
ULONGLONG g_loadTick = 0;
std::wstring g_exeDir;

void Log(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (!g_log) return;
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_log, "%02u:%02u:%02u.%03u [+%5.1f s] %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds,
            (GetTickCount64() - g_loadTick) / 1000.0, buf);
    fflush(g_log);
}

void OpenLog() {
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (!n || n >= MAX_PATH) return;
    g_exeDir.assign(path, n);
    g_exeDir.resize(g_exeDir.find_last_of(L"\\/") + 1);
    // ReShade loads add-ons again when the game recreates its device: the same process appends.
    wchar_t pid[32] = {}, mine[32] = {};
    swprintf(mine, 32, L"%lu", GetCurrentProcessId());
    const bool again = GetEnvironmentVariableW(L"TRUECOLORSDLSS_LOG_PID", pid, 32) && wcscmp(pid, mine) == 0;
    SetEnvironmentVariableW(L"TRUECOLORSDLSS_LOG_PID", mine);
    g_log = _wfopen((g_exeDir + L"TrueColorsDLSS.log").c_str(), again ? L"a" : L"w");
    Log(RD_PRODUCT " " RD_VERSION " loaded%s", again ? " again (same game session)" : "");
}

std::string Format(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

// ---------------------------------------------------------------- shader ids
//
// Pipeline handles change every run; a hash of the shader bytecode doesn't, so captures from
// different runs can be compared. Compute pipelines only (TSR is all compute).

std::mutex g_pipeMutex;
std::unordered_map<uint64_t, uint32_t> g_pipeHash;  // pipeline handle -> FNV-1a of its compute shader
// pipeline handle -> the registers its compute shader declares (DXBC: UE 4.25 ships SM5 bytecode
// on D3D12). UE 4's descriptor tables are fixed-size, and slots a shader doesn't use still hold
// old descriptors of resources that may be gone; describing those crashed the game (0.0.1,
// GetDesc on a freed resource). The capture only resolves declared registers.
std::unordered_map<uint64_t, ShaderDecl> g_pipeDecl;

uint32_t Fnv(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
    return h;
}

void OnInitPipeline(device*, pipeline_layout, uint32_t count, const pipeline_subobject* subobjects, pipeline pipe) {
    for (uint32_t i = 0; i < count; ++i) {
        if (subobjects[i].type != pipeline_subobject_type::compute_shader || !subobjects[i].data) continue;
        const shader_desc& s = *static_cast<const shader_desc*>(subobjects[i].data);
        if (!s.code || !s.code_size) continue;
        const ShaderDecl decl = ParseDxbc(s.code, s.code_size);
        std::lock_guard<std::mutex> lock(g_pipeMutex);
        g_pipeHash[pipe.handle] = Fnv(s.code, s.code_size);
        g_pipeDecl[pipe.handle] = decl;
    }
}

void OnDestroyPipeline(device*, pipeline pipe) {
    std::lock_guard<std::mutex> lock(g_pipeMutex);
    g_pipeHash.erase(pipe.handle);
    g_pipeDecl.erase(pipe.handle);
}

ShaderDecl PipeDecl(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_pipeMutex);
    auto it = g_pipeDecl.find(handle);
    return it == g_pipeDecl.end() ? ShaderDecl{} : it->second;
}

uint32_t PipeId(pipeline pipe) {
    std::lock_guard<std::mutex> lock(g_pipeMutex);
    auto it = g_pipeHash.find(pipe.handle);
    return it == g_pipeHash.end() ? 0 : it->second;
}

// ---------------------------------------------------------------- resource descriptions

const char* FormatName(format f) {
    switch (static_cast<uint32_t>(f)) {
        case 2: return "rgba32f";
        case 10: return "rgba16f";
        case 11: return "rgba16unorm";
        case 16: return "rg32f";
        case 19: return "r32g8x24_typeless";
        case 20: return "d32s8";
        case 23: return "rgb10a2_typeless";
        case 24: return "rgb10a2unorm";
        case 26: return "r11g11b10f";
        case 27: return "rgba8_typeless";
        case 28: return "rgba8unorm";
        case 29: return "rgba8srgb";
        case 34: return "rg16f";
        case 35: return "rg16unorm";
        case 39: return "r32_typeless";
        case 40: return "d32f";
        case 41: return "r32f";
        case 42: return "r32uint";
        case 44: return "r24g8_typeless";
        case 45: return "d24s8";
        case 49: return "rg8unorm";
        case 54: return "r16f";
        case 56: return "r16unorm";
        case 57: return "r16uint";
        case 61: return "r8unorm";
        case 62: return "r8uint";
        case 87: return "bgra8unorm";
        default: return nullptr;
    }
}

std::string Describe(device* dev, resource r) {
    if (!r.handle) return "null";
    resource_desc d = dev->get_resource_desc(r);
    std::string id = Format("#%05llx", (r.handle >> 4) & 0xfffff);  // short, enough to follow one resource
    if (d.type == resource_type::buffer) return id + Format(" buf %llu", d.buffer.size);
    const char* fn = FormatName(d.texture.format);
    std::string fmt = fn ? fn : Format("fmt%u", static_cast<uint32_t>(d.texture.format));
    std::string s = id + Format(" tex %ux%u %s", d.texture.width, d.texture.height, fmt.c_str());
    if (d.texture.depth_or_layers > 1) s += Format(" x%u", d.texture.depth_or_layers);
    if (d.texture.levels > 1) s += Format(" mips%u", d.texture.levels);
    return s;
}

const char* UsageName(resource_usage u) {
    switch (u) {
        case resource_usage::undefined: return "undef";
        case resource_usage::general: return "general";
        case resource_usage::depth_stencil: return "ds";
        case resource_usage::indirect_argument: return "indirect";
        case resource_usage::constant_buffer: return "cbv";
        case resource_usage::render_target: return "rt";
        case resource_usage::depth_stencil_write: return "dsv";
        case resource_usage::depth_stencil_read: return "dsv_ro";
        case resource_usage::unordered_access: return "uav";
        case resource_usage::shader_resource_pixel: return "srv_ps";
        case resource_usage::shader_resource_non_pixel: return "srv_cs";
        case resource_usage::shader_resource: return "srv";
        case resource_usage::copy_dest: return "copy_dst";
        case resource_usage::copy_source: return "copy_src";
        case resource_usage::present: return "present";
        default: return nullptr;
    }
}

std::string Usage(resource_usage u) {
    const char* n = UsageName(u);
    return n ? n : Format("0x%x", static_cast<uint32_t>(u));
}

// ---------------------------------------------------------------- capture

enum class Capture { Idle, Recording, Writing };
std::atomic<Capture> g_capture{Capture::Idle};
int g_framesLeft = 0;
int g_captureCount = 0;
std::string g_status;

struct ListLog {
    std::vector<std::string> lines;
    bool partial = false;  // reset before the capture began: its first events are missing
    uint32_t draws = 0;    // consecutive draws not yet written
};

// TSR's compute shaders in this game build (stable across runs; docs/FINDINGS.md), in order.
const uint32_t kTsrShaders[] = {rd::kTaaMain};

bool IsTsr(uint32_t id) {
    for (uint32_t t : kTsrShaders)
        if (t == id) return true;
    return false;
}

// Whether a compute pass gets its bound textures and constants written: TSR's once they are known,
// every one until then.
bool Detailed(uint32_t id) { return !rd::kTsrMapped || IsTsr(id); }

// Constant buffers already written in this capture (buffer and offset): the View uniform buffer is
// bound to most passes, so it's written once per frame.
std::unordered_set<std::string> g_dumped;

std::mutex g_capMutex;
std::unordered_map<uint64_t, uint32_t> g_listPipe;  // command list -> compute shader id bound now
std::unordered_map<uint64_t, uint64_t> g_listPipeHandle;  // command list -> compute pipeline bound now
std::unordered_map<uint64_t, ListLog> g_lists;  // command list -> events since its reset
std::vector<std::string> g_frame;                // executed lists in order, with frame boundaries
std::unordered_map<uint64_t, uint64_t> g_listIds;
uint64_t g_nextListId = 1;

bool Recording() { return g_capture.load(std::memory_order_relaxed) == Capture::Recording; }

ListLog& List(command_list* cl) {  // g_capMutex held
    auto it = g_lists.find(cl->get_native());
    if (it != g_lists.end()) return it->second;
    ListLog& l = g_lists[cl->get_native()];
    l.partial = true;
    return l;
}

void FlushDraws(ListLog& l) {
    if (!l.draws) return;
    l.lines.push_back(Format("  draw x%u", l.draws));
    l.draws = 0;
}

void Add(command_list* cl, std::string line) {
    std::lock_guard<std::mutex> lock(g_capMutex);
    ListLog& l = List(cl);
    FlushDraws(l);
    l.lines.push_back("  " + line);
}

void OnReset(command_list* cl) {
    if (!Recording()) return;
    std::lock_guard<std::mutex> lock(g_capMutex);
    ListLog& l = g_lists[cl->get_native()];
    l = {};
}

void OnExecute(command_queue* q, command_list* cl) {
    if (!Recording()) return;
    std::lock_guard<std::mutex> lock(g_capMutex);
    auto it = g_lists.find(cl->get_native());
    uint64_t& id = g_listIds[cl->get_native()];
    if (!id) id = g_nextListId++;
    const command_queue_type t = q->get_type();
    const char* qn = (t & command_queue_type::graphics) != 0  ? "graphics"
                     : (t & command_queue_type::compute) != 0 ? "compute"
                                                              : "copy";
    if (it == g_lists.end()) {
        g_frame.push_back(Format("EXECUTE list L%llu on %s queue %p (not recorded)", id, qn, q));
        return;
    }
    FlushDraws(it->second);
    g_frame.push_back(Format("EXECUTE list L%llu on %s queue %p, %zu events%s", id, qn, q, it->second.lines.size(),
                             it->second.partial ? " (started before the capture: partial)" : ""));
    for (std::string& s : it->second.lines) g_frame.push_back(std::move(s));
    g_lists.erase(it);
}

void OnBarrier(command_list* cl, uint32_t count, const resource* res, const resource_usage* from, const resource_usage* to) {
    if (!Recording()) return;
    device* dev = cl->get_device();
    for (uint32_t i = 0; i < count; ++i)
        Add(cl, Format("barrier %s: %s -> %s", Describe(dev, res[i]).c_str(), Usage(from[i]).c_str(), Usage(to[i]).c_str()));
}

void OnBindPipeline(command_list* cl, pipeline_stage stages, pipeline pipe) {
    if (!Recording()) return;
    if ((stages & pipeline_stage::compute_shader) == 0) return;
    const uint32_t id = PipeId(pipe);
    {
        std::lock_guard<std::mutex> lock(g_capMutex);
        g_listPipe[cl->get_native()] = id;
        g_listPipeHandle[cl->get_native()] = pipe.handle;
    }
    Add(cl, Format("compute pipeline cs%08x%s", id, IsTsr(id) ? " (TAA)" : ""));
}

bool OnDispatch(command_list* cl, uint32_t x, uint32_t y, uint32_t z) {
    if (Recording()) Add(cl, Format("DISPATCH %u x %u x %u", x, y, z));
    return false;
}

bool OnDraw(command_list* cl, uint32_t, uint32_t, uint32_t, uint32_t) {
    if (!Recording()) return false;
    std::lock_guard<std::mutex> lock(g_capMutex);
    ++List(cl).draws;
    return false;
}

bool OnDrawIndexed(command_list* cl, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
    return OnDraw(cl, 0, 0, 0, 0);
}

bool OnDispatchMesh(command_list* cl, uint32_t x, uint32_t y, uint32_t z) {
    if (Recording()) Add(cl, Format("dispatch_mesh %u x %u x %u", x, y, z));
    return false;
}

bool OnExecuteIndirect(command_list* cl, indirect_command type, resource, uint64_t, uint32_t count, uint32_t) {
    if (Recording()) Add(cl, Format("indirect type %u count %u", static_cast<uint32_t>(type), count));
    return false;
}

// Writes a constant buffer's contents (as 32-bit values: float, and hex where it isn't a plain
// float) to the list log. TSR passes only; the View uniform buffer is large, so up to 8 KB.
void DumpConstants(command_list* cl, uint32_t param, const buffer_range& b, bool tsr) {
    device* dev = cl->get_device();
    const resource_desc d = dev->get_resource_desc(b.buffer);
    if (d.type != resource_type::buffer || b.offset >= d.buffer.size) return;
    {
        const std::string key = Format("%llx+%llu", b.buffer.handle, b.offset);
        std::lock_guard<std::mutex> lock(g_capMutex);
        if (!g_dumped.insert(key).second) {
            ListLog& l = List(cl);
            FlushDraws(l);
            l.lines.push_back(Format("  constants param %u: written above", param));
            return;
        }
    }
    uint64_t size = d.buffer.size - b.offset;
    if (b.size != UINT64_MAX && b.size < size) size = b.size;
    // TSR's own parameters are a few hundred bytes; the View uniform buffer (needed whole for the
    // camera matrices) is larger. Unknown passes: the first 1 KB, except large buffers (the View).
    const uint64_t limit = tsr || size > 4096 ? 16384 : 1024;
    if (size > limit) size = limit;
    void* data = nullptr;
    if (!dev->map_buffer_region(b.buffer, b.offset, size, map_access::read_only, &data) || !data) {
        Add(cl, Format("constants param %u: can't read (heap %u)", param, static_cast<uint32_t>(d.heap)));
        return;
    }
    std::vector<uint32_t> v(static_cast<size_t>(size / 4));
    memcpy(v.data(), data, v.size() * 4);
    dev->unmap_buffer_region(b.buffer);
    Add(cl, Format("constants param %u: %zu bytes", param, v.size() * 4));
    for (size_t i = 0; i < v.size(); i += 4) {
        std::string line = Format("  c %4zu:", i * 4);
        for (size_t k = i; k < i + 4 && k < v.size(); ++k) {
            float f;
            memcpy(&f, &v[k], 4);
            const uint32_t e = (v[k] >> 23) & 0xff;
            if (v[k] == 0) line += "          0";
            else if (e > 90 && e < 165) line += Format(" %10.6g", f);  // a plausible float
            else line += Format("  0x%08x", v[k]);
        }
        Add(cl, line);
    }
}

void OnPushDescriptors(command_list* cl, shader_stage stages, pipeline_layout, uint32_t param, const descriptor_table_update& u) {
    if (!Recording()) return;
    device* dev = cl->get_device();
    bool tsr = false;
    if ((stages & shader_stage::compute) != 0 && u.type == descriptor_type::constant_buffer) {
        std::lock_guard<std::mutex> lock(g_capMutex);
        auto it = g_listPipe.find(cl->get_native());
        tsr = it != g_listPipe.end() && Detailed(it->second);
    }
    const char* st = (stages & shader_stage::compute) != 0 ? "cs" : "gfx";
    std::string s = Format("push %s param %u at %u:", st, param, u.binding);
    for (uint32_t i = 0; i < u.count && i < 16; ++i) {
        switch (u.type) {
            case descriptor_type::constant_buffer: {
                const buffer_range& b = static_cast<const buffer_range*>(u.descriptors)[i];
                s += Format(" cbv(%s +%llu size %llu)", Describe(dev, b.buffer).c_str(), b.offset, b.size);
                break;
            }
            case descriptor_type::shader_resource_view:
            case descriptor_type::unordered_access_view: {
                resource_view v = static_cast<const resource_view*>(u.descriptors)[i];
                s += Format(" %s(%s)", u.type == descriptor_type::unordered_access_view ? "uav" : "srv",
                            v.handle ? Describe(dev, dev->get_resource_from_view(v)).c_str() : "null");
                break;
            }
            case descriptor_type::buffer_shader_resource_view:
            case descriptor_type::buffer_unordered_access_view: {
                const buffer_range& b = static_cast<const buffer_range*>(u.descriptors)[i];
                s += Format(" %s(%s +%llu)", u.type == descriptor_type::buffer_unordered_access_view ? "buav" : "bsrv",
                            Describe(dev, b.buffer).c_str(), b.offset);
                break;
            }
            default:
                s += Format(" type%u", static_cast<uint32_t>(u.type));
                break;
        }
    }
    Add(cl, s);
    if (tsr)
        for (uint32_t i = 0; i < u.count && i < 4; ++i)
            DumpConstants(cl, param, static_cast<const buffer_range*>(u.descriptors)[i], rd::kTsrMapped);
}

char RegisterLetter(descriptor_type t) {
    switch (t) {
        case descriptor_type::sampler: return 's';
        case descriptor_type::constant_buffer: return 'b';
        case descriptor_type::buffer_unordered_access_view:
        case descriptor_type::texture_unordered_access_view: return 'u';
        default: return 't';
    }
}

// The resources in a TSR pass's descriptor tables, by shader register.
void ResolveTables(command_list* cl, pipeline_layout layout, uint32_t first, uint32_t count, const descriptor_table* tables) {
    device* dev = cl->get_device();
    uint64_t pipe = 0;
    {
        std::lock_guard<std::mutex> lock(g_capMutex);
        auto it = g_listPipeHandle.find(cl->get_native());
        if (it != g_listPipeHandle.end()) pipe = it->second;
    }
    const ShaderDecl decl = PipeDecl(pipe);
    if (!decl.valid) {
        Add(cl, "  tables: shader declarations unknown (not DXBC); slots not resolved");
        return;
    }
    descriptor_tracking* dt = dev->get_private_data<descriptor_tracking>();
    if (!dt) return;
    for (uint32_t i = 0; i < count; ++i) {
        const pipeline_layout_param p = dt->get_pipeline_layout_param(layout, first + i);
        if (p.type != pipeline_layout_param_type::descriptor_table || !p.descriptor_table.ranges) {
            Add(cl, Format("  table param %u: layout unknown (type %u)", first + i, static_cast<uint32_t>(p.type)));
            continue;
        }
        for (uint32_t r = 0; r < p.descriptor_table.count; ++r) {
            const descriptor_range& range = p.descriptor_table.ranges[r];
            const uint32_t n = range.count > 48 ? 48 : range.count;  // unbounded ranges: the first 48
            for (uint32_t k = 0; k < n; ++k) {
                descriptor_heap heap = {0};
                uint32_t offset = 0;
                dev->get_descriptor_heap_offset(tables[i], range.binding + k, 0, &heap, &offset);
                std::string what;
                if (range.type == descriptor_type::sampler) continue;
                const uint32_t reg = range.dx_register_index + k;
                const bool declared = range.type == descriptor_type::constant_buffer ? decl.HasCb(reg)
                                      : RegisterLetter(range.type) == 'u'          ? decl.HasUav(reg)
                                                                                   : decl.HasSrv(reg);
                if (!declared || range.dx_register_space != 0) continue;
                if (range.type == descriptor_type::constant_buffer) {
                    const buffer_range b = dt->get_buffer_range(heap, offset);
                    if (!b.buffer.handle) continue;
                    what = Format("cbv %s +%llu", Describe(dev, b.buffer).c_str(), b.offset);
                } else {
                    const resource_view v = dt->get_resource_view(heap, offset);
                    if (!v.handle) continue;
                    const resource_view_desc vd = dev->get_resource_view_desc(v);
                    what = Describe(dev, dev->get_resource_from_view(v));
                    if (const char* fn = FormatName(vd.format)) what += Format(" as %s", fn);
                }
                Add(cl, Format("  table param %u: %c%u space%u = %s", first + i, RegisterLetter(range.type),
                               range.dx_register_index + k, range.dx_register_space, what.c_str()));
            }
        }
    }
}

void OnBindTables(command_list* cl, shader_stage stages, pipeline_layout layout, uint32_t first, uint32_t count,
                  const descriptor_table* tables, uint32_t, const uint32_t*) {
    if (!Recording()) return;
    const bool compute = (stages & shader_stage::compute) != 0;
    const char* st = compute ? "cs" : "gfx";
    std::string s = Format("tables %s %u..%u:", st, first, first + count - 1);
    for (uint32_t i = 0; i < count && i < 8; ++i) s += Format(" %llx", tables[i].handle);
    Add(cl, s);
    if (!compute) return;
    bool tsr;
    {
        std::lock_guard<std::mutex> lock(g_capMutex);
        auto it = g_listPipe.find(cl->get_native());
        tsr = it != g_listPipe.end() && Detailed(it->second);
    }
    if (tsr) ResolveTables(cl, layout, first, count, tables);
}

void OnBindRenderTargets(command_list* cl, uint32_t count, const resource_view* rtvs, resource_view dsv) {
    if (!Recording()) return;
    device* dev = cl->get_device();
    std::string s = "render targets:";
    for (uint32_t i = 0; i < count; ++i)
        s += " " + (rtvs[i].handle ? Describe(dev, dev->get_resource_from_view(rtvs[i])) : std::string("null"));
    if (dsv.handle) s += " depth " + Describe(dev, dev->get_resource_from_view(dsv));
    Add(cl, s);
}

bool OnCopyResource(command_list* cl, resource src, resource dst) {
    if (Recording()) {
        device* dev = cl->get_device();
        Add(cl, Format("copy %s -> %s", Describe(dev, src).c_str(), Describe(dev, dst).c_str()));
    }
    return false;
}

bool OnCopyTexture(command_list* cl, resource src, uint32_t, const subresource_box*, resource dst, uint32_t,
                   const subresource_box*, filter_mode) {
    if (Recording()) {
        device* dev = cl->get_device();
        Add(cl, Format("copy region %s -> %s", Describe(dev, src).c_str(), Describe(dev, dst).c_str()));
    }
    return false;
}

void WriteCapture() {
    std::vector<std::string> frame;
    {
        std::lock_guard<std::mutex> lock(g_capMutex);
        frame.swap(g_frame);
        g_lists.clear();
    }
    ++g_captureCount;
    std::wstring name = g_exeDir + L"TrueColorsDLSS-frame-" + std::to_wstring(g_captureCount) + L".txt";
    FILE* f = _wfopen(name.c_str(), L"w");
    if (!f) {
        g_status = "couldn't write the capture file";
        return;
    }
    size_t dispatches = 0;
    for (const std::string& s : frame) {
        fputs(s.c_str(), f);
        fputc('\n', f);
        if (s.find("DISPATCH ") != std::string::npos) ++dispatches;
    }
    fclose(f);
    g_status = Format("Capture %d: %zu lines, %zu dispatches -> TrueColorsDLSS-frame-%d.txt", g_captureCount, frame.size(),
                      dispatches, g_captureCount);
    Log("%s", g_status.c_str());
}

void StartCapture() {
    if (g_capture.load() != Capture::Idle) return;
    {
        std::lock_guard<std::mutex> lock(g_capMutex);
        g_lists.clear();
        g_frame.clear();
        g_dumped.clear();
        g_frame.push_back("==== capture start (lists reset before this point are marked partial)");
    }
    g_framesLeft = kCaptureFrames;
    g_capture = Capture::Recording;
    Log("capture started");
}

// ---------------------------------------------------------------- present

bool g_keyWasDown = false;

void OnPresent(command_queue*, swapchain* sc, const rect*, const rect*, uint32_t, const rect*) {
    if (g_capture.load() == Capture::Recording) {
        resource back = sc->get_current_back_buffer();
        {
            std::lock_guard<std::mutex> lock(g_capMutex);
            g_frame.push_back(Format("==== PRESENT (back buffer %s), %d more", Describe(sc->get_device(), back).c_str(),
                                     g_framesLeft - 1));
        }
        if (--g_framesLeft <= 0) {
            g_capture = Capture::Writing;
            WriteCapture();
            g_capture = Capture::Idle;
        }
    }
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    bool down = pid == GetCurrentProcessId() && (GetAsyncKeyState(kCaptureHotkey) & 0x8000);
    if (down && !g_keyWasDown) StartCapture();
    g_keyWasDown = down;
}

// ---------------------------------------------------------------- overlay

void DrawOverlay(effect_runtime*) {
    ImGui::TextDisabled(RD_PRODUCT " " RD_VERSION);
    bool warn = false;
    const std::string up = rd::UpscalerHeadline(&warn);
    if (warn) ImGui::TextColored(rd::kWarn, "%s", up.c_str());
    else ImGui::TextUnformatted(up.c_str());
    const std::string fg = rd::FrameGenHeadline(&warn);
    if (warn) ImGui::TextColored(rd::kWarn, "%s", fg.c_str());
    else ImGui::TextUnformatted(fg.c_str());
    ImGui::Spacing();
    if (!ImGui::BeginTabBar("tabs")) return;
    if (ImGui::BeginTabItem("Settings")) {
        rd::DrawUpscalerSettings();
        rd::DrawFrameGenSettings();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Status")) {
        rd::DrawUpscalerStatus();
        rd::DrawFrameGenStatus();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Advanced")) {
        rd::DrawUpscalerAdvanced();
        ImGui::SeparatorText("Frame capture");
        ImGui::BeginDisabled(g_capture.load() != Capture::Idle);
        if (ImGui::Button("Capture frame (F10)")) StartCapture();
        ImGui::EndDisabled();
        rd::Help("Writes every command of two frames to TrueColorsDLSS-frame-N.txt next to the game's exe, for "
                 "finding TSR's passes again after a game update.");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", g_capture.load() == Capture::Recording ? "recording..." : g_status.c_str());
        ImGui::SeparatorText("Logs");
        ImGui::TextDisabled("TrueColorsDLSS.log and streamline\\sl.log next to the game's exe");
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

}  // namespace

namespace rd {

const ImVec4 kWarn(1.0f, 0.7f, 0.3f, 1.0f);

float ControlWidth() { return ImGui::GetFontSize() * 14.0f; }

void Help(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", text);
}

void Log(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    ::Log("%s", buf);
}

const std::wstring& ExeDir() { return g_exeDir; }

uint32_t PipeId(pipeline pipe) { return ::PipeId(pipe); }

bool PipeHasTaaRegisters(pipeline pipe) {
    const ShaderDecl d = PipeDecl(pipe.handle);
    return d.valid && d.srv[0] == 0x3f && d.srv[1] == 0 && d.uav == 1;
}

}  // namespace rd

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(module)) return FALSE;
            g_loadTick = GetTickCount64();
            OpenLog();
            descriptor_tracking::register_events();
            rd::RegisterUpscaler();
            rd::RegisterFrameGen();
            reshade::register_event<reshade::addon_event::init_pipeline>(&OnInitPipeline);
            reshade::register_event<reshade::addon_event::destroy_pipeline>(&OnDestroyPipeline);
            reshade::register_event<reshade::addon_event::reset_command_list>(&OnReset);
            reshade::register_event<reshade::addon_event::execute_command_list>(&OnExecute);
            reshade::register_event<reshade::addon_event::barrier>(&OnBarrier);
            reshade::register_event<reshade::addon_event::bind_pipeline>(&OnBindPipeline);
            reshade::register_event<reshade::addon_event::dispatch>(&OnDispatch);
            reshade::register_event<reshade::addon_event::draw>(&OnDraw);
            reshade::register_event<reshade::addon_event::draw_indexed>(&OnDrawIndexed);
            reshade::register_event<reshade::addon_event::dispatch_mesh>(&OnDispatchMesh);
            reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(&OnExecuteIndirect);
            reshade::register_event<reshade::addon_event::push_descriptors>(&OnPushDescriptors);
            reshade::register_event<reshade::addon_event::bind_descriptor_tables>(&OnBindTables);
            reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&OnBindRenderTargets);
            reshade::register_event<reshade::addon_event::copy_resource>(&OnCopyResource);
            reshade::register_event<reshade::addon_event::copy_texture_region>(&OnCopyTexture);
            reshade::register_event<reshade::addon_event::present>(&OnPresent);
            reshade::register_overlay(RD_PRODUCT, &DrawOverlay);
            break;
        case DLL_PROCESS_DETACH:
            rd::UnregisterFrameGen();
            rd::UnregisterUpscaler();
            descriptor_tracking::unregister_events();
            reshade::unregister_addon(module);
            if (g_log) fclose(g_log);
            break;
    }
    return TRUE;
}
