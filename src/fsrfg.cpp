// AMD FSR Frame Generation through the FidelityFX API. The signed runtime
// (amd_fidelityfx_loader_dx12.dll + amd_fidelityfx_framegeneration_dx12.dll, FSR SDK 2.4.0) sits
// next to the game exe.
//
// AMD's order, every frame: ffxConfigure (frame generation description: the swapchain, a callback
// that dispatches frame generation, the picture before the HUD, frame id + 1), then the prepare
// pass (depth, motion vectors, camera), then the swapchain's Present runs the callback. The
// configure and prepare happen right before the game's Present (framegen.cpp's present hook): all
// the frame's command lists are submitted by then, so the prepare pass goes in a command list of
// our own, submitted on the game's queue after them.
//
// Lifetime: the frame generation context is destroyed (after disabling it, which waits for
// presents) before FSR's swapchain context, and both before the game's last reference to the
// swapchain goes (framegen.cpp hooks the swapchain's Release).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.hpp"
#include "fsrfg.hpp"

#ifndef _WINDOWS
#define _WINDOWS
#endif
#include <ffx_api.h>
#include <ffx_api_loader.h>
#include <dx12/ffx_api_dx12.h>
#include <ffx_framegeneration.h>
#include <dx12/ffx_api_framegeneration_dx12.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#include "hud_compose_cs.h"  // kHudComposeCs: hud_compose.hlsl compiled by build.ps1

namespace fsrfg {
namespace {

// AMD: context create/destroy, the prepare dispatch and the swapchain's present must not overlap.
std::recursive_mutex g_mutex;
HMODULE g_module = nullptr;
ffxFunctions g_ffx = {};
bool g_loadTried = false, g_loaded = false;
std::string g_loadError;

ffxContext g_swapCtx = nullptr;   // FSR's swapchain
IDXGISwapChain4* g_proxy = nullptr;  // not a counted reference of ours: the game owns it
ULONG g_ownRefs = 0;
ID3D12CommandQueue* g_queue = nullptr;  // the game's (referenced)
ID3D12Device* g_device = nullptr;       // referenced

ffxContext g_fgCtx = nullptr;  // frame generation
unsigned g_ctxW = 0, g_ctxH = 0;
DXGI_FORMAT g_ctxFormat = DXGI_FORMAT_UNKNOWN;
bool g_ctxHdr = false;
uint64_t g_frameId = 0;
bool g_enabled = false;  // what the last configure said
std::string g_status = "not started", g_version;

FrameInputs g_last;  // the last frame's inputs, reused (with reset) for a frame without any
bool g_haveLast = false;

// Our command lists for the prepare pass: three allocators, in turn, behind a fence.
ID3D12CommandAllocator* g_alloc[3] = {};
uint64_t g_allocFence[3] = {};
ID3D12GraphicsCommandList* g_list = nullptr;
ID3D12Fence* g_fence = nullptr;
uint64_t g_fenceValue = 0;
int g_allocNext = 0;
HANDLE g_event = nullptr;

LARGE_INTEGER g_freq = {}, g_lastPresent = {};
float g_frameMs = 33.3f;

// The HUD composite (hud_compose.hlsl): its root signature, pipeline, a shader-visible heap with
// four descriptors per command allocator, and the output texture (back buffer size and format).
struct Compose {
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT step = 0;
    ID3D12Resource* out = nullptr;
    D3D12_RESOURCE_DESC outDesc = {};
    bool failed = false;
    uint64_t count = 0;
} g_compose;
constexpr D3D12_RESOURCE_STATES kReadState =
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

void ReleaseCompose() {
    if (g_compose.out) g_compose.out->Release();
    if (g_compose.heap) g_compose.heap->Release();
    if (g_compose.pso) g_compose.pso->Release();
    if (g_compose.root) g_compose.root->Release();
    g_compose = Compose();
}

bool ComposeObjects() {
    if (g_compose.pso) return true;
    if (g_compose.failed) return false;
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 2;
    rs.pParameters = params;
    ID3DBlob* blob = nullptr;
    bool ok = SUCCEEDED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) &&
              SUCCEEDED(g_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                      IID_PPV_ARGS(&g_compose.root)));
    if (blob) blob->Release();
    if (ok) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g_compose.root;
        pd.CS = {kHudComposeCs, sizeof(kHudComposeCs)};
        ok = SUCCEEDED(g_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_compose.pso)));
    }
    if (ok) {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 12;  // 4 per allocator
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ok = SUCCEEDED(g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_compose.heap)));
        g_compose.step = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    if (!ok) {
        ReleaseCompose();
        g_compose.failed = true;
        rd::Log("FSR frame generation: couldn't create the HUD composite shader: the picture before the HUD is used as is");
    }
    return ok;
}

// Our output texture like <bb> (with unordered access), in kReadState; null on failure.
ID3D12Resource* ComposeTarget(const D3D12_RESOURCE_DESC& bb) {
    if (g_compose.out && g_compose.outDesc.Width == bb.Width && g_compose.outDesc.Height == bb.Height &&
        g_compose.outDesc.Format == bb.Format)
        return g_compose.out;
    if (g_compose.out) g_compose.out->Release(), g_compose.out = nullptr;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = bb.Width;
    d.Height = bb.Height;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = bb.Format;
    d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, kReadState, nullptr,
                                                 IID_PPV_ARGS(&g_compose.out)))) {
        g_compose.out = nullptr;
        return nullptr;
    }
    g_compose.outDesc = d;
    rd::Log("FSR frame generation: HUD composite texture %llux%u, format %u", d.Width, d.Height, static_cast<unsigned>(d.Format));
    return g_compose.out;
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

// Records the composite on <list> (allocator slot <slot>) into ComposeTarget; returns it, or null
// (nothing recorded). The back buffer is in the present state at this point.
ID3D12Resource* RecordCompose(ID3D12GraphicsCommandList* list, int slot, ID3D12Resource* hudless, ID3D12Resource* mask) {
    if (!g_proxy || !ComposeObjects()) return nullptr;
    ID3D12Resource* bb = nullptr;
    if (FAILED(g_proxy->GetBuffer(g_proxy->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&bb))) || !bb) return nullptr;
    const D3D12_RESOURCE_DESC bd = bb->GetDesc(), hd = hudless->GetDesc(), md = mask->GetDesc();
    ID3D12Resource* out = nullptr;
    if (hd.Width == bd.Width && hd.Height == bd.Height && hd.Format == bd.Format && md.Width == bd.Width &&
        md.Height == bd.Height)
        out = ComposeTarget(bd);
    if (!out) {
        bb->Release();
        return nullptr;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_compose.heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_compose.heap->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(slot) * 4 * g_compose.step;
    gpu.ptr += static_cast<UINT64>(slot) * 4 * g_compose.step;
    ID3D12Resource* srvs[3] = {bb, hudless, mask};
    for (int i = 0; i < 3; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Format = i == 2 ? DXGI_FORMAT_R8_UNORM : bd.Format;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
        h.ptr += static_cast<SIZE_T>(i) * g_compose.step;
        g_device->CreateShaderResourceView(srvs[i], &s, h);
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
    u.Format = bd.Format;
    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
    h.ptr += 3 * static_cast<SIZE_T>(g_compose.step);
    g_device->CreateUnorderedAccessView(out, nullptr, &u, h);

    const D3D12_RESOURCE_BARRIER before[2] = {Transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                                              Transition(out, kReadState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    const D3D12_RESOURCE_BARRIER after[2] = {Transition(bb, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT),
                                             Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kReadState)};
    list->ResourceBarrier(2, before);
    list->SetDescriptorHeaps(1, &g_compose.heap);
    list->SetComputeRootSignature(g_compose.root);
    list->SetPipelineState(g_compose.pso);
    const uint32_t w = static_cast<uint32_t>(bd.Width), hgt = bd.Height;
    const float threshold = 1.0f / 255;
    uint32_t constants[4] = {w, hgt, 0, 0};
    memcpy(&constants[2], &threshold, 4);
    list->SetComputeRoot32BitConstants(0, 4, constants, 0);
    list->SetComputeRootDescriptorTable(1, gpu);
    list->Dispatch((w + 7) / 8, (hgt + 7) / 8, 1);
    list->ResourceBarrier(2, after);
    bb->Release();
    if (++g_compose.count == 1) rd::Log("FSR frame generation: first HUD composite recorded");
    return out;
}
std::string Code(ffxReturnCode_t r) {
    switch (r) {
        case FFX_API_RETURN_OK: return "ok";
        case FFX_API_RETURN_ERROR: return "error";
        case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "unknown description type";
        case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "runtime error";
        case FFX_API_RETURN_NO_PROVIDER: return "no provider (GPU, driver or DLL missing)";
        case FFX_API_RETURN_ERROR_MEMORY: return "out of memory";
        case FFX_API_RETURN_ERROR_PARAMETER: return "bad parameter";
        case FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "runtime too old";
        default: return "code " + std::to_string(r);
    }
}

ffxReturnCode_t DispatchCallback(ffxDispatchDescFrameGeneration* params, void* user) {
    return g_ffx.Dispatch(static_cast<ffxContext*>(user), &params->header);
}

bool CommandObjects() {
    if (g_list) return true;
    for (ID3D12CommandAllocator*& a : g_alloc)
        if (FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)))) return false;
    if (FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc[0], nullptr, IID_PPV_ARGS(&g_list))))
        return false;
    g_list->Close();
    if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)))) return false;
    g_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return g_event != nullptr;
}

void WaitFence(uint64_t value) {
    if (!g_fence || g_fence->GetCompletedValue() >= value) return;
    g_fence->SetEventOnCompletion(value, g_event);
    WaitForSingleObject(g_event, 2000);
}

void ReleaseCommandObjects() {
    WaitFence(g_fenceValue);
    for (ID3D12CommandAllocator*& a : g_alloc)
        if (a) a->Release(), a = nullptr;
    if (g_list) g_list->Release(), g_list = nullptr;
    if (g_fence) g_fence->Release(), g_fence = nullptr;
    if (g_event) CloseHandle(g_event), g_event = nullptr;
}

// Frame generation off and its context gone (configure first: it waits for the presents using it).
void DestroyFrameGen() {
    if (!g_fgCtx) return;
    ffxConfigureDescFrameGeneration c = {};
    c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    c.swapChain = g_proxy;
    c.frameGenerationEnabled = false;
    c.frameID = ++g_frameId;
    g_ffx.Configure(&g_fgCtx, &c.header);
    g_ffx.DestroyContext(&g_fgCtx, nullptr);
    g_fgCtx = nullptr;
    g_enabled = false;
    rd::Log("FSR frame generation: context released");
}

bool CreateFrameGen(unsigned w, unsigned h, DXGI_FORMAT format, bool hdr) {
    ffxCreateContextDescFrameGeneration d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    d.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE |
              (hdr ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0u);
    d.displaySize = {w, h};
    d.maxRenderSize = {w, h};
    d.backBufferFormat = ffxApiGetSurfaceFormatDX12(format);
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = g_device;
    ffxCreateContextDescFrameGenerationVersion version = {};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    version.version = FFX_FRAMEGENERATION_VERSION;
    d.header.pNext = &backend.header;
    backend.header.pNext = &version.header;
    const ffxReturnCode_t r = g_ffx.CreateContext(&g_fgCtx, &d.header, nullptr);
    if (r != FFX_API_RETURN_OK) {
        g_fgCtx = nullptr;
        g_status = "frame generation context creation failed (" + Code(r) + ")";
        rd::Log("FSR frame generation: %s (%ux%u, format %u, %s)", g_status.c_str(), w, h, static_cast<unsigned>(format),
                hdr ? "HDR" : "SDR");
        return false;
    }
    g_ctxW = w, g_ctxH = h, g_ctxFormat = format, g_ctxHdr = hdr;
    ffxQueryGetProviderVersion pv = {};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (g_ffx.Query(&g_fgCtx, &pv.header) == FFX_API_RETURN_OK && pv.versionName) {
        g_version = pv.versionName;
        if (g_version.rfind("FSR", 0) != 0) g_version = "FSR FG " + g_version;
    }
    g_status = "ready";
    rd::Log("FSR frame generation: context created, %ux%u, format %u, %s, %s", w, h, static_cast<unsigned>(format),
            hdr ? "HDR" : "SDR", g_version.empty() ? "version unknown" : g_version.c_str());
    return true;
}

FfxApiResource Res(ID3D12Resource* r, uint32_t state) { return ffxApiGetResourceDX12(r, state); }

// Opens our command list on the next allocator; its slot, or -1.
int OpenList() {
    if (!CommandObjects()) {
        g_status = "couldn't create command lists";
        return -1;
    }
    const int i = g_allocNext;
    g_allocNext = (g_allocNext + 1) % 3;
    WaitFence(g_allocFence[i]);
    g_alloc[i]->Reset();
    g_list->Reset(g_alloc[i], nullptr);
    return i;
}

// Closes and submits our command list on the game's queue.
void Submit(int slot) {
    g_list->Close();
    ID3D12CommandList* lists[] = {g_list};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    g_allocFence[slot] = g_fenceValue;
}

// Records the prepare pass on our open command list.
void Prepare(const FrameInputs& in, bool reset) {
    ffxDispatchDescFrameGenerationPrepareV2 p = {};
    p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    p.frameID = g_frameId;
    p.commandList = g_list;
    p.renderSize = {in.renderWidth, in.renderHeight};
    p.jitterOffset = {in.jitterX, in.jitterY};
    p.motionVectorScale = {1, 1};  // our motion vectors are in render pixels
    p.frameTimeDelta = g_frameMs;
    p.reset = reset;
    p.cameraNear = in.cameraNear;
    p.cameraFar = 1e8f;  // infinite far plane (flag set)
    p.cameraFovAngleVertical = in.cameraFovY;
    p.viewSpaceToMetersFactor = 0.01f;  // UE units are centimetres
    // Depth copy in depth read + shader resource, motion in non-pixel shader resource: both
    // readable as is, so FSR adds no barriers.
    p.depth = Res(in.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    p.motionVectors = Res(in.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    for (int k = 0; k < 3; ++k) {
        p.cameraPosition[k] = in.position[k];
        p.cameraUp[k] = in.up[k];
        p.cameraRight[k] = in.right[k];
        p.cameraForward[k] = in.forward[k];
    }
    const ffxReturnCode_t r = g_ffx.Dispatch(&g_fgCtx, &p.header);
    if (r != FFX_API_RETURN_OK) {
        static int logged = 0;
        if (logged++ < 10) rd::Log("FSR frame generation: prepare failed (%s)", Code(r).c_str());
        g_status = "prepare failed (" + Code(r) + ")";
    }
}

}  // namespace

bool Load(std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_loadTried) {
        error = g_loadError;
        return g_loaded;
    }
    g_loadTried = true;
    QueryPerformanceFrequency(&g_freq);
    const std::wstring fg = rd::ExeDir() + L"amd_fidelityfx_framegeneration_dx12.dll";
    if (GetFileAttributesW(fg.c_str()) == INVALID_FILE_ATTRIBUTES) {
        g_loadError = "amd_fidelityfx_framegeneration_dx12.dll missing next to the game exe";
    } else {
        g_module = LoadLibraryW((rd::ExeDir() + L"amd_fidelityfx_loader_dx12.dll").c_str());
        if (!g_module) g_loadError = "couldn't load amd_fidelityfx_loader_dx12.dll";
    }
    if (g_module) {
        ffxLoadFunctions(&g_ffx, g_module);
        if (!g_ffx.CreateContext || !g_ffx.DestroyContext || !g_ffx.Configure || !g_ffx.Dispatch || !g_ffx.Query)
            g_loadError = "amd_fidelityfx_loader_dx12.dll is incomplete";
    }
    g_loaded = g_loadError.empty();
    g_status = g_loaded ? "loaded" : g_loadError;
    rd::Log("FSR frame generation: %s", g_loaded ? "runtime loaded" : g_loadError.c_str());
    error = g_loadError;
    return g_loaded;
}

namespace {

// Shared by both creation paths once FSR's swapchain exists.
void Adopt(ID3D12CommandQueue* queue, IDXGISwapChain4* sc) {
    g_proxy = sc;
    queue->AddRef();
    g_queue = queue;
    if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&g_device)))) {
        // References on the swapchain besides the game's one (FSR's own): counted now.
        sc->AddRef();
        const ULONG n = sc->Release();
        g_ownRefs = n > 0 ? n - 1 : 0;
        g_status = "swapchain created";
        rd::Log("FSR frame generation: the game's swapchain is FSR's (references: %lu, FSR's own %lu)", n, g_ownRefs);
    }
}

}  // namespace

HRESULT CreateSwapChainForHwnd(IDXGIFactory* factory, ID3D12CommandQueue* queue, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                               const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen, IDXGISwapChain1** out, std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_loaded || g_swapCtx) return error = g_loaded ? "a swapchain exists" : g_loadError, E_FAIL;
    DXGI_SWAP_CHAIN_DESC1 d1 = *desc;
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fs = fullscreen ? *fullscreen : DXGI_SWAP_CHAIN_FULLSCREEN_DESC{};
    IDXGISwapChain4* sc = nullptr;
    ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 c = {};
    c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12;
    c.swapchain = &sc;
    c.hwnd = hwnd;
    c.desc = &d1;
    c.fullscreenDesc = fullscreen ? &fs : nullptr;
    c.dxgiFactory = factory;
    c.gameQueue = queue;
    ffxCreateContextDescFrameGenerationSwapChainVersionDX12 v = {};
    v.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
    v.version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
    c.header.pNext = &v.header;
    const ffxReturnCode_t r = g_ffx.CreateContext(&g_swapCtx, &c.header, nullptr);
    if (r != FFX_API_RETURN_OK || !sc) {
        g_swapCtx = nullptr;
        error = "FSR swapchain creation failed (" + Code(r) + ")";
        g_status = error;
        return E_FAIL;
    }
    Adopt(queue, sc);
    *out = sc;
    return S_OK;
}

HRESULT CreateSwapChain(IDXGIFactory* factory, ID3D12CommandQueue* queue, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** out,
                        std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_loaded || g_swapCtx) return error = g_loaded ? "a swapchain exists" : g_loadError, E_FAIL;
    IDXGISwapChain4* sc = nullptr;
    ffxCreateContextDescFrameGenerationSwapChainNewDX12 c = {};
    c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_NEW_DX12;
    c.swapchain = &sc;
    c.desc = desc;
    c.dxgiFactory = factory;
    c.gameQueue = queue;
    ffxCreateContextDescFrameGenerationSwapChainVersionDX12 v = {};
    v.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
    v.version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
    c.header.pNext = &v.header;
    const ffxReturnCode_t r = g_ffx.CreateContext(&g_swapCtx, &c.header, nullptr);
    if (r != FFX_API_RETURN_OK || !sc) {
        g_swapCtx = nullptr;
        error = "FSR swapchain creation failed (" + Code(r) + ")";
        g_status = error;
        return E_FAIL;
    }
    Adopt(queue, sc);
    *out = sc;
    return S_OK;
}

IUnknown* Proxy() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_proxy;
}

ULONG OwnReferences() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_ownRefs;
}

void BeforePresent(bool enable, const FrameInputs* in, ID3D12Resource* hudless, ID3D12Resource* uiMask, int hudMode,
                   unsigned bbWidth, unsigned bbHeight, DXGI_FORMAT bbFormat, bool hdr) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_proxy || !g_device) return;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (g_freq.QuadPart && g_lastPresent.QuadPart) {
        const float ms = static_cast<float>(now.QuadPart - g_lastPresent.QuadPart) * 1000.0f / g_freq.QuadPart;
        g_frameMs = ms > 0.5f && ms < 500 ? ms : 33.3f;
    }
    g_lastPresent = now;

    if (in) g_last = *in, g_haveLast = true;
    const bool want = enable && g_haveLast && bbWidth && bbHeight;
    if (!want) {
        if (g_enabled && g_fgCtx) {  // pause (this waits for the presents in flight once)
            ffxConfigureDescFrameGeneration c = {};
            c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            c.swapChain = g_proxy;
            c.frameGenerationEnabled = false;
            c.frameID = ++g_frameId;
            g_ffx.Configure(&g_fgCtx, &c.header);
            g_enabled = false;
            rd::Log("FSR frame generation paused");
        }
        return;
    }
    // A new back buffer size, format or HDR mode: a new context.
    if (g_fgCtx && (g_ctxW != bbWidth || g_ctxH != bbHeight || g_ctxFormat != bbFormat || g_ctxHdr != hdr)) DestroyFrameGen();
    if (!g_fgCtx && !CreateFrameGen(bbWidth, bbHeight, bbFormat, hdr)) return;

    const int slot = OpenList();
    if (slot < 0) return;
    // What FSR compares the back buffer with to find the HUD: the composite (only the HUD differs),
    // the game's picture before the HUD as is, or nothing. Without a HUD mask, the composite can't
    // be made and the picture before the HUD is used.
    ID3D12Resource* hud = nullptr;
    if (hudless && hudMode == kHudComposite && uiMask) hud = RecordCompose(g_list, slot, hudless, uiMask);
    if (!hud && hudless && hudMode != kHudOff) hud = hudless;
    static int lastKind = -1;
    const int kind = !hud ? 0 : hud == hudless ? 1 : 2;
    if (kind != lastKind) {
        lastKind = kind;
        rd::Log("FSR frame generation: HUD %s", kind == 2 ? "told apart with the composite (HUD mask)"
                                              : kind == 1 ? "told apart with the picture before the HUD"
                                                          : "not told apart");
    }

    ffxConfigureDescFrameGeneration c = {};
    c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    c.swapChain = g_proxy;
    c.frameGenerationCallback = &DispatchCallback;
    c.frameGenerationCallbackUserContext = &g_fgCtx;
    c.frameGenerationEnabled = true;
    c.allowAsyncWorkloads = false;
    c.HUDLessColor = hud ? Res(hud, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ) : FfxApiResource{};
    c.generationRect = {0, 0, static_cast<int32_t>(bbWidth), static_cast<int32_t>(bbHeight)};
    c.frameID = ++g_frameId;
    const ffxReturnCode_t r = g_ffx.Configure(&g_fgCtx, &c.header);
    if (r != FFX_API_RETURN_OK) {
        Submit(slot);  // whatever was recorded
        static int logged = 0;
        if (logged++ < 10) rd::Log("FSR frame generation: configure failed (%s)", Code(r).c_str());
        g_status = "configure failed (" + Code(r) + ")";
        return;
    }
    if (!g_enabled) rd::Log("FSR frame generation on (%ux%u)", bbWidth, bbHeight);
    g_enabled = true;
    // A frame without inputs of its own (a hitch) reuses the last ones, as a reset.
    Prepare(g_last, in ? in->reset : true);
    Submit(slot);
    if (g_status.rfind("prepare failed", 0) != 0) g_status = "running";
}

void BeforeSwapchainRelease() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_swapCtx) return;
    DestroyFrameGen();
    ReleaseCommandObjects();
    ReleaseCompose();
    // Cleared first: FSR may release its own references while its context goes, which comes back
    // through the hooked Release.
    ffxContext ctx = g_swapCtx;
    g_swapCtx = nullptr;
    g_proxy = nullptr;
    g_ffx.DestroyContext(&ctx, nullptr);
    g_haveLast = false;
    if (g_queue) g_queue->Release(), g_queue = nullptr;
    if (g_device) g_device->Release(), g_device = nullptr;
    g_status = "swapchain released";
    rd::Log("FSR frame generation: shut down with the game's swapchain");
}

void Shutdown() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_swapCtx) BeforeSwapchainRelease();
    ReleaseCommandObjects();
}

bool Running() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_fgCtx && g_enabled;
}

std::string Status() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_status;
}

std::string Version() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_version.empty() ? "FSR frame generation" : g_version;
}

}  // namespace fsrfg
