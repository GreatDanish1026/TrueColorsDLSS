// AMD FSR Frame Generation through the FidelityFX API (FSR SDK 2.4.0): FSR's frame generation
// swapchain in place of the game's, and a frame generation context fed with the inputs the add-on
// already makes for DLSS Frame Generation. FSR FG 4 (machine learning) on Radeon RX 9000 under
// Windows 11, FSR FG 3.1 everywhere else. One generated frame per rendered frame.
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>

#include <string>

namespace fsrfg {

// One frame's inputs, all valid until the frame's Present.
struct FrameInputs {
    ID3D12Resource* depth = nullptr;   // our copy of UE's depth (depth read + shader resource)
    ID3D12Resource* motion = nullptr;  // render pixels, previous minus current (non-pixel shader resource)
    unsigned renderWidth = 0, renderHeight = 0;
    float jitterX = 0, jitterY = 0;  // render pixels
    bool reset = false;
    float cameraNear = 10, cameraFovY = 1.0f;  // UE units (cm), radians
    float position[3] = {}, up[3] = {}, right[3] = {}, forward[3] = {};
};

// Loads the FidelityFX runtime (once). False with a reason.
bool Load(std::string& error);

// Creates the game's swapchain as FSR's frame generation swapchain through <factory> (whose
// CreateSwapChainForHwnd / CreateSwapChain must let FSR's own call through to the real one).
HRESULT CreateSwapChainForHwnd(IDXGIFactory* factory, ID3D12CommandQueue* queue, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                               const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen, IDXGISwapChain1** out, std::string& error);
HRESULT CreateSwapChain(IDXGIFactory* factory, ID3D12CommandQueue* queue, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** out,
                        std::string& error);

// The swapchain FSR made (null before / after), and how many references FSR itself holds on it.
IUnknown* Proxy();
ULONG OwnReferences();

// Right before the game's Present of each frame (on the presenting thread): configures frame
// generation for this frame and records + submits the prepare pass on the game's queue.
// <in> null: this frame has no inputs (frame generation then pauses after a few such frames).
// <hudless>: the picture before the HUD (back buffer format, shader resource), or null.
// <uiMask>: this frame's HUD opacity (r8, back buffer size, shader resource), or null.
// <hudMode>: what FSR gets to tell the HUD apart (HudMode).
enum HudMode : int { kHudComposite, kHudPreHud, kHudOff, kHudModeCount };
void BeforePresent(bool enable, const FrameInputs* in, ID3D12Resource* hudless, ID3D12Resource* uiMask, int hudMode,
                   unsigned bbWidth, unsigned bbHeight, DXGI_FORMAT bbFormat, bool hdr);

// The game is releasing its last reference to the swapchain: frame generation and FSR's swapchain
// context are shut down first, as AMD requires.
void BeforeSwapchainRelease();

// The device is going away: anything left.
void Shutdown();

bool Running();          // a frame generation context exists and was enabled for the last frame
std::string Status();    // for the overlay
std::string Version();   // "FSR FG 3.1.7" once known

}  // namespace fsrfg
