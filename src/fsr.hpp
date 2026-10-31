// AMD FSR upscaling through the FidelityFX API (FSR SDK 2.4.0): FSR 4 on Radeon GPUs that support
// it, FSR 3.1 everywhere else (any DirectX 12 GPU).
#pragma once

#include <d3d12.h>

#include <string>

namespace fsr {

struct Inputs {
    ID3D12Resource* color = nullptr;     // render resolution, linear HDR, jittered
    ID3D12Resource* depth = nullptr;     // render resolution, reversed Z (UE's depth buffer)
    ID3D12Resource* motion = nullptr;    // render resolution, previous minus current in pixels
    ID3D12Resource* output = nullptr;    // output resolution, UAV
    ID3D12Resource* reactive = nullptr;  // optional, render resolution r8: 1 = trust this frame over history
    unsigned renderWidth = 0, renderHeight = 0;
    unsigned outputWidth = 0, outputHeight = 0;
    float jitterX = 0, jitterY = 0;  // in render pixels
    float motionScaleX = 1, motionScaleY = 1;
    float cameraNear = 10;           // UE units (cm)
    float cameraFovY = 1.0f;         // radians
    float sharpness = 0;             // 0 = no extra sharpening, 1 = most
    bool reset = false;
    // FSR 3.1 tuning (FfxApiConfigureUpscaleKey; AMD's defaults). Ignored by FSR 4.
    float velocityFactor = 1.0f;          // 0..1; 0 steadies bright pixels in motion
    float minDisocclusion = -1.0f / 3;    // -1..1; higher: less flicker around swaying thin objects, more ghosting
    float shadingChangeScale = 1.0f;      // 0..; lower: less reaction to shading changes (less noise)
    bool debugView = false;  // AMD's debug overlay in the output (context recreated with debug support)
};

// Loads the FidelityFX runtime next to the game exe (once). False with a reason if FSR can't run.
bool Init(ID3D12Device* device, std::string& error);

// Records FSR on the command list. One context (history) per output size, created when first seen
// and recreated when the render size grows past it. The inputs must be in the non-pixel shader
// resource state (or one that includes it) and the output in the unordered access state.
// Changes the command list's compute state and descriptor heaps.
bool Evaluate(ID3D12GraphicsCommandList* list, const Inputs& in, std::string& error);

// Once per presented frame: frame time, and releases replaced contexts the GPU is done with.
void EndFrame();

// Releases every context (after the GPU is done with them): when DLSS takes over.
void ReleaseContexts();

void Shutdown();

// A line for the overlay ("running: FSR 3.1.6, 2560x1440 -> 3840x2160 ...").
std::string Status();
// "FSR 4.1.2" / "FSR 3.1.6" once a context runs, else "FSR".
std::string VersionName();

}  // namespace fsr
