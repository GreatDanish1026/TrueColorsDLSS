// The "picture without the HUD" for FSR Frame Generation: the finished frame everywhere, except
// where the HUD mask says there's HUD, where it's the game's copy from before the HUD.
//
// FSR finds the HUD by comparing that picture with the back buffer. UE's copy from before the HUD
// also differs from the finished frame around characters in Life is Strange: Reunion, so FSR took
// them for HUD and left them out of its generated frames (jitter around characters while the camera
// moves). Built this way, only the HUD differs.

cbuffer Params : register(b0) {
    uint2 Size;
    float Threshold;  // mask values above this are HUD
    uint Pad;
};

Texture2D<float4> BackBuffer : register(t0);  // the finished frame (with the HUD)
Texture2D<float4> Hudless : register(t1);     // the game's picture before the HUD
Texture2D<float> Mask : register(t2);         // HUD opacity, 1 = HUD
RWTexture2D<float4> Out : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= Size)) return;
    Out[id] = Mask[id] > Threshold ? Hudless[id] : BackBuffer[id];
}
