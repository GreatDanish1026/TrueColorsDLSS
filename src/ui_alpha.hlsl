// Frame generation's UI mask (opacity of the HUD per pixel) from the game's HUD layer. UE keeps
// that layer's alpha inverted in this game (1 where there's no HUD), so Invert flips it.

cbuffer Params : register(b0) {
    uint2 Size;
    uint Invert;
    uint Pad;
};

Texture2D<float4> Ui : register(t0);
RWTexture2D<float> Mask : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= Size)) return;
    const float a = Ui[id].a;
    Mask[id] = Invert ? 1 - a : a;
}
