// Debug view: tints the DLSS output red where the bias mask is set, so the masked areas (trees,
// cloth, moving objects) can be checked in game. The output is linear HDR scene colour, before
// the game's tone mapping.

cbuffer Params : register(b0) {
    float2 OutSize;  // output size in pixels
    float2 Scale;    // render size / output size
};

Texture2D<float> Mask : register(t0);     // render resolution
RWTexture2D<float3> Output : register(u0);  // output resolution

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(float2(id) >= OutSize)) return;
    const float k = Mask[uint2((float2(id) + 0.5) * Scale)];
    if (k <= 0) return;
    const float3 c = Output[id];
    const float l = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.05);
    Output[id] = lerp(c, float3(l * 2.0, 0, 0), 0.65 * saturate(k));
}
