// Composites UE 5.2's separate translucency over the scene colour before DLSS.
//
// UE 5.2 draws translucency (particles, light beams, volumetric shafts, outline materials) into its
// own texture and TSR blends it in UpdateHistory (t1). DLSS only sees the scene colour, so it's
// blended here the way UE does (ComposeSeparateTranslucency: rgb premultiplied, alpha is how much
// of the scene shows through).
//
// It also writes DLSS's bias-current-colour mask: the motion pass's mask (moving objects), plus
// where translucency covers the scene. Translucency has no motion vectors of its own, so DLSS would
// otherwise smear particles and beams along the motion of what's behind them.

cbuffer Params : register(b0) {
    float2 Size;           // render size in pixels
    float2 TransScale;     // translucency texture size / scene colour texture size
    float MaskStrength;    // 0 leaves translucency out of the mask
    float3 Pad;
};

Texture2D<float4> Scene : register(t0);
Texture2D<float4> Translucency : register(t1);
Texture2D<float> MotionMask : register(t2);
RWTexture2D<float4> Output : register(u0);
RWTexture2D<float> Mask : register(u1);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(float2(id) >= Size)) return;
    const float4 t = Translucency[uint2((float2(id) + 0.5) * TransScale)];
    const float4 s = Scene[id];
    Output[id] = float4(s.rgb * t.a + t.rgb, 1);
    const float coverage = saturate(1 - t.a);
    const float lum = max(max(t.r, t.g), t.b);
    // Additive effects (beams, glows) leave alpha at 1 and only add light.
    const float k = max(coverage, saturate(lum / max(max(max(s.r, s.g), s.b), 1e-3) * 0.5));
    Mask[id] = max(MotionMask[id], k > 0.02 ? k * MaskStrength : 0);
}
