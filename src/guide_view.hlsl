// Debug view: one Ray Reconstruction guide (render size) drawn over the upscaler's output, to check
// the G-buffer is read right. Mode 1 colour as is, 2 normals (xyz * 0.5 + 0.5), 3 w as grey.

cbuffer Params : register(b0) {
    float2 OutSize;  // output size in pixels
    float2 Scale;    // render size / output size
    uint Mode;
    uint3 Pad;
};

Texture2D<float4> Guide : register(t0);     // render resolution
RWTexture2D<float3> Output : register(u0);  // output resolution

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(float2(id) >= OutSize)) return;
    const float4 g = Guide[uint2((float2(id) + 0.5) * Scale)];
    Output[id] = Mode == 2 ? g.xyz * 0.5 + 0.5 : Mode == 3 ? g.www : g.rgb;
}
