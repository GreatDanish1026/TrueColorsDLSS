// Ray Reconstruction guides from UE 4.2x's G-buffer (DeferredShadingCommon.ush), at render size:
//   GBufferA rgb = world normal * 0.5 + 0.5
//   GBufferB r = metallic, g = specular, b = roughness, a = shading model id (low 4 bits) / 255
//   GBufferC rgb = base colour
// Outputs: diffuse albedo (base colour * (1 - metallic)), specular albedo (UE's EnvBRDFApprox of
// the specular colour, roughness and N.V), world normal with linear roughness in w.

cbuffer Params : register(b0) {
    float4 ViewRow0;   // View.TranslatedWorldToView rows (xyz): world directions to view space
    float4 ViewRow1;
    float4 ViewRow2;
    float2 Size;       // render size in pixels
    float2 InvSize;
    float2 ProjScale;  // ViewToClip [0][0], [1][1]
    uint SkyMode;      // unused, 0
    uint Pad;
};

Texture2D<float4> GBufferA : register(t0);
Texture2D<float4> GBufferB : register(t1);
Texture2D<float4> GBufferC : register(t2);
RWTexture2D<float4> DiffuseOut : register(u0);
RWTexture2D<float4> SpecularOut : register(u1);
RWTexture2D<float4> NormalOut : register(u2);

float3 EnvBRDFApprox(float3 specularColor, float roughness, float NoV) {
    const float4 c0 = float4(-1, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1, 0.0425, 1.04, -0.04);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
    AB.y *= saturate(50.0 * specularColor.g);
    return specularColor * AB.x + AB.y;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(float2(id) >= Size)) return;
    const float4 a = GBufferA[id];
    const float4 b = GBufferB[id];
    const float3 baseColor = GBufferC[id].rgb;
    const uint shadingModel = uint(round(b.a * 255.0)) & 0xF;

    // View direction (camera to pixel) in UE's view space: x right, y up, z forward.
    const float2 ndc = float2((id.x + 0.5) * InvSize.x * 2 - 1, 1 - (id.y + 0.5) * InvSize.y * 2);
    const float3 dirView = normalize(float3(ndc.x / ProjScale.x, ndc.y / ProjScale.y, 1));

    if (shadingModel == 0) {  // unlit (sky, emissive): nothing to demodulate
        DiffuseOut[id] = float4(1, 1, 1, 1);
        SpecularOut[id] = float4(0, 0, 0, 1);
        // Facing the camera: the view direction back in world space (the rotation is orthonormal, so
        // its inverse is the transpose).
        const float3 toCamera = -float3(dot(ViewRow0.xyz, dirView), dot(ViewRow1.xyz, dirView), dot(ViewRow2.xyz, dirView));
        NormalOut[id] = float4(toCamera, 1);
        return;
    }

    float3 N = a.rgb * 2 - 1;
    const float len = length(N);
    N = len > 1e-4 ? N / len : float3(0, 0, 1);
    const float metallic = saturate(b.r), specular = saturate(b.g), roughness = saturate(b.b);

    // World normal into view space (row vectors: n * M).
    const float3 nView = N.x * ViewRow0.xyz + N.y * ViewRow1.xyz + N.z * ViewRow2.xyz;
    const float NoV = saturate(dot(nView, -dirView));

    const float3 diffuse = baseColor * (1 - metallic);
    const float3 f0 = lerp(0.08 * specular.xxx, baseColor, metallic);
    const float3 spec = saturate(EnvBRDFApprox(f0, roughness, max(NoV, 1e-3)));

    DiffuseOut[id] = float4(diffuse, 1);
    SpecularOut[id] = float4(spec, 1);
    NormalOut[id] = float4(N, roughness);
}
