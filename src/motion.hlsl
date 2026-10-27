// Motion vectors for DLSS, in render pixels, pointing from the current pixel to where it was in
// the previous frame, and DLSS's "bias current colour" mask.
//
// The engine's velocity texture only has values for moving objects (x > 0 when written). For
// everything else the motion is the camera's: the pixel's clip position (screen xy, device z)
// through View.ClipToPrevClip gives its previous clip position. That's what TSR does itself.
//
// Mask: UE 5.1+ packs a "pixel animation" flag into the lowest bit of the velocity's w (16-bit
// unorm; the rest of z and w hold depth): set for materials whose pixels animate in ways velocity
// doesn't describe (wind, panning textures, cloth). TSR distrusts its history there; DLSS does the
// same where the mask is 1. MaskMode 0 writes 0 (no mask), 1 marks flagged pixels, 2 also marks
// every pixel with object velocity (moving objects: less smearing, a little less stable).

cbuffer Params : register(b0) {
    float4 Row0, Row1, Row2, Row3;  // View.ClipToPrevClip, one row each (UE: row vector * matrix)
    float2 Size;                    // render size in pixels
    float2 InvSize;
    uint MaskMode;
    float MaskStrength;
    uint ObjectMode;  // object velocity: 1 square-root ("gamma") decode, UE 5.2's encoding (default);
                      // 0 linear decode (UE 5.6, Reunion); 2 ignored (camera motion only)
    uint Pad;
};

Texture2D<float> Depth : register(t0);      // device z, reversed
Texture2D<float4> Velocity : register(t1);  // UE encoded velocity (rgba16unorm)
RWTexture2D<float2> Motion : register(u0);
RWTexture2D<float> Mask : register(u1);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(float2(id) >= Size)) return;
    const float z = Depth[id];
    const float2 uv = (float2(id) + 0.5) * InvSize;
    const float2 screen = float2(uv.x * 2 - 1, 1 - uv.y * 2);
    const float4 prev = screen.x * Row0 + screen.y * Row1 + z * Row2 + Row3;
    float2 v = screen - prev.xy / prev.w;  // UE velocity: current minus previous, screen units
    const float4 e = Velocity[id];
    const bool written = e.x > 0;
    if (written && ObjectMode != 2) {
        const float inv = 1.0 / (0.499 * 0.5);
        v = e.xy * inv - (32767.0 / 65535.0) * inv;
        // VELOCITY_ENCODE_GAMMA: encoded = sign(V) * sqrt(|V|) * (2 / sqrt(2)) = sign(V) * sqrt(2|V|)
        if (ObjectMode == 1) v = sign(v) * v * v * 0.5;
    }
    Motion[id] = -v * float2(0.5 * Size.x, -0.5 * Size.y);

    const bool animated = written && (uint(e.w * 65535.0 + 0.5) & 1u) != 0;
    const bool marked = (MaskMode >= 1 && animated) || (MaskMode >= 2 && written);
    Mask[id] = marked ? MaskStrength : 0;
}
