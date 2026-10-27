// Reads the resource declarations of a DXBC shader (SHEX/SHDR chunk): which
// t#, u# and cb# slots it uses, and how large each constant buffer is. The
// D3D11 RHI leaves old views bound between passes, so the probe only trusts
// slots the shader itself declares.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

struct ShaderDecl {
    bool valid = false;
    uint64_t srv[2] = {};    // t0..t127
    uint64_t uav = 0;        // u0..u63
    uint32_t cb = 0;         // cb0..cb13
    uint32_t cbVec4[14] = {};  // declared size of each constant buffer, in float4s

    bool HasSrv(uint32_t i) const { return i < 128 && (srv[i >> 6] >> (i & 63)) & 1; }
    bool HasUav(uint32_t i) const { return i < 64 && (uav >> i) & 1; }
    bool HasCb(uint32_t i) const { return i < 14 && (cb >> i) & 1; }
};

inline ShaderDecl ParseDxbc(const void* code, size_t size) {
    ShaderDecl d;
    auto p = static_cast<const uint8_t*>(code);
    if (!p || size < 32 || memcmp(p, "DXBC", 4) != 0) return d;
    auto u32 = [&](size_t off) { uint32_t v; memcpy(&v, p + off, 4); return v; };
    const uint32_t chunks = u32(28);
    if (32 + size_t(chunks) * 4 > size) return d;

    for (uint32_t c = 0; c < chunks; ++c) {
        const uint32_t off = u32(32 + c * 4);
        if (size_t(off) + 8 > size) return d;
        if (memcmp(p + off, "SHEX", 4) != 0 && memcmp(p + off, "SHDR", 4) != 0) continue;
        const uint32_t chunkSize = u32(off + 4);
        const size_t begin = off + 8;
        if (begin + chunkSize > size || chunkSize < 8) return d;
        const uint32_t* tok = reinterpret_cast<const uint32_t*>(p + begin);  // DXBC is 4-byte aligned
        const uint32_t count = std::min<uint32_t>(tok[1], chunkSize / 4);

        for (uint32_t i = 2; i < count;) {
            const uint32_t op = tok[i] & 0x7ff;
            uint32_t len = (tok[i] >> 24) & 0x7f;
            if (op == 53) len = i + 1 < count ? tok[i + 1] : 0;  // customdata: length in next token
            if (len == 0 || i + len > count) break;

            const bool isDecl = op == 88 || op == 89 || op == 161 || op == 162 || op == 156 || op == 157 || op == 158;
            if (isDecl) {
                uint32_t j = i + 1;
                if (tok[i] & 0x80000000u)  // extended opcode tokens
                    while (j < i + len && (tok[j++] & 0x80000000u)) {}
                const uint32_t operand = tok[j++];
                if (operand & 0x80000000u)  // extended operand token
                    ++j;
                const uint32_t dims = (operand >> 20) & 3;
                // SM 5.1 declares a range: [id][lower][upper]; SM 5.0: [slot]([size]).
                const uint32_t slot = dims == 3 ? (j + 1 < i + len ? tok[j + 1] : 0) : (j < i + len ? tok[j] : 0);
                switch (op) {
                    case 89:  // dcl_constantbuffer cb#[size]
                        if (slot < 14) {
                            d.cb |= 1u << slot;
                            d.cbVec4[slot] = dims == 2 && j + 1 < i + len ? tok[j + 1] : 0;
                        }
                        break;
                    case 88: case 161: case 162:  // dcl_resource(_raw/_structured) t#
                        if (slot < 128) d.srv[slot >> 6] |= 1ull << (slot & 63);
                        break;
                    default:  // dcl_uav_typed/raw/structured u#
                        if (slot < 64) d.uav |= 1ull << slot;
                        break;
                }
            }
            i += len;
        }
        d.valid = true;
        return d;
    }
    return d;
}
