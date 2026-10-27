// Finds Unreal console variables in the running game and reads and sets them. Adapted from
// ReunionTweaks (UE 5.6) and AITDTweaks (UE 4.27) for Life is Strange: True Colors
// (UE 5.2.1, project Chronos, no PDB). Offsets were read from Chronos-Win64-Shipping.exe
// (docs/FINDINGS.md).
//
// The exe isn't protected, but sections are still picked by their flags (as in ReunionTweaks):
// code = executable and not writable; read-only data = readable, not executable, not writable.
//
// IConsoleManager::Singleton is found from the game's own code: cvar lookups load it with
// `mov rcx, [rip+Singleton]` shortly before `lea rdx, [rip+L"r.ScreenPercentage"]`. The address
// most matches agree on wins.
//
// Vtable offsets. UE 5.2 keeps UE 4.27's IConsoleVariable layout; only FindConsoleVariable
// moved (0xb0 in 4.27, 0xb8 here, as in 5.5):
//   IConsoleManager  +0xb8  FindConsoleVariable(const TCHAR* Name, bool bTrackFrequentCalls)
//                    +0x50  CallAllConsoleVariableSinks
//   IConsoleObject   +0x18 flags field (not a slot; top byte = SetBy priority)
//                    IsVariableBool/Int/Float/String: found per game by reading their code
//                    (FindTypeSlots); expected at 0x30-0x48. AsVariableBool/Int/Float follow
//                    0x20 later (AsVariableInt +0x58: 536 call sites).
//   IConsoleVariable +0x80 Set(const TCHAR* Value, EConsoleVariableFlags SetBy)
//                    +0x88 GetBool  +0x90 GetInt  +0x98 GetFloat
//
// Setting from ReShade's thread: UE 5.2 has no ECVF_Set_SetOnly_Unsafe, so (as in the UE 4
// add-ons) Set runs normally, and because OnCVarChange only copies the value to the render
// thread's shadow on the game thread, MirrorToRenderer does that copy afterwards.
//
// Priority: the game's own code sets at 0x09000000 (SetByCode, 51 call sites) and the console at
// 0x0A000000 (SetByConsole, the highest value passed anywhere in the exe). The add-on sets at
// SetByConsole, so the menu can't override a held value; values go back to the game's value at
// the same priority.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <cwchar>
#include <map>
#include <vector>

namespace cvars {

constexpr uint32_t kFindConsoleVariable = 0xb0;  // UE 4.25 (True Colors)
constexpr uint32_t kCallAllConsoleVariableSinks = 0x50;
constexpr uint32_t kSet = 0x80;
constexpr uint32_t kGetInt = 0x90;
constexpr uint32_t kGetFloat = 0x98;
constexpr uint32_t kFlags = 0x18;
constexpr uint32_t kSetByConsole = 0x0a000000;

// IsVariableBool's slot; Int, Float and String follow it, then AsVariableBool/Int/Float.
// Set by FindTypeSlots.
inline uint32_t g_isVariableBool = 0;

struct Section {
    const uint8_t* begin = nullptr;
    size_t size = 0;
};

enum class SectionKind { Code, ReadOnly, Writable };

// The module's sections of one kind, in address order.
inline std::vector<Section> Sections(const uint8_t* base, SectionKind kind) {
    std::vector<Section> out;
    auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    auto sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const DWORD c = sec[i].Characteristics;
        const bool x = c & IMAGE_SCN_MEM_EXECUTE, r = c & IMAGE_SCN_MEM_READ, w = c & IMAGE_SCN_MEM_WRITE;
        const bool want = kind == SectionKind::Code       ? x && !w
                          : kind == SectionKind::ReadOnly ? r && !x && !w
                                                          : r && w && !x;
        if (want) out.push_back({base + sec[i].VirtualAddress, sec[i].Misc.VirtualSize});
    }
    return out;
}

inline bool Contains(const std::vector<Section>& v, const void* p) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (const Section& s : v)
        if (b >= s.begin && b < s.begin + s.size) return true;
    return false;
}

// A NUL-terminated UTF-16 string that starts right after another string's terminator.
inline const uint8_t* FindWideString(const Section& s, const wchar_t* text) {
    const size_t n = (wcslen(text) + 1) * 2;
    for (size_t i = 2; i + n <= s.size; i += 2)
        if (s.begin[i - 2] == 0 && s.begin[i - 1] == 0 && memcmp(s.begin + i, text, n) == 0) return s.begin + i;
    return nullptr;
}

inline const uint8_t* FindWideString(const std::vector<Section>& v, const wchar_t* text) {
    for (const Section& s : v)
        if (const uint8_t* p = FindWideString(s, text)) return p;
    return nullptr;
}

// IConsoleManager::Singleton (the variable holding the pointer), or null. *votes gets how
// many of the matches agreed, *matches how many there were.
inline void** FindConsoleManagerSingleton(const uint8_t* base, int* votes = nullptr, int* matches = nullptr) {
    const uint8_t* name = FindWideString(Sections(base, SectionKind::ReadOnly), L"r.ScreenPercentage");
    if (!name) return nullptr;
    std::map<void**, int> found;
    int total = 0;
    for (const Section& text : Sections(base, SectionKind::Code)) {
        for (size_t i = 0x40; i + 7 <= text.size; ++i) {
            const uint8_t* p = text.begin + i;
            // lea r64, [rip+disp32] -> the name
            if ((p[0] != 0x48 && p[0] != 0x4c) || p[1] != 0x8d || (p[2] & 0xc7) != 0x05) continue;
            int32_t disp;
            memcpy(&disp, p + 3, 4);
            if (p + 7 + disp != name) continue;
            // mov rcx, [rip+disp32] within the 0x40 bytes before it
            for (const uint8_t* q = p - 7; q >= p - 0x40; --q) {
                if (q[0] != 0x48 || q[1] != 0x8b || q[2] != 0x0d) continue;
                memcpy(&disp, q + 3, 4);
                ++found[reinterpret_cast<void**>(const_cast<uint8_t*>(q + 7 + disp))];
                ++total;
                break;
            }
        }
    }
    void** best = nullptr;
    int bestVotes = 0;
    for (auto& [addr, n] : found)
        if (n > bestVotes) best = addr, bestVotes = n;
    if (votes) *votes = bestVotes;
    if (matches) *matches = total;
    return best;
}

template <typename Fn>
inline Fn Virtual(void* object, uint32_t offset) {
    return *reinterpret_cast<Fn*>(*reinterpret_cast<uint8_t**>(object) + offset);
}

// memcpy that reports a fault instead of raising it.
inline bool SafeCopy(void* dst, const void* src, size_t n) {
    __try {
        memcpy(dst, src, n);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// True if the object's vtable lies in the module's read-only data (a sanity check before any
// virtual call on a pointer we found ourselves).
inline bool HasModuleVtable(const uint8_t* base, const void* object) {
    const uint8_t* vtable = nullptr;
    if (!object || !SafeCopy(&vtable, object, 8)) return false;
    return Contains(Sections(base, SectionKind::ReadOnly), vtable);
}

using FindFn = void*(__fastcall*)(void* self, const wchar_t* name, bool track);
using PtrFn = void*(__fastcall*)(void* self);
using IntFn = int32_t(__fastcall*)(void* self);
using FloatFn = float(__fastcall*)(void* self);
using SetFn = void(__fastcall*)(void* self, const wchar_t* value, uint32_t setBy);

// The IConsoleVariable object for <name> (null if absent or a command). Guarded: a wrong
// vtable layout faults here instead of taking the game down.
inline void* FindVariable(void* manager, const wchar_t* name) {
    __try {
        return Virtual<FindFn>(manager, kFindConsoleVariable)(manager, name, false);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
}

// The first instruction of the function in <object>'s vtable at <offset>, past a jmp thunk.
inline const uint8_t* SlotCode(const void* object, uint32_t offset) {
    const uint8_t* fn = nullptr;
    __try {
        fn = *reinterpret_cast<const uint8_t* const*>(*reinterpret_cast<const uint8_t* const*>(object) + offset);
        if (fn[0] == 0xe9) {
            int32_t disp;
            memcpy(&disp, fn + 1, 4);
            fn += 5 + disp;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
    return fn;
}

// What the function in <object>'s vtable at <offset> returns, read from its code: 1 for
// `return true`, 0 for `return false`, -1 for anything else. Nothing is called.
inline int ConstantReturn(const void* object, uint32_t offset) {
    const uint8_t* fn = SlotCode(object, offset);
    uint8_t b[6];
    if (!fn || !SafeCopy(b, fn, 6)) return -1;
    if (b[0] == 0xb0 && b[2] == 0xc3) return b[1] == 1 ? 1 : b[1] == 0 ? 0 : -1;  // mov al, imm8; ret
    if ((b[0] == 0x32 || b[0] == 0x30 || b[0] == 0x33 || b[0] == 0x31) && b[1] == 0xc0 && b[2] == 0xc3)
        return 0;  // xor al/eax, al/eax; ret
    if (b[0] == 0xb8 && b[5] == 0xc3) {  // mov eax, imm32; ret
        uint32_t v;
        memcpy(&v, b + 1, 4);
        return v == 1 ? 1 : v == 0 ? 0 : -1;
    }
    return -1;
}

// Finds the four IsVariable* slots from a known int and a known float variable: the slot where
// the int's code reads false, true, false, false and the float's false, false, true, false.
inline bool FindTypeSlots(const void* intVar, const void* floatVar) {
    for (uint32_t s = 0x28; s <= 0x60; s += 8) {
        int i[4], f[4];
        for (int k = 0; k < 4; ++k) {
            i[k] = ConstantReturn(intVar, s + 8 * k);
            f[k] = ConstantReturn(floatVar, s + 8 * k);
        }
        if (i[0] == 0 && i[1] == 1 && i[2] == 0 && i[3] == 0 && f[0] == 0 && f[1] == 0 && f[2] == 1 && f[3] == 0) {
            g_isVariableBool = s;
            return true;
        }
    }
    return false;
}

enum class Type { Unknown, Bool, Int, Float, String };

inline const char* TypeName(Type t) {
    switch (t) {
        case Type::Bool: return "bool";
        case Type::Int: return "int";
        case Type::Float: return "float";
        case Type::String: return "string";
        default: return "?";
    }
}

// The value type, read from the IsVariable* slots' code (no calls).
inline Type GetType(const void* var) {
    if (!g_isVariableBool || !var) return Type::Unknown;
    if (ConstantReturn(var, g_isVariableBool + 0x10) == 1) return Type::Float;
    if (ConstantReturn(var, g_isVariableBool + 0x08) == 1) return Type::Int;
    if (ConstantReturn(var, g_isVariableBool) == 1) return Type::Bool;
    if (ConstantReturn(var, g_isVariableBool + 0x18) == 1) return Type::String;
    return Type::Unknown;
}

inline bool GetFloat(void* var, float& out) {
    __try {
        out = Virtual<FloatFn>(var, kGetFloat)(var);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

inline bool GetInt(void* var, int32_t& out) {
    __try {
        out = Virtual<IntFn>(var, kGetInt)(var);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// The current value as text (as Set accepts it), or false.
inline bool GetText(void* var, Type t, wchar_t* out, size_t n) {
    if (t == Type::Float) {
        float v;
        if (!GetFloat(var, v)) return false;
        swprintf(out, n, L"%g", v);
        return true;
    }
    if (t == Type::Int || t == Type::Bool) {
        int32_t v;
        if (!GetInt(var, v)) return false;
        swprintf(out, n, L"%d", v);
        return true;
    }
    return false;
}

// The SetBy priority of the last Set (top byte of the flags), or -1.
inline int Priority(void* var) {
    uint32_t flags = 0;
    if (!SafeCopy(&flags, static_cast<uint8_t*>(var) + kFlags, 4)) return -1;
    return static_cast<int>(flags >> 24);
}

// ---------------------------------------------------------------- render thread copy
//
// Data cvars (FConsoleVariable<T>) hold TConsoleVariableData<T>: the game thread's value, then
// the render thread's copy right after it (+4 for int and float, +1 for bool). AsVariable*
// returns it; it lies inside the cvar object.
//
// Ref cvars (FConsoleVariableRef<T>, AsVariable* returns null) hold a pointer to the engine
// global the renderer reads and, right after it, their own value. The value's offset is read
// from the getter's code (mov eax, [rcx+X] / movzx eax, byte [rcx+X] / movss xmm0, [rcx+X]),
// and the pointer must point into the module's writable data.

inline void* AsData(void* var, Type t) {
    if (!g_isVariableBool || (t != Type::Bool && t != Type::Int && t != Type::Float)) return nullptr;
    const uint32_t slot = g_isVariableBool + 0x20 + (t == Type::Bool ? 0 : t == Type::Int ? 8 : 0x10);
    void* out = nullptr;
    __try {
        out = Virtual<PtrFn>(var, slot)(var);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
    // Must be inside the object, past the flags.
    auto p = static_cast<uint8_t*>(out), v = static_cast<uint8_t*>(var);
    return out && p > v + kFlags && p < v + 0x200 ? out : nullptr;
}

// The Ref cvar's own value offset, read from its getter's code, or -1.
inline int RefValueOffset(void* var, Type t) {
    const uint8_t* fn = SlotCode(var, t == Type::Float ? kGetFloat : kGetInt);
    uint8_t b[7];
    if (!fn || !SafeCopy(b, fn, 7)) return -1;
    if (t == Type::Int && b[0] == 0x8b && b[1] == 0x41 && b[3] == 0xc3) return b[2];
    if (t == Type::Bool && b[0] == 0x0f && b[1] == 0xb6 && b[2] == 0x41 && b[4] == 0xc3) return b[3];
    if (t == Type::Float && b[0] == 0xf3 && b[1] == 0x0f && b[2] == 0x10 && b[3] == 0x41 && b[5] == 0xc3) return b[4];
    return -1;
}

enum class Mirror { Data, Ref, NotNumeric, Unknown };

inline const char* MirrorName(Mirror m) {
    switch (m) {
        case Mirror::Data: return "data";
        case Mirror::Ref: return "ref";
        case Mirror::NotNumeric: return "not numeric";
        default: return "unknown layout";
    }
}

// Copies the main value to where the renderer reads it (what OnCVarChange does on the game
// thread). Aligned stores, so the render thread sees old or new, never a mix.
inline Mirror MirrorToRenderer(const uint8_t* base, void* var, Type t) {
    if (t != Type::Bool && t != Type::Int && t != Type::Float) return Mirror::NotNumeric;
    const size_t size = t == Type::Bool ? 1 : 4;
    uint8_t value[4];
    if (void* data = AsData(var, t))
        return SafeCopy(value, data, size) && SafeCopy(static_cast<uint8_t*>(data) + size, value, size) ? Mirror::Data
                                                                                                         : Mirror::Unknown;
    const int off = RefValueOffset(var, t);
    if (off < 0x20) return Mirror::Unknown;
    uint8_t* global = nullptr;
    auto obj = static_cast<uint8_t*>(var);
    if (!SafeCopy(&global, obj + off - 8, 8) || !Contains(Sections(base, SectionKind::Writable), global))
        return Mirror::Unknown;
    return SafeCopy(value, obj + off, size) && SafeCopy(global, value, size) ? Mirror::Ref : Mirror::Unknown;
}

// ---------------------------------------------------------------- sinks
//
// Some settings only reach the engine through a console variable sink. The game thread calls
// IConsoleManager::CallAllConsoleVariableSinks every tick, and it runs the sinks when
// FConsoleManager::bCallAllConsoleVariableSinks is set, then clears it. Setting that flag from
// any thread is safe; the sinks then run on the game thread. Its offset is read from the
// function's first instructions: cmp byte ptr [rcx+offset], 0.
inline int FindSinkFlagOffset(void* manager) {
    const uint8_t* fn = SlotCode(manager, kCallAllConsoleVariableSinks);
    uint8_t b[40];
    if (!fn || !SafeCopy(b, fn, sizeof(b))) return -1;
    for (int i = 0; i < 32; ++i) {
        const uint8_t* p = b + i;
        if (p[0] != 0x80) continue;
        if ((p[1] == 0x79 || p[1] == 0x7b) && p[3] == 0) return p[2];  // cmp byte [rcx/rbx+disp8], 0
        if ((p[1] == 0xb9 || p[1] == 0xbb) && p[6] == 0) {             // cmp byte [rcx/rbx+disp32], 0
            int32_t disp;
            memcpy(&disp, p + 2, 4);
            return disp > 0 && disp < 0x1000 ? disp : -1;
        }
    }
    return -1;
}

// The flag's value (0 or 1), or -1 if it doesn't look like a bool.
inline int ReadSinkFlag(void* manager, int offset) {
    uint8_t v = 0xff;
    if (offset < 0 || !SafeCopy(&v, static_cast<uint8_t*>(manager) + offset, 1)) return -1;
    return v <= 1 ? v : -1;
}

// Asks the game thread to run all sinks at its next tick.
inline bool RequestSinks(void* manager, int offset) {
    if (ReadSinkFlag(manager, offset) < 0) return false;
    __try {
        static_cast<volatile uint8_t*>(manager)[offset] = 1;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// IConsoleVariable::Set at console priority: stores the main value, records the priority and
// runs the change delegates. Follow it with MirrorToRenderer (off the game thread). False if it
// faulted.
inline bool Set(void* var, const wchar_t* value) {
    __try {
        Virtual<SetFn>(var, kSet)(var, value, kSetByConsole);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

}  // namespace cvars
