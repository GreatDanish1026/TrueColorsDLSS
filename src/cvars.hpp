// Finds Unreal console variables in the running game and reads and sets them, for Life is
// Strange: True Colors (UE 4.25Plus, project Siren, no PDB). The scanner comes from
// DoubleExposureTweaks; the game-thread setting from FrankStoneTweaks. Offsets were read from
// Siren-Win64-Shipping.exe's own call sites (docs/FINDINGS.md).
//
// The exe isn't protected; sections are still picked by their flags: code = executable and not
// writable; read-only data = readable, not executable, not writable.
//
// IConsoleManager::Singleton is found from the game's own code: cvar lookups load it with
// `mov rcx, [rip+Singleton]` shortly before `lea rdx, [rip+L"r.ScreenPercentage"]`. The address
// most matches agree on wins (RVA 0x492a1f0 in build 8050014).
//
// Vtable offsets (UE 4.25, same as 4.26/4.27):
//   IConsoleManager  +0x50  CallAllConsoleVariableSinks (6 call sites; +0x58 registers a sink)
//                    +0xb0  FindConsoleVariable(const TCHAR* Name, bool bTrackFrequentCalls)
//   IConsoleObject   +0x18 flags field (not a slot; top byte = SetBy priority)
//                    IsVariableBool/Int/Float/String: found by reading their code
//                    (FindTypeSlots); AsVariableBool/Int/Float follow 0x20 later
//                    (AsVariableInt +0x58: about 3000 call sites).
//   IConsoleVariable +0x80 Set(const TCHAR* Value, EConsoleVariableFlags SetBy)
//                    +0x88 GetBool  +0x90 GetInt  +0x98 GetFloat  +0xa0 GetString
//
// Priority: the game's code sets at 0x08000000 (SetByCode) and the console at 0x09000000
// (SetByConsole, the highest value passed to Set anywhere in the exe). The add-on sets at
// SetByConsole, so the menu can't override a held value; letting go puts the game's value and
// priority back.
//
// Threads: in UE 4 a Set reaches the renderer only when it runs on the game thread
// (OnCVarChange enqueues the render-thread copy there), and some change callbacks flush
// rendering, which hangs when called from the render or present thread (Jedi Survivor,
// r.MaxAnisotropy). So every Set runs on the game thread: the add-on calls it from its own
// CallAllConsoleVariableSinks (HookSinks below), which FEngineLoop::Tick calls once per frame.
// Off the game thread the getters return the renderer's copy.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <cwchar>
#include <map>
#include <vector>

namespace cvars {

constexpr uint32_t kFindConsoleVariable = 0xb0;
constexpr uint32_t kCallAllConsoleVariableSinks = 0x50;
constexpr uint32_t kSet = 0x80;
constexpr uint32_t kGetInt = 0x90;
constexpr uint32_t kGetFloat = 0x98;
constexpr uint32_t kFlags = 0x18;
constexpr uint32_t kSetByCode = 0x08000000;
constexpr uint32_t kSetByConsole = 0x09000000;
constexpr uint32_t kPriorityMask = 0xff000000;

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

// The flags word (top byte: priority of the last accepted Set), or 0 if unreadable.
inline uint32_t GetFlags(void* var) {
    uint32_t f = 0;
    SafeCopy(&f, static_cast<uint8_t*>(var) + kFlags, 4);
    return f;
}

// Puts the priority byte back to <priority> (the game's, when we let go of a variable), so the
// game's menu can set it again. Only the top byte changes.
inline bool RestorePriority(void* var, uint32_t priority) {
    uint32_t f = GetFlags(var);
    f = (f & ~kPriorityMask) | (priority & kPriorityMask);
    return SafeCopy(static_cast<uint8_t*>(var) + kFlags, &f, 4);
}

// Data cvars return their TConsoleVariableData from AsVariable*; Ref cvars (pointing at an
// engine global) return null. Logged only, to see the layout.
inline void* AsData(void* var, Type t) {
    if (!g_isVariableBool || (t != Type::Bool && t != Type::Int && t != Type::Float)) return nullptr;
    const uint32_t slot = g_isVariableBool + 0x20 + (t == Type::Bool ? 0 : t == Type::Int ? 8 : 0x10);
    void* out = nullptr;
    __try {
        out = Virtual<PtrFn>(var, slot)(var);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
    auto p = static_cast<uint8_t*>(out), v = static_cast<uint8_t*>(var);
    return out && p > v + kFlags && p < v + 0x200 ? out : nullptr;
}

// Sets at <setBy> priority: stores the value, records the priority, runs the change callbacks
// and (on the game thread) passes the value to the renderer. Game thread only. False if it
// faulted.
inline bool Set(void* var, const wchar_t* value, uint32_t setBy = kSetByConsole) {
    __try {
        Virtual<SetFn>(var, kSet)(var, value, setBy);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// ---------------------------------------------------------------- running on the game thread
//
// FEngineLoop::Tick calls IConsoleManager::CallAllConsoleVariableSinks (vtable +0x50) through the
// manager's vtable once per frame, on the game thread. HookSinks gives the manager a copy of its
// vtable with that slot pointing at <hook>; the hook does its work and then calls the original
// (returned in *original). The copy keeps the pointer in front of the vtable. Nothing in the
// game's code is changed, and UnhookSinks puts the original vtable back.
using SinksFn = void(__fastcall*)(void* self);

inline void* g_vtableCopy[1 + 256];
inline void** g_vtableOriginal = nullptr;

inline bool HookSinks(const uint8_t* base, void* manager, SinksFn hook, SinksFn* original) {
    const std::vector<Section> code = Sections(base, SectionKind::Code);
    void** vt = nullptr;
    if (!SafeCopy(&vt, manager, 8) || !vt) return false;
    size_t n = 0;
    while (n < 256) {  // slots up to the first one that isn't code
        const uint8_t* fn = nullptr;
        if (!SafeCopy(&fn, vt + n, 8) || !Contains(code, fn)) break;
        ++n;
    }
    const size_t slot = kCallAllConsoleVariableSinks / 8;
    if (n <= kFindConsoleVariable / 8 || !SafeCopy(&g_vtableCopy[0], vt - 1, 8)) return false;
    memcpy(&g_vtableCopy[1], vt, n * 8);
    *original = reinterpret_cast<SinksFn>(vt[slot]);
    g_vtableCopy[1 + slot] = reinterpret_cast<void*>(hook);
    g_vtableOriginal = vt;
    // TrueColorsTweaks and TrueColorsDLSS both hook this slot; the second copies the first's vtable,
    // so its chain runs through the first add-on's code. Pinning keeps every hooked add-on's code
    // loaded for the rest of the process, whatever order ReShade unloads them in.
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                       reinterpret_cast<LPCWSTR>(hook), &self);
    InterlockedExchangePointer(reinterpret_cast<PVOID volatile*>(manager), &g_vtableCopy[1]);
    return true;
}

// Puts the vtable back only if ours is still the one in place (nobody hooked on top of us).
inline void UnhookSinks(void* manager) {
    if (manager && g_vtableOriginal)
        InterlockedCompareExchangePointer(reinterpret_cast<PVOID volatile*>(manager), g_vtableOriginal, &g_vtableCopy[1]);
    g_vtableOriginal = nullptr;
}

}  // namespace cvars
