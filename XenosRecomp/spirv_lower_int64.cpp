#include "spirv_lower_int64.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace
{
    enum Op : uint16_t
    {
        OpName = 5,
        OpMemberName = 6,
        OpCapability = 17,
        OpTypeInt = 21,
        OpTypeVector = 23,
        OpTypeArray = 28,
        OpTypeRuntimeArray = 29,
        OpTypeStruct = 30,
        OpTypePointer = 32,
        OpTypeFunction = 33,
        OpConstant = 43,
        OpConstantComposite = 44,
        OpSpecConstant = 50,
        OpFunction = 54,
        OpStore = 62,
        OpLoad = 61,
        OpCopyMemory = 63,
        OpDecorate = 71,
        OpCompositeConstruct = 80,
        OpCompositeExtract = 81,
        OpCopyObject = 83,
        OpImageWrite = 99,
        OpUConvert = 113,
        OpSConvert = 114,
        OpBitcast = 124,
        OpIAdd = 128,
        OpIAddCarry = 149,
        OpShiftRightArithmetic = 195,
        OpControlBarrier = 224,
        OpMemoryBarrier = 225,
        OpLoopMerge = 246,
        OpSelectionMerge = 247,
        OpBranchConditional = 250,
        OpSwitch = 251,
        OpReturnValue = 254,
    };

    constexpr uint32_t CapabilityInt64 = 11;

    struct Inst
    {
        uint16_t op;
        std::vector<uint32_t> w; // including the word count / opcode word
    };

    Inst Make(uint16_t op, std::initializer_list<uint32_t> operands)
    {
        Inst inst{op, {}};
        inst.w.reserve(operands.size() + 1);
        inst.w.push_back((uint32_t(operands.size() + 1) << 16) | op);
        inst.w.insert(inst.w.end(), operands);
        return inst;
    }

    // Whether word k of a function-body instruction is an id operand rather
    // than a literal, for the opcodes whose literals could otherwise be taken
    // for an id that happens to share their value. Unlisted opcodes are treated
    // as all-id, which can only make the pass refuse, never miscompile.
    bool IsIdOperand(uint16_t op, size_t k)
    {
        switch (op)
        {
        case 12:  // OpExtInst: type, result, set, instruction literal, operands
            return k >= 5;
        case 61:  // OpLoad: type, result, pointer, memory operands
            return k == 3;
        case 62:  // OpStore: pointer, object, memory operands
            return k == 1 || k == 2;
        case 63:  // OpCopyMemory
            return k == 1 || k == 2;
        case 81:  // OpCompositeExtract: type, result, composite, indices
            return k == 3;
        case 82:  // OpCompositeInsert: type, result, object, composite, indices
            return k == 3 || k == 4;
        case 79:  // OpVectorShuffle: type, result, v1, v2, components
            return k == 3 || k == 4;
        case 87: case 88: case 91: case 92: case 95: case 98:
            // Sample/fetch/read: type, result, image, coordinate, operands
            // mask, operand ids.
            return k != 5;
        case 89: case 90: case 93: case 94: case 96: case 97:
            // Dref sample and gathers carry a Dref/component id first.
            return k != 6;
        case 99:  // OpImageWrite: image, coordinate, texel, image operands
            return k <= 3 || k >= 5;
        case 246: // OpLoopMerge: merge, continue, control literals
            return k == 1 || k == 2;
        case 247: // OpSelectionMerge: merge, control literal
            return k == 1;
        case 251: // OpSwitch: selector, default, (literal, label)...
            return k == 1 || k == 2 || (k > 2 && (k - 3) % 2 == 1);
        default:
            return true;
        }
    }

    // Instructions in a function body that carry no result type in word 1.
    bool HasNoResultType(uint16_t op)
    {
        switch (op)
        {
        case OpStore:
        case OpCopyMemory:
        case OpImageWrite:
        case OpControlBarrier:
        case OpMemoryBarrier:
        case OpLoopMerge:
        case OpSelectionMerge:
        case OpBranchConditional:
        case OpSwitch:
        case OpReturnValue:
            return true;
        default:
            return false;
        }
    }
}

bool LowerInt64(std::vector<uint32_t>& words, std::string& error)
{
    error.clear();
    if (words.size() < 5 || words[0] != 0x07230203)
    {
        error = "not a SPIR-V module";
        return false;
    }

    std::vector<Inst> insts;
    for (size_t i = 5; i < words.size();)
    {
        const uint32_t count = words[i] >> 16;
        if (count == 0 || i + count > words.size())
        {
            error = "truncated instruction stream";
            return false;
        }
        insts.push_back({uint16_t(words[i] & 0xFFFF), {words.begin() + i, words.begin() + i + count}});
        i += count;
    }

    // Locate the integer types.
    std::unordered_set<uint32_t> type64;
    uint32_t u32 = 0, i32 = 0, v2u32 = 0;
    size_t u32Index = SIZE_MAX, i32Index = SIZE_MAX, v2u32Index = SIZE_MAX, first64Index = SIZE_MAX;
    for (size_t i = 0; i < insts.size(); i++)
    {
        const Inst& inst = insts[i];
        if (inst.op == OpTypeInt)
        {
            const uint32_t width = inst.w[2], sign = inst.w[3];
            if (width == 64)
            {
                type64.insert(inst.w[1]);
                if (i < first64Index)
                    first64Index = i;
            }
            else if (width == 32 && sign == 0)
            {
                u32 = inst.w[1];
                u32Index = i;
            }
            else if (width == 32 && sign == 1)
            {
                i32 = inst.w[1];
                i32Index = i;
            }
        }
    }
    if (type64.empty())
        return false; // nothing to do

    for (size_t i = 0; i < insts.size(); i++)
    {
        const Inst& inst = insts[i];
        if (inst.op == OpTypeVector && inst.w[2] == u32 && inst.w[3] == 2)
        {
            v2u32 = inst.w[1];
            v2u32Index = i;
        }
    }

    uint32_t bound = words[3];
    auto newId = [&] { return bound++; };

    // The lowered values are built from uint32: declare it where the 64-bit
    // type was when the module has none, or move it there when it comes later.
    // A plain OpTypeInt depends on nothing, so moving it up is always legal.
    const bool createU32 = u32 == 0;
    const bool moveU32 = !createU32 && u32Index > first64Index;
    if (createU32)
        u32 = newId();

    // Declarations that replace the first 64-bit OpTypeInt.
    const bool moveVector = v2u32 != 0 && v2u32Index > first64Index;
    if (v2u32 == 0)
        v2u32 = newId();
    const uint32_t carryStruct = newId();
    const uint32_t zero = newId();
    uint32_t c31 = 0;
    if (i32 != 0)
        c31 = newId();

    std::vector<Inst> out;
    out.reserve(insts.size() * 2);

    std::unordered_set<uint32_t> value64; // result ids now holding a uvec2
    std::unordered_map<uint32_t, uint32_t> typeOf;
    std::unordered_map<uint32_t, uint32_t> u32Constants; // literal -> id

    auto u32Constant = [&](uint32_t value) -> uint32_t
    {
        auto it = u32Constants.find(value);
        if (it != u32Constants.end())
            return it->second;
        const uint32_t id = newId();
        out.push_back(Make(OpConstant, {u32, id, value}));
        u32Constants.emplace(value, id);
        return id;
    };

    auto lowLane = [&](uint32_t value) -> uint32_t
    {
        const uint32_t id = newId();
        out.push_back(Make(OpCompositeExtract, {u32, id, value, 0}));
        return id;
    };

    auto highLane = [&](uint32_t value) -> uint32_t
    {
        const uint32_t id = newId();
        out.push_back(Make(OpCompositeExtract, {u32, id, value, 1}));
        return id;
    };

    auto asU32 = [&](uint32_t value) -> uint32_t
    {
        const uint32_t id = newId();
        auto it = typeOf.find(value);
        if (it != typeOf.end() && it->second == u32)
            out.push_back(Make(OpCopyObject, {u32, id, value}));
        else
            out.push_back(Make(OpBitcast, {u32, id, value}));
        return id;
    };

    bool inFunctions = false;
    for (size_t i = 0; i < insts.size(); i++)
    {
        Inst inst = insts[i];
        auto& w = inst.w;

        if (inst.op == OpFunction)
            inFunctions = true;

        if (!inFunctions)
        {
            switch (inst.op)
            {
            case OpCapability:
                if (w[1] == CapabilityInt64)
                    continue;
                break;
            case OpName:
            case OpMemberName:
            case OpDecorate:
                if (type64.count(w[1]))
                    continue;
                break;
            case OpTypeInt:
                if (w[2] == 64)
                {
                    if (i == first64Index)
                    {
                        if (createU32 || moveU32)
                            out.push_back(Make(OpTypeInt, {u32, 32, 0}));
                        if (moveVector || v2u32Index == SIZE_MAX)
                            out.push_back(Make(OpTypeVector, {v2u32, u32, 2}));
                        out.push_back(Make(OpTypeStruct, {carryStruct, u32, u32}));
                        out.push_back(Make(OpConstant, {u32, zero, 0}));
                        u32Constants.emplace(0, zero);
                        if (c31 != 0 && i32Index < first64Index)
                            out.push_back(Make(OpConstant, {i32, c31, 31}));
                    }
                    continue;
                }
                if (moveU32 && w[1] == u32)
                    continue; // already emitted ahead of the 64-bit type
                break;
            case OpTypeVector:
                if (w[1] == v2u32 && moveVector)
                    continue; // already emitted ahead of its first use
                [[fallthrough]];
            case OpTypeArray:
            case OpTypeRuntimeArray:
                if (type64.count(w[2]))
                {
                    error = "64-bit integers inside a vector or array type";
                    return false;
                }
                break;
            case OpTypeFunction:
                for (size_t k = 2; k < w.size(); k++)
                    if (type64.count(w[k]))
                    {
                        error = "64-bit integers in a function signature";
                        return false;
                    }
                break;
            case OpTypeStruct:
                for (size_t k = 2; k < w.size(); k++)
                    if (type64.count(w[k]))
                        w[k] = v2u32;
                break;
            case OpTypePointer:
                if (type64.count(w[3]))
                    w[3] = v2u32;
                break;
            case OpSpecConstant:
                if (type64.count(w[1]))
                {
                    error = "64-bit specialization constant";
                    return false;
                }
                break;
            case OpConstant:
                if (type64.count(w[1]))
                {
                    const uint32_t lo = u32Constant(w[3]);
                    const uint32_t hi = u32Constant(w.size() > 4 ? w[4] : 0);
                    out.push_back(Make(OpConstantComposite, {v2u32, w[2], lo, hi}));
                    value64.insert(w[2]);
                    continue;
                }
                break;
            default:
                break;
            }

            if (inst.op == OpTypeInt && w[1] == i32 && c31 != 0 && i32Index > first64Index)
            {
                out.push_back(std::move(inst));
                out.push_back(Make(OpConstant, {i32, c31, 31}));
                continue;
            }

            if (inst.op == OpConstant || inst.op == OpConstantComposite)
                typeOf[w[2]] = w[1];
            out.push_back(std::move(inst));
            continue;
        }

        // Function bodies.
        const bool resultIs64 = w.size() >= 3 && !HasNoResultType(inst.op) && type64.count(w[1]);

        switch (inst.op)
        {
        case OpIAdd:
            if (resultIs64)
            {
                const uint32_t a = w[3], b = w[4], result = w[2];
                if (!value64.count(a) || !value64.count(b))
                {
                    error = "64-bit add of an operand that is not a lowered value";
                    return false;
                }
                const uint32_t aLo = lowLane(a), aHi = highLane(a);
                const uint32_t bLo = lowLane(b), bHi = highLane(b);
                const uint32_t sum = newId();
                out.push_back(Make(OpIAddCarry, {carryStruct, sum, aLo, bLo}));
                const uint32_t lo = lowLane(sum), carry = highLane(sum);
                const uint32_t partial = newId();
                out.push_back(Make(OpIAdd, {u32, partial, aHi, bHi}));
                const uint32_t hi = newId();
                out.push_back(Make(OpIAdd, {u32, hi, partial, carry}));
                out.push_back(Make(OpCompositeConstruct, {v2u32, result, lo, hi}));
                value64.insert(result);
                continue;
            }
            break;

        case OpSConvert:
            if (resultIs64)
            {
                if (c31 == 0)
                {
                    error = "sign extension without an int32 type";
                    return false;
                }
                const uint32_t x = w[3], result = w[2];
                const uint32_t lo = asU32(x);
                uint32_t signedX = x;
                auto it = typeOf.find(x);
                if (it != typeOf.end() && it->second != i32)
                {
                    signedX = newId();
                    out.push_back(Make(OpBitcast, {i32, signedX, x}));
                }
                const uint32_t sign = newId();
                out.push_back(Make(OpShiftRightArithmetic, {i32, sign, signedX, c31}));
                const uint32_t hi = newId();
                out.push_back(Make(OpBitcast, {u32, hi, sign}));
                out.push_back(Make(OpCompositeConstruct, {v2u32, result, lo, hi}));
                value64.insert(result);
                continue;
            }
            break;

        case OpUConvert:
            if (resultIs64)
            {
                const uint32_t lo = asU32(w[3]);
                out.push_back(Make(OpCompositeConstruct, {v2u32, w[2], lo, zero}));
                value64.insert(w[2]);
                continue;
            }
            break;

        case OpBitcast:
            if (resultIs64)
            {
                if (!value64.count(w[3]))
                {
                    error = "bitcast to a 64-bit integer from a non-integer value";
                    return false;
                }
                out.push_back(Make(OpCopyObject, {v2u32, w[2], w[3]}));
                value64.insert(w[2]);
                continue;
            }
            // A pointer made from a lowered value: uvec2 -> pointer is legal.
            if (w.size() >= 4 && value64.count(w[3]))
            {
                out.push_back(std::move(inst));
                continue;
            }
            break;

        case OpLoad:
            if (resultIs64)
            {
                w[1] = v2u32;
                value64.insert(w[2]);
                out.push_back(std::move(inst));
                continue;
            }
            break;

        default:
            break;
        }

        if (resultIs64)
        {
            error = "unsupported instruction producing a 64-bit integer (opcode " + std::to_string(inst.op) + ")";
            return false;
        }
        for (size_t k = 2; k < w.size(); k++)
        {
            if (IsIdOperand(inst.op, k) && value64.count(w[k]))
            {
                error = "unsupported use of a 64-bit integer (opcode " + std::to_string(inst.op) + ")";
                return false;
            }
        }

        if (w.size() >= 3 && !HasNoResultType(inst.op))
            typeOf[w[2]] = w[1];
        out.push_back(std::move(inst));
    }

    std::vector<uint32_t> result(words.begin(), words.begin() + 5);
    result[3] = bound;
    for (const Inst& inst : out)
        result.insert(result.end(), inst.w.begin(), inst.w.end());
    words = std::move(result);
    return true;
}
