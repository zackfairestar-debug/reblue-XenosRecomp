#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Rewrites the 64-bit integer address arithmetic DXC emits for
// vk::RawBufferLoad into 32-bit pairs, so the module no longer declares the
// Int64 capability. Mobile drivers (Qualcomm Adreno among them) report
// shaderInt64 = false and reject any pipeline whose shaders declare it, while
// still supporting bufferDeviceAddress.
//
// 64-bit integers become uvec2 {lo, hi}: the same 8 bytes in push constants,
// adds become OpIAddCarry, sign extension an arithmetic shift, and the final
// conversion to a PhysicalStorageBuffer pointer an OpBitcast from the uvec2,
// which SPV_KHR_physical_storage_buffer allows.
//
// Only the instruction shapes the recompiler's shaders produce are handled:
// OpConstant, OpLoad, OpIAdd, OpSConvert, OpUConvert and OpBitcast on 64-bit
// values. Anything else leaves the module untouched and returns false with the
// reason in `error`.
bool LowerInt64(std::vector<uint32_t>& words, std::string& error);
