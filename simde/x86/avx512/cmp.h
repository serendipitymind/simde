/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

/* This code is a lot touchier than you might think; lots of compilers
 * have small, annoying bugs or differences which make it a pain to get
 * everything right, so please be careful when making changes.
 *
 * One thing which has been particularly troublesome is the
 * _MM_CMPINT_ENUM enumeration.  On GCC it doesn't exist (they just use
 * ints, which frankly is more consistent with the rest of the API).
 * The documentation has _MM_CMPINT_{TRUE,FALSE}, but in at least ICC
 * and clang those values don't exist.  ICC will actually emit an error
 * if you try to pass the values 3 or 7 (TRUE/FALSE), which means we
 * can't use the normal CONSTIFY macros.
 *
 * Another problem is that some compilers (XL C/C++ and GCC 7 on
 * AArch64, for example) take a *long* time (and lots of memory)
 * to compile this file, so much so that CI isn't really feasible.
 * Fallbacks directly from the 512-bit functions to the 128-bit
 * functions instead of going through the 256-bit functions first
 * help, but don't fully solve the problem. */

#if !defined(SIMDE_X86_AVX512_CMP_H)
#define SIMDE_X86_AVX512_CMP_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#include "mov_mask.h"
#include "setzero.h"
#include "setone.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

/* GCC doesn't have an _MM_CMPINT_ENUM enum, and all the functions take
 * ints.  On other compilers, however, we want to use _MM_CMPINT_ENUM
 * when possible to avoid mismatches between SIMDE_MM_CMPINT_* and
 * _MM_CMPINT_* values. */
#if (defined(_MM_CMPINT_GE) || defined(_MM_CMPINT_NLT)) && !defined(HEDLEY_GCC_VERSION)
  typedef _MM_CMPINT_ENUM SIMDE_MM_CMPINT_ENUM;
  #define SIMDE_MM_CMPINT_EQ    _MM_CMPINT_EQ
  #define SIMDE_MM_CMPINT_LT    _MM_CMPINT_LT
  #define SIMDE_MM_CMPINT_LE    _MM_CMPINT_LE
  #define SIMDE_MM_CMPINT_NE    _MM_CMPINT_NE
  #define SIMDE_MM_CMPINT_NLT   _MM_CMPINT_NLT
  #define SIMDE_MM_CMPINT_GE    _MM_CMPINT_NLT
  #define SIMDE_MM_CMPINT_NLE   _MM_CMPINT_NLE
  #define SIMDE_MM_CMPINT_GT    _MM_CMPINT_NLE
  #define SIMDE_MM_CMPINT_FALSE (HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, 3))
  #define SIMDE_MM_CMPINT_TRUE  (HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, 7))
#else
  typedef enum {
    SIMDE_MM_CMPINT_EQ    = 0,
    SIMDE_MM_CMPINT_LT    = 1,
    SIMDE_MM_CMPINT_LE    = 2,
    SIMDE_MM_CMPINT_FALSE = 3,
    SIMDE_MM_CMPINT_NE    = 4,
    SIMDE_MM_CMPINT_NLT   = 5,
    #define SIMDE_MM_CMPINT_GE SIMDE_MM_CMPINT_NLT
    SIMDE_MM_CMPINT_NLE   = 6,
    #define SIMDE_MM_CMPINT_GT SIMDE_MM_CMPINT_NLE
    SIMDE_MM_CMPINT_TRUE  = 7,
  } SIMDE_MM_CMPINT_ENUM;
#endif

#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES) && !(defined(_MM_CMPINT_GE) || defined(_MM_CMPINT_NLT))
  typedef SIMDE_MM_CMPINT_ENUM _MM_CMPINT_ENUM;
  #define _MM_CMPINT_EQ    SIMDE_MM_CMPINT_EQ
  #define _MM_CMPINT_LT    SIMDE_MM_CMPINT_LT
  #define _MM_CMPINT_LE    SIMDE_MM_CMPINT_LE
  #define _MM_CMPINT_FALSE SIMDE_MM_CMPINT_FALSE
  #define _MM_CMPINT_NE    SIMDE_MM_CMPINT_NE
  #define _MM_CMPINT_NLT   SIMDE_MM_CMPINT_NLT
  #define _MM_CMPINT_GE    SIMDE_MM_CMPINT_GE
  #define _MM_CMPINT_NLE   SIMDE_MM_CMPINT_NLE
  #define _MM_CMPINT_GT    SIMDE_MM_CMPINT_GT
  #define _MM_CMPINT_TRUE  SIMDE_MM_CMPINT_TRUE
#endif

/* We can't use SIMDE_CONSTIFY_8_ here because ICC doesn't support th
 * _MM_CMPINT_FALSE and _MM_CMPINT_TRUE values.  If we used constify
 * there would be a call to _mm*_cmp_*_mask with a value of 3 or 7,
 * which is an error in ICC.
 *
 * Instead, we use this macro.  The idea is still somewhat similar,
 * we just handle the two problem cases separately. */
 #define SIMDE_MM_CMPINT_CONSTIFY_(return_value, prefix, type_name, si_type_name, mask, a, b, imm8) \
  do { \
    switch (HEDLEY_STATIC_CAST(int, imm8)) { \
      case SIMDE_MM_CMPINT_EQ: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_EQ); \
        break; \
      case SIMDE_MM_CMPINT_LT: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_LT); \
        break; \
      case SIMDE_MM_CMPINT_LE: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_LE); \
        break; \
      case SIMDE_MM_CMPINT_FALSE: \
        return_value = 0; \
        break; \
      case SIMDE_MM_CMPINT_NE: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_NE); \
        break; \
      case SIMDE_MM_CMPINT_NLT: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_NLT); \
        break; \
      case SIMDE_MM_CMPINT_NLE: \
        return_value = _##prefix##_cmp_##type_name##_mask(a, b, SIMDE_MM_CMPINT_NLE); \
        break; \
      case SIMDE_MM_CMPINT_TRUE: \
        return_value = mask; \
        break; \
      default: \
        HEDLEY_UNREACHABLE(); \
        break; \
    } \
  } while (0)

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_ps_mask (simde__m128 a, simde__m128 b, const int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE)
    SIMDE_CONSTIFY_32_(_mm_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    simde__m128 rm;
    SIMDE_CONSTIFY_32_(simde_mm_cmp_ps, rm, simde_mm_setzero_ps(), imm8, a, b);
    r = HEDLEY_STATIC_CAST(simde__mmask8, simde_mm_movemask_ps(rm));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_ps_mask
  #define _mm_cmp_ps_mask(a, b, imm8) simde_mm_cmp_ps_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_pd_mask (simde__m128d a, simde__m128d b, const int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE)
    SIMDE_CONSTIFY_32_(_mm_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    simde__m128d rm;
    SIMDE_CONSTIFY_32_(simde_mm_cmp_pd, rm, simde_mm_setzero_pd(), imm8, a, b);
    r = HEDLEY_STATIC_CAST(simde__mmask8, simde_mm_movemask_pd(rm));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_pd_mask
  #define _mm_cmp_pd_mask(a, b, imm8) simde_mm_cmp_pd_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm_cmp_epi8_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm, epi8, si128, HEDLEY_STATIC_CAST(simde__mmask16, UINT16_C(0xffff)), a, b, imm8);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi8_mask(simde_mm_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi8_mask(simde_mm_cmplt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi8_mask(simde_mm_or_si128(simde_mm_cmplt_epi8(a, b), simde_mm_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm_movepi8_mask(simde_mm_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = ~simde_mm_movepi8_mask(simde_mm_cmplt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = ~simde_mm_movepi8_mask(simde_mm_or_si128(simde_mm_cmplt_epi8(a, b), simde_mm_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vceqq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcltq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcleq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vmvnq_u8(vceqq_s8(a_.neon_i8, b_.neon_i8));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcgeq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcgtq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            default:
              HEDLEY_UNREACHABLE();
              break;
          }

          r = simde_mm_movepi8_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi8_mask
  #define _mm_cmp_epi8_mask(a, b, imm8) simde_mm_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi16_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm, epi16, si128, HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0xff)), a, b, imm8);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi16_mask(simde_mm_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi16_mask(simde_mm_cmplt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi16_mask(simde_mm_or_si128(simde_mm_cmplt_epi16(a, b), simde_mm_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm_movepi16_mask(simde_mm_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = ~simde_mm_movepi16_mask(simde_mm_cmplt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = ~simde_mm_movepi16_mask(simde_mm_or_si128(simde_mm_cmplt_epi16(a, b), simde_mm_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vceqq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcltq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcleq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vmvnq_u16(vceqq_s16(a_.neon_i16, b_.neon_i16));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcgeq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcgtq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            default:
              HEDLEY_UNREACHABLE();
              break;
          }

          r = simde_mm_movepi16_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi16_mask
  #define _mm_cmp_epi16_mask(a, b, imm8) simde_mm_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi32_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm, epi32, si128, HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0xf)), a, b, imm8);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi32_mask(simde_mm_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi32_mask(simde_mm_cmplt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi32_mask(simde_mm_or_si128(simde_mm_cmplt_epi32(a, b), simde_mm_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = simde_mm_movepi32_mask(simde_mm_cmpeq_epi32(a, b)) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm_movepi32_mask(simde_mm_cmplt_epi32(a, b)) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm_movepi32_mask(simde_mm_or_si128(simde_mm_cmplt_epi32(a, b), simde_mm_cmpeq_epi32(a, b))) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0x0f);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0x0f);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vceqq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcltq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcleq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vmvnq_u32(vceqq_s32(a_.neon_i32, b_.neon_i32));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcgeq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcgtq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            default:
              HEDLEY_UNREACHABLE();
              break;
          }

          r = simde_mm_movepi32_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi32_mask
  #define _mm_cmp_epi32_mask(a, b, imm8) simde_mm_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi64_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm, epi64, si128, HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0x3)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = HEDLEY_STATIC_CAST(simde__mmask8, 0x03);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vceqq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcltq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcleq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
              #elif defined(HEDLEY_GCC_VERSION) && !HEDLEY_GCC_VERSION_CHECK(4,8,0)
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #else
                SIMDE_VECTORIZE
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u32 = vmvnq_u32(vreinterpretq_u32_u64(vceqq_s64(a_.neon_i64, b_.neon_i64)));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcgeq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcgtq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            default:
              HEDLEY_UNREACHABLE();
              break;
          }

          r = simde_mm_movepi64_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi64_mask
  #define _mm_cmp_epi64_mask(a, b, imm8) simde_mm_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_ps_mask (simde__m256 a, simde__m256 b, const int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE)
    SIMDE_CONSTIFY_32_(_mm256_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif SIMDE_NATURAL_VECTOR_SIZE_LE(128)
    simde__m256_private
      a_ = simde__m256_to_private(a),
      b_ = simde__m256_to_private(b);
    r = UINT8_C(0);

    for (size_t i = 0 ; i < (sizeof(a_.m128) / sizeof(a_.m128[0])) ; i++) {
      simde__mmask8 tm;
      SIMDE_CONSTIFY_32_(simde_mm_cmp_ps_mask, tm, (HEDLEY_UNREACHABLE(), 0), imm8, a_.m128[i], b_.m128[i]);
      r |= tm << (4 * i);
    }
  #else
    simde__m256 rm;
    SIMDE_CONSTIFY_32_(simde_mm256_cmp_ps, rm, simde_mm256_setzero_ps(), imm8, a, b);
    r = HEDLEY_STATIC_CAST(simde__mmask8, simde_mm256_movemask_ps(rm));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_ps_mask
  #define _mm256_cmp_ps_mask(a, b, imm8) simde_mm256_cmp_ps_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_pd_mask (simde__m256d a, simde__m256d b, const int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE)
    SIMDE_CONSTIFY_32_(_mm256_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif SIMDE_NATURAL_VECTOR_SIZE_LE(128)
    simde__m256d_private
      a_ = simde__m256d_to_private(a),
      b_ = simde__m256d_to_private(b);
    r = UINT8_C(0);

    for (size_t i = 0 ; i < (sizeof(a_.m128d) / sizeof(a_.m128d[0])) ; i++) {
      simde__mmask8 tm;
      SIMDE_CONSTIFY_32_(simde_mm_cmp_pd_mask, tm, (HEDLEY_UNREACHABLE(), 0), imm8, a_.m128d[i], b_.m128d[i]);
      r |= tm << (2 * i);
    }
  #else
    simde__m256d rm;
    SIMDE_CONSTIFY_32_(simde_mm256_cmp_pd, rm, simde_mm256_setzero_pd(), imm8, a, b);
    r = HEDLEY_STATIC_CAST(simde__mmask8, simde_mm256_movemask_pd(rm));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_pd_mask
  #define _mm256_cmp_pd_mask(a, b, imm8) simde_mm256_cmp_pd_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask32
simde_mm256_cmp_epi8_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask32 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm256, epi8, si256, HEDLEY_STATIC_CAST(simde__mmask32, UINT32_C(0xffffffff)), a, b, imm8);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi8_mask(simde_mm256_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi8_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi8(a, b), simde_mm256_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi8_mask(simde_mm256_cmpgt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x00000000);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi8_mask(simde_mm256_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi8_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi8(a, b), simde_mm256_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi8_mask(simde_mm256_cmpgt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x00000000);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm256_movepi8_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi8_mask
  #define _mm256_cmp_epi8_mask(a, b, imm8) simde_mm256_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm256_cmp_epi16_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm256, epi16, si256, HEDLEY_STATIC_CAST(simde__mmask16, UINT16_C(0xffff)), a, b, imm8);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi16_mask(simde_mm256_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi16_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi16(a, b), simde_mm256_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi16_mask(simde_mm256_cmpgt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0000);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi16_mask(simde_mm256_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi16_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi16(a, b), simde_mm256_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi16_mask(simde_mm256_cmpgt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm256_movepi16_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi16_mask
  #define _mm256_cmp_epi16_mask(a, b, imm8) simde_mm256_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_epi32_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm256, epi32, si256, HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0xff)), a, b, imm8);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi32_mask(simde_mm256_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi32_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi32(a, b), simde_mm256_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi32_mask(simde_mm256_cmpgt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi32_mask(simde_mm256_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi32_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi32(a, b), simde_mm256_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi32_mask(simde_mm256_cmpgt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE) << (4 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm256_movepi32_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi32_mask
  #define _mm256_cmp_epi32_mask(a, b, imm8) simde_mm256_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_epi64_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm256, epi64, si256, HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0xf)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0x0f);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE) << (2 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm256_movepi64_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi64_mask
  #define _mm256_cmp_epi64_mask(a, b, imm8) simde_mm256_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm512_cmp_ps_mask (simde__m512 a, simde__m512 b, const int imm8)
    SIMDE_REQUIRE_CONSTANT(imm8)
    HEDLEY_REQUIRE_MSG(((imm8 >= 0) && (imm8 <= 31)), "imm8 must be one of the SIMDE_CMP_* macros (values: [0, 31])") {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_CONSTIFY_32_(_mm512_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), imm8, a, b);
  #else
    simde__m512_private
      r_,
      a_ = simde__m512_to_private(a),
      b_ = simde__m512_to_private(b);

    #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        SIMDE_CONSTIFY_32_(simde_mm_cmp_ps, r_.m128[i], simde_mm_setzero_ps(), imm8, a_.m128[i], b_.m128[i]);
      }
    #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        SIMDE_CONSTIFY_32_(simde_mm256_cmp_ps, r_.m256[i], simde_mm256_setzero_ps(), imm8, a_.m256[i], b_.m256[i]);
      }
    #else
      switch (HEDLEY_STATIC_CAST(int, imm8)) {
        case SIMDE_CMP_EQ_OQ:
        case SIMDE_CMP_EQ_UQ:
        case SIMDE_CMP_EQ_OS:
        case SIMDE_CMP_EQ_US:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 == b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] == b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_LT_OS:
        case SIMDE_CMP_NGE_US:
        case SIMDE_CMP_LT_OQ:
        case SIMDE_CMP_NGE_UQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 < b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_LE_OS:
        case SIMDE_CMP_NGT_US:
        case SIMDE_CMP_LE_OQ:
        case SIMDE_CMP_NGT_UQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 <= b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NEQ_UQ:
        case SIMDE_CMP_NEQ_OQ:
        case SIMDE_CMP_NEQ_US:
        case SIMDE_CMP_NEQ_OS:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 != b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] != b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NLT_US:
        case SIMDE_CMP_GE_OS:
        case SIMDE_CMP_NLT_UQ:
        case SIMDE_CMP_GE_OQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 >= b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NLE_US:
        case SIMDE_CMP_GT_OS:
        case SIMDE_CMP_NLE_UQ:
        case SIMDE_CMP_GT_OQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 > b_.f32);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_FALSE_OQ:
        case SIMDE_CMP_FALSE_OS:
          return HEDLEY_STATIC_CAST(simde__mmask16, UINT16_C(0));
          break;

        case SIMDE_CMP_TRUE_UQ:
        case SIMDE_CMP_TRUE_US:
          return HEDLEY_STATIC_CAST(simde__mmask16, ~UINT16_C(0));
          break;

        case SIMDE_CMP_UNORD_Q:
        case SIMDE_CMP_UNORD_S:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 != a_.f32) | (b_.f32 != b_.f32)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_ORD_Q:
        case SIMDE_CMP_ORD_S:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 == a_.f32) & (b_.f32 == b_.f32)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        default:
          HEDLEY_UNREACHABLE();
          break;
      }

      switch (HEDLEY_STATIC_CAST(int, imm8)) {
        case SIMDE_CMP_EQ_OQ:
        case SIMDE_CMP_LT_OQ:
        case SIMDE_CMP_LE_OQ:
        case SIMDE_CMP_NEQ_OQ:
        case SIMDE_CMP_GE_OQ:
        case SIMDE_CMP_GT_OQ:
        case SIMDE_CMP_EQ_OS:
        case SIMDE_CMP_LT_OS:
        case SIMDE_CMP_LE_OS:
        case SIMDE_CMP_NEQ_OS:
        case SIMDE_CMP_GE_OS:
        case SIMDE_CMP_GT_OS:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 == a_.f32) & (b_.f32 == b_.f32)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] &= ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_EQ_UQ:
        case SIMDE_CMP_NGE_UQ:
        case SIMDE_CMP_NGT_UQ:
        case SIMDE_CMP_NEQ_UQ:
        case SIMDE_CMP_NLT_UQ:
        case SIMDE_CMP_NLE_UQ:
        case SIMDE_CMP_EQ_US:
        case SIMDE_CMP_NGE_US:
        case SIMDE_CMP_NGT_US:
        case SIMDE_CMP_NEQ_US:
        case SIMDE_CMP_NLT_US:
        case SIMDE_CMP_NLE_US:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i32 |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 != a_.f32) | (b_.f32 != b_.f32)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.i32[i] |= ((a_.f32[i] != a_.f32[i]) || (b_.f32[i] != b_.f32[i]));
            }
          #endif
          break;
      }
    #endif

    r = simde_mm512_movepi32_mask(simde_mm512_castps_si512(simde__m512_from_private(r_)));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_ps_mask
  #define _mm512_cmp_ps_mask(a, b, imm8) simde_mm512_cmp_ps_mask(a, b, imm8)
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm512_cmp_pd_mask (simde__m512d a, simde__m512d b, const int imm8)
    SIMDE_REQUIRE_CONSTANT(imm8)
    HEDLEY_REQUIRE_MSG(((imm8 >= 0) && (imm8 <= 31)), "imm8 must be one of the SIMDE_CMP_* macros (values: [0, 31])") {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_CONSTIFY_32_(_mm512_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), imm8, a, b);
  #else
    simde__m512d_private
      r_,
      a_ = simde__m512d_to_private(a),
      b_ = simde__m512d_to_private(b);

    #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(r_.m128d) / sizeof(r_.m128d[0])) ; i++) {
        SIMDE_CONSTIFY_32_(simde_mm_cmp_pd, r_.m128d[i], simde_mm_setzero_pd(), imm8, a_.m128d[i], b_.m128d[i]);
      }
    #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        SIMDE_CONSTIFY_32_(simde_mm256_cmp_pd, r_.m256d[i], simde_mm256_setzero_pd(), imm8, a_.m256d[i], b_.m256d[i]);
      }
    #else
      switch (HEDLEY_STATIC_CAST(int, imm8)) {
        case SIMDE_CMP_EQ_OQ:
        case SIMDE_CMP_EQ_UQ:
        case SIMDE_CMP_EQ_OS:
        case SIMDE_CMP_EQ_US:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 == b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] == b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_LT_OS:
        case SIMDE_CMP_NGE_US:
        case SIMDE_CMP_LT_OQ:
        case SIMDE_CMP_NGE_UQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 < b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] < b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_LE_OS:
        case SIMDE_CMP_NGT_US:
        case SIMDE_CMP_LE_OQ:
        case SIMDE_CMP_NGT_UQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 <= b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] <= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NEQ_UQ:
        case SIMDE_CMP_NEQ_OQ:
        case SIMDE_CMP_NEQ_US:
        case SIMDE_CMP_NEQ_OS:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 != b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] != b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NLT_US:
        case SIMDE_CMP_GE_OS:
        case SIMDE_CMP_NLT_UQ:
        case SIMDE_CMP_GE_OQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 >= b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] >= b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_NLE_US:
        case SIMDE_CMP_GT_OS:
        case SIMDE_CMP_NLE_UQ:
        case SIMDE_CMP_GT_OQ:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.f64 > b_.f64);
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (a_.f64[i] > b_.f64[i]) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_FALSE_OQ:
        case SIMDE_CMP_FALSE_OS:
          return HEDLEY_STATIC_CAST(simde__mmask8, UINT8_C(0));
          break;

        case SIMDE_CMP_TRUE_UQ:
        case SIMDE_CMP_TRUE_US:
          return HEDLEY_STATIC_CAST(simde__mmask8, ~UINT8_C(0));
          break;

        case SIMDE_CMP_UNORD_Q:
        case SIMDE_CMP_UNORD_S:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 != a_.f64) | (b_.f64 != b_.f64)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_ORD_Q:
        case SIMDE_CMP_ORD_S:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 == a_.f64) & (b_.f64 == b_.f64)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        default:
          HEDLEY_UNREACHABLE();
          break;
      }

      switch (HEDLEY_STATIC_CAST(int, imm8)) {
        case SIMDE_CMP_EQ_OQ:
        case SIMDE_CMP_LT_OQ:
        case SIMDE_CMP_LE_OQ:
        case SIMDE_CMP_NEQ_OQ:
        case SIMDE_CMP_GE_OQ:
        case SIMDE_CMP_GT_OQ:
        case SIMDE_CMP_EQ_OS:
        case SIMDE_CMP_LT_OS:
        case SIMDE_CMP_LE_OS:
        case SIMDE_CMP_NEQ_OS:
        case SIMDE_CMP_GE_OS:
        case SIMDE_CMP_GT_OS:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 == a_.f64) & (b_.f64 == b_.f64)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] &= ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i])) ? ~INT64_C(0) : INT64_C(0);
            }
          #endif
          break;

        case SIMDE_CMP_EQ_UQ:
        case SIMDE_CMP_NGE_UQ:
        case SIMDE_CMP_NGT_UQ:
        case SIMDE_CMP_NEQ_UQ:
        case SIMDE_CMP_NLT_UQ:
        case SIMDE_CMP_NLE_UQ:
        case SIMDE_CMP_EQ_US:
        case SIMDE_CMP_NGE_US:
        case SIMDE_CMP_NGT_US:
        case SIMDE_CMP_NEQ_US:
        case SIMDE_CMP_NLT_US:
        case SIMDE_CMP_NLE_US:
          #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
            r_.i64 |= HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 != a_.f64) | (b_.f64 != b_.f64)));
          #else
            SIMDE_VECTORIZE
            for (size_t i = 0; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.i64[i] |= ((a_.f64[i] != a_.f64[i]) || (b_.f64[i] != b_.f64[i]));
            }
          #endif
          break;
      }
    #endif

    r = simde_mm512_movepi64_mask(simde_mm512_castpd_si512(simde__m512d_from_private(r_)));
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_pd_mask
  #define _mm512_cmp_pd_mask(a, b, imm8) simde_mm512_cmp_pd_mask(a, b, imm8)
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask64
simde_mm512_cmp_epi8_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask64 r;

  #if defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm512, epi8, si512, HEDLEY_STATIC_CAST(simde__mmask64, UINT64_C(0xffffffffffffffff)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT64_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT64_C(0xffffffffffffffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (16 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm512_movepi8_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi8_mask
  #define _mm512_cmp_epi8_mask(a, b, imm8) simde_mm512_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask32
simde_mm512_cmp_epi16_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask32 r;

  #if defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm512, epi16, si512, HEDLEY_STATIC_CAST(simde__mmask32, UINT32_C(0xffffffff)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (8 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm512_movepi16_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi16_mask
  #define _mm512_cmp_epi16_mask(a, b, imm8) simde_mm512_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm512_cmp_epi32_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm512, epi32, si512, HEDLEY_STATIC_CAST(simde__mmask16, UINT16_C(0xffff)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (4 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              default:
                HEDLEY_UNREACHABLE();
                break;
            }

            r = simde_mm512_movepi32_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi32_mask
  #define _mm512_cmp_epi32_mask(a, b, imm8) simde_mm512_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm512_cmp_epi64_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_MM_CMPINT_CONSTIFY_(r, mm512, epi64, si512, HEDLEY_STATIC_CAST(simde__mmask16, UINT16_C(0xff)), a, b, imm8);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE) << (2 * i);
                }
                break;
            default:
              HEDLEY_UNREACHABLE();
              break;
            }
          #elif SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE) << (4 * i);
                }
                break;
            default:
              HEDLEY_UNREACHABLE();
              break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
            default:
              HEDLEY_UNREACHABLE();
              break;
            }

            r = simde_mm512_movepi64_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi64_mask
  #define _mm512_cmp_epi64_mask(a, b, imm8) simde_mm512_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_CMP_H) */
