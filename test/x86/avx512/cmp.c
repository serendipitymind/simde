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

#define SIMDE_TEST_X86_AVX512_INSN cmp

#include <test/x86/avx512/test-avx512.h>
#include <simde/x86/avx512/blend.h>
#include <simde/x86/avx512/set.h>
#include <simde/x86/avx512/set1.h>
#include <simde/x86/avx512/cmp.h>

#if !defined(SIMDE_NATIVE_ALIASES_TESTING)

static int
test_simde_mm_cmp_ps_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float32 a[4];
    const simde_float32 b[4];
    const simde__mmask8 r;
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(   -89.72), SIMDE_FLOAT32_C(  -431.81), SIMDE_FLOAT32_C(  -199.17), SIMDE_FLOAT32_C(   603.72) },
      { SIMDE_FLOAT32_C(  -475.47), SIMDE_FLOAT32_C(   -95.61), SIMDE_FLOAT32_C(   415.45), SIMDE_FLOAT32_C(   603.72) },
      UINT8_C(  8) },
    { { SIMDE_FLOAT32_C(  -946.59), SIMDE_FLOAT32_C(  -533.28), SIMDE_FLOAT32_C(  -601.80), SIMDE_FLOAT32_C(   415.02) },
      { SIMDE_FLOAT32_C(  -644.79), SIMDE_FLOAT32_C(  -775.55), SIMDE_FLOAT32_C(  -601.80), SIMDE_FLOAT32_C(   265.50) },
      UINT8_C(  1) },
    { { SIMDE_FLOAT32_C(  -719.78), SIMDE_FLOAT32_C(   455.23), SIMDE_FLOAT32_C(   160.95), SIMDE_FLOAT32_C(  -670.00) },
      { SIMDE_FLOAT32_C(  -719.78), SIMDE_FLOAT32_C(   455.23), SIMDE_FLOAT32_C(  -738.06), SIMDE_FLOAT32_C(  -666.38) },
      UINT8_C( 11) },
    { { SIMDE_FLOAT32_C(   846.15), SIMDE_FLOAT32_C(   -95.42), SIMDE_FLOAT32_C(   -26.68), SIMDE_FLOAT32_C(  -342.81) },
      { SIMDE_FLOAT32_C(   846.15), SIMDE_FLOAT32_C(   -95.42), SIMDE_FLOAT32_C(   -26.68), SIMDE_FLOAT32_C(  -342.81) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   890.08), SIMDE_FLOAT32_C(    94.37), SIMDE_FLOAT32_C(  -894.07), SIMDE_FLOAT32_C(  -648.44) },
      { SIMDE_FLOAT32_C(   -83.12), SIMDE_FLOAT32_C(    64.19), SIMDE_FLOAT32_C(  -894.07), SIMDE_FLOAT32_C(  -648.44) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT32_C(   544.44), SIMDE_FLOAT32_C(    52.77), SIMDE_FLOAT32_C(  -139.98), SIMDE_FLOAT32_C(  -675.71) },
      { SIMDE_FLOAT32_C(  -565.79), SIMDE_FLOAT32_C(    52.77), SIMDE_FLOAT32_C(  -486.60), SIMDE_FLOAT32_C(  -675.71) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(  -211.15), SIMDE_FLOAT32_C(   931.06), SIMDE_FLOAT32_C(   493.39), SIMDE_FLOAT32_C(   241.68) },
      { SIMDE_FLOAT32_C(  -211.15), SIMDE_FLOAT32_C(   931.06), SIMDE_FLOAT32_C(   493.39), SIMDE_FLOAT32_C(  -666.72) },
      UINT8_C(  8) },
    { { SIMDE_FLOAT32_C(  -571.19), SIMDE_FLOAT32_C(  -191.58), SIMDE_FLOAT32_C(  -642.61), SIMDE_FLOAT32_C(   154.77) },
      { SIMDE_FLOAT32_C(  -571.19), SIMDE_FLOAT32_C(   302.82), SIMDE_FLOAT32_C(   968.63), SIMDE_FLOAT32_C(   235.28) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(   808.41), SIMDE_FLOAT32_C(   900.63), SIMDE_FLOAT32_C(  -944.16), SIMDE_FLOAT32_C(  -883.57) },
      { SIMDE_FLOAT32_C(   933.56), SIMDE_FLOAT32_C(   442.75), SIMDE_FLOAT32_C(   909.63), SIMDE_FLOAT32_C(  -512.15) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(  -906.82), SIMDE_FLOAT32_C(   282.07), SIMDE_FLOAT32_C(  -583.71), SIMDE_FLOAT32_C(   901.60) },
      { SIMDE_FLOAT32_C(  -906.82), SIMDE_FLOAT32_C(   699.57), SIMDE_FLOAT32_C(   -85.30), SIMDE_FLOAT32_C(   901.60) },
      UINT8_C(  6) },
    { { SIMDE_FLOAT32_C(  -554.57), SIMDE_FLOAT32_C(  -551.43), SIMDE_FLOAT32_C(  -814.87), SIMDE_FLOAT32_C(   585.36) },
      { SIMDE_FLOAT32_C(  -554.57), SIMDE_FLOAT32_C(  -551.43), SIMDE_FLOAT32_C(  -299.17), SIMDE_FLOAT32_C(   585.36) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(   518.40), SIMDE_FLOAT32_C(   176.54), SIMDE_FLOAT32_C(  -615.93), SIMDE_FLOAT32_C(   -88.03) },
      { SIMDE_FLOAT32_C(   518.40), SIMDE_FLOAT32_C(   176.54), SIMDE_FLOAT32_C(  -615.93), SIMDE_FLOAT32_C(   170.32) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(  -958.89), SIMDE_FLOAT32_C(  -813.04), SIMDE_FLOAT32_C(   577.98), SIMDE_FLOAT32_C(   795.69) },
      { SIMDE_FLOAT32_C(  -840.79), SIMDE_FLOAT32_C(   354.04), SIMDE_FLOAT32_C(   577.98), SIMDE_FLOAT32_C(   795.69) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT32_C(  -135.50), SIMDE_FLOAT32_C(  -569.57), SIMDE_FLOAT32_C(  -805.90), SIMDE_FLOAT32_C(   217.93) },
      { SIMDE_FLOAT32_C(  -702.99), SIMDE_FLOAT32_C(  -569.57), SIMDE_FLOAT32_C(   407.89), SIMDE_FLOAT32_C(  -661.87) },
      UINT8_C( 11) },
    { { SIMDE_FLOAT32_C(   733.35), SIMDE_FLOAT32_C(   -47.60), SIMDE_FLOAT32_C(   479.92), SIMDE_FLOAT32_C(   105.26) },
      { SIMDE_FLOAT32_C(  -670.54), SIMDE_FLOAT32_C(  -861.59), SIMDE_FLOAT32_C(   479.92), SIMDE_FLOAT32_C(   193.95) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT32_C(   876.28), SIMDE_FLOAT32_C(   629.20), SIMDE_FLOAT32_C(  -490.52), SIMDE_FLOAT32_C(   609.63) },
      { SIMDE_FLOAT32_C(   876.28), SIMDE_FLOAT32_C(   620.97), SIMDE_FLOAT32_C(   754.93), SIMDE_FLOAT32_C(   609.63) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(  -735.73), SIMDE_FLOAT32_C(  -806.45), SIMDE_FLOAT32_C(  -487.00), SIMDE_FLOAT32_C(   224.87) },
      { SIMDE_FLOAT32_C(  -735.73), SIMDE_FLOAT32_C(  -451.41), SIMDE_FLOAT32_C(  -487.00), SIMDE_FLOAT32_C(   614.93) },
      UINT8_C(  5) },
    { { SIMDE_FLOAT32_C(  -967.35), SIMDE_FLOAT32_C(   110.62), SIMDE_FLOAT32_C(    49.38), SIMDE_FLOAT32_C(    53.52) },
      { SIMDE_FLOAT32_C(  -967.35), SIMDE_FLOAT32_C(   110.62), SIMDE_FLOAT32_C(    49.38), SIMDE_FLOAT32_C(  -194.04) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(  -262.90), SIMDE_FLOAT32_C(   546.20), SIMDE_FLOAT32_C(   752.72), SIMDE_FLOAT32_C(  -512.29) },
      { SIMDE_FLOAT32_C(   762.62), SIMDE_FLOAT32_C(  -760.33), SIMDE_FLOAT32_C(   752.72), SIMDE_FLOAT32_C(  -512.29) },
      UINT8_C( 13) },
    { { SIMDE_FLOAT32_C(   855.46), SIMDE_FLOAT32_C(  -696.49), SIMDE_FLOAT32_C(  -360.52), SIMDE_FLOAT32_C(   468.27) },
      { SIMDE_FLOAT32_C(  -337.97), SIMDE_FLOAT32_C(   -61.91), SIMDE_FLOAT32_C(  -360.52), SIMDE_FLOAT32_C(   399.13) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   243.09), SIMDE_FLOAT32_C(  -356.08), SIMDE_FLOAT32_C(  -719.61), SIMDE_FLOAT32_C(   -21.41) },
      { SIMDE_FLOAT32_C(   978.76), SIMDE_FLOAT32_C(   948.08), SIMDE_FLOAT32_C(  -719.61), SIMDE_FLOAT32_C(   834.22) },
      UINT8_C( 11) },
    { { SIMDE_FLOAT32_C(  -528.32), SIMDE_FLOAT32_C(   555.29), SIMDE_FLOAT32_C(  -975.26), SIMDE_FLOAT32_C(   338.14) },
      { SIMDE_FLOAT32_C(   606.69), SIMDE_FLOAT32_C(  -482.81), SIMDE_FLOAT32_C(  -290.42), SIMDE_FLOAT32_C(  -150.22) },
      UINT8_C( 10) },
    { { SIMDE_FLOAT32_C(   497.43), SIMDE_FLOAT32_C(   -71.92), SIMDE_FLOAT32_C(    97.48), SIMDE_FLOAT32_C(   973.73) },
      { SIMDE_FLOAT32_C(   502.05), SIMDE_FLOAT32_C(   -71.92), SIMDE_FLOAT32_C(    97.48), SIMDE_FLOAT32_C(   973.73) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   953.46), SIMDE_FLOAT32_C(   478.42), SIMDE_FLOAT32_C(   543.86), SIMDE_FLOAT32_C(   409.61) },
      { SIMDE_FLOAT32_C(   953.46), SIMDE_FLOAT32_C(   253.40), SIMDE_FLOAT32_C(   543.86), SIMDE_FLOAT32_C(   450.89) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(   947.61), SIMDE_FLOAT32_C(   822.74), SIMDE_FLOAT32_C(  -493.39), SIMDE_FLOAT32_C(   138.68) },
      { SIMDE_FLOAT32_C(   947.61), SIMDE_FLOAT32_C(   822.74), SIMDE_FLOAT32_C(  -328.51), SIMDE_FLOAT32_C(   138.68) },
      UINT8_C( 11) },
    { { SIMDE_FLOAT32_C(   410.63), SIMDE_FLOAT32_C(   162.05), SIMDE_FLOAT32_C(   614.07), SIMDE_FLOAT32_C(  -572.56) },
      { SIMDE_FLOAT32_C(   410.63), SIMDE_FLOAT32_C(  -293.20), SIMDE_FLOAT32_C(   145.06), SIMDE_FLOAT32_C(  -572.56) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   874.24), SIMDE_FLOAT32_C(  -746.64), SIMDE_FLOAT32_C(   865.24), SIMDE_FLOAT32_C(   580.53) },
      { SIMDE_FLOAT32_C(  -756.69), SIMDE_FLOAT32_C(   853.99), SIMDE_FLOAT32_C(   865.24), SIMDE_FLOAT32_C(   580.53) },
      UINT8_C( 14) },
    { { SIMDE_FLOAT32_C(  -901.01), SIMDE_FLOAT32_C(   945.37), SIMDE_FLOAT32_C(   646.58), SIMDE_FLOAT32_C(   973.24) },
      { SIMDE_FLOAT32_C(  -901.01), SIMDE_FLOAT32_C(   945.37), SIMDE_FLOAT32_C(  -353.05), SIMDE_FLOAT32_C(   973.24) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   973.76), SIMDE_FLOAT32_C(  -285.53), SIMDE_FLOAT32_C(   381.32), SIMDE_FLOAT32_C(  -858.17) },
      { SIMDE_FLOAT32_C(  -973.37), SIMDE_FLOAT32_C(  -285.53), SIMDE_FLOAT32_C(   382.52), SIMDE_FLOAT32_C(   840.91) },
      UINT8_C( 13) },
    { { SIMDE_FLOAT32_C(   946.81), SIMDE_FLOAT32_C(   -69.84), SIMDE_FLOAT32_C(  -488.92), SIMDE_FLOAT32_C(   152.14) },
      { SIMDE_FLOAT32_C(   178.39), SIMDE_FLOAT32_C(   -69.84), SIMDE_FLOAT32_C(  -488.92), SIMDE_FLOAT32_C(   152.14) },
      UINT8_C( 15) },
    { { SIMDE_FLOAT32_C(   106.22), SIMDE_FLOAT32_C(    45.26), SIMDE_FLOAT32_C(   636.09), SIMDE_FLOAT32_C(  -273.51) },
      { SIMDE_FLOAT32_C(   106.22), SIMDE_FLOAT32_C(    45.26), SIMDE_FLOAT32_C(   636.09), SIMDE_FLOAT32_C(    53.02) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(  -525.29), SIMDE_FLOAT32_C(   296.81), SIMDE_FLOAT32_C(  -888.15), SIMDE_FLOAT32_C(   749.88) },
      { SIMDE_FLOAT32_C(  -691.76), SIMDE_FLOAT32_C(   965.07), SIMDE_FLOAT32_C(   441.46), SIMDE_FLOAT32_C(   986.28) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128 a = simde_mm_loadu_ps(test_vec[i].a);
    simde__m128 b = simde_mm_loadu_ps(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_ps_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 32 ; i++) {
    simde_float32 a_[4];
    simde_float32 b_[4];

    simde_test_codegen_random_vf32((sizeof(a_) / sizeof(a_[0])), a_, SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    simde_test_codegen_random_vf32((sizeof(b_) / sizeof(b_[0])), b_, SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 1))
        a_[j] = b_[j];

    simde__m128 a = simde_mm_loadu_ps(a_);
    simde__m128 b = simde_mm_loadu_ps(b_);
    simde__mmask8 r = simde_mm_cmp_ps_mask(a, b, HEDLEY_STATIC_CAST(int, i));

    simde_test_x86_write_f32x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f32x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_pd_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float64 a[2];
    const simde_float64 b[2];
    const simde__mmask8 r;
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(   822.54), SIMDE_FLOAT64_C(   541.72) },
      { SIMDE_FLOAT64_C(   822.54), SIMDE_FLOAT64_C(   541.72) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   671.70), SIMDE_FLOAT64_C(   627.40) },
      { SIMDE_FLOAT64_C(  -245.87), SIMDE_FLOAT64_C(   627.40) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(   561.27), SIMDE_FLOAT64_C(   627.88) },
      { SIMDE_FLOAT64_C(  -298.22), SIMDE_FLOAT64_C(   782.49) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(    45.92), SIMDE_FLOAT64_C(   299.08) },
      { SIMDE_FLOAT64_C(  -337.63), SIMDE_FLOAT64_C(  -883.67) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(    63.80), SIMDE_FLOAT64_C(   817.58) },
      { SIMDE_FLOAT64_C(    63.80), SIMDE_FLOAT64_C(   817.58) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -526.59), SIMDE_FLOAT64_C(   798.72) },
      { SIMDE_FLOAT64_C(  -526.59), SIMDE_FLOAT64_C(   -74.55) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   475.11), SIMDE_FLOAT64_C(   879.26) },
      { SIMDE_FLOAT64_C(   749.42), SIMDE_FLOAT64_C(   185.63) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(  -140.06), SIMDE_FLOAT64_C(   615.14) },
      { SIMDE_FLOAT64_C(  -140.06), SIMDE_FLOAT64_C(   615.14) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   168.68), SIMDE_FLOAT64_C(  -903.71) },
      { SIMDE_FLOAT64_C(   168.68), SIMDE_FLOAT64_C(  -554.27) },
      UINT8_C(  1) },
    { { SIMDE_FLOAT64_C(   201.71), SIMDE_FLOAT64_C(   736.83) },
      { SIMDE_FLOAT64_C(   201.71), SIMDE_FLOAT64_C(  -444.18) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(   -81.86), SIMDE_FLOAT64_C(  -731.77) },
      { SIMDE_FLOAT64_C(  -115.89), SIMDE_FLOAT64_C(   -47.32) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(  -424.96), SIMDE_FLOAT64_C(  -282.52) },
      { SIMDE_FLOAT64_C(   858.00), SIMDE_FLOAT64_C(  -170.66) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -747.95), SIMDE_FLOAT64_C(   585.38) },
      { SIMDE_FLOAT64_C(   251.28), SIMDE_FLOAT64_C(  -887.99) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   382.23), SIMDE_FLOAT64_C(   169.84) },
      { SIMDE_FLOAT64_C(   382.23), SIMDE_FLOAT64_C(   -22.40) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -508.71), SIMDE_FLOAT64_C(   164.15) },
      { SIMDE_FLOAT64_C(  -508.71), SIMDE_FLOAT64_C(  -459.62) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(   293.20), SIMDE_FLOAT64_C(  -265.96) },
      { SIMDE_FLOAT64_C(   293.20), SIMDE_FLOAT64_C(  -837.02) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -244.59), SIMDE_FLOAT64_C(  -287.33) },
      { SIMDE_FLOAT64_C(   916.29), SIMDE_FLOAT64_C(  -846.31) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -816.63), SIMDE_FLOAT64_C(  -976.95) },
      { SIMDE_FLOAT64_C(   618.31), SIMDE_FLOAT64_C(  -976.95) },
      UINT8_C(  1) },
    { { SIMDE_FLOAT64_C(   626.66), SIMDE_FLOAT64_C(   456.89) },
      { SIMDE_FLOAT64_C(   626.66), SIMDE_FLOAT64_C(   456.89) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -256.46), SIMDE_FLOAT64_C(   889.76) },
      { SIMDE_FLOAT64_C(  -256.46), SIMDE_FLOAT64_C(   889.76) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -143.31), SIMDE_FLOAT64_C(   710.00) },
      { SIMDE_FLOAT64_C(   163.83), SIMDE_FLOAT64_C(  -671.11) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   466.41), SIMDE_FLOAT64_C(  -125.85) },
      { SIMDE_FLOAT64_C(   466.41), SIMDE_FLOAT64_C(  -574.48) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(   846.43), SIMDE_FLOAT64_C(  -356.19) },
      { SIMDE_FLOAT64_C(   846.43), SIMDE_FLOAT64_C(   935.22) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(   589.68), SIMDE_FLOAT64_C(   -18.65) },
      { SIMDE_FLOAT64_C(  -741.40), SIMDE_FLOAT64_C(  -893.69) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -340.78), SIMDE_FLOAT64_C(   575.99) },
      { SIMDE_FLOAT64_C(    51.15), SIMDE_FLOAT64_C(   575.99) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(  -780.09), SIMDE_FLOAT64_C(   156.88) },
      { SIMDE_FLOAT64_C(  -780.09), SIMDE_FLOAT64_C(  -725.65) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -492.97), SIMDE_FLOAT64_C(    85.64) },
      { SIMDE_FLOAT64_C(  -381.49), SIMDE_FLOAT64_C(    85.64) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -443.24), SIMDE_FLOAT64_C(   956.25) },
      { SIMDE_FLOAT64_C(  -443.24), SIMDE_FLOAT64_C(   967.40) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -966.57), SIMDE_FLOAT64_C(  -894.80) },
      { SIMDE_FLOAT64_C(  -966.57), SIMDE_FLOAT64_C(  -894.80) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT64_C(  -979.95), SIMDE_FLOAT64_C(   903.47) },
      { SIMDE_FLOAT64_C(  -979.95), SIMDE_FLOAT64_C(   559.82) },
      UINT8_C(  3) },
    { { SIMDE_FLOAT64_C(  -457.84), SIMDE_FLOAT64_C(   502.12) },
      { SIMDE_FLOAT64_C(  -457.84), SIMDE_FLOAT64_C(   104.05) },
      UINT8_C(  2) },
    { { SIMDE_FLOAT64_C(  -388.65), SIMDE_FLOAT64_C(   -38.82) },
      { SIMDE_FLOAT64_C(  -388.65), SIMDE_FLOAT64_C(   -38.82) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128d a = simde_mm_loadu_pd(test_vec[i].a);
    simde__m128d b = simde_mm_loadu_pd(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_pd_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 32 ; i++) {
    simde_float64 a_[2];
    simde_float64 b_[2];

    simde_test_codegen_random_vf64((sizeof(a_) / sizeof(a_[0])), a_, SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    simde_test_codegen_random_vf64((sizeof(b_) / sizeof(b_[0])), b_, SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 1))
        a_[j] = b_[j];

    simde__m128d a = simde_mm_loadu_pd(a_);
    simde__m128d b = simde_mm_loadu_pd(b_);
    simde__mmask8 r = simde_mm_cmp_pd_mask(a, b, HEDLEY_STATIC_CAST(int, i));

    simde_test_x86_write_f64x2(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f64x2(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[16];
    const int8_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { { -INT8_C(  24), -INT8_C(  41), -INT8_C(  37), -INT8_C(  97),  INT8_C( 107),  INT8_C(  15),  INT8_C( 121),  INT8_C( 120),
        -INT8_C(   1), -INT8_C(  79), -INT8_C(  17),  INT8_C(  41),  INT8_C(   3), -INT8_C(  45),  INT8_C(  37),  INT8_C( 110) },
      { -INT8_C(  24),  INT8_C(  26), -INT8_C(  20),  INT8_C(  87),  INT8_C(   7), -INT8_C(  15),  INT8_C( 121), -INT8_C(  17),
        -INT8_C(   1),  INT8_C(  40), -INT8_C(  17),  INT8_C( 124),  INT8_C(   4),  INT8_C(  93), -INT8_C(  25),  INT8_C(  95) },
      UINT16_C( 1345) },
    { {  INT8_C(  56), -INT8_C( 107), -INT8_C( 115),      INT8_MAX, -INT8_C( 121),  INT8_C( 103),  INT8_C(  47), -INT8_C( 122),
         INT8_C(  46),  INT8_C(  30), -INT8_C( 118),  INT8_C(  50),  INT8_C( 123), -INT8_C(  22), -INT8_C( 111), -INT8_C(  99) },
      { -INT8_C(  83), -INT8_C( 113),  INT8_C(  79),      INT8_MAX,  INT8_C(  22),  INT8_C( 103),  INT8_C(  79),  INT8_C(  79),
        -INT8_C(  31),  INT8_C(  71), -INT8_C( 118),  INT8_C(  46),  INT8_C( 101),  INT8_C(  52),  INT8_C( 100), -INT8_C(  99) },
      UINT16_C(25300) },
    { {  INT8_C( 121), -INT8_C(  92), -INT8_C(  73),  INT8_C( 104),  INT8_C(  11),  INT8_C(  63), -INT8_C(  34), -INT8_C(  20),
        -INT8_C( 122), -INT8_C(  26),  INT8_C(  27),  INT8_C(  43), -INT8_C(  99),      INT8_MAX, -INT8_C( 119),  INT8_C(  72) },
      {  INT8_C( 113),  INT8_C( 102), -INT8_C(  73),  INT8_C( 104),  INT8_C( 114), -INT8_C( 114), -INT8_C( 114), -INT8_C(  99),
         INT8_C( 103), -INT8_C(  26),  INT8_C(  67),  INT8_C(  43), -INT8_C(  49), -INT8_C( 104), -INT8_C( 101),  INT8_C(  72) },
      UINT16_C(57118) },
    { {  INT8_C(  33), -INT8_C(  38), -INT8_C(  87), -INT8_C(  20),  INT8_C( 104),  INT8_C(  55),  INT8_C(  61), -INT8_C(  49),
         INT8_C(  29),      INT8_MAX, -INT8_C(  97), -INT8_C(  20),  INT8_C(  64), -INT8_C( 106),  INT8_C(  53),  INT8_C(  85) },
      {  INT8_C(  33),  INT8_C(  13), -INT8_C(  99), -INT8_C(  20), -INT8_C(  61), -INT8_C(  46),  INT8_C(  61), -INT8_C(  29),
         INT8_C(  34),  INT8_C( 122), -INT8_C(  97), -INT8_C(  14),  INT8_C(  64), -INT8_C(  62),  INT8_C(  50),  INT8_C( 109) },
      UINT16_C(    0) },
    { {  INT8_C(  94), -INT8_C(  45),  INT8_C( 114),  INT8_C(  33), -INT8_C(  90), -INT8_C(  81),  INT8_C(   4), -INT8_C(  56),
         INT8_C(  41), -INT8_C(  92), -INT8_C(  41),  INT8_C( 105),  INT8_C( 102), -INT8_C(  19), -INT8_C(  41),  INT8_C(   3) },
      { -INT8_C(  56), -INT8_C(  28),  INT8_C(   7), -INT8_C(  37),  INT8_C(  45), -INT8_C(  37),  INT8_C(  10), -INT8_C(  10),
        -INT8_C(  86),  INT8_C(  38), -INT8_C(  41),  INT8_C(  14),  INT8_C( 119),  INT8_C(  13), -INT8_C( 108), -INT8_C(  43) },
      UINT16_C(64511) },
    { {  INT8_C(   3),  INT8_C(  16), -INT8_C( 102),  INT8_C(  48), -INT8_C(  20), -INT8_C(  91), -INT8_C(  45), -INT8_C( 106),
        -INT8_C(  53),  INT8_C(  27), -INT8_C(  92),  INT8_C(  67),  INT8_C(  12),  INT8_C(  57), -INT8_C( 124), -INT8_C(  19) },
      {  INT8_C(  63),  INT8_C(  15),  INT8_C( 116), -INT8_C(  11),  INT8_C(  11), -INT8_C(  61), -INT8_C(  45), -INT8_C(  86),
        -INT8_C(  51),  INT8_C(  27), -INT8_C(  80), -INT8_C(  60),  INT8_C(  58), -INT8_C(  71), -INT8_C( 124),  INT8_C(  61) },
      UINT16_C(27210) },
    { { -INT8_C(   8),  INT8_C(   0),  INT8_C(  94), -INT8_C(  68), -INT8_C(  56),  INT8_C(  49), -INT8_C(  81), -INT8_C( 111),
         INT8_C(  77),  INT8_C(  96), -INT8_C(   5), -INT8_C( 121),  INT8_C(  25), -INT8_C(  38), -INT8_C(  59), -INT8_C(  29) },
      { -INT8_C(   8),  INT8_C(  51), -INT8_C( 103), -INT8_C(  68), -INT8_C(  56), -INT8_C(  27),  INT8_C(  75),  INT8_C(  91),
        -INT8_C(  42),  INT8_C(  29), -INT8_C(   5), -INT8_C(   1),  INT8_C(   7), -INT8_C( 121),  INT8_C( 104),  INT8_C(   1) },
      UINT16_C(13092) },
    { { -INT8_C( 106), -INT8_C(  84), -INT8_C(  62), -INT8_C( 116), -INT8_C( 110),  INT8_C(  13), -INT8_C(  24),  INT8_C(   5),
         INT8_C(  42), -INT8_C(  29),  INT8_C( 103),  INT8_C(  93),  INT8_C( 106),  INT8_C(  72),  INT8_C(  51), -INT8_C(  10) },
      { -INT8_C( 106),  INT8_C(  57),  INT8_C(  62), -INT8_C( 114), -INT8_C(  17),  INT8_C(  28), -INT8_C(  45),  INT8_C(   5),
         INT8_C(  79), -INT8_C(  96),  INT8_C(  53),  INT8_C(  93),  INT8_C(  49),  INT8_C(  72),  INT8_C(  99), -INT8_C(  10) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi8(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi8(test_vec[i].b);
    simde__mmask16 r = simde_mm_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[16];
    int8_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi8(a_);
    simde__m128i b = simde_mm_loadu_epi8(b_);
    simde__mmask16 r = simde_mm_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[8];
    const int16_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT16_C(  4039),  INT16_C( 12767),  INT16_C( 15592),  INT16_C( 15098),  INT16_C(  6901),  INT16_C(  3425),  INT16_C( 11846),  INT16_C(  3579) },
      { -INT16_C(  4039),  INT16_C( 31584),  INT16_C(  3681),  INT16_C( 15098),  INT16_C(  6901),  INT16_C(  6239),  INT16_C( 11846), -INT16_C( 10122) },
      UINT8_C( 89) },
    { { -INT16_C(  9544),  INT16_C(  6413),  INT16_C( 29956), -INT16_C(  2798), -INT16_C(  8221),  INT16_C( 15202),  INT16_C( 27617),  INT16_C( 12607) },
      {  INT16_C( 18624),  INT16_C( 21097),  INT16_C( 29956), -INT16_C(  2798), -INT16_C(  8221),  INT16_C( 15202), -INT16_C(  9049),  INT16_C( 24780) },
      UINT8_C(131) },
    { { -INT16_C(  1040), -INT16_C(  2832),  INT16_C(   625), -INT16_C( 20457),  INT16_C( 32619), -INT16_C( 30577),  INT16_C( 23335), -INT16_C(  8472) },
      {  INT16_C( 25141),  INT16_C(  5501), -INT16_C(  1745), -INT16_C( 20457),  INT16_C( 32619),  INT16_C( 18705), -INT16_C( 23513),  INT16_C(  6119) },
      UINT8_C(187) },
    { {  INT16_C( 31023), -INT16_C( 17255), -INT16_C( 20292), -INT16_C( 20542),  INT16_C( 32304),  INT16_C( 22385),  INT16_C( 22562), -INT16_C( 16018) },
      {  INT16_C( 31023),  INT16_C(  2258),  INT16_C( 13934), -INT16_C( 20542),  INT16_C(  1066),  INT16_C( 30999),  INT16_C( 23855), -INT16_C( 17155) },
      UINT8_C(  0) },
    { {  INT16_C(  6581),  INT16_C(  9220), -INT16_C( 14769),  INT16_C( 31187),  INT16_C(  4797), -INT16_C( 29698), -INT16_C(  4281),  INT16_C( 26551) },
      {  INT16_C( 12422), -INT16_C( 12989),  INT16_C( 18453),  INT16_C( 31044),  INT16_C(  4797), -INT16_C( 29698),  INT16_C( 17742),  INT16_C(   903) },
      UINT8_C(207) },
    { { -INT16_C( 10497), -INT16_C( 13090), -INT16_C( 24546),  INT16_C(  2794), -INT16_C( 29518),  INT16_C( 10550), -INT16_C( 14127),  INT16_C( 12292) },
      {  INT16_C( 11130), -INT16_C( 13090),  INT16_C(  1318),  INT16_C(  2794),  INT16_C(   543),  INT16_C( 10550), -INT16_C( 14127), -INT16_C( 12104) },
      UINT8_C(234) },
    { { -INT16_C(  8808), -INT16_C( 16845),  INT16_C(  7651),  INT16_C(   712),  INT16_C( 32696), -INT16_C(  4053), -INT16_C(  6969),  INT16_C( 26048) },
      { -INT16_C( 23303), -INT16_C( 20958), -INT16_C( 17898),  INT16_C(  5142),  INT16_C( 32696), -INT16_C( 23068), -INT16_C(  7189), -INT16_C( 31989) },
      UINT8_C(231) },
    { {  INT16_C( 22479),  INT16_C(  5158),  INT16_C( 29713), -INT16_C( 18802), -INT16_C(  1633), -INT16_C(  8594), -INT16_C( 17885), -INT16_C(  3580) },
      { -INT16_C( 23624),  INT16_C(  5158), -INT16_C( 12883), -INT16_C( 18802), -INT16_C(  1633),  INT16_C( 21893), -INT16_C( 17885), -INT16_C(  3580) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi16(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi16(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[8];
    int16_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi16(a_);
    simde__m128i b = simde_mm_loadu_epi16(b_);
    simde__mmask8 r = simde_mm_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(   145519450),  INT32_C(   296776414), -INT32_C(  1519279458), -INT32_C(   952076455) },
      { -INT32_C(   145519450),  INT32_C(  1257134046), -INT32_C(   998466168), -INT32_C(  1248046432) },
      UINT8_C(  1) },
    { { -INT32_C(   111893733), -INT32_C(   835616865), -INT32_C(  2142781472),  INT32_C(   718579215) },
      { -INT32_C(  2018504068), -INT32_C(   835616865),  INT32_C(  2002157759),  INT32_C(   718579215) },
      UINT8_C(  4) },
    { { -INT32_C(  1292576493),  INT32_C(   411051352), -INT32_C(  1265641819), -INT32_C(   975281284) },
      { -INT32_C(   133627508), -INT32_C(  1924870297), -INT32_C(   859565860),  INT32_C(   607776273) },
      UINT8_C( 13) },
    { { -INT32_C(   173049715), -INT32_C(  1801235353),  INT32_C(  1161711899), -INT32_C(    63297228) },
      { -INT32_C(  1441001129), -INT32_C(  1801235353),  INT32_C(  1161711899),  INT32_C(  1794944477) },
      UINT8_C(  0) },
    { {  INT32_C(  1839821830), -INT32_C(   486388536),  INT32_C(  2049523869),  INT32_C(   317007341) },
      {  INT32_C(  1904100561),  INT32_C(  1310068006),  INT32_C(   866832523), -INT32_C(   885980219) },
      UINT8_C( 15) },
    { { -INT32_C(  1900677016), -INT32_C(  1864578299),  INT32_C(   595851953),  INT32_C(    19723658) },
      { -INT32_C(   163486257),  INT32_C(   465266080),  INT32_C(   595851953), -INT32_C(  2040005346) },
      UINT8_C( 12) },
    { {  INT32_C(   369706357), -INT32_C(   496226967), -INT32_C(   595610178), -INT32_C(  1469847886) },
      { -INT32_C(   375228414), -INT32_C(   496226967), -INT32_C(  1075556484), -INT32_C(   707598241) },
      UINT8_C(  5) },
    { { -INT32_C(  2138649066),  INT32_C(  2019750652), -INT32_C(  1590213054),  INT32_C(  1568016943) },
      { -INT32_C(  2043321882),  INT32_C(  1340435070), -INT32_C(  1194122978), -INT32_C(   699007041) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi32(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi32(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[4];
    int32_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi32(a_);
    simde__m128i b = simde_mm_loadu_epi32(b_);
    simde__mmask8 r = simde_mm_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const simde__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 1933546251694643553), -INT64_C( 6674426040188389969) },
      {  INT64_C( 3316585011335538905), -INT64_C( 1397919465970533867) },
      UINT8_C(  0) },
    { {  INT64_C( 5277093564638603313), -INT64_C( 1315097322887719273) },
      { -INT64_C( 6872449015084218189), -INT64_C( 4405021074379975170) },
      UINT8_C(  0) },
    { {  INT64_C( 6616989965536034123), -INT64_C( 2178074625616115189) },
      {  INT64_C( 6616989965536034123),  INT64_C( 8492415186196017232) },
      UINT8_C(  3) },
    { {  INT64_C( 2989634292083645491),  INT64_C( 8976485222817187412) },
      { -INT64_C( 4528093087150505358),  INT64_C( 8976485222817187412) },
      UINT8_C(  0) },
    { { -INT64_C( 8348038954527816028),  INT64_C( 9188914379591628719) },
      { -INT64_C( 2868702623850210345), -INT64_C(  974227990175860470) },
      UINT8_C(  3) },
    { { -INT64_C( 3067000787965063319), -INT64_C( 8674135663512655445) },
      {  INT64_C( 4593578800997777889),  INT64_C( 5885977995198498453) },
      UINT8_C(  0) },
    { { -INT64_C( 3151990947610216115),  INT64_C( 6097220498204270054) },
      { -INT64_C( 3151990947610216115), -INT64_C( 6408752416989317890) },
      UINT8_C(  2) },
    { { -INT64_C( 1073687052990391200), -INT64_C( 6099308486802790392) },
      { -INT64_C( 5069770386641178014),  INT64_C(  189502087319862710) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi64(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi64(a_);
    simde__m128i b = simde_mm_loadu_epi64(b_);
    simde__mmask8 r = simde_mm_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x2(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x2(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_ps_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float32 a[8];
    const simde_float32 b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(   831.44), SIMDE_FLOAT32_C(  -410.27), SIMDE_FLOAT32_C(  -112.75), SIMDE_FLOAT32_C(  -159.93),
        SIMDE_FLOAT32_C(   364.37), SIMDE_FLOAT32_C(   379.51), SIMDE_FLOAT32_C(  -514.59), SIMDE_FLOAT32_C(  -220.73) },
      { SIMDE_FLOAT32_C(   -95.60), SIMDE_FLOAT32_C(  -410.27), SIMDE_FLOAT32_C(  -112.75), SIMDE_FLOAT32_C(  -291.39),
        SIMDE_FLOAT32_C(   364.37), SIMDE_FLOAT32_C(   379.51), SIMDE_FLOAT32_C(  -514.59), SIMDE_FLOAT32_C(  -940.91) },
      UINT8_C(118) },
    { { SIMDE_FLOAT32_C(  -996.40), SIMDE_FLOAT32_C(   545.10), SIMDE_FLOAT32_C(  -357.87), SIMDE_FLOAT32_C(   967.66),
        SIMDE_FLOAT32_C(  -981.04), SIMDE_FLOAT32_C(  -884.43), SIMDE_FLOAT32_C(  -225.57), SIMDE_FLOAT32_C(  -130.50) },
      { SIMDE_FLOAT32_C(  -125.62), SIMDE_FLOAT32_C(   545.10), SIMDE_FLOAT32_C(  -357.87), SIMDE_FLOAT32_C(   908.68),
        SIMDE_FLOAT32_C(  -981.04), SIMDE_FLOAT32_C(  -884.43), SIMDE_FLOAT32_C(  -225.57), SIMDE_FLOAT32_C(  -815.18) },
      UINT8_C(  1) },
    { { SIMDE_FLOAT32_C(   155.30), SIMDE_FLOAT32_C(   481.86), SIMDE_FLOAT32_C(   806.27), SIMDE_FLOAT32_C(   927.59),
        SIMDE_FLOAT32_C(  -913.38), SIMDE_FLOAT32_C(   740.99), SIMDE_FLOAT32_C(   581.65), SIMDE_FLOAT32_C(  -877.78) },
      { SIMDE_FLOAT32_C(  -305.64), SIMDE_FLOAT32_C(   481.86), SIMDE_FLOAT32_C(     5.51), SIMDE_FLOAT32_C(  -184.53),
        SIMDE_FLOAT32_C(   687.30), SIMDE_FLOAT32_C(   740.99), SIMDE_FLOAT32_C(   -59.83), SIMDE_FLOAT32_C(   213.72) },
      UINT8_C(178) },
    { { SIMDE_FLOAT32_C(   298.39), SIMDE_FLOAT32_C(  -581.89), SIMDE_FLOAT32_C(  -101.63), SIMDE_FLOAT32_C(  -220.39),
        SIMDE_FLOAT32_C(  -514.76), SIMDE_FLOAT32_C(  -851.94), SIMDE_FLOAT32_C(   -75.08), SIMDE_FLOAT32_C(   989.15) },
      { SIMDE_FLOAT32_C(   298.39), SIMDE_FLOAT32_C(  -581.89), SIMDE_FLOAT32_C(   371.67), SIMDE_FLOAT32_C(   123.77),
        SIMDE_FLOAT32_C(   444.03), SIMDE_FLOAT32_C(  -806.99), SIMDE_FLOAT32_C(  -895.55), SIMDE_FLOAT32_C(   989.15) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(    42.02), SIMDE_FLOAT32_C(   537.55), SIMDE_FLOAT32_C(  -318.59), SIMDE_FLOAT32_C(  -560.36),
        SIMDE_FLOAT32_C(   -22.12), SIMDE_FLOAT32_C(   -43.21), SIMDE_FLOAT32_C(   838.15), SIMDE_FLOAT32_C(  -490.84) },
      { SIMDE_FLOAT32_C(    42.02), SIMDE_FLOAT32_C(   -28.38), SIMDE_FLOAT32_C(  -318.59), SIMDE_FLOAT32_C(  -560.36),
        SIMDE_FLOAT32_C(   -22.12), SIMDE_FLOAT32_C(   -43.21), SIMDE_FLOAT32_C(   800.49), SIMDE_FLOAT32_C(  -490.84) },
      UINT8_C( 66) },
    { { SIMDE_FLOAT32_C(   940.76), SIMDE_FLOAT32_C(   712.28), SIMDE_FLOAT32_C(  -948.28), SIMDE_FLOAT32_C(   605.67),
        SIMDE_FLOAT32_C(  -349.92), SIMDE_FLOAT32_C(   565.86), SIMDE_FLOAT32_C(  -111.05), SIMDE_FLOAT32_C(  -488.05) },
      { SIMDE_FLOAT32_C(   940.76), SIMDE_FLOAT32_C(  -693.68), SIMDE_FLOAT32_C(  -929.95), SIMDE_FLOAT32_C(   605.67),
        SIMDE_FLOAT32_C(  -359.64), SIMDE_FLOAT32_C(   565.86), SIMDE_FLOAT32_C(  -868.29), SIMDE_FLOAT32_C(  -488.05) },
      UINT8_C(251) },
    { { SIMDE_FLOAT32_C(   908.21), SIMDE_FLOAT32_C(   -92.67), SIMDE_FLOAT32_C(   173.44), SIMDE_FLOAT32_C(   331.87),
        SIMDE_FLOAT32_C(   206.80), SIMDE_FLOAT32_C(   715.17), SIMDE_FLOAT32_C(  -219.12), SIMDE_FLOAT32_C(   245.73) },
      { SIMDE_FLOAT32_C(   908.21), SIMDE_FLOAT32_C(   -92.67), SIMDE_FLOAT32_C(   -47.13), SIMDE_FLOAT32_C(   553.06),
        SIMDE_FLOAT32_C(  -779.61), SIMDE_FLOAT32_C(   659.70), SIMDE_FLOAT32_C(  -409.32), SIMDE_FLOAT32_C(   245.73) },
      UINT8_C(116) },
    { { SIMDE_FLOAT32_C(  -815.41), SIMDE_FLOAT32_C(  -895.46), SIMDE_FLOAT32_C(   -73.50), SIMDE_FLOAT32_C(   299.36),
        SIMDE_FLOAT32_C(  -782.68), SIMDE_FLOAT32_C(  -461.75), SIMDE_FLOAT32_C(   996.57), SIMDE_FLOAT32_C(   339.39) },
      { SIMDE_FLOAT32_C(  -815.41), SIMDE_FLOAT32_C(  -824.51), SIMDE_FLOAT32_C(   -73.50), SIMDE_FLOAT32_C(   299.36),
        SIMDE_FLOAT32_C(  -782.68), SIMDE_FLOAT32_C(  -461.75), SIMDE_FLOAT32_C(   996.57), SIMDE_FLOAT32_C(   339.39) },
         UINT8_MAX },
    { { SIMDE_FLOAT32_C(   604.80), SIMDE_FLOAT32_C(   483.90), SIMDE_FLOAT32_C(  -853.75), SIMDE_FLOAT32_C(  -559.59),
        SIMDE_FLOAT32_C(   157.89), SIMDE_FLOAT32_C(   853.80), SIMDE_FLOAT32_C(  -879.85), SIMDE_FLOAT32_C(   -63.58) },
      { SIMDE_FLOAT32_C(  -952.20), SIMDE_FLOAT32_C(   483.90), SIMDE_FLOAT32_C(  -853.75), SIMDE_FLOAT32_C(  -205.89),
        SIMDE_FLOAT32_C(   157.89), SIMDE_FLOAT32_C(   107.97), SIMDE_FLOAT32_C(  -879.85), SIMDE_FLOAT32_C(   -63.58) },
      UINT8_C(214) },
    { { SIMDE_FLOAT32_C(  -178.78), SIMDE_FLOAT32_C(  -476.17), SIMDE_FLOAT32_C(   278.39), SIMDE_FLOAT32_C(   895.68),
        SIMDE_FLOAT32_C(  -679.02), SIMDE_FLOAT32_C(  -733.78), SIMDE_FLOAT32_C(  -714.60), SIMDE_FLOAT32_C(  -945.93) },
      { SIMDE_FLOAT32_C(  -223.70), SIMDE_FLOAT32_C(   763.33), SIMDE_FLOAT32_C(  -738.67), SIMDE_FLOAT32_C(   895.68),
        SIMDE_FLOAT32_C(   150.33), SIMDE_FLOAT32_C(  -733.78), SIMDE_FLOAT32_C(  -380.15), SIMDE_FLOAT32_C(  -885.34) },
      UINT8_C(210) },
    { { SIMDE_FLOAT32_C(   -65.18), SIMDE_FLOAT32_C(  -674.01), SIMDE_FLOAT32_C(  -188.40), SIMDE_FLOAT32_C(  -400.37),
        SIMDE_FLOAT32_C(   -51.44), SIMDE_FLOAT32_C(   712.19), SIMDE_FLOAT32_C(    85.66), SIMDE_FLOAT32_C(    89.72) },
      { SIMDE_FLOAT32_C(   -65.18), SIMDE_FLOAT32_C(  -674.01), SIMDE_FLOAT32_C(   362.51), SIMDE_FLOAT32_C(  -400.37),
        SIMDE_FLOAT32_C(  -554.06), SIMDE_FLOAT32_C(  -812.53), SIMDE_FLOAT32_C(    85.66), SIMDE_FLOAT32_C(    89.72) },
      UINT8_C(207) },
    { { SIMDE_FLOAT32_C(   988.51), SIMDE_FLOAT32_C(    20.52), SIMDE_FLOAT32_C(  -514.76), SIMDE_FLOAT32_C(  -471.09),
        SIMDE_FLOAT32_C(  -141.89), SIMDE_FLOAT32_C(   685.92), SIMDE_FLOAT32_C(   181.30), SIMDE_FLOAT32_C(  -552.71) },
      { SIMDE_FLOAT32_C(   988.51), SIMDE_FLOAT32_C(   609.72), SIMDE_FLOAT32_C(  -652.86), SIMDE_FLOAT32_C(  -622.93),
        SIMDE_FLOAT32_C(  -141.89), SIMDE_FLOAT32_C(   685.92), SIMDE_FLOAT32_C(   181.30), SIMDE_FLOAT32_C(  -217.72) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   240.10), SIMDE_FLOAT32_C(  -220.27), SIMDE_FLOAT32_C(  -894.96), SIMDE_FLOAT32_C(   616.06),
        SIMDE_FLOAT32_C(   448.96), SIMDE_FLOAT32_C(  -429.81), SIMDE_FLOAT32_C(   398.26), SIMDE_FLOAT32_C(  -505.09) },
      { SIMDE_FLOAT32_C(   240.10), SIMDE_FLOAT32_C(  -847.08), SIMDE_FLOAT32_C(  -894.96), SIMDE_FLOAT32_C(   616.06),
        SIMDE_FLOAT32_C(   873.64), SIMDE_FLOAT32_C(  -429.81), SIMDE_FLOAT32_C(   423.54), SIMDE_FLOAT32_C(  -385.13) },
      UINT8_C(210) },
    { { SIMDE_FLOAT32_C(   499.12), SIMDE_FLOAT32_C(   280.53), SIMDE_FLOAT32_C(  -792.46), SIMDE_FLOAT32_C(   646.93),
        SIMDE_FLOAT32_C(   866.53), SIMDE_FLOAT32_C(  -920.25), SIMDE_FLOAT32_C(  -133.27), SIMDE_FLOAT32_C(   232.26) },
      { SIMDE_FLOAT32_C(   499.12), SIMDE_FLOAT32_C(   280.53), SIMDE_FLOAT32_C(  -999.50), SIMDE_FLOAT32_C(   646.93),
        SIMDE_FLOAT32_C(   866.53), SIMDE_FLOAT32_C(  -170.66), SIMDE_FLOAT32_C(  -133.27), SIMDE_FLOAT32_C(   232.26) },
      UINT8_C(223) },
    { { SIMDE_FLOAT32_C(   167.93), SIMDE_FLOAT32_C(  -852.32), SIMDE_FLOAT32_C(  -632.81), SIMDE_FLOAT32_C(  -158.28),
        SIMDE_FLOAT32_C(  -872.97), SIMDE_FLOAT32_C(  -831.53), SIMDE_FLOAT32_C(  -102.83), SIMDE_FLOAT32_C(   335.73) },
      { SIMDE_FLOAT32_C(   852.13), SIMDE_FLOAT32_C(  -852.32), SIMDE_FLOAT32_C(   885.08), SIMDE_FLOAT32_C(  -158.28),
        SIMDE_FLOAT32_C(  -872.97), SIMDE_FLOAT32_C(  -853.53), SIMDE_FLOAT32_C(  -102.83), SIMDE_FLOAT32_C(   335.73) },
      UINT8_C( 32) },
    { { SIMDE_FLOAT32_C(  -570.69), SIMDE_FLOAT32_C(  -452.90), SIMDE_FLOAT32_C(   -76.46), SIMDE_FLOAT32_C(  -353.70),
        SIMDE_FLOAT32_C(   612.31), SIMDE_FLOAT32_C(  -728.38), SIMDE_FLOAT32_C(  -258.82), SIMDE_FLOAT32_C(   106.17) },
      { SIMDE_FLOAT32_C(   178.23), SIMDE_FLOAT32_C(  -452.90), SIMDE_FLOAT32_C(   -76.46), SIMDE_FLOAT32_C(  -353.70),
        SIMDE_FLOAT32_C(   307.45), SIMDE_FLOAT32_C(   573.14), SIMDE_FLOAT32_C(   282.29), SIMDE_FLOAT32_C(    69.17) },
         UINT8_MAX },
    { { SIMDE_FLOAT32_C(  -983.86), SIMDE_FLOAT32_C(   408.46), SIMDE_FLOAT32_C(  -959.03), SIMDE_FLOAT32_C(  -657.05),
        SIMDE_FLOAT32_C(   845.99), SIMDE_FLOAT32_C(    16.18), SIMDE_FLOAT32_C(   -89.37), SIMDE_FLOAT32_C(   363.78) },
      { SIMDE_FLOAT32_C(   545.29), SIMDE_FLOAT32_C(  -944.94), SIMDE_FLOAT32_C(  -731.09), SIMDE_FLOAT32_C(    13.30),
        SIMDE_FLOAT32_C(   -96.88), SIMDE_FLOAT32_C(    16.18), SIMDE_FLOAT32_C(   -89.37), SIMDE_FLOAT32_C(   363.78) },
      UINT8_C(224) },
    { { SIMDE_FLOAT32_C(   876.99), SIMDE_FLOAT32_C(   678.73), SIMDE_FLOAT32_C(   656.39), SIMDE_FLOAT32_C(   250.65),
        SIMDE_FLOAT32_C(  -705.37), SIMDE_FLOAT32_C(   815.10), SIMDE_FLOAT32_C(  -612.18), SIMDE_FLOAT32_C(   -32.64) },
      { SIMDE_FLOAT32_C(   876.99), SIMDE_FLOAT32_C(   609.87), SIMDE_FLOAT32_C(   407.63), SIMDE_FLOAT32_C(   124.33),
        SIMDE_FLOAT32_C(   584.38), SIMDE_FLOAT32_C(  -932.19), SIMDE_FLOAT32_C(  -612.18), SIMDE_FLOAT32_C(   -32.64) },
      UINT8_C( 16) },
    { { SIMDE_FLOAT32_C(   524.88), SIMDE_FLOAT32_C(   869.75), SIMDE_FLOAT32_C(  -196.64), SIMDE_FLOAT32_C(   212.86),
        SIMDE_FLOAT32_C(  -120.74), SIMDE_FLOAT32_C(   987.87), SIMDE_FLOAT32_C(   102.21), SIMDE_FLOAT32_C(   114.24) },
      { SIMDE_FLOAT32_C(   524.88), SIMDE_FLOAT32_C(   869.75), SIMDE_FLOAT32_C(  -196.64), SIMDE_FLOAT32_C(  -914.16),
        SIMDE_FLOAT32_C(  -120.74), SIMDE_FLOAT32_C(   247.04), SIMDE_FLOAT32_C(   136.12), SIMDE_FLOAT32_C(   567.49) },
      UINT8_C(215) },
    { { SIMDE_FLOAT32_C(   696.44), SIMDE_FLOAT32_C(   755.81), SIMDE_FLOAT32_C(   674.18), SIMDE_FLOAT32_C(   991.20),
        SIMDE_FLOAT32_C(    85.71), SIMDE_FLOAT32_C(   725.49), SIMDE_FLOAT32_C(   545.57), SIMDE_FLOAT32_C(  -373.78) },
      { SIMDE_FLOAT32_C(   696.44), SIMDE_FLOAT32_C(  -940.91), SIMDE_FLOAT32_C(   674.18), SIMDE_FLOAT32_C(   -28.29),
        SIMDE_FLOAT32_C(    85.71), SIMDE_FLOAT32_C(   336.40), SIMDE_FLOAT32_C(   545.57), SIMDE_FLOAT32_C(  -373.78) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(   341.85), SIMDE_FLOAT32_C(   580.90), SIMDE_FLOAT32_C(  -515.68), SIMDE_FLOAT32_C(   644.24),
        SIMDE_FLOAT32_C(  -129.57), SIMDE_FLOAT32_C(   449.33), SIMDE_FLOAT32_C(  -655.19), SIMDE_FLOAT32_C(   -16.68) },
      { SIMDE_FLOAT32_C(   341.85), SIMDE_FLOAT32_C(   580.90), SIMDE_FLOAT32_C(   630.00), SIMDE_FLOAT32_C(   644.24),
        SIMDE_FLOAT32_C(   404.20), SIMDE_FLOAT32_C(   449.33), SIMDE_FLOAT32_C(  -655.19), SIMDE_FLOAT32_C(   -16.68) },
      UINT8_C( 20) },
    { { SIMDE_FLOAT32_C(   510.54), SIMDE_FLOAT32_C(  -865.91), SIMDE_FLOAT32_C(  -220.86), SIMDE_FLOAT32_C(  -411.08),
        SIMDE_FLOAT32_C(   921.04), SIMDE_FLOAT32_C(   336.67), SIMDE_FLOAT32_C(  -676.55), SIMDE_FLOAT32_C(   827.53) },
      { SIMDE_FLOAT32_C(   798.79), SIMDE_FLOAT32_C(  -865.91), SIMDE_FLOAT32_C(  -321.49), SIMDE_FLOAT32_C(   180.18),
        SIMDE_FLOAT32_C(  -687.20), SIMDE_FLOAT32_C(   336.67), SIMDE_FLOAT32_C(  -676.55), SIMDE_FLOAT32_C(   827.53) },
      UINT8_C(246) },
    { { SIMDE_FLOAT32_C(  -627.13), SIMDE_FLOAT32_C(    64.98), SIMDE_FLOAT32_C(  -970.49), SIMDE_FLOAT32_C(  -723.12),
        SIMDE_FLOAT32_C(  -117.77), SIMDE_FLOAT32_C(  -863.38), SIMDE_FLOAT32_C(   486.41), SIMDE_FLOAT32_C(   934.05) },
      { SIMDE_FLOAT32_C(  -627.13), SIMDE_FLOAT32_C(   770.44), SIMDE_FLOAT32_C(   669.77), SIMDE_FLOAT32_C(    83.09),
        SIMDE_FLOAT32_C(  -887.70), SIMDE_FLOAT32_C(  -970.91), SIMDE_FLOAT32_C(   303.83), SIMDE_FLOAT32_C(   376.67) },
      UINT8_C(240) },
    { { SIMDE_FLOAT32_C(  -558.58), SIMDE_FLOAT32_C(   -35.80), SIMDE_FLOAT32_C(   180.97), SIMDE_FLOAT32_C(   208.99),
        SIMDE_FLOAT32_C(   986.84), SIMDE_FLOAT32_C(   450.10), SIMDE_FLOAT32_C(   136.00), SIMDE_FLOAT32_C(  -410.08) },
      { SIMDE_FLOAT32_C(  -980.36), SIMDE_FLOAT32_C(   -35.80), SIMDE_FLOAT32_C(  -951.91), SIMDE_FLOAT32_C(   208.99),
        SIMDE_FLOAT32_C(  -755.43), SIMDE_FLOAT32_C(   450.10), SIMDE_FLOAT32_C(   136.00), SIMDE_FLOAT32_C(  -410.08) },
         UINT8_MAX },
    { { SIMDE_FLOAT32_C(  -799.22), SIMDE_FLOAT32_C(   469.27), SIMDE_FLOAT32_C(   -70.04), SIMDE_FLOAT32_C(   943.53),
        SIMDE_FLOAT32_C(  -698.60), SIMDE_FLOAT32_C(   629.00), SIMDE_FLOAT32_C(  -872.30), SIMDE_FLOAT32_C(  -173.15) },
      { SIMDE_FLOAT32_C(  -799.22), SIMDE_FLOAT32_C(   287.73), SIMDE_FLOAT32_C(   -70.04), SIMDE_FLOAT32_C(   943.53),
        SIMDE_FLOAT32_C(  -698.60), SIMDE_FLOAT32_C(   629.00), SIMDE_FLOAT32_C(  -872.30), SIMDE_FLOAT32_C(  -173.15) },
      UINT8_C(253) },
    { { SIMDE_FLOAT32_C(  -143.01), SIMDE_FLOAT32_C(  -471.18), SIMDE_FLOAT32_C(   534.70), SIMDE_FLOAT32_C(   723.54),
        SIMDE_FLOAT32_C(   844.10), SIMDE_FLOAT32_C(  -894.69), SIMDE_FLOAT32_C(  -211.48), SIMDE_FLOAT32_C(   952.24) },
      { SIMDE_FLOAT32_C(  -143.01), SIMDE_FLOAT32_C(    15.45), SIMDE_FLOAT32_C(   534.70), SIMDE_FLOAT32_C(   971.43),
        SIMDE_FLOAT32_C(   844.10), SIMDE_FLOAT32_C(    62.65), SIMDE_FLOAT32_C(  -211.48), SIMDE_FLOAT32_C(  -857.97) },
      UINT8_C( 42) },
    { { SIMDE_FLOAT32_C(   107.45), SIMDE_FLOAT32_C(   -70.47), SIMDE_FLOAT32_C(   432.82), SIMDE_FLOAT32_C(  -780.42),
        SIMDE_FLOAT32_C(   655.60), SIMDE_FLOAT32_C(   899.15), SIMDE_FLOAT32_C(   286.51), SIMDE_FLOAT32_C(  -960.27) },
      { SIMDE_FLOAT32_C(   981.94), SIMDE_FLOAT32_C(   -70.47), SIMDE_FLOAT32_C(   432.82), SIMDE_FLOAT32_C(  -578.67),
        SIMDE_FLOAT32_C(   655.60), SIMDE_FLOAT32_C(   899.15), SIMDE_FLOAT32_C(   286.51), SIMDE_FLOAT32_C(   -89.52) },
         UINT8_MAX },
    { { SIMDE_FLOAT32_C(   884.88), SIMDE_FLOAT32_C(   -35.17), SIMDE_FLOAT32_C(   -20.28), SIMDE_FLOAT32_C(  -812.58),
        SIMDE_FLOAT32_C(   -88.96), SIMDE_FLOAT32_C(   511.58), SIMDE_FLOAT32_C(  -580.64), SIMDE_FLOAT32_C(   326.32) },
      { SIMDE_FLOAT32_C(   599.03), SIMDE_FLOAT32_C(   -35.17), SIMDE_FLOAT32_C(   -20.28), SIMDE_FLOAT32_C(  -544.53),
        SIMDE_FLOAT32_C(   425.84), SIMDE_FLOAT32_C(  -142.94), SIMDE_FLOAT32_C(  -580.64), SIMDE_FLOAT32_C(  -795.92) },
      UINT8_C(  0) },
    { { SIMDE_FLOAT32_C(  -655.95), SIMDE_FLOAT32_C(   800.64), SIMDE_FLOAT32_C(  -708.69), SIMDE_FLOAT32_C(  -476.50),
        SIMDE_FLOAT32_C(   975.99), SIMDE_FLOAT32_C(  -898.03), SIMDE_FLOAT32_C(   209.09), SIMDE_FLOAT32_C(  -433.20) },
      { SIMDE_FLOAT32_C(   564.29), SIMDE_FLOAT32_C(  -943.57), SIMDE_FLOAT32_C(   750.81), SIMDE_FLOAT32_C(   571.99),
        SIMDE_FLOAT32_C(   975.99), SIMDE_FLOAT32_C(  -898.03), SIMDE_FLOAT32_C(   464.58), SIMDE_FLOAT32_C(  -433.20) },
      UINT8_C( 79) },
    { { SIMDE_FLOAT32_C(  -309.14), SIMDE_FLOAT32_C(   799.08), SIMDE_FLOAT32_C(   -89.08), SIMDE_FLOAT32_C(   270.31),
        SIMDE_FLOAT32_C(   266.85), SIMDE_FLOAT32_C(   458.14), SIMDE_FLOAT32_C(  -217.02), SIMDE_FLOAT32_C(  -386.85) },
      { SIMDE_FLOAT32_C(  -784.25), SIMDE_FLOAT32_C(   721.19), SIMDE_FLOAT32_C(  -229.72), SIMDE_FLOAT32_C(   270.31),
        SIMDE_FLOAT32_C(   266.85), SIMDE_FLOAT32_C(   458.14), SIMDE_FLOAT32_C(  -331.09), SIMDE_FLOAT32_C(  -386.85) },
         UINT8_MAX },
    { { SIMDE_FLOAT32_C(  -843.26), SIMDE_FLOAT32_C(   225.68), SIMDE_FLOAT32_C(   -21.81), SIMDE_FLOAT32_C(   134.84),
        SIMDE_FLOAT32_C(   283.09), SIMDE_FLOAT32_C(  -559.67), SIMDE_FLOAT32_C(   746.09), SIMDE_FLOAT32_C(  -730.92) },
      { SIMDE_FLOAT32_C(   214.45), SIMDE_FLOAT32_C(  -137.75), SIMDE_FLOAT32_C(   -21.81), SIMDE_FLOAT32_C(   134.84),
        SIMDE_FLOAT32_C(  -333.88), SIMDE_FLOAT32_C(  -424.40), SIMDE_FLOAT32_C(   746.09), SIMDE_FLOAT32_C(  -730.92) },
      UINT8_C( 18) },
    { { SIMDE_FLOAT32_C(   519.34), SIMDE_FLOAT32_C(   620.57), SIMDE_FLOAT32_C(  -706.76), SIMDE_FLOAT32_C(   767.22),
        SIMDE_FLOAT32_C(  -959.03), SIMDE_FLOAT32_C(  -585.58), SIMDE_FLOAT32_C(  -243.41), SIMDE_FLOAT32_C(  -221.88) },
      { SIMDE_FLOAT32_C(   432.41), SIMDE_FLOAT32_C(   620.57), SIMDE_FLOAT32_C(   443.15), SIMDE_FLOAT32_C(  -983.87),
        SIMDE_FLOAT32_C(   106.03), SIMDE_FLOAT32_C(  -526.63), SIMDE_FLOAT32_C(  -243.41), SIMDE_FLOAT32_C(   817.18) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256 a = simde_mm256_loadu_ps(test_vec[i].a);
    simde__m256 b = simde_mm256_loadu_ps(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_ps_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 32 ; i++) {
    simde_float32 a_[8];
    simde_float32 b_[8];

    simde_test_codegen_random_vf32((sizeof(a_) / sizeof(a_[0])), a_, SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    simde_test_codegen_random_vf32((sizeof(b_) / sizeof(b_[0])), b_, SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 1))
        a_[j] = b_[j];

    simde__m256 a = simde_mm256_loadu_ps(a_);
    simde__m256 b = simde_mm256_loadu_ps(b_);
    simde__mmask8 r = simde_mm256_cmp_ps_mask(a, b, HEDLEY_STATIC_CAST(int, i));

    simde_test_x86_write_f32x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f32x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_pd_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float64 a[4];
    const simde_float64 b[4];
    const simde__mmask8 r;
    const simde_float64 an[4];
    const simde_float64 bn[4];
    const simde__mmask8 rn;
  } test_vec[] = {
    /* 0 */
    { { SIMDE_FLOAT64_C(   -92.56), SIMDE_FLOAT64_C(  -784.23), SIMDE_FLOAT64_C(   201.94), SIMDE_FLOAT64_C(  -524.10) },
      { SIMDE_FLOAT64_C(   -92.56), SIMDE_FLOAT64_C(  -860.14), SIMDE_FLOAT64_C(  -142.10), SIMDE_FLOAT64_C(  -398.14) },
      UINT8_C(  1),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   696.02),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -638.84) },
      { SIMDE_FLOAT64_C(  -298.55),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -216.53) },
      UINT8_C(  0) },
    /* 1 */
    { { SIMDE_FLOAT64_C(   900.78), SIMDE_FLOAT64_C(  -468.88), SIMDE_FLOAT64_C(  -599.62), SIMDE_FLOAT64_C(   756.84) },
      { SIMDE_FLOAT64_C(   900.78), SIMDE_FLOAT64_C(  -427.16), SIMDE_FLOAT64_C(  -105.22), SIMDE_FLOAT64_C(    86.11) },
      UINT8_C(  6),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   574.43),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   424.95) },
      { SIMDE_FLOAT64_C(  -126.69),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   780.76) },
      UINT8_C(  8) },
    /* 2 */
    { { SIMDE_FLOAT64_C(   846.95), SIMDE_FLOAT64_C(   939.89), SIMDE_FLOAT64_C(  -743.34), SIMDE_FLOAT64_C(  -192.23) },
      { SIMDE_FLOAT64_C(   846.95), SIMDE_FLOAT64_C(   114.56), SIMDE_FLOAT64_C(   409.63), SIMDE_FLOAT64_C(  -515.67) },
      UINT8_C(  5),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   294.12),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   512.04) },
      { SIMDE_FLOAT64_C(   656.02),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   556.79) },
      UINT8_C(  8) },
    /* 3 */
    { { SIMDE_FLOAT64_C(   497.48), SIMDE_FLOAT64_C(  -304.12), SIMDE_FLOAT64_C(   313.63), SIMDE_FLOAT64_C(  -148.20) },
      { SIMDE_FLOAT64_C(   497.48), SIMDE_FLOAT64_C(  -791.59), SIMDE_FLOAT64_C(   937.92), SIMDE_FLOAT64_C(   795.78) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   781.79),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -343.84) },
      { SIMDE_FLOAT64_C(   412.97),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   259.92) },
      UINT8_C(  7) },
    /* 4 */
    { { SIMDE_FLOAT64_C(   898.56), SIMDE_FLOAT64_C(  -306.42), SIMDE_FLOAT64_C(  -932.31), SIMDE_FLOAT64_C(   978.31) },
      { SIMDE_FLOAT64_C(   898.56), SIMDE_FLOAT64_C(   477.32), SIMDE_FLOAT64_C(  -537.36), SIMDE_FLOAT64_C(  -381.27) },
      UINT8_C( 14),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   308.12),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -572.54) },
      { SIMDE_FLOAT64_C(  -725.52),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   771.96) },
      UINT8_C( 15) },
    /* 5 */
    { { SIMDE_FLOAT64_C(   122.15), SIMDE_FLOAT64_C(   297.88), SIMDE_FLOAT64_C(  -376.24), SIMDE_FLOAT64_C(  -609.13) },
      { SIMDE_FLOAT64_C(   122.15), SIMDE_FLOAT64_C(  -438.32), SIMDE_FLOAT64_C(  -813.35), SIMDE_FLOAT64_C(   289.14) },
      UINT8_C(  7),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   407.38),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   756.44) },
      { SIMDE_FLOAT64_C(   366.06),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   264.62) },
      UINT8_C( 15) },
    /* 6 */
    { { SIMDE_FLOAT64_C(    75.79), SIMDE_FLOAT64_C(    84.05), SIMDE_FLOAT64_C(   242.93), SIMDE_FLOAT64_C(  -116.07) },
      { SIMDE_FLOAT64_C(    75.79), SIMDE_FLOAT64_C(   705.57), SIMDE_FLOAT64_C(   502.66), SIMDE_FLOAT64_C(   332.82) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   633.42),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   288.17) },
      { SIMDE_FLOAT64_C(  -940.31),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   181.84) },
      UINT8_C( 15) },
    /* 7 */
    { { SIMDE_FLOAT64_C(    42.43), SIMDE_FLOAT64_C(   683.89), SIMDE_FLOAT64_C(   572.71), SIMDE_FLOAT64_C(  -451.28) },
      { SIMDE_FLOAT64_C(    42.43), SIMDE_FLOAT64_C(   759.36), SIMDE_FLOAT64_C(   837.86), SIMDE_FLOAT64_C(  -410.96) },
      UINT8_C( 15),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   783.16),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -467.20) },
      { SIMDE_FLOAT64_C(  -834.62),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   241.17) },
      UINT8_C(  8) },
    /* 8 */
    { { SIMDE_FLOAT64_C(  -554.10), SIMDE_FLOAT64_C(    40.35), SIMDE_FLOAT64_C(  -874.90), SIMDE_FLOAT64_C(     7.28) },
      { SIMDE_FLOAT64_C(  -554.10), SIMDE_FLOAT64_C(   627.75), SIMDE_FLOAT64_C(  -659.90), SIMDE_FLOAT64_C(   759.62) },
      UINT8_C(  1),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -899.61),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   320.87) },
      { SIMDE_FLOAT64_C(   844.93),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -112.64) },
      UINT8_C(  7) },
    /* 9 */
    { { SIMDE_FLOAT64_C(   791.80), SIMDE_FLOAT64_C(  -924.57), SIMDE_FLOAT64_C(   436.08), SIMDE_FLOAT64_C(  -962.62) },
      { SIMDE_FLOAT64_C(   791.80), SIMDE_FLOAT64_C(   273.94), SIMDE_FLOAT64_C(  -373.58), SIMDE_FLOAT64_C(     1.53) },
      UINT8_C( 10),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   -28.09),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   222.48) },
      { SIMDE_FLOAT64_C(  -666.24),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -220.34) },
      UINT8_C(  7) },
    /* 10 */
    { { SIMDE_FLOAT64_C(  -627.90), SIMDE_FLOAT64_C(  -411.26), SIMDE_FLOAT64_C(   786.94), SIMDE_FLOAT64_C(   118.02) },
      { SIMDE_FLOAT64_C(  -627.90), SIMDE_FLOAT64_C(  -872.96), SIMDE_FLOAT64_C(  -122.36), SIMDE_FLOAT64_C(   477.67) },
      UINT8_C(  9),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   925.42),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -927.64) },
      { SIMDE_FLOAT64_C(  -966.66),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   825.14) },
      UINT8_C( 15) },
    /* 11 */
    { { SIMDE_FLOAT64_C(   376.68), SIMDE_FLOAT64_C(  -604.21), SIMDE_FLOAT64_C(   862.52), SIMDE_FLOAT64_C(   211.47) },
      { SIMDE_FLOAT64_C(   376.68), SIMDE_FLOAT64_C(  -511.06), SIMDE_FLOAT64_C(  -787.00), SIMDE_FLOAT64_C(  -273.17) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   747.32),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   794.60) },
      { SIMDE_FLOAT64_C(    79.07),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   451.16) },
      UINT8_C(  0) },
    /* 12 */
    { { SIMDE_FLOAT64_C(     1.69), SIMDE_FLOAT64_C(  -638.80), SIMDE_FLOAT64_C(  -430.82), SIMDE_FLOAT64_C(   218.18) },
      { SIMDE_FLOAT64_C(     1.69), SIMDE_FLOAT64_C(   446.82), SIMDE_FLOAT64_C(  -304.15), SIMDE_FLOAT64_C(  -284.32) },
      UINT8_C( 14),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   494.39),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   405.58) },
      { SIMDE_FLOAT64_C(  -204.35),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -827.67) },
      UINT8_C(  8) },
    /* 13 */
    { { SIMDE_FLOAT64_C(  -856.45), SIMDE_FLOAT64_C(    93.24), SIMDE_FLOAT64_C(   383.80), SIMDE_FLOAT64_C(   813.29) },
      { SIMDE_FLOAT64_C(  -856.45), SIMDE_FLOAT64_C(   596.80), SIMDE_FLOAT64_C(  -459.88), SIMDE_FLOAT64_C(    43.02) },
      UINT8_C( 13),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -510.57),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -576.81) },
      { SIMDE_FLOAT64_C(   -97.62),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   904.07) },
      UINT8_C(  0) },
    /* 14 */
    { { SIMDE_FLOAT64_C(   773.08), SIMDE_FLOAT64_C(  -556.47), SIMDE_FLOAT64_C(   122.25), SIMDE_FLOAT64_C(  -738.68) },
      { SIMDE_FLOAT64_C(   773.08), SIMDE_FLOAT64_C(   818.10), SIMDE_FLOAT64_C(   -23.00), SIMDE_FLOAT64_C(   262.58) },
      UINT8_C(  4),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   765.04),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -891.86) },
      { SIMDE_FLOAT64_C(   512.80),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   656.36) },
      UINT8_C(  0) },
    /* 15 */
    { { SIMDE_FLOAT64_C(    -7.89), SIMDE_FLOAT64_C(   664.27), SIMDE_FLOAT64_C(   469.65), SIMDE_FLOAT64_C(  -425.71) },
      { SIMDE_FLOAT64_C(    -7.89), SIMDE_FLOAT64_C(  -990.23), SIMDE_FLOAT64_C(   617.30), SIMDE_FLOAT64_C(  -394.81) },
      UINT8_C( 15),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -545.08),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   401.58) },
      { SIMDE_FLOAT64_C(   866.79),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   639.87) },
      UINT8_C( 15) },
    /* 16 */
    { { SIMDE_FLOAT64_C(   346.25), SIMDE_FLOAT64_C(  -572.10), SIMDE_FLOAT64_C(   901.19), SIMDE_FLOAT64_C(   236.59) },
      { SIMDE_FLOAT64_C(   346.25), SIMDE_FLOAT64_C(  -121.81), SIMDE_FLOAT64_C(  -500.83), SIMDE_FLOAT64_C(   558.48) },
      UINT8_C(  1),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   167.33),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -843.96) },
      { SIMDE_FLOAT64_C(    66.21),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -941.68) },
      UINT8_C(  0) },
    /* 17 */
    { { SIMDE_FLOAT64_C(   611.36), SIMDE_FLOAT64_C(   282.04), SIMDE_FLOAT64_C(  -367.39), SIMDE_FLOAT64_C(  -127.58) },
      { SIMDE_FLOAT64_C(   611.36), SIMDE_FLOAT64_C(  -750.09), SIMDE_FLOAT64_C(   477.61), SIMDE_FLOAT64_C(   791.01) },
      UINT8_C( 12),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -494.01),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -428.38) },
      { SIMDE_FLOAT64_C(   408.71),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -245.05) },
      UINT8_C(  8) },
    /* 18 */
    { { SIMDE_FLOAT64_C(   -73.86), SIMDE_FLOAT64_C(  -887.33), SIMDE_FLOAT64_C(   991.54), SIMDE_FLOAT64_C(   172.14) },
      { SIMDE_FLOAT64_C(   -73.86), SIMDE_FLOAT64_C(  -509.28), SIMDE_FLOAT64_C(  -269.38), SIMDE_FLOAT64_C(   634.10) },
      UINT8_C( 11),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -602.76),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -275.74) },
      { SIMDE_FLOAT64_C(  -655.67),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   955.69) },
      UINT8_C(  8) },
    /* 19 */
    { { SIMDE_FLOAT64_C(  -115.42), SIMDE_FLOAT64_C(   415.19), SIMDE_FLOAT64_C(  -171.89), SIMDE_FLOAT64_C(  -823.61) },
      { SIMDE_FLOAT64_C(  -115.42), SIMDE_FLOAT64_C(  -694.28), SIMDE_FLOAT64_C(   967.40), SIMDE_FLOAT64_C(  -630.06) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   159.99),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -779.58) },
      { SIMDE_FLOAT64_C(   658.23),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -415.63) },
      UINT8_C(  7) },
    /* 20 */
    { { SIMDE_FLOAT64_C(   265.71), SIMDE_FLOAT64_C(   -33.09), SIMDE_FLOAT64_C(   756.51), SIMDE_FLOAT64_C(  -743.43) },
      { SIMDE_FLOAT64_C(   265.71), SIMDE_FLOAT64_C(  -512.87), SIMDE_FLOAT64_C(   890.67), SIMDE_FLOAT64_C(   115.68) },
      UINT8_C( 14),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   680.81),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   228.70) },
      { SIMDE_FLOAT64_C(   283.34),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -832.08) },
      UINT8_C( 15) },
    /* 21 */
    { { SIMDE_FLOAT64_C(  -962.28), SIMDE_FLOAT64_C(  -987.50), SIMDE_FLOAT64_C(  -655.69), SIMDE_FLOAT64_C(   702.82) },
      { SIMDE_FLOAT64_C(  -962.28), SIMDE_FLOAT64_C(  -688.29), SIMDE_FLOAT64_C(  -927.24), SIMDE_FLOAT64_C(   129.93) },
      UINT8_C( 13),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    14.31),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   129.93) },
      { SIMDE_FLOAT64_C(  -832.65),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   433.06) },
      UINT8_C(  7) },
    /* 22 */
    { { SIMDE_FLOAT64_C(   292.63), SIMDE_FLOAT64_C(   470.81), SIMDE_FLOAT64_C(   689.64), SIMDE_FLOAT64_C(  -249.74) },
      { SIMDE_FLOAT64_C(   292.63), SIMDE_FLOAT64_C(   580.31), SIMDE_FLOAT64_C(   865.93), SIMDE_FLOAT64_C(  -157.69) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   705.87),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -455.55) },
      { SIMDE_FLOAT64_C(  -671.61),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -633.89) },
      UINT8_C( 15) },
    /* 23 */
    { { SIMDE_FLOAT64_C(   267.90), SIMDE_FLOAT64_C(    56.67), SIMDE_FLOAT64_C(  -931.07), SIMDE_FLOAT64_C(   586.12) },
      { SIMDE_FLOAT64_C(   267.90), SIMDE_FLOAT64_C(  -858.31), SIMDE_FLOAT64_C(  -283.94), SIMDE_FLOAT64_C(  -159.92) },
      UINT8_C( 15),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -933.59),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   323.35) },
      { SIMDE_FLOAT64_C(  -607.88),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   684.75) },
      UINT8_C(  8) },
    /* 24 */
    { { SIMDE_FLOAT64_C(   155.10), SIMDE_FLOAT64_C(  -553.96), SIMDE_FLOAT64_C(  -564.99), SIMDE_FLOAT64_C(   113.04) },
      { SIMDE_FLOAT64_C(   155.10), SIMDE_FLOAT64_C(  -699.06), SIMDE_FLOAT64_C(   955.35), SIMDE_FLOAT64_C(   287.46) },
      UINT8_C(  1),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -973.64),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -664.80) },
      { SIMDE_FLOAT64_C(   281.76),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -450.34) },
      UINT8_C(  7) },
    /* 25 */
    { { SIMDE_FLOAT64_C(   600.95), SIMDE_FLOAT64_C(  -229.77), SIMDE_FLOAT64_C(  -864.22), SIMDE_FLOAT64_C(   -30.67) },
      { SIMDE_FLOAT64_C(   600.95), SIMDE_FLOAT64_C(  -148.17), SIMDE_FLOAT64_C(   809.41), SIMDE_FLOAT64_C(  -932.08) },
      UINT8_C(  6),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   779.41),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   310.36) },
      { SIMDE_FLOAT64_C(   463.71),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -381.19) },
      UINT8_C(  7) },
    /* 26 */
    { { SIMDE_FLOAT64_C(  -406.29), SIMDE_FLOAT64_C(   430.12), SIMDE_FLOAT64_C(   731.85), SIMDE_FLOAT64_C(  -379.94) },
      { SIMDE_FLOAT64_C(  -406.29), SIMDE_FLOAT64_C(   687.20), SIMDE_FLOAT64_C(   907.52), SIMDE_FLOAT64_C(   737.87) },
      UINT8_C( 15),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   739.44),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    -4.69) },
      { SIMDE_FLOAT64_C(  -716.28),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   884.67) },
      UINT8_C( 15) },
    /* 27 */
    { { SIMDE_FLOAT64_C(   544.60), SIMDE_FLOAT64_C(   680.75), SIMDE_FLOAT64_C(  -145.99), SIMDE_FLOAT64_C(  -543.48) },
      { SIMDE_FLOAT64_C(   544.60), SIMDE_FLOAT64_C(  -336.58), SIMDE_FLOAT64_C(  -475.57), SIMDE_FLOAT64_C(   450.82) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   915.70),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   906.54) },
      { SIMDE_FLOAT64_C(  -936.63),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -342.92) },
      UINT8_C(  0) },
    /* 28 */
    { { SIMDE_FLOAT64_C(   186.41), SIMDE_FLOAT64_C(  -742.80), SIMDE_FLOAT64_C(   277.14), SIMDE_FLOAT64_C(   -82.53) },
      { SIMDE_FLOAT64_C(   186.41), SIMDE_FLOAT64_C(   184.66), SIMDE_FLOAT64_C(  -344.67), SIMDE_FLOAT64_C(   657.96) },
      UINT8_C( 14),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -271.60),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   207.82) },
      { SIMDE_FLOAT64_C(   502.77),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    47.37) },
      UINT8_C(  8) },
    /* 29 */
    { { SIMDE_FLOAT64_C(   878.98), SIMDE_FLOAT64_C(   946.50), SIMDE_FLOAT64_C(   503.89), SIMDE_FLOAT64_C(  -588.44) },
      { SIMDE_FLOAT64_C(   878.98), SIMDE_FLOAT64_C(  -971.68), SIMDE_FLOAT64_C(   862.38), SIMDE_FLOAT64_C(    52.75) },
      UINT8_C(  3),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -376.44),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -992.61) },
      { SIMDE_FLOAT64_C(  -620.14),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   566.27) },
      UINT8_C(  0) },
    /* 30 */
    { { SIMDE_FLOAT64_C(   741.84), SIMDE_FLOAT64_C(   941.61), SIMDE_FLOAT64_C(  -516.27), SIMDE_FLOAT64_C(   686.24) },
      { SIMDE_FLOAT64_C(   741.84), SIMDE_FLOAT64_C(   139.07), SIMDE_FLOAT64_C(   344.20), SIMDE_FLOAT64_C(  -949.63) },
      UINT8_C( 10),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   997.47),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   370.23) },
      { SIMDE_FLOAT64_C(  -804.29),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -925.31) },
      UINT8_C(  8) },
    /* 31 */
    { { SIMDE_FLOAT64_C(  -702.81), SIMDE_FLOAT64_C(   921.49), SIMDE_FLOAT64_C(  -513.75), SIMDE_FLOAT64_C(   -92.88) },
      { SIMDE_FLOAT64_C(  -702.81), SIMDE_FLOAT64_C(  -651.37), SIMDE_FLOAT64_C(   959.86), SIMDE_FLOAT64_C(   893.83) },
      UINT8_C( 15),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   -80.85),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   352.05) },
      { SIMDE_FLOAT64_C(  -596.21),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -854.37) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256d a = simde_mm256_loadu_pd(test_vec[i].a);
    simde__m256d b = simde_mm256_loadu_pd(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_pd_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  const simde__m256d nans = simde_mm256_set1_pd(SIMDE_MATH_NAN);
  for (int i = 0 ; i < 32 ; i++) {
    simde__m256d a, b;
    simde__mmask8 r;

    fprintf(stdout, "    /* %d */\n", i);
    a = simde_test_x86_random_f64x4(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_test_x86_random_f64x4(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_mm256_blend_pd(b, a, 1);
    SIMDE_CONSTIFY_32_(simde_mm256_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f64x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f64x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_MIDDLE);

    a = simde_test_x86_random_f64x4(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_test_x86_random_f64x4(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    a = simde_mm256_blend_pd(a, nans, 5);
    b = simde_mm256_blend_pd(b, nans, 6);
    SIMDE_CONSTIFY_32_(simde_mm256_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f64x4(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_f64x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[32];
    const int8_t b[32];
    const simde__mmask32 r;
  } test_vec[] = {
    { {  INT8_C(  43),  INT8_C( 113),  INT8_C(  56),  INT8_C(  27),  INT8_C(  80),  INT8_C(  65), -INT8_C(  20), -INT8_C(  77),
         INT8_C(   3),  INT8_C(  93),  INT8_C( 124), -INT8_C(  58), -INT8_C(  98), -INT8_C(  78), -INT8_C( 108), -INT8_C(  78),
        -INT8_C(  85), -INT8_C(  87),  INT8_C(  33), -INT8_C(  42),  INT8_C( 115),  INT8_C(  64),  INT8_C(  72), -INT8_C(  41),
         INT8_C(  93), -INT8_C(  68),  INT8_C(  27),  INT8_C(  15), -INT8_C(  77), -INT8_C(  43),  INT8_C(  56), -INT8_C(  33) },
      { -INT8_C( 113),  INT8_C( 113), -INT8_C(   6), -INT8_C(  33), -INT8_C(  78), -INT8_C(  26), -INT8_C( 109), -INT8_C(  75),
         INT8_C(  67),  INT8_C(  44),  INT8_C( 124), -INT8_C(  30), -INT8_C(  42), -INT8_C(  78), -INT8_C( 108), -INT8_C( 127),
         INT8_C(  92), -INT8_C(  75),  INT8_C(  87), -INT8_C(  49), -INT8_C(  11), -INT8_C(  97), -INT8_C(  90),  INT8_C(  82),
         INT8_C(  91), -INT8_C(  63), -INT8_C(  82),  INT8_C(  15), -INT8_C( 105), -INT8_C(  26), -INT8_C(  18),  INT8_C(  38) },
      UINT32_C( 134243330) },
    { {  INT8_C(  84),  INT8_C(  68),  INT8_C( 123),  INT8_C(  35), -INT8_C(  35), -INT8_C(  27),  INT8_C(  53), -INT8_C(  93),
         INT8_C( 117),  INT8_C(  42),  INT8_C(  62),  INT8_C(  98), -INT8_C(  78),  INT8_C(  91), -INT8_C(  84), -INT8_C( 119),
         INT8_C( 101), -INT8_C(  59),  INT8_C(  35), -INT8_C(  31),  INT8_C( 125), -INT8_C(  88),  INT8_C(  80),  INT8_C(  89),
        -INT8_C(  36), -INT8_C(  51),  INT8_C(  29), -INT8_C(  10),  INT8_C(  57),  INT8_C(  92),  INT8_C( 103), -INT8_C( 115) },
      { -INT8_C(  96), -INT8_C(  30), -INT8_C(  80),  INT8_C( 126),  INT8_C(  28), -INT8_C(  27),  INT8_C(  33), -INT8_C( 111),
         INT8_C(  15),  INT8_C(  95), -INT8_C(  12), -INT8_C(  62), -INT8_C(  70), -INT8_C(  96), -INT8_C(  78), -INT8_C( 119),
         INT8_C( 101), -INT8_C(  43),  INT8_C( 106), -INT8_C(  23),  INT8_C( 125), -INT8_C(  70), -INT8_C(  17),  INT8_C(  89),
        -INT8_C( 120),  INT8_C(  12),  INT8_C(  79), -INT8_C(  63),  INT8_C( 104), -INT8_C(  73),  INT8_C(  78),  INT8_C(   9) },
      UINT32_C(2519618072) },
    { {  INT8_C(   5), -INT8_C( 104),  INT8_C( 105), -INT8_C(  24), -INT8_C(  58), -INT8_C(  80),  INT8_C(  71), -INT8_C(  51),
         INT8_C(  17), -INT8_C(  42), -INT8_C(   4), -INT8_C(  57), -INT8_C(  79),  INT8_C(   4),  INT8_C(   8),  INT8_C(  51),
        -INT8_C( 112),  INT8_C(  50), -INT8_C(  19),  INT8_C(   0),  INT8_C(  41),  INT8_C(  57), -INT8_C(   8),  INT8_C( 112),
        -INT8_C(  22), -INT8_C(  79),  INT8_C(   8),  INT8_C( 124), -INT8_C(  72), -INT8_C( 107),  INT8_C(  47), -INT8_C(  67) },
      { -INT8_C(  77), -INT8_C( 104), -INT8_C(  91),  INT8_C( 121),  INT8_C(  72), -INT8_C( 127),  INT8_C(  71),  INT8_C(  52),
         INT8_C(  17), -INT8_C(  42), -INT8_C(   4),  INT8_C(   9), -INT8_C(  79),  INT8_C(   4),  INT8_C(  60),  INT8_C(  65),
         INT8_C(  54),  INT8_C(  41),  INT8_C(  65), -INT8_C( 127),  INT8_C(  41),  INT8_C(  57), -INT8_C(  15), -INT8_C( 116),
        -INT8_C(  22), -INT8_C(  16),  INT8_C(   8), -INT8_C(  94), -INT8_C( 123),  INT8_C(  55),  INT8_C(  95),  INT8_C(  57) },
      UINT32_C(3879075802) },
    { { -INT8_C( 127),  INT8_C(  85), -INT8_C(  31), -INT8_C(  79),  INT8_C(  79),  INT8_C(  45),  INT8_C(  72), -INT8_C(  64),
         INT8_C( 117), -INT8_C( 113), -INT8_C(  96), -INT8_C(  63), -INT8_C(  15),  INT8_C(  98),  INT8_C(  67), -INT8_C(  54),
         INT8_C(  10),  INT8_C(  70), -INT8_C(   6), -INT8_C(  95),  INT8_C(  58), -INT8_C(  75), -INT8_C(  57),  INT8_C(  31),
         INT8_C( 121), -INT8_C( 113), -INT8_C(  19),  INT8_C( 115), -INT8_C(  74),  INT8_C(  44),  INT8_C(  61), -INT8_C( 102) },
      { -INT8_C( 127),  INT8_C(  30), -INT8_C( 110), -INT8_C(  48),  INT8_C(  75), -INT8_C(  38), -INT8_C(  17), -INT8_C(  64),
        -INT8_C(  62), -INT8_C( 113), -INT8_C( 127), -INT8_C( 121), -INT8_C(  15), -INT8_C(  60),  INT8_C(  81),  INT8_C(  86),
         INT8_C(  10),  INT8_C(  75), -INT8_C(   9),  INT8_C(  68),  INT8_C(   1), -INT8_C(  66),  INT8_C(  99),  INT8_C( 122),
         INT8_C(  77), -INT8_C(  20), -INT8_C(  19),  INT8_C(   3),  INT8_C(  24),  INT8_C(  42), -INT8_C(  27), -INT8_C( 102) },
      UINT32_C(         0) },
    { { -INT8_C( 107), -INT8_C(  47), -INT8_C(  89), -INT8_C(  25), -INT8_C(  82), -INT8_C(   5), -INT8_C(   5), -INT8_C( 105),
        -INT8_C(  49), -INT8_C( 105),  INT8_C( 114),  INT8_C( 104), -INT8_C( 124), -INT8_C(  92),  INT8_C(  10), -INT8_C(  68),
        -INT8_C(  51), -INT8_C(  15), -INT8_C(  10),  INT8_C(  60),  INT8_C(  60), -INT8_C(  57), -INT8_C(  23),  INT8_C( 115),
         INT8_C(  74),  INT8_C(  34), -INT8_C( 114),  INT8_C(  22),  INT8_C(  29), -INT8_C(  84), -INT8_C(   3), -INT8_C(  44) },
      {  INT8_C(   0), -INT8_C(  47), -INT8_C(  69), -INT8_C(  82), -INT8_C(  52), -INT8_C(  73),  INT8_C(  69), -INT8_C( 100),
         INT8_C(  78), -INT8_C(  72),  INT8_C(   4), -INT8_C(  46),  INT8_C(  92),  INT8_C(  14), -INT8_C( 114),  INT8_C(  41),
         INT8_C(   0), -INT8_C( 124), -INT8_C(  35),  INT8_C(  60),  INT8_C(  46), -INT8_C(  57), -INT8_C(  81),  INT8_C( 120),
        -INT8_C(  23),  INT8_C( 113), -INT8_C( 114),  INT8_C(  40),  INT8_C(  29), -INT8_C(  72), -INT8_C(   3),  INT8_C(  29) },
      UINT32_C(2883059709) },
    { { -INT8_C(  71),  INT8_C( 104),  INT8_C( 112),  INT8_C(  41), -INT8_C(  91),  INT8_C(  99), -INT8_C(  26),  INT8_C(  66),
         INT8_C(  89),  INT8_C( 118),  INT8_C( 103),  INT8_C(  94), -INT8_C( 108), -INT8_C(  75),  INT8_C(  99),  INT8_C(  49),
        -INT8_C(  32), -INT8_C(  92),  INT8_C(   7), -INT8_C(  45), -INT8_C( 108),  INT8_C(  80), -INT8_C(  82), -INT8_C(  10),
         INT8_C(  39),  INT8_C( 100),  INT8_C( 117),  INT8_C(  24), -INT8_C(  77),  INT8_C(  17), -INT8_C(  47),  INT8_C( 109) },
      {  INT8_C( 121),  INT8_C(  66), -INT8_C( 106), -INT8_C(  14), -INT8_C(  91),  INT8_C( 124),  INT8_C(  52), -INT8_C(   2),
        -INT8_C(  14), -INT8_C( 101),  INT8_C(  93), -INT8_C( 122),  INT8_C(  80), -INT8_C(  64), -INT8_C(  67),  INT8_C(  49),
         INT8_C( 101), -INT8_C(  60),  INT8_C(   4), -INT8_C(   7),  INT8_C(  20), -INT8_C(  78), -INT8_C(  17),  INT8_C(  59),
         INT8_C( 101),  INT8_C( 100), -INT8_C(  13),  INT8_C(  24),  INT8_C( 118), -INT8_C(  60), -INT8_C( 123), -INT8_C(  17) },
      UINT32_C(3995389854) },
    { {  INT8_C(   8), -INT8_C(  63),  INT8_C(  75), -INT8_C(  96),  INT8_C( 112), -INT8_C(  11),  INT8_C(  43), -INT8_C( 118),
        -INT8_C( 107),  INT8_C(  60),  INT8_C(  48), -INT8_C(  61),  INT8_C(  10), -INT8_C(  64), -INT8_C(  16), -INT8_C(  36),
         INT8_C(  54),  INT8_C(  22),  INT8_C(  66),  INT8_C(  97),  INT8_C(  43),  INT8_C(  35),  INT8_C(  48), -INT8_C( 104),
        -INT8_C(   8), -INT8_C( 104),  INT8_C(  41), -INT8_C( 111),  INT8_C(  17),  INT8_C( 117),  INT8_C(  48), -INT8_C( 115) },
      {  INT8_C(  54),  INT8_C( 123),  INT8_C(  46),  INT8_C(  14),  INT8_C( 112),  INT8_C(  89), -INT8_C( 104),  INT8_C( 108),
        -INT8_C( 107),  INT8_C(  37),  INT8_C(  48), -INT8_C(  97), -INT8_C(  27),  INT8_C(  32),  INT8_C(  59), -INT8_C(  36),
         INT8_C(  54),  INT8_C( 125), -INT8_C(  66),  INT8_C(  97), -INT8_C(  96), -INT8_C(  18),  INT8_C(   7), -INT8_C( 104),
        -INT8_C( 122), -INT8_C( 100),  INT8_C(  41),  INT8_C(  11),  INT8_C(  17),  INT8_C(  90), -INT8_C( 103),  INT8_C(  72) },
      UINT32_C(1634998852) },
    { {  INT8_C( 116), -INT8_C(  30), -INT8_C(  28), -INT8_C(  36), -INT8_C(  36), -INT8_C( 106),  INT8_C(  73), -INT8_C(  16),
         INT8_C( 121), -INT8_C(  74), -INT8_C(  23),  INT8_C( 123),  INT8_C(  44), -INT8_C(  65), -INT8_C(  76),  INT8_C(  56),
         INT8_C( 111),  INT8_C(  78), -INT8_C(  28), -INT8_C(  44), -INT8_C(   2),  INT8_C(  41), -INT8_C( 117),  INT8_C(  44),
        -INT8_C( 104), -INT8_C(  15),  INT8_C( 123),  INT8_C(  96), -INT8_C(  98),  INT8_C(  18), -INT8_C(   2),  INT8_C(  18) },
      {  INT8_C(   0), -INT8_C(  30), -INT8_C(  90), -INT8_C(  36),  INT8_C( 121), -INT8_C(  17), -INT8_C(  51), -INT8_C(  14),
        -INT8_C( 116), -INT8_C(  74),  INT8_C( 109), -INT8_C(  72),  INT8_C( 117),  INT8_C(  33), -INT8_C(  16), -INT8_C(  56),
         INT8_C( 111), -INT8_C(  44), -INT8_C( 100),  INT8_C(  94), -INT8_C(   2),  INT8_C( 121), -INT8_C( 117), -INT8_C( 106),
         INT8_C( 106),  INT8_C(   6), -INT8_C(  10),  INT8_C(   8),  INT8_C(  25), -INT8_C(  11),  INT8_C(  26),  INT8_C(  25) },
                UINT32_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi8(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi8(test_vec[i].b);
    simde__mmask32 r = simde_mm256_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[32];
    int8_t b_[32];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi8(a_);
    simde__m256i b = simde_mm256_loadu_epi8(b_);
    simde__mmask32 r = simde_mm256_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x32(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask32(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { {  INT16_C(  9101), -INT16_C( 20347),  INT16_C( 28106), -INT16_C( 22715), -INT16_C( 11474),  INT16_C( 18560), -INT16_C(  8264), -INT16_C( 13503),
        -INT16_C( 16546),  INT16_C( 30132),  INT16_C( 13426), -INT16_C( 26683),  INT16_C( 22930),  INT16_C(  2747), -INT16_C( 17032),  INT16_C(  1379) },
      { -INT16_C(  5664), -INT16_C( 21578), -INT16_C(  1194),  INT16_C( 21842), -INT16_C( 11474),  INT16_C( 16285), -INT16_C(  8264), -INT16_C( 22262),
        -INT16_C( 16546), -INT16_C( 12258), -INT16_C(  6925), -INT16_C( 13720),  INT16_C( 22930),  INT16_C(  2747),  INT16_C(  7703), -INT16_C(  2288) },
      UINT16_C(12624) },
    { { -INT16_C( 11874),  INT16_C( 24565), -INT16_C( 22330), -INT16_C( 29106),  INT16_C( 32249), -INT16_C( 31065),  INT16_C( 14262), -INT16_C( 20808),
         INT16_C(  6015), -INT16_C( 21863),  INT16_C( 20446), -INT16_C(   408),  INT16_C( 12480),  INT16_C( 20534),  INT16_C( 25864),  INT16_C(  9281) },
      { -INT16_C(  9162), -INT16_C(   763),  INT16_C( 21380),  INT16_C( 32395), -INT16_C( 25904), -INT16_C( 31065),  INT16_C( 24529),  INT16_C( 20532),
        -INT16_C( 12682),  INT16_C( 21755),  INT16_C( 25373), -INT16_C(  8621), -INT16_C( 30317), -INT16_C( 25810),  INT16_C(  5614),  INT16_C(  9281) },
      UINT16_C( 1741) },
    { { -INT16_C( 28886),  INT16_C(  5528),  INT16_C(  3099),  INT16_C( 16758),  INT16_C( 28710),  INT16_C(  8674),  INT16_C( 19349),  INT16_C( 21145),
         INT16_C( 22447),  INT16_C( 30360), -INT16_C( 30864),  INT16_C( 23757), -INT16_C( 14941), -INT16_C( 10847),  INT16_C( 22011),  INT16_C( 30711) },
      { -INT16_C( 28886), -INT16_C( 21108),  INT16_C(  3099),  INT16_C( 16758),  INT16_C( 22652),  INT16_C(  4450), -INT16_C(  2909),  INT16_C( 21145),
         INT16_C( 12620), -INT16_C( 17208), -INT16_C( 26440),  INT16_C( 23757),  INT16_C( 28253),  INT16_C( 22577),  INT16_C( 10435), -INT16_C(  4401) },
      UINT16_C(15501) },
    { { -INT16_C(   862),  INT16_C(  2744),  INT16_C( 17946),  INT16_C( 30702), -INT16_C(  7356), -INT16_C( 25842), -INT16_C(  6799),  INT16_C( 12647),
        -INT16_C(  9715),  INT16_C( 27494),  INT16_C( 32027), -INT16_C( 13331),  INT16_C( 25730), -INT16_C( 20674), -INT16_C( 24662),  INT16_C( 19861) },
      {  INT16_C( 19867), -INT16_C( 22441),  INT16_C( 17946),  INT16_C( 24096), -INT16_C( 23255), -INT16_C( 25842),  INT16_C( 30090), -INT16_C( 26676),
         INT16_C( 30031),  INT16_C( 27494),  INT16_C( 21490),  INT16_C( 29750),  INT16_C( 29879),  INT16_C( 24867), -INT16_C( 18413), -INT16_C( 20818) },
      UINT16_C(    0) },
    { { -INT16_C( 32690),  INT16_C(  4026),  INT16_C( 14836),  INT16_C( 32594), -INT16_C( 27194), -INT16_C( 14155), -INT16_C(  3471), -INT16_C( 28636),
        -INT16_C( 10603), -INT16_C( 29946), -INT16_C( 16696),  INT16_C( 12088),  INT16_C( 18329),  INT16_C( 11432), -INT16_C( 12284), -INT16_C(  2970) },
      { -INT16_C( 32690),  INT16_C( 17110),  INT16_C( 23225),  INT16_C( 32594),  INT16_C(  2287),  INT16_C( 24903),  INT16_C( 24826), -INT16_C( 28636),
         INT16_C( 10806), -INT16_C(   229),  INT16_C( 21736), -INT16_C( 32466), -INT16_C( 10597), -INT16_C( 24658),  INT16_C( 29862), -INT16_C(  2970) },
      UINT16_C(32630) },
    { { -INT16_C(  8345), -INT16_C(   414),  INT16_C( 26142),  INT16_C(  8396),  INT16_C( 27393), -INT16_C( 10874), -INT16_C( 23946), -INT16_C( 21537),
        -INT16_C(  5926), -INT16_C(  9420),  INT16_C( 26911),  INT16_C( 11404),  INT16_C( 20918),  INT16_C( 30944), -INT16_C( 32399), -INT16_C(  7123) },
      { -INT16_C( 28568), -INT16_C( 11806),  INT16_C( 26142),  INT16_C(  8396),  INT16_C( 21201),  INT16_C( 18421), -INT16_C( 11019), -INT16_C( 12301),
        -INT16_C( 17220), -INT16_C(  9420), -INT16_C( 16347), -INT16_C(  9209), -INT16_C(  6126), -INT16_C( 28844), -INT16_C( 32399), -INT16_C(  9869) },
      UINT16_C(65311) },
    { { -INT16_C( 27155), -INT16_C( 22115), -INT16_C(  4852), -INT16_C(  4712),  INT16_C(  3378),  INT16_C( 19348),  INT16_C(  8661),  INT16_C( 23072),
        -INT16_C( 15063),  INT16_C( 26117), -INT16_C( 29816),  INT16_C( 10234),  INT16_C(  7782),  INT16_C( 25976),  INT16_C(  8885), -INT16_C(  5625) },
      { -INT16_C( 12873), -INT16_C( 15541), -INT16_C( 31814), -INT16_C(  4712),  INT16_C( 11408),  INT16_C( 25912),  INT16_C( 22862),  INT16_C( 12736),
        -INT16_C( 15063), -INT16_C( 20073), -INT16_C( 28080), -INT16_C( 18727),  INT16_C(  4528),  INT16_C( 25976), -INT16_C( 22477), -INT16_C(  5625) },
      UINT16_C(23172) },
    { { -INT16_C( 19996),  INT16_C( 24348),  INT16_C( 14466), -INT16_C(  2876),  INT16_C(  4024),  INT16_C( 15284), -INT16_C( 23014),  INT16_C( 27155),
        -INT16_C( 25553),  INT16_C( 24622),  INT16_C( 17322),  INT16_C( 28949),  INT16_C( 17713), -INT16_C( 22249), -INT16_C( 22660),  INT16_C(  1429) },
      { -INT16_C( 19996),  INT16_C( 26212),  INT16_C( 10730),  INT16_C( 30555),  INT16_C(  4024), -INT16_C( 11341), -INT16_C( 14667), -INT16_C(  7107),
         INT16_C( 18530),  INT16_C( 24622),  INT16_C( 17322), -INT16_C(  9007), -INT16_C(  6008),  INT16_C(  1157),  INT16_C(  6799),  INT16_C( 29450) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi16(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi16(test_vec[i].b);
    simde__mmask16 r = simde_mm256_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[16];
    int16_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi16(a_);
    simde__m256i b = simde_mm256_loadu_epi16(b_);
    simde__mmask16 r = simde_mm256_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(  1520606249), -INT32_C(  1370141982), -INT32_C(   676213503), -INT32_C(  1831730832),  INT32_C(    73771260),  INT32_C(  2021947825), -INT32_C(   182757216),  INT32_C(   835755084) },
      { -INT32_C(   948442395),  INT32_C(   896912692),  INT32_C(   168568474),  INT32_C(   597483047),  INT32_C(   925303174),  INT32_C(   447720314), -INT32_C(    32519677),  INT32_C(   835755084) },
      UINT8_C(128) },
    { {  INT32_C(  1166098769), -INT32_C(  1986934796), -INT32_C(  1689957438), -INT32_C(   117824095), -INT32_C(   523595731), -INT32_C(   459884587), -INT32_C(   109648088), -INT32_C(  1637233711) },
      {  INT32_C(  1166098769), -INT32_C(   892464632), -INT32_C(    26930339),  INT32_C(  1727487801), -INT32_C(  1001995793),  INT32_C(   732486776), -INT32_C(   109648088),  INT32_C(    60284850) },
      UINT8_C(174) },
    { { -INT32_C(   410847300),  INT32_C(    20002357), -INT32_C(  2086996020), -INT32_C(  1272421515), -INT32_C(   503266497), -INT32_C(  1670295389), -INT32_C(   481853782), -INT32_C(   566364894) },
      { -INT32_C(    87637780),  INT32_C(    20002357), -INT32_C(  1568355283), -INT32_C(  1571377873), -INT32_C(   503266497), -INT32_C(  1115786989),  INT32_C(   295749103), -INT32_C(  1477386565) },
      UINT8_C(119) },
    { {  INT32_C(  1972662153), -INT32_C(   510760894),  INT32_C(   711556559),  INT32_C(   789323866),  INT32_C(  1930835089),  INT32_C(  1162132442),  INT32_C(   139277582), -INT32_C(  2082194694) },
      { -INT32_C(   319260503), -INT32_C(  1563588896),  INT32_C(   711556559),  INT32_C(   794392222), -INT32_C(  2069729366),  INT32_C(   717874730),  INT32_C(   139277582),  INT32_C(   428618095) },
      UINT8_C(  0) },
    { { -INT32_C(  1816974597), -INT32_C(  1065589462), -INT32_C(  1579764074), -INT32_C(   514329836),  INT32_C(  1490587826), -INT32_C(   871630189),  INT32_C(   192202602),  INT32_C(  1292873298) },
      { -INT32_C(  1816974597), -INT32_C(  1065589462),  INT32_C(   811749916), -INT32_C(   133056186), -INT32_C(   581899959),  INT32_C(  1319722212),  INT32_C(   861478179),  INT32_C(  1292873298) },
      UINT8_C(124) },
    { { -INT32_C(   360494398),  INT32_C(  1445873575),  INT32_C(  1263247980), -INT32_C(  1787611097),  INT32_C(   283670646), -INT32_C(   238122970),  INT32_C(   705609346), -INT32_C(  2136311618) },
      {  INT32_C(  1768566466), -INT32_C(   507537291),  INT32_C(  1630276154), -INT32_C(  1175019709),  INT32_C(  1036705303), -INT32_C(   382822297),  INT32_C(   336804950),  INT32_C(  2006236596) },
      UINT8_C( 98) },
    { { -INT32_C(  1653351929),  INT32_C(   611235194), -INT32_C(  2001592127), -INT32_C(   833111609),  INT32_C(   859552298),  INT32_C(   429036107),  INT32_C(   580650747),  INT32_C(   372610319) },
      { -INT32_C(   374101905), -INT32_C(  2146557603), -INT32_C(  2001592127),  INT32_C(    55970265),  INT32_C(   238457539),  INT32_C(   522766372), -INT32_C(   566049585), -INT32_C(   906725543) },
      UINT8_C(210) },
    { { -INT32_C(   862828029),  INT32_C(   415531994), -INT32_C(   173774531), -INT32_C(   848338478),  INT32_C(   506165109),  INT32_C(  1279116449), -INT32_C(  1561526101),  INT32_C(  1622770780) },
      { -INT32_C(   651408434),  INT32_C(   415531994), -INT32_C(  1106417428),  INT32_C(   495680713),  INT32_C(   506165109),  INT32_C(  1279116449),  INT32_C(   569322180),  INT32_C(  2105649326) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi32(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi32(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi32(a_);
    simde__m256i b = simde_mm256_loadu_epi32(b_);
    simde__mmask8 r = simde_mm256_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 3061706303324098721), -INT64_C( 7075376414461670625), -INT64_C( 8248903588976154960),  INT64_C(  754087591936956394) },
      {  INT64_C(  227787899817509374), -INT64_C( 1923559078301860082), -INT64_C( 2649140287693721396),  INT64_C( 4010940600286325747) },
      UINT8_C(  0) },
    { { -INT64_C( 1626301166014909010), -INT64_C( 7000663457614807716),  INT64_C( 3786838713518589141),  INT64_C( 6290469473638956404) },
      { -INT64_C( 4571573207251538703),  INT64_C(  327285041734901055),  INT64_C( 3786838713518589141),  INT64_C( 7767282717047275210) },
      UINT8_C( 10) },
    { { -INT64_C(  104664725362663840), -INT64_C( 2679084206771791897),  INT64_C( 7360512122289395314),  INT64_C( 7536310226373420634) },
      { -INT64_C( 8778927866836519120), -INT64_C( 5690558540880401134),  INT64_C( 7360512122289395314),  INT64_C( 7536310226373420634) },
      UINT8_C( 12) },
    { {  INT64_C( 1458947386577960448), -INT64_C( 6095539528045473816), -INT64_C( 3407628869727227501),  INT64_C( 3691192106094222744) },
      { -INT64_C( 4500607351733994358), -INT64_C( 6095539528045473816), -INT64_C( 6726885128119193027),  INT64_C( 3691192106094222744) },
      UINT8_C(  0) },
    { { -INT64_C( 7319844171321101507),  INT64_C( 6993771106197164796),  INT64_C( 4155832417833431850), -INT64_C( 4636994022111648454) },
      { -INT64_C( 8906344863618508727), -INT64_C( 5456731580380894176), -INT64_C(  124091139766138754),  INT64_C( 1389883622969095176) },
      UINT8_C( 15) },
    { {  INT64_C( 6215649242152406996), -INT64_C( 2227888630144339662), -INT64_C( 7251913801718489655),  INT64_C( 5678914690226300385) },
      {  INT64_C( 6215649242152406996), -INT64_C( 2227888630144339662), -INT64_C( 3639453938778642927), -INT64_C( 8010442780220248652) },
      UINT8_C( 11) },
    { {  INT64_C( 2495064606853273644), -INT64_C( 8707789476991879884), -INT64_C( 8213802854753458746), -INT64_C( 4291239295976548379) },
      {  INT64_C(  736836437842040077), -INT64_C( 5000272174874936199), -INT64_C( 8539889611890626788),  INT64_C( 2047907544154741262) },
      UINT8_C(  5) },
    { {  INT64_C(  363786789876318920),  INT64_C( 1001922729367468764),  INT64_C( 2617708196441159240), -INT64_C( 4347279376536444313) },
      {  INT64_C(  363786789876318920), -INT64_C( 5809673286333930776),  INT64_C( 2617708196441159240), -INT64_C( 6914029452691079975) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi64(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi64(a_);
    simde__m256i b = simde_mm256_loadu_epi64(b_);
    simde__mmask8 r = simde_mm256_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[64];
    const int8_t b[64];
    const simde__mmask64 r;
  } test_vec[] = {
    { {  INT8_C(  75), -INT8_C(  47), -INT8_C(  35), -INT8_C(  54), -INT8_C(  26), -INT8_C( 123), -INT8_C(  57),  INT8_C(  53),
         INT8_C(  88), -INT8_C(  81),  INT8_C(   5),  INT8_C( 110),  INT8_C(  66),  INT8_C( 102), -INT8_C(  78), -INT8_C(  36),
         INT8_C(  43), -INT8_C( 105), -INT8_C(  97),  INT8_C(  48),  INT8_C(  21),  INT8_C(  36), -INT8_C(  27),  INT8_C( 125),
        -INT8_C(  62), -INT8_C(  12),  INT8_C(  19), -INT8_C(  67),  INT8_C(   6),  INT8_C( 116), -INT8_C(  26), -INT8_C(  81),
         INT8_C(  46), -INT8_C( 116),  INT8_C(  91),  INT8_C( 122), -INT8_C(  35),  INT8_C(  34), -INT8_C(  81),  INT8_C( 106),
        -INT8_C(  54),  INT8_C(  34), -INT8_C(  40),  INT8_C(  85), -INT8_C( 119),  INT8_C(   3),  INT8_C(  49), -INT8_C(  76),
        -INT8_C( 102), -INT8_C(  48),  INT8_C(   4),  INT8_C(  54), -INT8_C(  31), -INT8_C(  23), -INT8_C( 117), -INT8_C(   4),
        -INT8_C(  34), -INT8_C(  98),  INT8_C(  15), -INT8_C(  92), -INT8_C(   5), -INT8_C(  66),  INT8_C(  83),  INT8_C(  41) },
      {  INT8_C(  75), -INT8_C(  52), -INT8_C(  93),  INT8_C(  93), -INT8_C(  26),  INT8_C(  83), -INT8_C(  57), -INT8_C(  80),
         INT8_C( 117), -INT8_C(  97),  INT8_C(   5), -INT8_C(   2),  INT8_C(  66),  INT8_C(  55), -INT8_C(  78),  INT8_C(  99),
         INT8_C(   7), -INT8_C( 105), -INT8_C( 103), -INT8_C(  23), -INT8_C(  33),  INT8_C(  36), -INT8_C(  27),  INT8_C( 125),
        -INT8_C(  62), -INT8_C(  12),  INT8_C(  33), -INT8_C(  67), -INT8_C(  77),  INT8_C( 116), -INT8_C(  26), -INT8_C(   2),
         INT8_C(  64), -INT8_C( 118),  INT8_C(  91),  INT8_C(  38), -INT8_C(  35),  INT8_C(  34), -INT8_C(  41),  INT8_C(  82),
        -INT8_C(  63), -INT8_C(  36),  INT8_C(  81),  INT8_C(   3),  INT8_C(  19),  INT8_C(   3),  INT8_C( 102),  INT8_C(  27),
        -INT8_C( 102), -INT8_C(   1),  INT8_C(   4),  INT8_C( 122),  INT8_C(  36), -INT8_C(  23), -INT8_C(   9), -INT8_C(  26),
        -INT8_C(  34),  INT8_C(  24), -INT8_C(  92), -INT8_C( 111), -INT8_C( 116), -INT8_C( 118), -INT8_C( 113), -INT8_C(  52) },
      UINT64_C(   82507577696605265) },
    { { -INT8_C( 120), -INT8_C(  94), -INT8_C( 113),  INT8_C(  88), -INT8_C( 125),  INT8_C(  18),  INT8_C(  66), -INT8_C(  88),
             INT8_MIN, -INT8_C(   2), -INT8_C(  70), -INT8_C( 118),  INT8_C(   2), -INT8_C(  89), -INT8_C(  83), -INT8_C(  34),
         INT8_C( 109), -INT8_C(  59),  INT8_C( 125), -INT8_C(  31), -INT8_C( 100), -INT8_C(  69), -INT8_C(   3),  INT8_C(  89),
        -INT8_C(  65),  INT8_C( 107),  INT8_C(   3),  INT8_C(  90), -INT8_C(  19),  INT8_C(  49),  INT8_C(  52),  INT8_C( 117),
        -INT8_C(  45),  INT8_C( 102), -INT8_C( 100),  INT8_C(  45), -INT8_C(  42), -INT8_C(  34), -INT8_C(  43),  INT8_C(  59),
        -INT8_C( 102),  INT8_C(  99),  INT8_C(  42), -INT8_C( 100),  INT8_C(  10), -INT8_C( 113),  INT8_C( 108), -INT8_C(  75),
        -INT8_C( 100), -INT8_C(   8), -INT8_C( 106), -INT8_C(  71), -INT8_C(  82),  INT8_C(  75),  INT8_C(  18),  INT8_C( 110),
         INT8_C(  78), -INT8_C(  60),  INT8_C( 116), -INT8_C(  93),  INT8_C(  70),  INT8_C( 109),  INT8_C(  25),  INT8_C(  25) },
      {  INT8_C(  48), -INT8_C(  25),  INT8_C(  71),  INT8_C(   6), -INT8_C( 125),  INT8_C(  28),  INT8_C(  66),  INT8_C(  30),
             INT8_MIN,  INT8_C( 108), -INT8_C(  70), -INT8_C( 118),  INT8_C(  67),  INT8_C(  53),  INT8_C(   2), -INT8_C(  33),
         INT8_C(  45),  INT8_C(  80), -INT8_C( 104), -INT8_C(  31), -INT8_C( 100), -INT8_C(  86), -INT8_C(  65),  INT8_C(  82),
        -INT8_C(  65), -INT8_C(   8), -INT8_C(  10),  INT8_C(   6),  INT8_C( 101),  INT8_C(  15),  INT8_C(  31), -INT8_C( 106),
        -INT8_C(  10),  INT8_C( 102), -INT8_C( 100),  INT8_C( 121), -INT8_C( 125), -INT8_C(  34), -INT8_C( 105),  INT8_C(   3),
         INT8_C(  74),  INT8_C(  82), -INT8_C( 115), -INT8_C( 115), -INT8_C( 121), -INT8_C( 113),  INT8_C( 108), -INT8_C(  75),
        -INT8_C(  32),  INT8_C(   4), -INT8_C( 106),  INT8_C( 124), -INT8_C(  82),  INT8_C(  85), -INT8_C(  50),  INT8_C( 110),
         INT8_C(  78), -INT8_C(  60),  INT8_C( 116), -INT8_C(  77), -INT8_C(  45), -INT8_C( 109),  INT8_C(  73), -INT8_C(  55) },
      UINT64_C( 5200251333164331687) },
    { { -INT8_C( 111),  INT8_C(  44),  INT8_C(  75), -INT8_C(  41), -INT8_C(  66), -INT8_C(  92),  INT8_C(  74), -INT8_C(  12),
        -INT8_C(  52),  INT8_C( 113), -INT8_C(  57), -INT8_C( 104),  INT8_C(   1),  INT8_C(  23), -INT8_C(  92), -INT8_C( 109),
         INT8_C(  96),  INT8_C( 108),  INT8_C( 116),  INT8_C( 112),  INT8_C(  94), -INT8_C(   6),  INT8_C(  13), -INT8_C( 118),
        -INT8_C(  42),  INT8_C(   9), -INT8_C(  81),  INT8_C(  81),  INT8_C(  46), -INT8_C( 100),  INT8_C(  82), -INT8_C(  65),
         INT8_C(  50), -INT8_C(  99), -INT8_C(  57), -INT8_C(  16),  INT8_C(  65),  INT8_C(  78), -INT8_C(  28), -INT8_C(  37),
         INT8_C(  82), -INT8_C(  85),  INT8_C( 115),  INT8_C(  84), -INT8_C(  62),  INT8_C(  23), -INT8_C(  28), -INT8_C(  53),
        -INT8_C(  44),  INT8_C(  88), -INT8_C( 110), -INT8_C(  31),  INT8_C(  82), -INT8_C(  97), -INT8_C(  49),  INT8_C( 101),
         INT8_C(  95),  INT8_C( 126),  INT8_C( 122), -INT8_C(  42), -INT8_C( 104), -INT8_C(  52), -INT8_C( 107),  INT8_C(  77) },
      {  INT8_C( 106),  INT8_C(  44),  INT8_C(  61), -INT8_C(  85),  INT8_C(  13),  INT8_C(  33), -INT8_C( 122),  INT8_C(  95),
        -INT8_C(  52), -INT8_C(   7), -INT8_C(  77), -INT8_C( 114),  INT8_C(  16), -INT8_C( 104), -INT8_C(  80), -INT8_C( 109),
        -INT8_C(  16),  INT8_C(  66),  INT8_C( 116),  INT8_C(  67), -INT8_C(  30),  INT8_C(  67),  INT8_C( 107), -INT8_C( 118),
        -INT8_C(  62), -INT8_C(  27),  INT8_C(  97), -INT8_C(  35), -INT8_C(  78), -INT8_C(  10),  INT8_C(  42),  INT8_C(  28),
         INT8_C(  34),  INT8_C( 103), -INT8_C(  57),  INT8_C(  47), -INT8_C( 120),  INT8_C(  78), -INT8_C( 113),  INT8_C(  84),
         INT8_C(  71),  INT8_C(  66), -INT8_C(  30),  INT8_C(  88), -INT8_C(  38), -INT8_C( 110), -INT8_C(  21), -INT8_C(  53),
        -INT8_C(  44),  INT8_C(  96),  INT8_C(  14), -INT8_C(  74), -INT8_C(  93),  INT8_C( 121),  INT8_C(  65),  INT8_C( 101),
         INT8_C(  95), -INT8_C(  94),  INT8_C(  66),  INT8_C(  17), -INT8_C( 104),  INT8_C( 108),  INT8_C(  45), -INT8_C(  69) },
      UINT64_C( 8784230041835065779) },
    { {      INT8_MAX,  INT8_C(  87), -INT8_C(  76), -INT8_C(  56),  INT8_C(  27), -INT8_C(   5), -INT8_C(  51), -INT8_C(  90),
         INT8_C(  36),  INT8_C(  70), -INT8_C(  22),  INT8_C(  26), -INT8_C(  95),  INT8_C(  91),  INT8_C( 123), -INT8_C(  46),
         INT8_C(   4),  INT8_C(  51),  INT8_C(  39), -INT8_C( 119),  INT8_C(  95), -INT8_C(  59),  INT8_C(  72),  INT8_C(   3),
        -INT8_C(  69),      INT8_MIN, -INT8_C( 115),  INT8_C(  56), -INT8_C(  25), -INT8_C(  48),  INT8_C(  47),  INT8_C(  35),
         INT8_C(  40), -INT8_C(  29), -INT8_C(  21),  INT8_C(  67), -INT8_C(  34), -INT8_C(  72), -INT8_C(  76),  INT8_C(   2),
         INT8_C(  32), -INT8_C(  98),  INT8_C(  28), -INT8_C( 122), -INT8_C(   7),  INT8_C(  91), -INT8_C( 101),  INT8_C( 116),
        -INT8_C( 114),      INT8_MAX, -INT8_C(   3), -INT8_C(  19),  INT8_C(  68),  INT8_C(  39), -INT8_C(  15),  INT8_C(   0),
         INT8_C(  28),  INT8_C( 126),  INT8_C(  48), -INT8_C(  64), -INT8_C(  69),  INT8_C( 103), -INT8_C(  29),  INT8_C( 119) },
      {  INT8_C(  74), -INT8_C(  50), -INT8_C(  70),  INT8_C(  40), -INT8_C( 122),  INT8_C( 111),  INT8_C(  42), -INT8_C(  90),
         INT8_C(  13),  INT8_C(  70),  INT8_C(  44),  INT8_C(   7), -INT8_C(  95), -INT8_C( 124),  INT8_C( 123),  INT8_C(  48),
         INT8_C(   4),  INT8_C( 120),  INT8_C(  29),  INT8_C(  72), -INT8_C(  96),  INT8_C(  14),  INT8_C(  72), -INT8_C(  68),
        -INT8_C( 115),      INT8_MIN,  INT8_C( 124), -INT8_C(  36), -INT8_C(  25),  INT8_C(  95),  INT8_C(  83),  INT8_C(  49),
         INT8_C(  45),  INT8_C(  13),  INT8_C(  89), -INT8_C(  77),  INT8_C( 124), -INT8_C( 125),  INT8_C(  89), -INT8_C( 118),
        -INT8_C(  54), -INT8_C( 122), -INT8_C( 111),  INT8_C( 107),  INT8_C(  10),  INT8_C(  12), -INT8_C( 101),  INT8_C(  14),
        -INT8_C( 124), -INT8_C(  71),  INT8_C(  87),  INT8_C(  36), -INT8_C(  57), -INT8_C(  97), -INT8_C(  32),  INT8_C(  84),
         INT8_C(  32),  INT8_C(  92),  INT8_C(  48),  INT8_C(   7), -INT8_C(  69), -INT8_C( 125),  INT8_C(  57), -INT8_C(  24) },
      UINT64_C(                   0) },
    { {  INT8_C(  96), -INT8_C(  43), -INT8_C(  79),  INT8_C(   5),  INT8_C( 123),  INT8_C(  37),  INT8_C( 121),  INT8_C(  74),
        -INT8_C(  28), -INT8_C( 101),  INT8_C(  77), -INT8_C( 118), -INT8_C(  84),  INT8_C(  86), -INT8_C(  48), -INT8_C(  40),
        -INT8_C(  94),  INT8_C( 125), -INT8_C(  44), -INT8_C(   6), -INT8_C(  25), -INT8_C(  41),  INT8_C( 108),  INT8_C(  75),
        -INT8_C(  93), -INT8_C(  76),  INT8_C(   3),  INT8_C(  45), -INT8_C(  70), -INT8_C(  54), -INT8_C(  33),  INT8_C(  27),
        -INT8_C(  97), -INT8_C(   3), -INT8_C(  54),  INT8_C(  27),  INT8_C(  34),  INT8_C(  68), -INT8_C(  91),  INT8_C(   6),
         INT8_C(  13), -INT8_C(  78), -INT8_C( 112), -INT8_C(  71), -INT8_C(  69),  INT8_C(  10),  INT8_C(  52),  INT8_C(  93),
        -INT8_C(  35),  INT8_C(  43), -INT8_C(  68), -INT8_C(  19), -INT8_C(  32), -INT8_C(  61),  INT8_C(  56),  INT8_C( 109),
         INT8_C( 119), -INT8_C(  75), -INT8_C( 102),  INT8_C(  50), -INT8_C(  50), -INT8_C(  26),  INT8_C(  77),  INT8_C( 110) },
      { -INT8_C(  29),  INT8_C(  23), -INT8_C( 119),  INT8_C(   5),  INT8_C(  91), -INT8_C(  18),  INT8_C(  11),  INT8_C( 104),
        -INT8_C(  96), -INT8_C( 101),  INT8_C(  34),  INT8_C(  91), -INT8_C(   5),  INT8_C(  86), -INT8_C(  72), -INT8_C(  40),
         INT8_C(  95),  INT8_C(  16), -INT8_C(  99),  INT8_C(  63), -INT8_C(  45), -INT8_C(  98), -INT8_C(  84),  INT8_C(  75),
        -INT8_C(  93),  INT8_C(  70),  INT8_C( 125),  INT8_C( 113),  INT8_C(  44), -INT8_C(  54), -INT8_C(  33),  INT8_C(  15),
        -INT8_C(  31),  INT8_C( 104),  INT8_C(  20),  INT8_C(  61),  INT8_C(  86),  INT8_C(  31), -INT8_C(  91), -INT8_C(  10),
        -INT8_C(  70), -INT8_C(  57),  INT8_C(  81), -INT8_C(  75),  INT8_C(  30),  INT8_C(  10), -INT8_C( 114),  INT8_C( 125),
         INT8_C(  26),  INT8_C(  43), -INT8_C(  68), -INT8_C(  19), -INT8_C(  55),  INT8_C( 104),  INT8_C(  56),  INT8_C( 108),
        -INT8_C(  82), -INT8_C(  75), -INT8_C(  34), -INT8_C(  37),      INT8_MAX, -INT8_C(  67), -INT8_C(  22),  INT8_C(  97) },
      UINT64_C(18280638376564448759) },
    { {  INT8_C( 125),  INT8_C(  36),  INT8_C(  46), -INT8_C(  85),  INT8_C(  83),  INT8_C(  99), -INT8_C( 108),  INT8_C(  70),
         INT8_C(  38),  INT8_C(  78),  INT8_C(   7), -INT8_C(  78), -INT8_C(  82),  INT8_C(  25),  INT8_C(  44),  INT8_C( 112),
         INT8_C(  49),  INT8_C( 100), -INT8_C( 122), -INT8_C(  41),  INT8_C(  26),  INT8_C(   0), -INT8_C( 113),  INT8_C(  43),
         INT8_C(  88),  INT8_C(  54),  INT8_C(  85), -INT8_C(  18), -INT8_C(  29), -INT8_C(  66), -INT8_C(  63),  INT8_C(  96),
        -INT8_C(  30), -INT8_C(  16),  INT8_C(  99),  INT8_C(  69),  INT8_C(  83), -INT8_C(  59),  INT8_C( 123),  INT8_C( 121),
         INT8_C( 103), -INT8_C(  66),  INT8_C( 126), -INT8_C(  13),  INT8_C(  52), -INT8_C(  85),  INT8_C(  99),  INT8_C( 102),
         INT8_C(  15),  INT8_C(  95),  INT8_C(  38),  INT8_C(  41),  INT8_C(  95),  INT8_C(  56),  INT8_C(  84), -INT8_C(  72),
        -INT8_C(  31), -INT8_C(  87), -INT8_C(  28), -INT8_C(  60),  INT8_C(  99),  INT8_C( 104),      INT8_MIN,  INT8_C(  74) },
      {  INT8_C(  88), -INT8_C( 121),      INT8_MAX, -INT8_C(  85),  INT8_C( 126), -INT8_C(   5),  INT8_C(  36), -INT8_C(  61),
         INT8_C( 126), -INT8_C(  94), -INT8_C(  74), -INT8_C(  78),  INT8_C(  77),  INT8_C(  25),  INT8_C(  24),  INT8_C(  93),
         INT8_C( 120),  INT8_C(  62), -INT8_C( 122), -INT8_C(  41),  INT8_C( 119), -INT8_C(  37), -INT8_C( 113),  INT8_C(  88),
        -INT8_C( 124),  INT8_C(  54),  INT8_C(  28), -INT8_C(  20), -INT8_C(  98),  INT8_C(  64),  INT8_C(  54), -INT8_C(  10),
        -INT8_C(  57), -INT8_C(  75), -INT8_C(  95),  INT8_C(  69), -INT8_C(  80), -INT8_C(  59),  INT8_C(   8),  INT8_C(  46),
         INT8_C( 103), -INT8_C(  66), -INT8_C(  31), -INT8_C(  75), -INT8_C(  41), -INT8_C(   7),  INT8_C(  18),  INT8_C(  79),
         INT8_C(  56), -INT8_C( 104),  INT8_C(  38), -INT8_C(  81),  INT8_C( 115), -INT8_C(  74),  INT8_C(   7), -INT8_C(   8),
        -INT8_C(  20),  INT8_C(  35), -INT8_C(  28), -INT8_C( 118),  INT8_C(  99),  INT8_C(  26),      INT8_MIN,  INT8_C(  42) },
      UINT64_C(18189722233980513963) },
    { {  INT8_C(  41), -INT8_C(  34), -INT8_C(  29),  INT8_C(  54), -INT8_C(  14),  INT8_C( 101),  INT8_C( 120), -INT8_C( 106),
        -INT8_C(  22),  INT8_C(  36), -INT8_C(  61), -INT8_C( 125),  INT8_C( 111), -INT8_C(  79),  INT8_C( 122), -INT8_C(  40),
         INT8_C(  15),  INT8_C(  47), -INT8_C(   5), -INT8_C(  28), -INT8_C(  43), -INT8_C( 127),  INT8_C(  83),  INT8_C(  42),
        -INT8_C(  76), -INT8_C(  65), -INT8_C(  68),  INT8_C(  20),  INT8_C(  82),  INT8_C(  52), -INT8_C(  61),  INT8_C( 123),
        -INT8_C(  47), -INT8_C(  89), -INT8_C(  79), -INT8_C( 111),  INT8_C(  12), -INT8_C(  38), -INT8_C( 101), -INT8_C(  10),
        -INT8_C(  17),  INT8_C(  94), -INT8_C(  77),  INT8_C(  94),  INT8_C(  16),  INT8_C(  66), -INT8_C(   8),  INT8_C(  31),
         INT8_C( 114),  INT8_C(  54),  INT8_C(   4),  INT8_C(  32),  INT8_C( 116),  INT8_C(  83),  INT8_C( 109),  INT8_C(  40),
         INT8_C(  23),  INT8_C(   6),  INT8_C(  61),  INT8_C( 105), -INT8_C(   2),  INT8_C(  25), -INT8_C(  27),  INT8_C(  76) },
      { -INT8_C(  89), -INT8_C( 106),  INT8_C(  81), -INT8_C(  76), -INT8_C(  64), -INT8_C(  20), -INT8_C(  86), -INT8_C(  81),
         INT8_C(  74),  INT8_C(  36),  INT8_C(  14),  INT8_C(  90),  INT8_C( 102),  INT8_C(   6),  INT8_C( 122), -INT8_C(  40),
        -INT8_C(   7),  INT8_C( 126), -INT8_C(   7),  INT8_C( 110), -INT8_C(  43),  INT8_C(  67), -INT8_C( 106), -INT8_C(  20),
         INT8_C(  74), -INT8_C(  45),  INT8_C(  86), -INT8_C( 124), -INT8_C(  44),  INT8_C(  59), -INT8_C(  47),  INT8_C( 123),
        -INT8_C(  47),  INT8_C(  34),  INT8_C(  47), -INT8_C( 111),  INT8_C(  14), -INT8_C(  38),  INT8_C(  65),  INT8_C(  88),
        -INT8_C(   2),  INT8_C(  79), -INT8_C(  77),  INT8_C( 100),  INT8_C(  85),  INT8_C(  45),  INT8_C(  61),  INT8_C(  78),
        -INT8_C(  85),  INT8_C(  54), -INT8_C(  68),      INT8_MIN,  INT8_C( 121),  INT8_C(  83),  INT8_C( 109), -INT8_C(  61),
         INT8_C(  38), -INT8_C(  61),  INT8_C(  72), -INT8_C(   6), -INT8_C(   2),  INT8_C(  25),  INT8_C( 118), -INT8_C(  49) },
      UINT64_C( 9983673332761170043) },
    { {  INT8_C(  80),  INT8_C(   6),  INT8_C( 121), -INT8_C(   3),  INT8_C(  30),  INT8_C(  86), -INT8_C(  72), -INT8_C( 117),
        -INT8_C(  20),  INT8_C(  75),  INT8_C( 122), -INT8_C(  12),  INT8_C(   6), -INT8_C( 107),  INT8_C(  40), -INT8_C(  61),
         INT8_C(  93),  INT8_C(  42),  INT8_C(  49),  INT8_C(  63), -INT8_C(  66),  INT8_C( 106), -INT8_C(   2),  INT8_C(  44),
        -INT8_C(  71), -INT8_C( 104), -INT8_C( 115),  INT8_C(  49), -INT8_C(  36),  INT8_C(  28), -INT8_C(  71),  INT8_C(  44),
         INT8_C(  34),  INT8_C(  50),  INT8_C(  28),  INT8_C(  64),  INT8_C(  76), -INT8_C(  44), -INT8_C(  52),  INT8_C(  57),
         INT8_C(  32),  INT8_C(  70), -INT8_C( 109),  INT8_C(  64), -INT8_C(  37), -INT8_C(  69),  INT8_C(  18),  INT8_C(  56),
        -INT8_C(  27),  INT8_C(  53),  INT8_C( 119), -INT8_C(  93), -INT8_C(   2),  INT8_C(  58),  INT8_C(  90), -INT8_C(  73),
         INT8_C(  13),  INT8_C(  92), -INT8_C(  90), -INT8_C(  68),  INT8_C(  52),  INT8_C(  95),  INT8_C(  22), -INT8_C( 102) },
      { -INT8_C( 111),  INT8_C(  64), -INT8_C(  38),  INT8_C(  26),  INT8_C(   6), -INT8_C(  90), -INT8_C(  72),  INT8_C(  76),
        -INT8_C(  20),  INT8_C(  75), -INT8_C( 116), -INT8_C(  57),  INT8_C(   6), -INT8_C( 112), -INT8_C(   1), -INT8_C(  21),
        -INT8_C(  59),  INT8_C( 118), -INT8_C( 114),  INT8_C( 100), -INT8_C(  21),  INT8_C(  93),  INT8_C( 106), -INT8_C(   8),
        -INT8_C(  71),  INT8_C(  17), -INT8_C(  30),  INT8_C(  49),  INT8_C( 112), -INT8_C(   8), -INT8_C(  53),  INT8_C(   2),
         INT8_C(  56), -INT8_C(  90),  INT8_C(  28),  INT8_C(  62),  INT8_C(  76), -INT8_C(  44), -INT8_C( 118),  INT8_C(  57),
         INT8_C(  32),  INT8_C(  23),  INT8_C(   0),  INT8_C(  38), -INT8_C(  89),  INT8_C(   0),  INT8_C(  18),  INT8_C( 108),
         INT8_C( 118), -INT8_C(  96), -INT8_C(  48),  INT8_C(  98), -INT8_C(   2),  INT8_C(  58),  INT8_C(  90), -INT8_C(  73),
         INT8_C(  75),  INT8_C(  60), -INT8_C(  23), -INT8_C(  68),  INT8_C(  52), -INT8_C(  76), -INT8_C(  66),  INT8_C( 108) },
                         UINT64_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi8(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi8(test_vec[i].b);
    simde__mmask64 r = simde_mm512_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[64];
    int8_t b_[64];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi8(a_);
    simde__m512i b = simde_mm512_loadu_epi8(b_);
    simde__mmask64 r = simde_mm512_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x64(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x64(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask64(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const simde__mmask32 r;
  } test_vec[] = {
    { { -INT16_C( 26663),  INT16_C( 31647), -INT16_C(  3155), -INT16_C(  4891),  INT16_C( 22831), -INT16_C( 24110), -INT16_C( 21561), -INT16_C(  9836),
         INT16_C( 21550),  INT16_C( 13739), -INT16_C( 24088),  INT16_C( 17127),  INT16_C( 13677), -INT16_C(  1207), -INT16_C(  3618), -INT16_C( 18423),
        -INT16_C( 17743), -INT16_C(  2608),  INT16_C(  6300), -INT16_C( 13534), -INT16_C(  2959),  INT16_C( 14700),  INT16_C(   160), -INT16_C( 12782),
        -INT16_C( 32428), -INT16_C( 28335),  INT16_C( 11699),  INT16_C(  8438),  INT16_C( 16226),  INT16_C( 16668),  INT16_C(  9521), -INT16_C( 17927) },
      {  INT16_C( 11470),  INT16_C( 27375),  INT16_C(  4676), -INT16_C( 19146), -INT16_C( 24058), -INT16_C( 22802),  INT16_C(   419), -INT16_C(  2188),
        -INT16_C( 17534),  INT16_C( 13739), -INT16_C( 24088),  INT16_C( 19285),  INT16_C( 29152),  INT16_C(  4492), -INT16_C( 31337),  INT16_C( 26059),
        -INT16_C( 17743), -INT16_C(  2608),  INT16_C(  1740), -INT16_C( 11350), -INT16_C( 26200),  INT16_C( 19321), -INT16_C(  4454),  INT16_C(  7235),
        -INT16_C(  4439), -INT16_C( 28335), -INT16_C( 22897),  INT16_C( 28636),  INT16_C( 26648), -INT16_C( 20607),  INT16_C( 19693), -INT16_C( 25068) },
      UINT32_C(  33752576) },
    { {  INT16_C( 32375), -INT16_C( 24025), -INT16_C( 24616),  INT16_C( 17685),  INT16_C( 22080),  INT16_C(   730),  INT16_C(  6165), -INT16_C( 19501),
         INT16_C( 17310),  INT16_C( 12972), -INT16_C( 27711),  INT16_C( 23888),  INT16_C( 30363), -INT16_C(  4850), -INT16_C( 32515),  INT16_C( 29878),
        -INT16_C( 27650), -INT16_C( 10538), -INT16_C(  1485),  INT16_C(  2153),  INT16_C(   848),  INT16_C( 25973),  INT16_C( 18459), -INT16_C( 18152),
        -INT16_C(  7227),  INT16_C( 19713),  INT16_C(  2987),  INT16_C(  7708), -INT16_C( 11597), -INT16_C( 20442), -INT16_C(  9134),  INT16_C( 20517) },
      { -INT16_C(  1169), -INT16_C( 24025),  INT16_C( 20725),  INT16_C( 17685), -INT16_C( 30125),  INT16_C( 28331), -INT16_C( 15405),  INT16_C( 24359),
         INT16_C( 10417),  INT16_C( 12972), -INT16_C( 14166),  INT16_C( 23888),  INT16_C( 30363), -INT16_C(  4850),  INT16_C( 13138), -INT16_C( 15810),
         INT16_C( 25902),  INT16_C(  9316),  INT16_C( 31413),  INT16_C(  2153),  INT16_C(  5124), -INT16_C( 10378), -INT16_C( 25128), -INT16_C( 30410),
        -INT16_C(  7227),  INT16_C( 28603),  INT16_C(  2987),  INT16_C( 18125), -INT16_C(  9342), -INT16_C( 11212),  INT16_C( 29198),  INT16_C( 15510) },
      UINT32_C(2048345252) },
    { {  INT16_C( 25775), -INT16_C( 22282),  INT16_C(  9022),  INT16_C( 10385),  INT16_C(  3584), -INT16_C( 19189),  INT16_C(  8923), -INT16_C( 23595),
         INT16_C(  2581), -INT16_C( 20111), -INT16_C( 14436), -INT16_C( 16431), -INT16_C( 31878),  INT16_C(  8164),  INT16_C(  7449), -INT16_C( 19533),
        -INT16_C( 22202), -INT16_C( 23460), -INT16_C(  9092),  INT16_C( 32005),  INT16_C(  4330), -INT16_C(  6186),  INT16_C(  1843),  INT16_C( 18537),
        -INT16_C(  9710), -INT16_C( 23503), -INT16_C( 13406),  INT16_C(  7277),  INT16_C( 20815),  INT16_C( 21307),  INT16_C(  3485),  INT16_C( 31238) },
      {  INT16_C( 25240),  INT16_C(  5150),  INT16_C(  9022),  INT16_C( 10385), -INT16_C( 15308),  INT16_C( 26606),  INT16_C( 22475), -INT16_C(  8785),
        -INT16_C( 22223), -INT16_C( 11380), -INT16_C(  1675), -INT16_C( 15121),  INT16_C( 11083),  INT16_C( 32535),  INT16_C(  7449), -INT16_C( 19975),
         INT16_C(  6272), -INT16_C( 16698),  INT16_C( 22331),  INT16_C( 28647), -INT16_C( 10981), -INT16_C(  6186), -INT16_C( 31188),  INT16_C( 24004),
         INT16_C( 20527), -INT16_C( 23503),  INT16_C(  8266), -INT16_C( 27288),  INT16_C( 32587),  INT16_C( 25876),  INT16_C(  3485),  INT16_C(  7446) },
      UINT32_C(2007465710) },
    { { -INT16_C( 10798),  INT16_C(  4092),  INT16_C( 25518), -INT16_C( 27928), -INT16_C( 31643),  INT16_C( 29865), -INT16_C( 26633),  INT16_C( 20551),
        -INT16_C(  3151), -INT16_C(  4143), -INT16_C( 11810), -INT16_C( 12693), -INT16_C( 16676),  INT16_C(  3903), -INT16_C( 19245), -INT16_C(  7743),
        -INT16_C( 24951), -INT16_C(    15), -INT16_C( 22270), -INT16_C(   190), -INT16_C( 26019), -INT16_C( 24855),  INT16_C( 12406),  INT16_C( 10222),
        -INT16_C( 30527), -INT16_C(  5098),  INT16_C( 27481),  INT16_C( 17611),  INT16_C(  2799), -INT16_C(   722),  INT16_C(  5404), -INT16_C( 11292) },
      { -INT16_C( 10798),  INT16_C(  9682),  INT16_C( 25518),  INT16_C(  3038), -INT16_C( 14339),  INT16_C( 29865), -INT16_C( 26633),  INT16_C(  6811),
        -INT16_C( 19882),  INT16_C(  9990), -INT16_C( 11810), -INT16_C( 12693), -INT16_C( 16676), -INT16_C(  1840), -INT16_C( 19245), -INT16_C( 22836),
        -INT16_C( 24951),  INT16_C( 14283), -INT16_C( 22270), -INT16_C(   190), -INT16_C(  5264),  INT16_C( 26483),  INT16_C(  3970), -INT16_C(  9855),
        -INT16_C( 30527), -INT16_C( 24832),  INT16_C( 27481),  INT16_C( 13677),  INT16_C( 15913), -INT16_C(   722), -INT16_C(  1294),  INT16_C( 31907) },
      UINT32_C(         0) },
    { {  INT16_C( 18961),  INT16_C( 21137),  INT16_C(  4808),  INT16_C( 21364), -INT16_C(  1406),  INT16_C( 25850),  INT16_C( 32701), -INT16_C( 17088),
         INT16_C( 28289), -INT16_C( 25053), -INT16_C( 29064), -INT16_C( 27773), -INT16_C( 19674), -INT16_C(  8015),  INT16_C( 20070), -INT16_C( 29831),
         INT16_C( 13646),  INT16_C( 10973),  INT16_C( 20765),  INT16_C( 10813),  INT16_C( 30795),  INT16_C(  2052),  INT16_C( 17655),  INT16_C( 31173),
         INT16_C( 10930), -INT16_C( 10009),  INT16_C( 27145), -INT16_C( 28902),  INT16_C( 17965), -INT16_C( 26865), -INT16_C(  7787), -INT16_C(  7282) },
      {  INT16_C(   148), -INT16_C( 20031),  INT16_C( 15953), -INT16_C( 25264),  INT16_C( 21686), -INT16_C( 20827),  INT16_C( 27544),  INT16_C( 19239),
         INT16_C(  3733), -INT16_C( 25053), -INT16_C( 29064), -INT16_C( 27186), -INT16_C(  8790), -INT16_C(  8660),  INT16_C( 20070), -INT16_C(  1420),
         INT16_C( 13646), -INT16_C( 24405), -INT16_C(   908),  INT16_C( 10813), -INT16_C(  7600), -INT16_C(  5672), -INT16_C(   179), -INT16_C(  7372),
         INT16_C( 22285), -INT16_C( 31359),  INT16_C( 20453), -INT16_C( 28902),  INT16_C( 17965), -INT16_C( 27795), -INT16_C(  7787), -INT16_C(  7282) },
      UINT32_C( 670480895) },
    { {  INT16_C(  2613),  INT16_C( 29526), -INT16_C( 12959), -INT16_C( 16300),  INT16_C(  9877), -INT16_C( 28688), -INT16_C(  7848), -INT16_C( 14973),
         INT16_C( 23009), -INT16_C( 13469),  INT16_C(  9481), -INT16_C( 19597),  INT16_C( 12712),  INT16_C( 28812),  INT16_C( 14569),  INT16_C( 31816),
        -INT16_C(  3305),  INT16_C( 24978), -INT16_C(  6406), -INT16_C( 24603),  INT16_C(  4364), -INT16_C( 10465), -INT16_C( 23860),  INT16_C( 29084),
         INT16_C(   249), -INT16_C( 14430), -INT16_C( 20269),  INT16_C( 31812), -INT16_C( 12063), -INT16_C( 13588),  INT16_C( 20744),  INT16_C( 19176) },
      {  INT16_C( 31245),  INT16_C(  1963), -INT16_C( 12959),  INT16_C( 28054), -INT16_C( 18978), -INT16_C( 21947), -INT16_C(  7848),  INT16_C( 20764),
         INT16_C( 23009), -INT16_C( 18975),  INT16_C(  9481), -INT16_C(  5583),  INT16_C(  7669), -INT16_C(   588), -INT16_C( 25489),  INT16_C( 31816),
        -INT16_C(  3305),  INT16_C( 30851),  INT16_C(  6592), -INT16_C( 24603),  INT16_C( 10959),  INT16_C( 10057),  INT16_C( 25868), -INT16_C(  4744),
         INT16_C( 22974), -INT16_C( 14430), -INT16_C( 11393),  INT16_C( 29873),  INT16_C( 26097),  INT16_C( 24690), -INT16_C( 17918),  INT16_C(  6620) },
      UINT32_C(3398039382) },
    { {  INT16_C( 29047), -INT16_C(  3325),  INT16_C( 24825), -INT16_C( 10481), -INT16_C(  9941), -INT16_C(  9102), -INT16_C( 30915), -INT16_C(  8498),
         INT16_C( 11829), -INT16_C( 19804),  INT16_C( 16994), -INT16_C(  4916), -INT16_C( 13737), -INT16_C( 22059),  INT16_C(  7191), -INT16_C( 14485),
        -INT16_C( 25796),  INT16_C( 13735), -INT16_C( 18437),  INT16_C(  3789),  INT16_C( 32400), -INT16_C( 13054),  INT16_C( 28439), -INT16_C(  7253),
         INT16_C( 20382), -INT16_C( 20212), -INT16_C( 10115), -INT16_C( 11107),  INT16_C( 29346), -INT16_C(  8323),  INT16_C(  5181),  INT16_C( 31124) },
      {  INT16_C( 15279), -INT16_C( 21842), -INT16_C( 17422), -INT16_C( 32048), -INT16_C( 11463),  INT16_C( 16207), -INT16_C(  1373), -INT16_C(  3550),
         INT16_C( 11829), -INT16_C( 19804),  INT16_C( 16646), -INT16_C( 22138),  INT16_C(   948), -INT16_C(  3704),  INT16_C(  7191), -INT16_C( 14485),
         INT16_C(  6488),  INT16_C( 19057),  INT16_C( 17108),  INT16_C(  3789),  INT16_C(  7189), -INT16_C( 18355),  INT16_C( 28439),  INT16_C( 19627),
         INT16_C( 20382), -INT16_C( 23298), -INT16_C( 31600),  INT16_C( 17485), -INT16_C( 10617), -INT16_C( 25034), -INT16_C( 24078),  INT16_C( 19045) },
      UINT32_C(4130343967) },
    { {  INT16_C(  8243),  INT16_C(  9778),  INT16_C( 16019), -INT16_C(  3944), -INT16_C( 24527), -INT16_C( 12553),  INT16_C( 28428), -INT16_C( 31409),
        -INT16_C(  3681), -INT16_C(  4229), -INT16_C( 27524), -INT16_C(  3131),  INT16_C( 27187), -INT16_C( 15881), -INT16_C( 25152),  INT16_C( 16149),
         INT16_C( 18274), -INT16_C(  2715), -INT16_C(   379), -INT16_C( 18458), -INT16_C( 13673),  INT16_C( 10093), -INT16_C( 30494), -INT16_C( 32392),
        -INT16_C(  2439), -INT16_C(  2733),  INT16_C(  6538), -INT16_C( 16919),  INT16_C( 13671), -INT16_C( 28801), -INT16_C( 27615), -INT16_C( 31794) },
      {  INT16_C( 13531),  INT16_C( 24697),  INT16_C( 24370), -INT16_C( 12265), -INT16_C( 31508), -INT16_C( 12553),  INT16_C( 28428), -INT16_C( 31409),
        -INT16_C( 23963), -INT16_C(  4229),  INT16_C( 25787),  INT16_C( 16301),  INT16_C( 11332),  INT16_C( 26062), -INT16_C( 25152), -INT16_C( 25623),
         INT16_C( 25297),  INT16_C(  1019),  INT16_C(  5057), -INT16_C( 21037), -INT16_C( 13673), -INT16_C( 23429), -INT16_C( 13767), -INT16_C( 24791),
        -INT16_C( 23444),  INT16_C( 10382),  INT16_C( 15112),  INT16_C( 19559),  INT16_C( 13671),  INT16_C( 10162), -INT16_C( 25646), -INT16_C( 23614) },
                UINT32_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi16(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi16(test_vec[i].b);
    simde__mmask32 r = simde_mm512_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[32];
    int16_t b_[32];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi16(a_);
    simde__m512i b = simde_mm512_loadu_epi16(b_);
    simde__mmask32 r = simde_mm512_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x32(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask32(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { { -INT32_C(  1234438653),  INT32_C(  1498911642), -INT32_C(  1552866568),  INT32_C(  1529431642),  INT32_C(   839041684), -INT32_C(  1606076101), -INT32_C(  1385333212), -INT32_C(  1959327209),
         INT32_C(  1668099529),  INT32_C(  1973279853), -INT32_C(    44290330),  INT32_C(  1875525395),  INT32_C(  1936815799), -INT32_C(  2030508190), -INT32_C(   298621595),  INT32_C(  1081843288) },
      { -INT32_C(   487145868),  INT32_C(  1498911642), -INT32_C(  1552866568), -INT32_C(   840702698), -INT32_C(  1103080356), -INT32_C(  1606076101), -INT32_C(   158369634), -INT32_C(  1959327209),
         INT32_C(  1668099529), -INT32_C(  1178811200), -INT32_C(    44290330),  INT32_C(  1875525395), -INT32_C(   819065964), -INT32_C(   495946940),  INT32_C(    47841259), -INT32_C(   762441719) },
      UINT16_C( 3494) },
    { {  INT32_C(  1452446509), -INT32_C(  1717521140), -INT32_C(  1351082373),  INT32_C(   861683008),  INT32_C(   837484339),  INT32_C(  1361171151), -INT32_C(   967043476),  INT32_C(  1448856793),
         INT32_C(  1861811973), -INT32_C(  1620824028), -INT32_C(  1907434418), -INT32_C(   742282726),  INT32_C(  1076702320),  INT32_C(  2109633555),  INT32_C(  1882254692), -INT32_C(   660009703) },
      { -INT32_C(  1583072899), -INT32_C(  1220474775), -INT32_C(    12152859), -INT32_C(  1429076166),  INT32_C(   837484339), -INT32_C(  1298432946), -INT32_C(  2094805911),  INT32_C(  1448856793),
         INT32_C(  1861811973), -INT32_C(   316262136),  INT32_C(    49048776), -INT32_C(  1481785741),  INT32_C(   198743997),  INT32_C(  2109633555),  INT32_C(   419488064), -INT32_C(  1284482131) },
      UINT16_C( 1542) },
    { {  INT32_C(   528531700),  INT32_C(  1375655787),  INT32_C(  1966212249),  INT32_C(  1131518837),  INT32_C(    32002341), -INT32_C(   220637185), -INT32_C(  2111665093), -INT32_C(  1877571428),
        -INT32_C(  1263495607),  INT32_C(   117812845), -INT32_C(  1954793549),  INT32_C(   652115201), -INT32_C(  1339570482), -INT32_C(   647731554), -INT32_C(  1889045783), -INT32_C(  2059555821) },
      {  INT32_C(  1963773960),  INT32_C(   830214526),  INT32_C(  1253963849),  INT32_C(  1131518837),  INT32_C(    15964258), -INT32_C(   220637185), -INT32_C(   310299426), -INT32_C(  1371168346),
        -INT32_C(   266120847), -INT32_C(  1423859614),  INT32_C(   251059864), -INT32_C(   598644870), -INT32_C(    19118593),  INT32_C(  2062595484), -INT32_C(  1889045783), -INT32_C(  2059555821) },
      UINT16_C(62953) },
    { { -INT32_C(  1397461441),  INT32_C(   344900160),  INT32_C(   642801527),  INT32_C(  1041429755), -INT32_C(     1738154), -INT32_C(   829767179),  INT32_C(  1192202619),  INT32_C(   965130234),
         INT32_C(  1726298662),  INT32_C(  1954182140), -INT32_C(   845493550), -INT32_C(  1123308354),  INT32_C(  2065557638),  INT32_C(   608806825),  INT32_C(   862673209),  INT32_C(   913109520) },
      {  INT32_C(   698175788), -INT32_C(  1768089660), -INT32_C(  1604110366),  INT32_C(  1767730915), -INT32_C(     1738154),  INT32_C(  1529097762), -INT32_C(  1735487609), -INT32_C(  1362167167),
         INT32_C(   265775947),  INT32_C(  1705342083), -INT32_C(  1912272725), -INT32_C(   856136842), -INT32_C(     3416611), -INT32_C(  1822757109),  INT32_C(     2877567),  INT32_C(   766441954) },
      UINT16_C(    0) },
    { {  INT32_C(   409156628),  INT32_C(  1493539745), -INT32_C(  1664724730),  INT32_C(   943376073), -INT32_C(  1481140450),  INT32_C(  1831684419), -INT32_C(  1621018006),  INT32_C(  1097827884),
         INT32_C(  1683608258),  INT32_C(  1941790317),  INT32_C(  2031059887),  INT32_C(    83865318),  INT32_C(  2021211956), -INT32_C(   219830648),  INT32_C(   227690208), -INT32_C(  1890713140) },
      {  INT32_C(  1089710035), -INT32_C(  1263292411),  INT32_C(  2083373619),  INT32_C(  1058922251), -INT32_C(  1481140450),  INT32_C(   513383485), -INT32_C(  1356125213), -INT32_C(    12682964),
         INT32_C(   624898336),  INT32_C(   349827809), -INT32_C(  1081079884),  INT32_C(    83865318), -INT32_C(  2102675899),  INT32_C(   916473171), -INT32_C(  1645884560),  INT32_C(  1687954500) },
      UINT16_C(63471) },
    { {  INT32_C(  1141001457), -INT32_C(   318693713),  INT32_C(   515247741), -INT32_C(   776581777),  INT32_C(   293050934), -INT32_C(   911551490),  INT32_C(   522142850),  INT32_C(   914904133),
         INT32_C(  1350208161),  INT32_C(   641563560), -INT32_C(   112921719),  INT32_C(  1912841723), -INT32_C(  1819721324), -INT32_C(  1755565291), -INT32_C(  1112114313),  INT32_C(  1911766736) },
      {  INT32_C(  1908502216),  INT32_C(  1939341033),  INT32_C(   862772210),  INT32_C(  1789540053), -INT32_C(  1929563273), -INT32_C(   568108698), -INT32_C(  1516512555), -INT32_C(   518615528),
        -INT32_C(   430778372), -INT32_C(   950408747), -INT32_C(  1711618620),  INT32_C(  1912841723),  INT32_C(  1073676504),  INT32_C(   790438490),  INT32_C(   366262524),  INT32_C(  1140255302) },
      UINT16_C(36816) },
    { {  INT32_C(   848044500),  INT32_C(   296764458), -INT32_C(  1963567074),  INT32_C(  1196844100), -INT32_C(  2086046591),  INT32_C(  1247589409),  INT32_C(  2101382144),  INT32_C(  1210100252),
        -INT32_C(  1564842691),  INT32_C(  2088780893),  INT32_C(  1342601116), -INT32_C(  1441558744),  INT32_C(  2116926467),  INT32_C(   518954269), -INT32_C(  1061193971),  INT32_C(   495827928) },
      {  INT32_C(   848044500),  INT32_C(    28183909), -INT32_C(  1403931516),  INT32_C(  1196844100),  INT32_C(  1053131809),  INT32_C(  1247589409), -INT32_C(  1148192542), -INT32_C(  1999102797),
        -INT32_C(  1564842691), -INT32_C(   157063054),  INT32_C(  1621292060),  INT32_C(  2057894233), -INT32_C(  1632080515),  INT32_C(    82318625), -INT32_C(  1061193971), -INT32_C(   783771757) },
      UINT16_C(45762) },
    { {  INT32_C(   199368610), -INT32_C(  1216309145),  INT32_C(  1661360438),  INT32_C(  1603130128), -INT32_C(  1520839238), -INT32_C(  1928786624), -INT32_C(   179247884),  INT32_C(  1859947980),
         INT32_C(   343588780), -INT32_C(  1916011945),  INT32_C(  1676726611),  INT32_C(  1606581668), -INT32_C(  1828447149),  INT32_C(  1663044905),  INT32_C(  1435267977), -INT32_C(  1379702784) },
      {  INT32_C(  2143370535), -INT32_C(  1978889161),  INT32_C(    49216861), -INT32_C(   849235846), -INT32_C(   178166324), -INT32_C(    78085774), -INT32_C(   179247884),  INT32_C(  1956779085),
        -INT32_C(  1980538031),  INT32_C(  1276313839),  INT32_C(  2001601021),  INT32_C(  2118496178), -INT32_C(  2022398444),  INT32_C(   478333991), -INT32_C(    32386127),  INT32_C(   947041255) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi32(test_vec[i].b);
    simde__mmask16 r = simde_mm512_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[16];
    int32_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi32(a_);
    simde__m512i b = simde_mm512_loadu_epi32(b_);
    simde__mmask16 r = simde_mm512_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { {  INT64_C(  815426814384182538), -INT64_C( 3171329385968114620), -INT64_C( 8230427869076701561),  INT64_C( 2084909417815204586),
         INT64_C( 4992461192020513197), -INT64_C( 6997972495571653353),  INT64_C( 1544794234400857551), -INT64_C( 1133014558105260776) },
      {  INT64_C(  815426814384182538),  INT64_C( 5599850644529834206), -INT64_C( 8230427869076701561),  INT64_C( 6054806343208374738),
        -INT64_C( 7999325655028221805), -INT64_C( 6997972495571653353), -INT64_C( 8109267974603401356), -INT64_C( 4663851055046119768) },
      UINT8_C( 37) },
    { {  INT64_C(  314943195812753306), -INT64_C( 2501697685225398533), -INT64_C(  123825240566403747), -INT64_C( 4792185540628318021),
        -INT64_C( 6267670406683684139),  INT64_C( 6191148726960572633), -INT64_C( 8366236954574013875),  INT64_C( 7659196439573376538) },
      {  INT64_C(  314943195812753306),  INT64_C( 2593227683338024623),  INT64_C( 2383792892642253496),  INT64_C( 7293974096267030761),
        -INT64_C( 6267670406683684139), -INT64_C( 3661061373052316104),  INT64_C( 1533872782177549350),  INT64_C( 7659196439573376538) },
      UINT8_C( 78) },
    { {  INT64_C( 7306782805314696068),  INT64_C( 4568011377797348132),  INT64_C(  577426160441628932), -INT64_C( 9189351345856662814),
        -INT64_C(  559186510474580840), -INT64_C( 6521585401834452454), -INT64_C(  937168664740034858),  INT64_C( 1723173858224888238) },
      {  INT64_C( 6533081766334226909),  INT64_C( 7294731470644048630),  INT64_C( 7779697587739917241), -INT64_C( 9189351345856662814),
        -INT64_C( 8621616388207647041), -INT64_C( 4408079514205162801), -INT64_C(  937168664740034858), -INT64_C( 8117514230509080012) },
      UINT8_C(110) },
    { { -INT64_C( 7087406018461340711),  INT64_C( 1578307511195951783), -INT64_C( 3206819524879919840), -INT64_C( 3656953735145910880),
        -INT64_C( 2472103061205556146),  INT64_C( 3466626087279976835), -INT64_C( 8054510983497852476),  INT64_C( 4195864689923673698) },
      { -INT64_C(  723172496797065285),  INT64_C( 1578307511195951783),  INT64_C( 6169846445109806879), -INT64_C( 3948700237086378111),
        -INT64_C( 5337372428842416251),  INT64_C( 9052608391539333179),  INT64_C( 5645028817443232602),  INT64_C( 4195864689923673698) },
      UINT8_C(  0) },
    { { -INT64_C( 1344772335003037575),  INT64_C( 6912793192118502986), -INT64_C( 4364594109481085826), -INT64_C(  687703303177053746),
        -INT64_C( 6194245874054605388), -INT64_C( 7983050271184920002), -INT64_C( 6835846668000897898),  INT64_C(  640229329692288366) },
      { -INT64_C( 1344772335003037575),  INT64_C( 6912793192118502986), -INT64_C( 5382118327105768703), -INT64_C(  687703303177053746),
        -INT64_C( 8757538391267814438), -INT64_C( 7392990187872883649), -INT64_C( 3377535878610606970),  INT64_C(  640229329692288366) },
      UINT8_C(116) },
    { {  INT64_C( 6819904318137959795), -INT64_C( 1142813134477988786), -INT64_C( 3851044372280088878), -INT64_C( 8900216187008922246),
        -INT64_C( 5106163422351965971),  INT64_C( 2639724926454197296),  INT64_C( 2668071249071461296), -INT64_C(  581321648712506563) },
      { -INT64_C(   62804312891552516), -INT64_C( 4123797131256320821), -INT64_C( 3851044372280088878), -INT64_C( 8900216187008922246),
        -INT64_C( 1986355611550664288),  INT64_C( 5141222314171768888),  INT64_C( 2668071249071461296), -INT64_C( 2252320706351453007) },
      UINT8_C(207) },
    { { -INT64_C( 2622263577888628257),  INT64_C( 8339754642517624376),  INT64_C( 3200790277648262730),  INT64_C( 6542041936396860199),
         INT64_C( 5829708661834795745),  INT64_C( 7463910560856800826), -INT64_C(  449693853726649143),  INT64_C( 5182974009089025561) },
      {  INT64_C( 2188251220115954666),  INT64_C( 8339754642517624376),  INT64_C( 2657909982116818455),  INT64_C( 4746412852921284401),
        -INT64_C( 6682025271411920575),  INT64_C( 5427035690136793395),  INT64_C( 6166015185547188578),  INT64_C( 3559474213266672333) },
      UINT8_C(188) },
    { {  INT64_C( 5977711872274790846), -INT64_C( 6731727139327459557),  INT64_C( 5009458858281107556), -INT64_C( 9018258734263313439),
         INT64_C( 2992163722505636791), -INT64_C( 4112083580961091908),  INT64_C( 6344837915668859517),  INT64_C( 2681739022452898111) },
      {  INT64_C( 5977711872274790846), -INT64_C( 2821303568077464282),  INT64_C( 6615978186106779393), -INT64_C( 7063534414381159578),
        -INT64_C( 3757586874036024255), -INT64_C( 4112083580961091908), -INT64_C( 6764671361622241623), -INT64_C( 5046441195229204357) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm512_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[8];
    int64_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi64(a_);
    simde__m512i b = simde_mm512_loadu_epi64(b_);
    simde__mmask8 r = simde_mm512_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_ps_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float32 a[16];
    const simde_float32 b[16];
    const simde__mmask16 r;
    const simde_float32 an[16];
    const simde_float32 bn[16];
    const simde__mmask16 rn;
  } test_vec[] = {
    /* 0 */
    { { SIMDE_FLOAT32_C(    87.16), SIMDE_FLOAT32_C(   508.14), SIMDE_FLOAT32_C(   603.96), SIMDE_FLOAT32_C(   170.45),
        SIMDE_FLOAT32_C(  -616.81), SIMDE_FLOAT32_C(   316.02), SIMDE_FLOAT32_C(   562.38), SIMDE_FLOAT32_C(  -988.78),
        SIMDE_FLOAT32_C(   -65.87), SIMDE_FLOAT32_C(  -847.04), SIMDE_FLOAT32_C(  -447.11), SIMDE_FLOAT32_C(    45.04),
        SIMDE_FLOAT32_C(    80.53), SIMDE_FLOAT32_C(   -28.80), SIMDE_FLOAT32_C(   668.21), SIMDE_FLOAT32_C(   829.87) },
      { SIMDE_FLOAT32_C(    87.16), SIMDE_FLOAT32_C(  -430.72), SIMDE_FLOAT32_C(   603.96), SIMDE_FLOAT32_C(   542.42),
        SIMDE_FLOAT32_C(  -616.81), SIMDE_FLOAT32_C(   950.31), SIMDE_FLOAT32_C(   562.38), SIMDE_FLOAT32_C(   695.54),
        SIMDE_FLOAT32_C(   -65.87), SIMDE_FLOAT32_C(   924.58), SIMDE_FLOAT32_C(  -447.11), SIMDE_FLOAT32_C(  -680.51),
        SIMDE_FLOAT32_C(    80.53), SIMDE_FLOAT32_C(  -823.81), SIMDE_FLOAT32_C(   668.21), SIMDE_FLOAT32_C(   796.61) },
      UINT16_C(21845),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -450.16),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -932.48),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -470.55),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -200.01),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -368.38),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   762.93),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   513.24),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   381.67) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   147.76), SIMDE_FLOAT32_C(   -75.91),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(    98.06),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -930.84),
        SIMDE_FLOAT32_C(  -213.72),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -611.35),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   334.52),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -21.93) },
      UINT16_C(    0) },
    /* 1 */
    { { SIMDE_FLOAT32_C(   884.36), SIMDE_FLOAT32_C(  -740.60), SIMDE_FLOAT32_C(    45.59), SIMDE_FLOAT32_C(   750.22),
        SIMDE_FLOAT32_C(  -211.15), SIMDE_FLOAT32_C(   124.32), SIMDE_FLOAT32_C(  -449.79), SIMDE_FLOAT32_C(   471.26),
        SIMDE_FLOAT32_C(   755.95), SIMDE_FLOAT32_C(  -604.76), SIMDE_FLOAT32_C(   234.19), SIMDE_FLOAT32_C(   358.77),
        SIMDE_FLOAT32_C(   908.47), SIMDE_FLOAT32_C(  -173.01), SIMDE_FLOAT32_C(  -259.56), SIMDE_FLOAT32_C(   990.99) },
      { SIMDE_FLOAT32_C(   884.36), SIMDE_FLOAT32_C(   664.53), SIMDE_FLOAT32_C(    45.59), SIMDE_FLOAT32_C(    72.81),
        SIMDE_FLOAT32_C(  -211.15), SIMDE_FLOAT32_C(  -832.69), SIMDE_FLOAT32_C(  -449.79), SIMDE_FLOAT32_C(   312.51),
        SIMDE_FLOAT32_C(   755.95), SIMDE_FLOAT32_C(   530.63), SIMDE_FLOAT32_C(   234.19), SIMDE_FLOAT32_C(  -421.40),
        SIMDE_FLOAT32_C(   908.47), SIMDE_FLOAT32_C(  -899.42), SIMDE_FLOAT32_C(  -259.56), SIMDE_FLOAT32_C(  -250.49) },
      UINT16_C(  514),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -397.75),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   148.84),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    49.94),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   482.52),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   854.28),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   353.65),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   581.73),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -343.98) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   576.02), SIMDE_FLOAT32_C(   728.84),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   743.33),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    28.19),
        SIMDE_FLOAT32_C(  -598.56),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   606.79),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   163.45),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -646.19) },
      UINT16_C(    2) },
    /* 2 */
    { { SIMDE_FLOAT32_C(   765.71), SIMDE_FLOAT32_C(  -484.17), SIMDE_FLOAT32_C(   502.64), SIMDE_FLOAT32_C(   492.28),
        SIMDE_FLOAT32_C(   565.77), SIMDE_FLOAT32_C(  -877.26), SIMDE_FLOAT32_C(   -25.20), SIMDE_FLOAT32_C(    10.95),
        SIMDE_FLOAT32_C(   977.02), SIMDE_FLOAT32_C(   816.09), SIMDE_FLOAT32_C(  -635.40), SIMDE_FLOAT32_C(  -341.71),
        SIMDE_FLOAT32_C(   397.82), SIMDE_FLOAT32_C(   709.25), SIMDE_FLOAT32_C(   314.32), SIMDE_FLOAT32_C(  -355.93) },
      { SIMDE_FLOAT32_C(   765.71), SIMDE_FLOAT32_C(    43.15), SIMDE_FLOAT32_C(   502.64), SIMDE_FLOAT32_C(    28.60),
        SIMDE_FLOAT32_C(   565.77), SIMDE_FLOAT32_C(   501.56), SIMDE_FLOAT32_C(   -25.20), SIMDE_FLOAT32_C(  -684.60),
        SIMDE_FLOAT32_C(   977.02), SIMDE_FLOAT32_C(   663.58), SIMDE_FLOAT32_C(  -635.40), SIMDE_FLOAT32_C(   388.62),
        SIMDE_FLOAT32_C(   397.82), SIMDE_FLOAT32_C(   598.09), SIMDE_FLOAT32_C(   314.32), SIMDE_FLOAT32_C(  -407.26) },
      UINT16_C(23927),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   245.07),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   679.69),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    59.82),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   344.82),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    55.24),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -726.27),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   317.43),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -950.24) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   334.36), SIMDE_FLOAT32_C(    78.36),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -164.08),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -769.28),
        SIMDE_FLOAT32_C(  -201.27),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   619.34),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   361.76),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   883.94) },
      UINT16_C(40962) },
    /* 3 */
    { { SIMDE_FLOAT32_C(  -393.18), SIMDE_FLOAT32_C(  -696.47), SIMDE_FLOAT32_C(   563.63), SIMDE_FLOAT32_C(   974.63),
        SIMDE_FLOAT32_C(   363.35), SIMDE_FLOAT32_C(  -745.73), SIMDE_FLOAT32_C(   319.45), SIMDE_FLOAT32_C(  -760.73),
        SIMDE_FLOAT32_C(   309.52), SIMDE_FLOAT32_C(   322.56), SIMDE_FLOAT32_C(  -487.01), SIMDE_FLOAT32_C(  -925.99),
        SIMDE_FLOAT32_C(  -360.01), SIMDE_FLOAT32_C(   430.80), SIMDE_FLOAT32_C(  -876.23), SIMDE_FLOAT32_C(     0.57) },
      { SIMDE_FLOAT32_C(  -393.18), SIMDE_FLOAT32_C(   202.13), SIMDE_FLOAT32_C(   563.63), SIMDE_FLOAT32_C(   601.08),
        SIMDE_FLOAT32_C(   363.35), SIMDE_FLOAT32_C(   865.07), SIMDE_FLOAT32_C(   319.45), SIMDE_FLOAT32_C(   136.02),
        SIMDE_FLOAT32_C(   309.52), SIMDE_FLOAT32_C(   451.14), SIMDE_FLOAT32_C(  -487.01), SIMDE_FLOAT32_C(   807.02),
        SIMDE_FLOAT32_C(  -360.01), SIMDE_FLOAT32_C(   -19.70), SIMDE_FLOAT32_C(  -876.23), SIMDE_FLOAT32_C(   419.72) },
      UINT16_C(    0),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   254.59),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -352.82),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -286.21),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -181.61),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   399.45),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -323.65),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    16.17),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   595.40) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -47.96), SIMDE_FLOAT32_C(   196.48),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -182.89),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -145.89),
        SIMDE_FLOAT32_C(  -520.57),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -338.87),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -647.91),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -282.48) },
      UINT16_C(24445) },
    /* 4 */
    { { SIMDE_FLOAT32_C(   606.69), SIMDE_FLOAT32_C(  -893.60), SIMDE_FLOAT32_C(   364.71), SIMDE_FLOAT32_C(   115.56),
        SIMDE_FLOAT32_C(  -179.81), SIMDE_FLOAT32_C(  -748.84), SIMDE_FLOAT32_C(   933.95), SIMDE_FLOAT32_C(  -143.45),
        SIMDE_FLOAT32_C(   650.60), SIMDE_FLOAT32_C(  -173.66), SIMDE_FLOAT32_C(   532.89), SIMDE_FLOAT32_C(  -519.15),
        SIMDE_FLOAT32_C(   842.51), SIMDE_FLOAT32_C(   209.81), SIMDE_FLOAT32_C(  -923.75), SIMDE_FLOAT32_C(  -939.19) },
      { SIMDE_FLOAT32_C(   606.69), SIMDE_FLOAT32_C(   272.73), SIMDE_FLOAT32_C(   364.71), SIMDE_FLOAT32_C(   -21.04),
        SIMDE_FLOAT32_C(  -179.81), SIMDE_FLOAT32_C(  -692.00), SIMDE_FLOAT32_C(   933.95), SIMDE_FLOAT32_C(  -219.56),
        SIMDE_FLOAT32_C(   650.60), SIMDE_FLOAT32_C(  -505.80), SIMDE_FLOAT32_C(   532.89), SIMDE_FLOAT32_C(  -804.91),
        SIMDE_FLOAT32_C(   842.51), SIMDE_FLOAT32_C(  -215.18), SIMDE_FLOAT32_C(  -923.75), SIMDE_FLOAT32_C(  -547.02) },
      UINT16_C(43690),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -722.68),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   711.40),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   502.48),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -820.92),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -899.16),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -828.67),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -263.83),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   472.50) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -151.46), SIMDE_FLOAT32_C(  -548.55),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   156.54),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   917.93),
        SIMDE_FLOAT32_C(  -221.29),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -886.99),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    25.62),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -160.85) },
           UINT16_MAX },
    /* 5 */
    { { SIMDE_FLOAT32_C(   302.94), SIMDE_FLOAT32_C(   646.52), SIMDE_FLOAT32_C(  -449.44), SIMDE_FLOAT32_C(   831.42),
        SIMDE_FLOAT32_C(   149.00), SIMDE_FLOAT32_C(   118.50), SIMDE_FLOAT32_C(  -989.50), SIMDE_FLOAT32_C(   477.82),
        SIMDE_FLOAT32_C(   219.34), SIMDE_FLOAT32_C(  -329.57), SIMDE_FLOAT32_C(   649.15), SIMDE_FLOAT32_C(  -470.01),
        SIMDE_FLOAT32_C(   406.60), SIMDE_FLOAT32_C(   881.30), SIMDE_FLOAT32_C(  -997.51), SIMDE_FLOAT32_C(   415.50) },
      { SIMDE_FLOAT32_C(   302.94), SIMDE_FLOAT32_C(  -546.06), SIMDE_FLOAT32_C(  -449.44), SIMDE_FLOAT32_C(   886.37),
        SIMDE_FLOAT32_C(   149.00), SIMDE_FLOAT32_C(  -184.22), SIMDE_FLOAT32_C(  -989.50), SIMDE_FLOAT32_C(   517.17),
        SIMDE_FLOAT32_C(   219.34), SIMDE_FLOAT32_C(   917.32), SIMDE_FLOAT32_C(   649.15), SIMDE_FLOAT32_C(   926.85),
        SIMDE_FLOAT32_C(   406.60), SIMDE_FLOAT32_C(   220.16), SIMDE_FLOAT32_C(  -997.51), SIMDE_FLOAT32_C(  -754.12) },
      UINT16_C(62839),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   316.56),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -984.33),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -912.20),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   654.41),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   142.65),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -835.18),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -813.11),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   753.79) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   305.74), SIMDE_FLOAT32_C(   640.16),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -878.49),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -899.58),
        SIMDE_FLOAT32_C(   361.78),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -972.73),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -206.73),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -874.53) },
      UINT16_C(57343) },
    /* 6 */
    { { SIMDE_FLOAT32_C(  -890.17), SIMDE_FLOAT32_C(  -372.09), SIMDE_FLOAT32_C(  -858.85), SIMDE_FLOAT32_C(  -455.11),
        SIMDE_FLOAT32_C(  -284.28), SIMDE_FLOAT32_C(   634.64), SIMDE_FLOAT32_C(  -800.70), SIMDE_FLOAT32_C(   473.94),
        SIMDE_FLOAT32_C(  -222.71), SIMDE_FLOAT32_C(  -616.30), SIMDE_FLOAT32_C(   638.77), SIMDE_FLOAT32_C(   801.24),
        SIMDE_FLOAT32_C(  -429.41), SIMDE_FLOAT32_C(   219.09), SIMDE_FLOAT32_C(   555.03), SIMDE_FLOAT32_C(   211.42) },
      { SIMDE_FLOAT32_C(  -890.17), SIMDE_FLOAT32_C(   195.19), SIMDE_FLOAT32_C(  -858.85), SIMDE_FLOAT32_C(  -353.66),
        SIMDE_FLOAT32_C(  -284.28), SIMDE_FLOAT32_C(   487.16), SIMDE_FLOAT32_C(  -800.70), SIMDE_FLOAT32_C(  -998.56),
        SIMDE_FLOAT32_C(  -222.71), SIMDE_FLOAT32_C(  -225.98), SIMDE_FLOAT32_C(   638.77), SIMDE_FLOAT32_C(  -215.40),
        SIMDE_FLOAT32_C(  -429.41), SIMDE_FLOAT32_C(  -143.22), SIMDE_FLOAT32_C(   555.03), SIMDE_FLOAT32_C(   677.12) },
      UINT16_C(10400),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    51.22),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -799.59),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -578.69),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   463.15),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   313.12),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   375.60),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -180.58),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    57.03) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   177.72), SIMDE_FLOAT32_C(   703.37),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -335.12),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -809.32),
        SIMDE_FLOAT32_C(   224.15),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -24.72),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   885.35),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   303.33) },
      UINT16_C(24573) },
    /* 7 */
    { { SIMDE_FLOAT32_C(   -63.43), SIMDE_FLOAT32_C(  -309.43), SIMDE_FLOAT32_C(   503.74), SIMDE_FLOAT32_C(   622.43),
        SIMDE_FLOAT32_C(   111.88), SIMDE_FLOAT32_C(   178.09), SIMDE_FLOAT32_C(    85.59), SIMDE_FLOAT32_C(   916.89),
        SIMDE_FLOAT32_C(  -508.80), SIMDE_FLOAT32_C(  -650.02), SIMDE_FLOAT32_C(   292.49), SIMDE_FLOAT32_C(    23.41),
        SIMDE_FLOAT32_C(   169.40), SIMDE_FLOAT32_C(   879.51), SIMDE_FLOAT32_C(  -919.56), SIMDE_FLOAT32_C(   184.00) },
      { SIMDE_FLOAT32_C(   -63.43), SIMDE_FLOAT32_C(   783.81), SIMDE_FLOAT32_C(   503.74), SIMDE_FLOAT32_C(   722.11),
        SIMDE_FLOAT32_C(   111.88), SIMDE_FLOAT32_C(  -506.04), SIMDE_FLOAT32_C(    85.59), SIMDE_FLOAT32_C(   458.08),
        SIMDE_FLOAT32_C(  -508.80), SIMDE_FLOAT32_C(  -111.92), SIMDE_FLOAT32_C(   292.49), SIMDE_FLOAT32_C(   274.46),
        SIMDE_FLOAT32_C(   169.40), SIMDE_FLOAT32_C(  -281.92), SIMDE_FLOAT32_C(  -919.56), SIMDE_FLOAT32_C(   710.01) },
           UINT16_MAX,
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -918.47),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -479.46),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -581.97),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   750.83),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   729.93),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   937.40),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -145.33),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -333.33) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -40.33), SIMDE_FLOAT32_C(  -611.22),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   453.63),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -90.54),
        SIMDE_FLOAT32_C(   189.66),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -816.08),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -238.28),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -293.25) },
      UINT16_C(41090) },
    /* 8 */
    { { SIMDE_FLOAT32_C(  -156.75), SIMDE_FLOAT32_C(     5.54), SIMDE_FLOAT32_C(   227.29), SIMDE_FLOAT32_C(  -897.12),
        SIMDE_FLOAT32_C(   423.57), SIMDE_FLOAT32_C(   664.72), SIMDE_FLOAT32_C(   853.70), SIMDE_FLOAT32_C(  -808.43),
        SIMDE_FLOAT32_C(   394.65), SIMDE_FLOAT32_C(  -372.06), SIMDE_FLOAT32_C(  -871.02), SIMDE_FLOAT32_C(     4.09),
        SIMDE_FLOAT32_C(   482.61), SIMDE_FLOAT32_C(   250.38), SIMDE_FLOAT32_C(   670.75), SIMDE_FLOAT32_C(  -878.92) },
      { SIMDE_FLOAT32_C(  -156.75), SIMDE_FLOAT32_C(  -940.47), SIMDE_FLOAT32_C(   227.29), SIMDE_FLOAT32_C(   663.69),
        SIMDE_FLOAT32_C(   423.57), SIMDE_FLOAT32_C(   323.98), SIMDE_FLOAT32_C(   853.70), SIMDE_FLOAT32_C(   550.77),
        SIMDE_FLOAT32_C(   394.65), SIMDE_FLOAT32_C(  -242.93), SIMDE_FLOAT32_C(  -871.02), SIMDE_FLOAT32_C(   202.09),
        SIMDE_FLOAT32_C(   482.61), SIMDE_FLOAT32_C(   186.97), SIMDE_FLOAT32_C(   670.75), SIMDE_FLOAT32_C(  -637.96) },
      UINT16_C(21845),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   136.13),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   616.08),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -681.38),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -804.49),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   936.64),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -570.84),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -129.66),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   397.08) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   543.73), SIMDE_FLOAT32_C(    60.76),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -132.29),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -228.30),
        SIMDE_FLOAT32_C(  -609.02),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   973.80),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   882.64),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   735.12) },
      UINT16_C(24445) },
    /* 9 */
    { { SIMDE_FLOAT32_C(    18.78), SIMDE_FLOAT32_C(  -263.29), SIMDE_FLOAT32_C(   351.21), SIMDE_FLOAT32_C(   819.63),
        SIMDE_FLOAT32_C(    55.33), SIMDE_FLOAT32_C(   158.87), SIMDE_FLOAT32_C(  -984.86), SIMDE_FLOAT32_C(  -998.12),
        SIMDE_FLOAT32_C(    95.50), SIMDE_FLOAT32_C(   214.73), SIMDE_FLOAT32_C(  -568.96), SIMDE_FLOAT32_C(  -717.48),
        SIMDE_FLOAT32_C(  -914.93), SIMDE_FLOAT32_C(   -18.73), SIMDE_FLOAT32_C(   679.60), SIMDE_FLOAT32_C(    14.95) },
      { SIMDE_FLOAT32_C(    18.78), SIMDE_FLOAT32_C(  -259.63), SIMDE_FLOAT32_C(   351.21), SIMDE_FLOAT32_C(   392.71),
        SIMDE_FLOAT32_C(    55.33), SIMDE_FLOAT32_C(   147.70), SIMDE_FLOAT32_C(  -984.86), SIMDE_FLOAT32_C(  -234.75),
        SIMDE_FLOAT32_C(    95.50), SIMDE_FLOAT32_C(  -861.79), SIMDE_FLOAT32_C(  -568.96), SIMDE_FLOAT32_C(    45.94),
        SIMDE_FLOAT32_C(  -914.93), SIMDE_FLOAT32_C(   -53.19), SIMDE_FLOAT32_C(   679.60), SIMDE_FLOAT32_C(    39.63) },
      UINT16_C(34946),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -867.73),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -261.15),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -125.60),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -613.36),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   171.77),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -825.80),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -651.24),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -321.95) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   495.09), SIMDE_FLOAT32_C(  -929.24),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -357.21),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -853.87),
        SIMDE_FLOAT32_C(  -626.62),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   192.07),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   973.13),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   534.00) },
      UINT16_C(65407) },
    /* 10 */
    { { SIMDE_FLOAT32_C(  -894.60), SIMDE_FLOAT32_C(  -706.88), SIMDE_FLOAT32_C(  -727.15), SIMDE_FLOAT32_C(   396.53),
        SIMDE_FLOAT32_C(   167.52), SIMDE_FLOAT32_C(    13.58), SIMDE_FLOAT32_C(   783.16), SIMDE_FLOAT32_C(   256.64),
        SIMDE_FLOAT32_C(  -814.65), SIMDE_FLOAT32_C(  -547.68), SIMDE_FLOAT32_C(   430.84), SIMDE_FLOAT32_C(  -661.60),
        SIMDE_FLOAT32_C(  -198.92), SIMDE_FLOAT32_C(  -380.01), SIMDE_FLOAT32_C(    16.45), SIMDE_FLOAT32_C(   890.21) },
      { SIMDE_FLOAT32_C(  -894.60), SIMDE_FLOAT32_C(    87.20), SIMDE_FLOAT32_C(  -727.15), SIMDE_FLOAT32_C(  -242.13),
        SIMDE_FLOAT32_C(   167.52), SIMDE_FLOAT32_C(   582.26), SIMDE_FLOAT32_C(   783.16), SIMDE_FLOAT32_C(   695.75),
        SIMDE_FLOAT32_C(  -814.65), SIMDE_FLOAT32_C(  -903.94), SIMDE_FLOAT32_C(   430.84), SIMDE_FLOAT32_C(  -663.61),
        SIMDE_FLOAT32_C(  -198.92), SIMDE_FLOAT32_C(   523.84), SIMDE_FLOAT32_C(    16.45), SIMDE_FLOAT32_C(  -825.41) },
      UINT16_C(30199),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -856.76),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -15.52),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   354.28),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   342.17),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   671.97),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -392.32),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -302.99),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -592.96) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   851.50), SIMDE_FLOAT32_C(   164.91),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   433.76),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   919.68),
        SIMDE_FLOAT32_C(  -835.03),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -743.92),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -873.54),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -766.87) },
      UINT16_C(24575) },
    /* 11 */
    { { SIMDE_FLOAT32_C(  -730.30), SIMDE_FLOAT32_C(   979.86), SIMDE_FLOAT32_C(   217.61), SIMDE_FLOAT32_C(   426.52),
        SIMDE_FLOAT32_C(   334.14), SIMDE_FLOAT32_C(   458.74), SIMDE_FLOAT32_C(  -231.32), SIMDE_FLOAT32_C(   140.74),
        SIMDE_FLOAT32_C(   130.70), SIMDE_FLOAT32_C(  -550.75), SIMDE_FLOAT32_C(   748.42), SIMDE_FLOAT32_C(   422.66),
        SIMDE_FLOAT32_C(   146.26), SIMDE_FLOAT32_C(  -753.68), SIMDE_FLOAT32_C(   829.70), SIMDE_FLOAT32_C(   -69.52) },
      { SIMDE_FLOAT32_C(  -730.30), SIMDE_FLOAT32_C(    -5.38), SIMDE_FLOAT32_C(   217.61), SIMDE_FLOAT32_C(   531.59),
        SIMDE_FLOAT32_C(   334.14), SIMDE_FLOAT32_C(  -160.59), SIMDE_FLOAT32_C(  -231.32), SIMDE_FLOAT32_C(   228.49),
        SIMDE_FLOAT32_C(   130.70), SIMDE_FLOAT32_C(   707.35), SIMDE_FLOAT32_C(   748.42), SIMDE_FLOAT32_C(  -852.09),
        SIMDE_FLOAT32_C(   146.26), SIMDE_FLOAT32_C(   871.39), SIMDE_FLOAT32_C(   829.70), SIMDE_FLOAT32_C(  -896.49) },
      UINT16_C(    0),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   598.65),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   185.39),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -701.29),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -811.91),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -925.45),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   894.23),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   440.46),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -581.30) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -138.23), SIMDE_FLOAT32_C(   950.28),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   701.18),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   432.91),
        SIMDE_FLOAT32_C(   108.90),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   580.82),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   961.86),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   912.38) },
      UINT16_C(    0) },
    /* 12 */
    { { SIMDE_FLOAT32_C(   560.51), SIMDE_FLOAT32_C(  -423.75), SIMDE_FLOAT32_C(    97.77), SIMDE_FLOAT32_C(  -382.10),
        SIMDE_FLOAT32_C(  -125.04), SIMDE_FLOAT32_C(   423.90), SIMDE_FLOAT32_C(  -194.01), SIMDE_FLOAT32_C(   622.93),
        SIMDE_FLOAT32_C(   498.45), SIMDE_FLOAT32_C(  -583.26), SIMDE_FLOAT32_C(   517.15), SIMDE_FLOAT32_C(   819.33),
        SIMDE_FLOAT32_C(   857.20), SIMDE_FLOAT32_C(  -658.15), SIMDE_FLOAT32_C(  -761.98), SIMDE_FLOAT32_C(  -707.73) },
      { SIMDE_FLOAT32_C(   560.51), SIMDE_FLOAT32_C(  -811.70), SIMDE_FLOAT32_C(    97.77), SIMDE_FLOAT32_C(   -95.20),
        SIMDE_FLOAT32_C(  -125.04), SIMDE_FLOAT32_C(  -482.05), SIMDE_FLOAT32_C(  -194.01), SIMDE_FLOAT32_C(  -301.24),
        SIMDE_FLOAT32_C(   498.45), SIMDE_FLOAT32_C(   918.53), SIMDE_FLOAT32_C(   517.15), SIMDE_FLOAT32_C(  -231.19),
        SIMDE_FLOAT32_C(   857.20), SIMDE_FLOAT32_C(  -312.31), SIMDE_FLOAT32_C(  -761.98), SIMDE_FLOAT32_C(   440.90) },
      UINT16_C(43690),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   778.95),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -861.10),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -135.22),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -298.70),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   278.98),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   138.72),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -241.35),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -175.54) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   221.86), SIMDE_FLOAT32_C(   729.25),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   739.82),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -552.50),
        SIMDE_FLOAT32_C(   985.49),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   216.31),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   897.50),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -171.32) },
      UINT16_C(41090) },
    /* 13 */
    { { SIMDE_FLOAT32_C(   676.45), SIMDE_FLOAT32_C(   365.56), SIMDE_FLOAT32_C(   -32.42), SIMDE_FLOAT32_C(  -120.70),
        SIMDE_FLOAT32_C(  -769.66), SIMDE_FLOAT32_C(  -270.59), SIMDE_FLOAT32_C(   580.61), SIMDE_FLOAT32_C(   511.86),
        SIMDE_FLOAT32_C(  -991.61), SIMDE_FLOAT32_C(  -898.76), SIMDE_FLOAT32_C(  -349.42), SIMDE_FLOAT32_C(   629.22),
        SIMDE_FLOAT32_C(  -140.11), SIMDE_FLOAT32_C(  -918.42), SIMDE_FLOAT32_C(  -546.32), SIMDE_FLOAT32_C(   806.84) },
      { SIMDE_FLOAT32_C(   676.45), SIMDE_FLOAT32_C(  -817.06), SIMDE_FLOAT32_C(   -32.42), SIMDE_FLOAT32_C(    43.26),
        SIMDE_FLOAT32_C(  -769.66), SIMDE_FLOAT32_C(   579.23), SIMDE_FLOAT32_C(   580.61), SIMDE_FLOAT32_C(  -764.61),
        SIMDE_FLOAT32_C(  -991.61), SIMDE_FLOAT32_C(  -292.93), SIMDE_FLOAT32_C(  -349.42), SIMDE_FLOAT32_C(  -979.00),
        SIMDE_FLOAT32_C(  -140.11), SIMDE_FLOAT32_C(   408.03), SIMDE_FLOAT32_C(  -546.32), SIMDE_FLOAT32_C(  -718.98) },
      UINT16_C(56791),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   817.26),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(     3.94),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -259.07),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -444.94),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   166.38),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   702.05),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -362.03),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -448.60) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   852.55), SIMDE_FLOAT32_C(   594.65),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   431.78),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   888.05),
        SIMDE_FLOAT32_C(   792.48),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   909.05),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -241.27),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   589.08) },
      UINT16_C(    0) },
    /* 14 */
    { { SIMDE_FLOAT32_C(  -424.01), SIMDE_FLOAT32_C(   838.39), SIMDE_FLOAT32_C(  -406.98), SIMDE_FLOAT32_C(   122.66),
        SIMDE_FLOAT32_C(  -420.68), SIMDE_FLOAT32_C(   108.82), SIMDE_FLOAT32_C(   677.72), SIMDE_FLOAT32_C(   421.49),
        SIMDE_FLOAT32_C(  -724.80), SIMDE_FLOAT32_C(   862.01), SIMDE_FLOAT32_C(   123.54), SIMDE_FLOAT32_C(   523.16),
        SIMDE_FLOAT32_C(  -500.03), SIMDE_FLOAT32_C(  -367.57), SIMDE_FLOAT32_C(  -925.45), SIMDE_FLOAT32_C(   320.87) },
      { SIMDE_FLOAT32_C(  -424.01), SIMDE_FLOAT32_C(   669.21), SIMDE_FLOAT32_C(  -406.98), SIMDE_FLOAT32_C(   916.76),
        SIMDE_FLOAT32_C(  -420.68), SIMDE_FLOAT32_C(   697.86), SIMDE_FLOAT32_C(   677.72), SIMDE_FLOAT32_C(  -452.90),
        SIMDE_FLOAT32_C(  -724.80), SIMDE_FLOAT32_C(   713.86), SIMDE_FLOAT32_C(   123.54), SIMDE_FLOAT32_C(   920.80),
        SIMDE_FLOAT32_C(  -500.03), SIMDE_FLOAT32_C(   622.22), SIMDE_FLOAT32_C(  -925.45), SIMDE_FLOAT32_C(    48.58) },
      UINT16_C(33410),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -897.11),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -960.07),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   848.97),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   486.90),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -415.04),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -789.05),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    84.61),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   702.38) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   923.49), SIMDE_FLOAT32_C(   619.14),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   621.36),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -273.33),
        SIMDE_FLOAT32_C(   137.81),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -352.53),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -842.66),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -917.48) },
      UINT16_C(41088) },
    /* 15 */
    { { SIMDE_FLOAT32_C(  -739.77), SIMDE_FLOAT32_C(  -169.77), SIMDE_FLOAT32_C(  -877.55), SIMDE_FLOAT32_C(   471.94),
        SIMDE_FLOAT32_C(  -320.80), SIMDE_FLOAT32_C(   583.87), SIMDE_FLOAT32_C(   -41.16), SIMDE_FLOAT32_C(  -609.83),
        SIMDE_FLOAT32_C(  -831.17), SIMDE_FLOAT32_C(   968.90), SIMDE_FLOAT32_C(  -398.88), SIMDE_FLOAT32_C(   386.22),
        SIMDE_FLOAT32_C(    53.52), SIMDE_FLOAT32_C(  -867.06), SIMDE_FLOAT32_C(    88.60), SIMDE_FLOAT32_C(   807.34) },
      { SIMDE_FLOAT32_C(  -739.77), SIMDE_FLOAT32_C(  -292.26), SIMDE_FLOAT32_C(  -877.55), SIMDE_FLOAT32_C(   677.79),
        SIMDE_FLOAT32_C(  -320.80), SIMDE_FLOAT32_C(   371.32), SIMDE_FLOAT32_C(   -41.16), SIMDE_FLOAT32_C(   269.51),
        SIMDE_FLOAT32_C(  -831.17), SIMDE_FLOAT32_C(    51.93), SIMDE_FLOAT32_C(  -398.88), SIMDE_FLOAT32_C(    -7.08),
        SIMDE_FLOAT32_C(    53.52), SIMDE_FLOAT32_C(   538.90), SIMDE_FLOAT32_C(    88.60), SIMDE_FLOAT32_C(   469.50) },
           UINT16_MAX,
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   197.89),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    48.34),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   900.27),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -49.41),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -960.37),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -77.31),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   425.42),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -771.00) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -954.19), SIMDE_FLOAT32_C(   906.80),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   417.13),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   788.14),
        SIMDE_FLOAT32_C(   363.18),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -218.94),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   856.50),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -677.67) },
           UINT16_MAX },
    /* 16 */
    { { SIMDE_FLOAT32_C(    54.40), SIMDE_FLOAT32_C(   -16.62), SIMDE_FLOAT32_C(   370.66), SIMDE_FLOAT32_C(   836.16),
        SIMDE_FLOAT32_C(  -116.35), SIMDE_FLOAT32_C(  -190.82), SIMDE_FLOAT32_C(  -213.25), SIMDE_FLOAT32_C(  -247.17),
        SIMDE_FLOAT32_C(  -151.19), SIMDE_FLOAT32_C(   123.57), SIMDE_FLOAT32_C(   675.52), SIMDE_FLOAT32_C(    21.38),
        SIMDE_FLOAT32_C(  -451.01), SIMDE_FLOAT32_C(  -594.45), SIMDE_FLOAT32_C(   250.38), SIMDE_FLOAT32_C(  -317.84) },
      { SIMDE_FLOAT32_C(    54.40), SIMDE_FLOAT32_C(   157.18), SIMDE_FLOAT32_C(   370.66), SIMDE_FLOAT32_C(   868.49),
        SIMDE_FLOAT32_C(  -116.35), SIMDE_FLOAT32_C(  -518.61), SIMDE_FLOAT32_C(  -213.25), SIMDE_FLOAT32_C(  -168.38),
        SIMDE_FLOAT32_C(  -151.19), SIMDE_FLOAT32_C(  -562.31), SIMDE_FLOAT32_C(   675.52), SIMDE_FLOAT32_C(  -151.14),
        SIMDE_FLOAT32_C(  -451.01), SIMDE_FLOAT32_C(   446.01), SIMDE_FLOAT32_C(   250.38), SIMDE_FLOAT32_C(   348.59) },
      UINT16_C(21845),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -458.15),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   313.05),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   971.50),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -800.16),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   741.39),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   644.06),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -528.40),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -401.71) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   273.23), SIMDE_FLOAT32_C(  -533.22),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   754.62),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -349.70),
        SIMDE_FLOAT32_C(  -438.90),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   499.16),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -329.65),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   208.30) },
      UINT16_C(    0) },
    /* 17 */
    { { SIMDE_FLOAT32_C(   212.20), SIMDE_FLOAT32_C(   388.63), SIMDE_FLOAT32_C(  -478.65), SIMDE_FLOAT32_C(  -436.77),
        SIMDE_FLOAT32_C(   360.13), SIMDE_FLOAT32_C(  -412.78), SIMDE_FLOAT32_C(  -236.94), SIMDE_FLOAT32_C(  -544.80),
        SIMDE_FLOAT32_C(  -671.39), SIMDE_FLOAT32_C(   984.28), SIMDE_FLOAT32_C(  -900.74), SIMDE_FLOAT32_C(  -524.46),
        SIMDE_FLOAT32_C(  -544.12), SIMDE_FLOAT32_C(  -574.52), SIMDE_FLOAT32_C(    73.83), SIMDE_FLOAT32_C(  -915.35) },
      { SIMDE_FLOAT32_C(   212.20), SIMDE_FLOAT32_C(   540.61), SIMDE_FLOAT32_C(  -478.65), SIMDE_FLOAT32_C(   453.33),
        SIMDE_FLOAT32_C(   360.13), SIMDE_FLOAT32_C(   110.69), SIMDE_FLOAT32_C(  -236.94), SIMDE_FLOAT32_C(  -774.89),
        SIMDE_FLOAT32_C(  -671.39), SIMDE_FLOAT32_C(   602.79), SIMDE_FLOAT32_C(  -900.74), SIMDE_FLOAT32_C(   222.49),
        SIMDE_FLOAT32_C(  -544.12), SIMDE_FLOAT32_C(   284.28), SIMDE_FLOAT32_C(    73.83), SIMDE_FLOAT32_C(   485.33) },
      UINT16_C(43050),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -47.87),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -966.96),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -188.38),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   867.96),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -412.50),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   251.78),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   417.34),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -288.31) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -481.71), SIMDE_FLOAT32_C(  -834.98),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   628.98),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -927.44),
        SIMDE_FLOAT32_C(   871.44),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   295.05),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   725.84),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   884.67) },
      UINT16_C(40960) },
    /* 18 */
    { { SIMDE_FLOAT32_C(  -322.02), SIMDE_FLOAT32_C(   678.46), SIMDE_FLOAT32_C(   917.71), SIMDE_FLOAT32_C(  -782.67),
        SIMDE_FLOAT32_C(  -509.92), SIMDE_FLOAT32_C(  -594.05), SIMDE_FLOAT32_C(  -914.71), SIMDE_FLOAT32_C(   285.99),
        SIMDE_FLOAT32_C(    -6.55), SIMDE_FLOAT32_C(  -571.21), SIMDE_FLOAT32_C(  -462.23), SIMDE_FLOAT32_C(  -993.57),
        SIMDE_FLOAT32_C(   846.13), SIMDE_FLOAT32_C(   874.20), SIMDE_FLOAT32_C(  -281.89), SIMDE_FLOAT32_C(  -195.92) },
      { SIMDE_FLOAT32_C(  -322.02), SIMDE_FLOAT32_C(  -116.87), SIMDE_FLOAT32_C(   917.71), SIMDE_FLOAT32_C(  -978.52),
        SIMDE_FLOAT32_C(  -509.92), SIMDE_FLOAT32_C(  -726.89), SIMDE_FLOAT32_C(  -914.71), SIMDE_FLOAT32_C(    23.22),
        SIMDE_FLOAT32_C(    -6.55), SIMDE_FLOAT32_C(   389.09), SIMDE_FLOAT32_C(  -462.23), SIMDE_FLOAT32_C(   412.34),
        SIMDE_FLOAT32_C(   846.13), SIMDE_FLOAT32_C(  -202.30), SIMDE_FLOAT32_C(  -281.89), SIMDE_FLOAT32_C(   792.91) },
      UINT16_C(57173),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   214.72),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -33.76),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -904.47),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -385.87),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -210.00),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -629.55),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   338.67),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    56.69) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -399.44), SIMDE_FLOAT32_C(    78.17),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -126.32),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    74.26),
        SIMDE_FLOAT32_C(  -438.70),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -513.39),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   783.62),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -161.54) },
      UINT16_C( 8320) },
    /* 19 */
    { { SIMDE_FLOAT32_C(    -1.66), SIMDE_FLOAT32_C(   479.39), SIMDE_FLOAT32_C(   804.70), SIMDE_FLOAT32_C(  -380.98),
        SIMDE_FLOAT32_C(   574.92), SIMDE_FLOAT32_C(  -943.07), SIMDE_FLOAT32_C(   233.15), SIMDE_FLOAT32_C(  -900.76),
        SIMDE_FLOAT32_C(  -153.07), SIMDE_FLOAT32_C(   853.70), SIMDE_FLOAT32_C(  -530.31), SIMDE_FLOAT32_C(   511.13),
        SIMDE_FLOAT32_C(   192.37), SIMDE_FLOAT32_C(   644.22), SIMDE_FLOAT32_C(  -432.18), SIMDE_FLOAT32_C(   414.16) },
      { SIMDE_FLOAT32_C(    -1.66), SIMDE_FLOAT32_C(   645.98), SIMDE_FLOAT32_C(   804.70), SIMDE_FLOAT32_C(   118.46),
        SIMDE_FLOAT32_C(   574.92), SIMDE_FLOAT32_C(   184.56), SIMDE_FLOAT32_C(   233.15), SIMDE_FLOAT32_C(   379.49),
        SIMDE_FLOAT32_C(  -153.07), SIMDE_FLOAT32_C(  -320.68), SIMDE_FLOAT32_C(  -530.31), SIMDE_FLOAT32_C(   111.46),
        SIMDE_FLOAT32_C(   192.37), SIMDE_FLOAT32_C(  -475.12), SIMDE_FLOAT32_C(  -432.18), SIMDE_FLOAT32_C(   461.29) },
      UINT16_C(    0),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   754.63),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   579.19),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   313.45),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -341.51),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -851.88),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -640.49),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -262.56),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -962.89) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -438.57), SIMDE_FLOAT32_C(   155.57),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   745.99),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -504.85),
        SIMDE_FLOAT32_C(  -972.39),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   606.61),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   556.54),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -834.01) },
      UINT16_C(24445) },
    /* 20 */
    { { SIMDE_FLOAT32_C(   311.16), SIMDE_FLOAT32_C(  -967.85), SIMDE_FLOAT32_C(   745.17), SIMDE_FLOAT32_C(   122.73),
        SIMDE_FLOAT32_C(   345.60), SIMDE_FLOAT32_C(   423.60), SIMDE_FLOAT32_C(   781.22), SIMDE_FLOAT32_C(  -487.25),
        SIMDE_FLOAT32_C(   571.71), SIMDE_FLOAT32_C(   950.85), SIMDE_FLOAT32_C(  -127.74), SIMDE_FLOAT32_C(   364.05),
        SIMDE_FLOAT32_C(  -311.72), SIMDE_FLOAT32_C(  -354.06), SIMDE_FLOAT32_C(   401.16), SIMDE_FLOAT32_C(    71.71) },
      { SIMDE_FLOAT32_C(   311.16), SIMDE_FLOAT32_C(  -443.27), SIMDE_FLOAT32_C(   745.17), SIMDE_FLOAT32_C(   -46.65),
        SIMDE_FLOAT32_C(   345.60), SIMDE_FLOAT32_C(  -145.57), SIMDE_FLOAT32_C(   781.22), SIMDE_FLOAT32_C(   932.63),
        SIMDE_FLOAT32_C(   571.71), SIMDE_FLOAT32_C(    55.11), SIMDE_FLOAT32_C(  -127.74), SIMDE_FLOAT32_C(  -347.01),
        SIMDE_FLOAT32_C(  -311.72), SIMDE_FLOAT32_C(  -624.97), SIMDE_FLOAT32_C(   401.16), SIMDE_FLOAT32_C(   922.81) },
      UINT16_C(43690),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -435.85),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   752.77),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -173.23),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   559.46),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   137.77),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   465.89),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -675.33),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    -8.92) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -189.07), SIMDE_FLOAT32_C(   944.44),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   665.36),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -843.37),
        SIMDE_FLOAT32_C(  -551.95),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -190.38),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   628.59),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   924.44) },
           UINT16_MAX },
    /* 21 */
    { { SIMDE_FLOAT32_C(  -807.26), SIMDE_FLOAT32_C(  -971.94), SIMDE_FLOAT32_C(   677.21), SIMDE_FLOAT32_C(  -819.52),
        SIMDE_FLOAT32_C(  -145.18), SIMDE_FLOAT32_C(   942.72), SIMDE_FLOAT32_C(   739.94), SIMDE_FLOAT32_C(   632.43),
        SIMDE_FLOAT32_C(    80.49), SIMDE_FLOAT32_C(  -336.55), SIMDE_FLOAT32_C(    98.33), SIMDE_FLOAT32_C(  -135.79),
        SIMDE_FLOAT32_C(   -11.88), SIMDE_FLOAT32_C(   635.93), SIMDE_FLOAT32_C(   855.29), SIMDE_FLOAT32_C(   869.51) },
      { SIMDE_FLOAT32_C(  -807.26), SIMDE_FLOAT32_C(   799.72), SIMDE_FLOAT32_C(   677.21), SIMDE_FLOAT32_C(  -887.78),
        SIMDE_FLOAT32_C(  -145.18), SIMDE_FLOAT32_C(   374.97), SIMDE_FLOAT32_C(   739.94), SIMDE_FLOAT32_C(   640.71),
        SIMDE_FLOAT32_C(    80.49), SIMDE_FLOAT32_C(    78.46), SIMDE_FLOAT32_C(    98.33), SIMDE_FLOAT32_C(  -965.53),
        SIMDE_FLOAT32_C(   -11.88), SIMDE_FLOAT32_C(   682.92), SIMDE_FLOAT32_C(   855.29), SIMDE_FLOAT32_C(  -100.21) },
      UINT16_C(56701),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   636.11),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -434.20),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -179.78),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -340.67),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   296.56),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -528.22),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   378.83),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   379.34) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    -2.78), SIMDE_FLOAT32_C(   491.56),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -627.80),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -110.59),
        SIMDE_FLOAT32_C(  -161.13),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -76.12),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -117.21),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -893.76) },
      UINT16_C(65407) },
    /* 22 */
    { { SIMDE_FLOAT32_C(  -481.10), SIMDE_FLOAT32_C(   525.99), SIMDE_FLOAT32_C(  -327.96), SIMDE_FLOAT32_C(  -902.27),
        SIMDE_FLOAT32_C(  -653.79), SIMDE_FLOAT32_C(  -129.73), SIMDE_FLOAT32_C(  -242.94), SIMDE_FLOAT32_C(   829.87),
        SIMDE_FLOAT32_C(  -833.18), SIMDE_FLOAT32_C(  -719.40), SIMDE_FLOAT32_C(  -698.35), SIMDE_FLOAT32_C(    99.31),
        SIMDE_FLOAT32_C(   659.43), SIMDE_FLOAT32_C(  -357.05), SIMDE_FLOAT32_C(  -521.35), SIMDE_FLOAT32_C(  -162.02) },
      { SIMDE_FLOAT32_C(  -481.10), SIMDE_FLOAT32_C(   970.21), SIMDE_FLOAT32_C(  -327.96), SIMDE_FLOAT32_C(  -987.63),
        SIMDE_FLOAT32_C(  -653.79), SIMDE_FLOAT32_C(   221.11), SIMDE_FLOAT32_C(  -242.94), SIMDE_FLOAT32_C(  -430.52),
        SIMDE_FLOAT32_C(  -833.18), SIMDE_FLOAT32_C(   825.65), SIMDE_FLOAT32_C(  -698.35), SIMDE_FLOAT32_C(  -671.30),
        SIMDE_FLOAT32_C(   659.43), SIMDE_FLOAT32_C(  -438.88), SIMDE_FLOAT32_C(  -521.35), SIMDE_FLOAT32_C(   227.34) },
      UINT16_C(10376),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   106.98),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -566.68),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -917.86),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -855.93),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -435.16),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -977.83),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   722.03),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -152.04) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -930.66), SIMDE_FLOAT32_C(  -139.67),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   290.45),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   223.90),
        SIMDE_FLOAT32_C(   587.75),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   552.60),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   987.54),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -244.02) },
      UINT16_C(57215) },
    /* 23 */
    { { SIMDE_FLOAT32_C(    94.52), SIMDE_FLOAT32_C(  -151.40), SIMDE_FLOAT32_C(   189.29), SIMDE_FLOAT32_C(    71.77),
        SIMDE_FLOAT32_C(   -69.27), SIMDE_FLOAT32_C(   452.48), SIMDE_FLOAT32_C(   215.84), SIMDE_FLOAT32_C(   293.47),
        SIMDE_FLOAT32_C(  -982.68), SIMDE_FLOAT32_C(  -540.78), SIMDE_FLOAT32_C(   315.64), SIMDE_FLOAT32_C(   225.11),
        SIMDE_FLOAT32_C(  -818.75), SIMDE_FLOAT32_C(  -824.22), SIMDE_FLOAT32_C(  -926.93), SIMDE_FLOAT32_C(   873.50) },
      { SIMDE_FLOAT32_C(    94.52), SIMDE_FLOAT32_C(   -66.61), SIMDE_FLOAT32_C(   189.29), SIMDE_FLOAT32_C(   535.57),
        SIMDE_FLOAT32_C(   -69.27), SIMDE_FLOAT32_C(   288.70), SIMDE_FLOAT32_C(   215.84), SIMDE_FLOAT32_C(  -716.76),
        SIMDE_FLOAT32_C(  -982.68), SIMDE_FLOAT32_C(  -687.93), SIMDE_FLOAT32_C(   315.64), SIMDE_FLOAT32_C(  -934.68),
        SIMDE_FLOAT32_C(  -818.75), SIMDE_FLOAT32_C(  -897.05), SIMDE_FLOAT32_C(  -926.93), SIMDE_FLOAT32_C(   394.14) },
           UINT16_MAX,
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -989.40),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   882.28),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   681.74),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   480.40),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -508.61),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -677.79),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -221.42),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -87.71) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   492.06), SIMDE_FLOAT32_C(  -552.13),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -219.24),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -822.78),
        SIMDE_FLOAT32_C(   519.41),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -757.46),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    63.84),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -675.37) },
      UINT16_C(41090) },
    /* 24 */
    { { SIMDE_FLOAT32_C(    74.44), SIMDE_FLOAT32_C(   679.07), SIMDE_FLOAT32_C(  -793.09), SIMDE_FLOAT32_C(  -462.48),
        SIMDE_FLOAT32_C(   360.82), SIMDE_FLOAT32_C(   382.66), SIMDE_FLOAT32_C(  -982.08), SIMDE_FLOAT32_C(   501.77),
        SIMDE_FLOAT32_C(   874.05), SIMDE_FLOAT32_C(  -276.57), SIMDE_FLOAT32_C(   823.98), SIMDE_FLOAT32_C(  -458.77),
        SIMDE_FLOAT32_C(   502.02), SIMDE_FLOAT32_C(  -980.32), SIMDE_FLOAT32_C(   453.52), SIMDE_FLOAT32_C(   213.99) },
      { SIMDE_FLOAT32_C(    74.44), SIMDE_FLOAT32_C(   901.39), SIMDE_FLOAT32_C(  -793.09), SIMDE_FLOAT32_C(  -707.50),
        SIMDE_FLOAT32_C(   360.82), SIMDE_FLOAT32_C(  -687.83), SIMDE_FLOAT32_C(  -982.08), SIMDE_FLOAT32_C(  -371.87),
        SIMDE_FLOAT32_C(   874.05), SIMDE_FLOAT32_C(  -287.74), SIMDE_FLOAT32_C(   823.98), SIMDE_FLOAT32_C(   -44.61),
        SIMDE_FLOAT32_C(   502.02), SIMDE_FLOAT32_C(   660.33), SIMDE_FLOAT32_C(   453.52), SIMDE_FLOAT32_C(  -149.46) },
      UINT16_C(21845),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   486.94),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -299.78),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   405.98),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -256.34),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -974.03),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   631.43),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -261.59),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   557.40) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -533.11), SIMDE_FLOAT32_C(   849.90),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -220.94),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -638.64),
        SIMDE_FLOAT32_C(    31.88),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   316.75),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -403.22),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   823.55) },
      UINT16_C(24445) },
    /* 25 */
    { { SIMDE_FLOAT32_C(  -916.28), SIMDE_FLOAT32_C(    46.58), SIMDE_FLOAT32_C(  -476.22), SIMDE_FLOAT32_C(   -46.67),
        SIMDE_FLOAT32_C(  -547.44), SIMDE_FLOAT32_C(  -274.23), SIMDE_FLOAT32_C(   696.99), SIMDE_FLOAT32_C(  -418.02),
        SIMDE_FLOAT32_C(  -248.26), SIMDE_FLOAT32_C(   -18.13), SIMDE_FLOAT32_C(  -786.59), SIMDE_FLOAT32_C(  -202.60),
        SIMDE_FLOAT32_C(   720.28), SIMDE_FLOAT32_C(    58.84), SIMDE_FLOAT32_C(  -645.21), SIMDE_FLOAT32_C(  -639.93) },
      { SIMDE_FLOAT32_C(  -916.28), SIMDE_FLOAT32_C(  -795.31), SIMDE_FLOAT32_C(  -476.22), SIMDE_FLOAT32_C(  -695.22),
        SIMDE_FLOAT32_C(  -547.44), SIMDE_FLOAT32_C(  -514.75), SIMDE_FLOAT32_C(   696.99), SIMDE_FLOAT32_C(   556.19),
        SIMDE_FLOAT32_C(  -248.26), SIMDE_FLOAT32_C(   982.89), SIMDE_FLOAT32_C(  -786.59), SIMDE_FLOAT32_C(   793.21),
        SIMDE_FLOAT32_C(   720.28), SIMDE_FLOAT32_C(  -977.32), SIMDE_FLOAT32_C(  -645.21), SIMDE_FLOAT32_C(  -336.60) },
      UINT16_C(35456),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -859.46),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   521.82),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   313.71),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   618.05),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -682.79),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -984.14),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -229.76),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -98.23) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -515.47), SIMDE_FLOAT32_C(   206.55),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   -30.23),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -721.17),
        SIMDE_FLOAT32_C(   855.59),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -927.95),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   688.81),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -488.46) },
      UINT16_C(32639) },
    /* 26 */
    { { SIMDE_FLOAT32_C(   829.35), SIMDE_FLOAT32_C(  -284.62), SIMDE_FLOAT32_C(  -966.64), SIMDE_FLOAT32_C(  -304.34),
        SIMDE_FLOAT32_C(  -970.91), SIMDE_FLOAT32_C(  -862.84), SIMDE_FLOAT32_C(  -686.29), SIMDE_FLOAT32_C(  -675.32),
        SIMDE_FLOAT32_C(  -545.63), SIMDE_FLOAT32_C(  -270.84), SIMDE_FLOAT32_C(  -659.46), SIMDE_FLOAT32_C(   830.42),
        SIMDE_FLOAT32_C(   499.40), SIMDE_FLOAT32_C(  -283.53), SIMDE_FLOAT32_C(  -267.81), SIMDE_FLOAT32_C(  -525.66) },
      { SIMDE_FLOAT32_C(   829.35), SIMDE_FLOAT32_C(   938.75), SIMDE_FLOAT32_C(  -966.64), SIMDE_FLOAT32_C(  -829.23),
        SIMDE_FLOAT32_C(  -970.91), SIMDE_FLOAT32_C(    29.02), SIMDE_FLOAT32_C(  -686.29), SIMDE_FLOAT32_C(   667.03),
        SIMDE_FLOAT32_C(  -545.63), SIMDE_FLOAT32_C(  -478.35), SIMDE_FLOAT32_C(  -659.46), SIMDE_FLOAT32_C(   890.90),
        SIMDE_FLOAT32_C(   499.40), SIMDE_FLOAT32_C(   200.96), SIMDE_FLOAT32_C(  -267.81), SIMDE_FLOAT32_C(  -960.19) },
      UINT16_C(32247),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -564.19),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   945.43),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    49.18),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    27.34),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -389.35),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   277.74),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   589.96),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -471.89) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   725.67), SIMDE_FLOAT32_C(  -301.12),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -245.31),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   203.31),
        SIMDE_FLOAT32_C(   670.13),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    94.21),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   496.66),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -773.23) },
      UINT16_C(24575) },
    /* 27 */
    { { SIMDE_FLOAT32_C(   932.47), SIMDE_FLOAT32_C(  -344.14), SIMDE_FLOAT32_C(  -827.80), SIMDE_FLOAT32_C(  -494.56),
        SIMDE_FLOAT32_C(   705.04), SIMDE_FLOAT32_C(  -557.69), SIMDE_FLOAT32_C(   532.78), SIMDE_FLOAT32_C(   483.37),
        SIMDE_FLOAT32_C(    52.96), SIMDE_FLOAT32_C(  -609.46), SIMDE_FLOAT32_C(  -238.89), SIMDE_FLOAT32_C(  -619.92),
        SIMDE_FLOAT32_C(   980.50), SIMDE_FLOAT32_C(  -486.81), SIMDE_FLOAT32_C(   -91.81), SIMDE_FLOAT32_C(   509.20) },
      { SIMDE_FLOAT32_C(   932.47), SIMDE_FLOAT32_C(   607.07), SIMDE_FLOAT32_C(  -827.80), SIMDE_FLOAT32_C(    -6.45),
        SIMDE_FLOAT32_C(   705.04), SIMDE_FLOAT32_C(   856.52), SIMDE_FLOAT32_C(   532.78), SIMDE_FLOAT32_C(  -574.32),
        SIMDE_FLOAT32_C(    52.96), SIMDE_FLOAT32_C(   291.07), SIMDE_FLOAT32_C(  -238.89), SIMDE_FLOAT32_C(   276.42),
        SIMDE_FLOAT32_C(   980.50), SIMDE_FLOAT32_C(  -773.35), SIMDE_FLOAT32_C(   -91.81), SIMDE_FLOAT32_C(  -279.80) },
      UINT16_C(    0),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   675.39),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -412.45),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -241.58),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   170.67),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -167.96),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   129.46),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -541.07),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   584.09) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   488.00), SIMDE_FLOAT32_C(  -422.36),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   344.53),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -689.48),
        SIMDE_FLOAT32_C(  -934.43),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   586.95),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    90.14),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -337.34) },
      UINT16_C(    0) },
    /* 28 */
    { { SIMDE_FLOAT32_C(  -234.47), SIMDE_FLOAT32_C(  -200.85), SIMDE_FLOAT32_C(   250.20), SIMDE_FLOAT32_C(  -116.76),
        SIMDE_FLOAT32_C(   557.57), SIMDE_FLOAT32_C(   321.13), SIMDE_FLOAT32_C(  -946.09), SIMDE_FLOAT32_C(  -293.47),
        SIMDE_FLOAT32_C(  -846.84), SIMDE_FLOAT32_C(   604.65), SIMDE_FLOAT32_C(   835.99), SIMDE_FLOAT32_C(   498.39),
        SIMDE_FLOAT32_C(  -936.41), SIMDE_FLOAT32_C(  -525.36), SIMDE_FLOAT32_C(    82.48), SIMDE_FLOAT32_C(  -870.41) },
      { SIMDE_FLOAT32_C(  -234.47), SIMDE_FLOAT32_C(   660.12), SIMDE_FLOAT32_C(   250.20), SIMDE_FLOAT32_C(   307.17),
        SIMDE_FLOAT32_C(   557.57), SIMDE_FLOAT32_C(  -801.63), SIMDE_FLOAT32_C(  -946.09), SIMDE_FLOAT32_C(  -499.82),
        SIMDE_FLOAT32_C(  -846.84), SIMDE_FLOAT32_C(   204.64), SIMDE_FLOAT32_C(   835.99), SIMDE_FLOAT32_C(   532.00),
        SIMDE_FLOAT32_C(  -936.41), SIMDE_FLOAT32_C(   -73.01), SIMDE_FLOAT32_C(    82.48), SIMDE_FLOAT32_C(    60.32) },
      UINT16_C(43690),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   444.86),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   283.71),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   997.47),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -80.85),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   826.23),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   665.71),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   500.03),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -736.49) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -253.56), SIMDE_FLOAT32_C(   570.69),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   -55.20),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -303.34),
        SIMDE_FLOAT32_C(  -606.97),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -771.34),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -576.68),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -898.43) },
      UINT16_C(41090) },
    /* 29 */
    { { SIMDE_FLOAT32_C(   868.18), SIMDE_FLOAT32_C(   691.69), SIMDE_FLOAT32_C(   385.28), SIMDE_FLOAT32_C(  -365.83),
        SIMDE_FLOAT32_C(   689.16), SIMDE_FLOAT32_C(   375.52), SIMDE_FLOAT32_C(   553.32), SIMDE_FLOAT32_C(   291.29),
        SIMDE_FLOAT32_C(   201.74), SIMDE_FLOAT32_C(   970.87), SIMDE_FLOAT32_C(   -43.00), SIMDE_FLOAT32_C(   502.61),
        SIMDE_FLOAT32_C(   470.90), SIMDE_FLOAT32_C(  -247.71), SIMDE_FLOAT32_C(   766.12), SIMDE_FLOAT32_C(  -368.95) },
      { SIMDE_FLOAT32_C(   868.18), SIMDE_FLOAT32_C(   336.81), SIMDE_FLOAT32_C(   385.28), SIMDE_FLOAT32_C(  -556.47),
        SIMDE_FLOAT32_C(   689.16), SIMDE_FLOAT32_C(  -679.24), SIMDE_FLOAT32_C(   553.32), SIMDE_FLOAT32_C(   -81.78),
        SIMDE_FLOAT32_C(   201.74), SIMDE_FLOAT32_C(   368.86), SIMDE_FLOAT32_C(   -43.00), SIMDE_FLOAT32_C(  -855.37),
        SIMDE_FLOAT32_C(   470.90), SIMDE_FLOAT32_C(   354.17), SIMDE_FLOAT32_C(   766.12), SIMDE_FLOAT32_C(   660.36) },
      UINT16_C(24575),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   631.47),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -264.97),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   847.86),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -791.27),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -16.69),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   289.62),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   477.47),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   234.34) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -853.51), SIMDE_FLOAT32_C(   677.87),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -532.75),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -763.55),
        SIMDE_FLOAT32_C(  -813.08),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -618.92),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -372.72),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -736.24) },
      UINT16_C(40962) },
    /* 30 */
    { { SIMDE_FLOAT32_C(  -741.25), SIMDE_FLOAT32_C(   933.99), SIMDE_FLOAT32_C(    -1.21), SIMDE_FLOAT32_C(   265.74),
        SIMDE_FLOAT32_C(   781.85), SIMDE_FLOAT32_C(    25.11), SIMDE_FLOAT32_C(   474.47), SIMDE_FLOAT32_C(   600.57),
        SIMDE_FLOAT32_C(  -991.58), SIMDE_FLOAT32_C(   185.81), SIMDE_FLOAT32_C(  -109.80), SIMDE_FLOAT32_C(   744.03),
        SIMDE_FLOAT32_C(  -336.72), SIMDE_FLOAT32_C(  -189.13), SIMDE_FLOAT32_C(   -21.63), SIMDE_FLOAT32_C(   477.56) },
      { SIMDE_FLOAT32_C(  -741.25), SIMDE_FLOAT32_C(  -343.76), SIMDE_FLOAT32_C(    -1.21), SIMDE_FLOAT32_C(   424.60),
        SIMDE_FLOAT32_C(   781.85), SIMDE_FLOAT32_C(    74.73), SIMDE_FLOAT32_C(   474.47), SIMDE_FLOAT32_C(   661.22),
        SIMDE_FLOAT32_C(  -991.58), SIMDE_FLOAT32_C(  -957.87), SIMDE_FLOAT32_C(  -109.80), SIMDE_FLOAT32_C(  -843.64),
        SIMDE_FLOAT32_C(  -336.72), SIMDE_FLOAT32_C(   279.78), SIMDE_FLOAT32_C(   -21.63), SIMDE_FLOAT32_C(   -71.84) },
      UINT16_C(35330),
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   418.92),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    -4.38),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   668.37),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -547.55),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   486.39),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   517.46),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   174.85),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   254.61) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   812.05), SIMDE_FLOAT32_C(  -320.78),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(  -113.23),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   825.23),
        SIMDE_FLOAT32_C(  -617.60),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   981.60),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -598.28),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   100.46) },
      UINT16_C(40960) },
    /* 31 */
    { { SIMDE_FLOAT32_C(   820.64), SIMDE_FLOAT32_C(  -826.14), SIMDE_FLOAT32_C(  -903.92), SIMDE_FLOAT32_C(  -735.34),
        SIMDE_FLOAT32_C(   842.23), SIMDE_FLOAT32_C(  -307.72), SIMDE_FLOAT32_C(  -282.89), SIMDE_FLOAT32_C(  -303.59),
        SIMDE_FLOAT32_C(  -821.33), SIMDE_FLOAT32_C(   -86.41), SIMDE_FLOAT32_C(  -786.13), SIMDE_FLOAT32_C(  -524.07),
        SIMDE_FLOAT32_C(  -911.56), SIMDE_FLOAT32_C(   208.89), SIMDE_FLOAT32_C(   730.54), SIMDE_FLOAT32_C(   919.52) },
      { SIMDE_FLOAT32_C(   820.64), SIMDE_FLOAT32_C(  -590.24), SIMDE_FLOAT32_C(  -903.92), SIMDE_FLOAT32_C(   907.71),
        SIMDE_FLOAT32_C(   842.23), SIMDE_FLOAT32_C(  -808.49), SIMDE_FLOAT32_C(  -282.89), SIMDE_FLOAT32_C(   132.43),
        SIMDE_FLOAT32_C(  -821.33), SIMDE_FLOAT32_C(   714.54), SIMDE_FLOAT32_C(  -786.13), SIMDE_FLOAT32_C(  -314.88),
        SIMDE_FLOAT32_C(  -911.56), SIMDE_FLOAT32_C(  -835.79), SIMDE_FLOAT32_C(   730.54), SIMDE_FLOAT32_C(   936.90) },
           UINT16_MAX,
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   881.66),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -819.70),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -81.32),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -247.39),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    90.58),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   920.71),
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   -40.93),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   320.41) },
      {            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(    65.14), SIMDE_FLOAT32_C(   228.12),            SIMDE_MATH_NANF,
        SIMDE_FLOAT32_C(   256.65),            SIMDE_MATH_NANF,            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -944.93),
        SIMDE_FLOAT32_C(  -324.40),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -259.81),            SIMDE_MATH_NANF,
                   SIMDE_MATH_NANF, SIMDE_FLOAT32_C(  -474.23),            SIMDE_MATH_NANF, SIMDE_FLOAT32_C(   937.81) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 b = simde_mm512_loadu_ps(test_vec[i].b);
    simde__mmask16 r = simde_mm512_cmp_ps_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  const simde__m512 nans = simde_mm512_set1_ps(SIMDE_MATH_NAN);
  for (int i = 0 ; i < 32 ; i++) {
    simde__m512 a, b;
    simde__mmask16 r;

    fprintf(stdout, "    /* %d */\n", i);
    a = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    b = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    b = simde_mm512_mask_blend_ps(UINT16_C(0x5555), b, a);
    SIMDE_CONSTIFY_32_(simde_mm512_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f32x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f32x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_MIDDLE);

    a = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    b = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    a = simde_mm512_mask_blend_ps(UINT16_C(0x5555), a, nans);
    b = simde_mm512_mask_blend_ps(UINT16_C(0x5a69), b, nans);
    SIMDE_CONSTIFY_32_(simde_mm512_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f32x16(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_f32x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_pd_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float64 a[8];
    const simde_float64 b[8];
    const simde__mmask8 r;
    const simde_float64 an[8];
    const simde_float64 bn[8];
    const simde__mmask8 rn;
  } test_vec[] = {
    /* 0 */
    { { SIMDE_FLOAT64_C(   551.79), SIMDE_FLOAT64_C(  -108.45), SIMDE_FLOAT64_C(  -398.72), SIMDE_FLOAT64_C(   368.79),
        SIMDE_FLOAT64_C(   -43.58), SIMDE_FLOAT64_C(  -218.58), SIMDE_FLOAT64_C(   931.70), SIMDE_FLOAT64_C(  -486.02) },
      { SIMDE_FLOAT64_C(   551.79), SIMDE_FLOAT64_C(   997.86), SIMDE_FLOAT64_C(  -398.72), SIMDE_FLOAT64_C(  -815.22),
        SIMDE_FLOAT64_C(   -43.58), SIMDE_FLOAT64_C(   -45.15), SIMDE_FLOAT64_C(   931.70), SIMDE_FLOAT64_C(  -140.51) },
      UINT8_C( 85),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   512.71),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   376.68),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   785.98),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -995.75) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   642.20), SIMDE_FLOAT64_C(   401.69),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -521.66),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -969.87) },
      UINT8_C(  0) },
    /* 1 */
    { { SIMDE_FLOAT64_C(  -865.44), SIMDE_FLOAT64_C(   227.90), SIMDE_FLOAT64_C(   398.92), SIMDE_FLOAT64_C(    90.99),
        SIMDE_FLOAT64_C(  -990.69), SIMDE_FLOAT64_C(   330.61), SIMDE_FLOAT64_C(   604.97), SIMDE_FLOAT64_C(  -770.49) },
      { SIMDE_FLOAT64_C(  -865.44), SIMDE_FLOAT64_C(  -245.16), SIMDE_FLOAT64_C(   398.92), SIMDE_FLOAT64_C(  -933.19),
        SIMDE_FLOAT64_C(  -990.69), SIMDE_FLOAT64_C(   205.09), SIMDE_FLOAT64_C(   604.97), SIMDE_FLOAT64_C(   822.25) },
      UINT8_C(128),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -547.40),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -963.97),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   625.06),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -949.70) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   441.97), SIMDE_FLOAT64_C(  -981.98),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -315.02),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -180.46) },
      UINT8_C(130) },
    /* 2 */
    { { SIMDE_FLOAT64_C(   872.54), SIMDE_FLOAT64_C(   174.64), SIMDE_FLOAT64_C(   910.53), SIMDE_FLOAT64_C(   881.85),
        SIMDE_FLOAT64_C(  -494.74), SIMDE_FLOAT64_C(   515.50), SIMDE_FLOAT64_C(  -888.63), SIMDE_FLOAT64_C(   833.73) },
      { SIMDE_FLOAT64_C(   872.54), SIMDE_FLOAT64_C(  -474.34), SIMDE_FLOAT64_C(   910.53), SIMDE_FLOAT64_C(   980.03),
        SIMDE_FLOAT64_C(  -494.74), SIMDE_FLOAT64_C(  -173.16), SIMDE_FLOAT64_C(  -888.63), SIMDE_FLOAT64_C(  -551.45) },
      UINT8_C( 93),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(     1.21),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   518.01),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -475.13),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   893.52) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   586.34), SIMDE_FLOAT64_C(   639.11),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -769.02),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -896.48) },
      UINT8_C(  2) },
    /* 3 */
    { { SIMDE_FLOAT64_C(  -410.52), SIMDE_FLOAT64_C(   381.88), SIMDE_FLOAT64_C(   985.37), SIMDE_FLOAT64_C(    94.73),
        SIMDE_FLOAT64_C(  -102.62), SIMDE_FLOAT64_C(  -903.26), SIMDE_FLOAT64_C(   -71.53), SIMDE_FLOAT64_C(   167.72) },
      { SIMDE_FLOAT64_C(  -410.52), SIMDE_FLOAT64_C(  -170.99), SIMDE_FLOAT64_C(   985.37), SIMDE_FLOAT64_C(  -646.85),
        SIMDE_FLOAT64_C(  -102.62), SIMDE_FLOAT64_C(   -49.97), SIMDE_FLOAT64_C(   -71.53), SIMDE_FLOAT64_C(   -64.72) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   286.29),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -422.49),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -978.40),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   777.99) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -889.86), SIMDE_FLOAT64_C(  -570.20),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   524.98),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -885.54) },
      UINT8_C(125) },
    /* 4 */
    { { SIMDE_FLOAT64_C(   283.04), SIMDE_FLOAT64_C(   927.80), SIMDE_FLOAT64_C(   209.19), SIMDE_FLOAT64_C(  -819.58),
        SIMDE_FLOAT64_C(  -975.45), SIMDE_FLOAT64_C(  -862.35), SIMDE_FLOAT64_C(   348.14), SIMDE_FLOAT64_C(  -353.05) },
      { SIMDE_FLOAT64_C(   283.04), SIMDE_FLOAT64_C(  -504.10), SIMDE_FLOAT64_C(   209.19), SIMDE_FLOAT64_C(  -377.49),
        SIMDE_FLOAT64_C(  -975.45), SIMDE_FLOAT64_C(   801.80), SIMDE_FLOAT64_C(   348.14), SIMDE_FLOAT64_C(   397.17) },
      UINT8_C(170),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -988.93),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -100.76),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -554.28),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   640.60) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   107.03), SIMDE_FLOAT64_C(   479.52),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(     8.19),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -708.77) },
         UINT8_MAX },
    /* 5 */
    { { SIMDE_FLOAT64_C(   349.75), SIMDE_FLOAT64_C(   404.48), SIMDE_FLOAT64_C(  -528.34), SIMDE_FLOAT64_C(   374.30),
        SIMDE_FLOAT64_C(   542.14), SIMDE_FLOAT64_C(   819.80), SIMDE_FLOAT64_C(  -978.75), SIMDE_FLOAT64_C(  -491.20) },
      { SIMDE_FLOAT64_C(   349.75), SIMDE_FLOAT64_C(    21.34), SIMDE_FLOAT64_C(  -528.34), SIMDE_FLOAT64_C(   761.63),
        SIMDE_FLOAT64_C(   542.14), SIMDE_FLOAT64_C(  -310.91), SIMDE_FLOAT64_C(  -978.75), SIMDE_FLOAT64_C(   911.23) },
      UINT8_C(119),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   133.48),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -267.17),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   487.70),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -864.94) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   852.95), SIMDE_FLOAT64_C(  -784.10),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   274.90),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -375.35) },
      UINT8_C(125) },
    /* 6 */
    { { SIMDE_FLOAT64_C(   815.67), SIMDE_FLOAT64_C(  -634.19), SIMDE_FLOAT64_C(   998.95), SIMDE_FLOAT64_C(   357.81),
        SIMDE_FLOAT64_C(  -814.39), SIMDE_FLOAT64_C(  -979.80), SIMDE_FLOAT64_C(   866.61), SIMDE_FLOAT64_C(  -498.70) },
      { SIMDE_FLOAT64_C(   815.67), SIMDE_FLOAT64_C(    -2.09), SIMDE_FLOAT64_C(   998.95), SIMDE_FLOAT64_C(   864.68),
        SIMDE_FLOAT64_C(  -814.39), SIMDE_FLOAT64_C(   421.73), SIMDE_FLOAT64_C(   866.61), SIMDE_FLOAT64_C(  -612.85) },
      UINT8_C(128),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -413.62),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -865.59),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   493.41),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -331.19) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -514.63), SIMDE_FLOAT64_C(  -728.27),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -103.44),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -287.76) },
      UINT8_C(127) },
    /* 7 */
    { { SIMDE_FLOAT64_C(  -468.30), SIMDE_FLOAT64_C(   244.87), SIMDE_FLOAT64_C(  -929.95), SIMDE_FLOAT64_C(  -282.70),
        SIMDE_FLOAT64_C(   265.06), SIMDE_FLOAT64_C(   936.65), SIMDE_FLOAT64_C(   218.61), SIMDE_FLOAT64_C(  -693.40) },
      { SIMDE_FLOAT64_C(  -468.30), SIMDE_FLOAT64_C(   481.54), SIMDE_FLOAT64_C(  -929.95), SIMDE_FLOAT64_C(  -378.44),
        SIMDE_FLOAT64_C(   265.06), SIMDE_FLOAT64_C(   947.20), SIMDE_FLOAT64_C(   218.61), SIMDE_FLOAT64_C(   458.48) },
         UINT8_MAX,
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -871.31),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -392.33),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   862.36),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   -31.54) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   548.21), SIMDE_FLOAT64_C(   589.72),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   714.09),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -754.21) },
      UINT8_C(130) },
    /* 8 */
    { { SIMDE_FLOAT64_C(  -919.50), SIMDE_FLOAT64_C(  -973.42), SIMDE_FLOAT64_C(   -36.91), SIMDE_FLOAT64_C(   345.56),
        SIMDE_FLOAT64_C(   963.23), SIMDE_FLOAT64_C(  -818.30), SIMDE_FLOAT64_C(   652.16), SIMDE_FLOAT64_C(  -102.20) },
      { SIMDE_FLOAT64_C(  -919.50), SIMDE_FLOAT64_C(   823.45), SIMDE_FLOAT64_C(   -36.91), SIMDE_FLOAT64_C(  -433.49),
        SIMDE_FLOAT64_C(   963.23), SIMDE_FLOAT64_C(  -471.94), SIMDE_FLOAT64_C(   652.16), SIMDE_FLOAT64_C(  -695.76) },
      UINT8_C( 85),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   617.88),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -721.15),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -811.62),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   827.98) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -162.97), SIMDE_FLOAT64_C(    72.27),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   672.66),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   753.15) },
      UINT8_C(125) },
    /* 9 */
    { { SIMDE_FLOAT64_C(    55.38), SIMDE_FLOAT64_C(  -340.43), SIMDE_FLOAT64_C(    98.71), SIMDE_FLOAT64_C(    18.61),
        SIMDE_FLOAT64_C(  -158.72), SIMDE_FLOAT64_C(  -249.13), SIMDE_FLOAT64_C(   916.41), SIMDE_FLOAT64_C(  -495.48) },
      { SIMDE_FLOAT64_C(    55.38), SIMDE_FLOAT64_C(   435.76), SIMDE_FLOAT64_C(    98.71), SIMDE_FLOAT64_C(  -655.02),
        SIMDE_FLOAT64_C(  -158.72), SIMDE_FLOAT64_C(    96.02), SIMDE_FLOAT64_C(   916.41), SIMDE_FLOAT64_C(  -379.43) },
      UINT8_C(162),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   561.12),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -805.86),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -853.27),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   486.09) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    94.38), SIMDE_FLOAT64_C(   -63.22),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -876.82),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   178.56) },
      UINT8_C(125) },
    /* 10 */
    { { SIMDE_FLOAT64_C(   292.83), SIMDE_FLOAT64_C(   508.27), SIMDE_FLOAT64_C(  -802.82), SIMDE_FLOAT64_C(  -865.89),
        SIMDE_FLOAT64_C(  -740.85), SIMDE_FLOAT64_C(  -886.42), SIMDE_FLOAT64_C(  -361.38), SIMDE_FLOAT64_C(  -166.53) },
      { SIMDE_FLOAT64_C(   292.83), SIMDE_FLOAT64_C(   709.65), SIMDE_FLOAT64_C(  -802.82), SIMDE_FLOAT64_C(   513.17),
        SIMDE_FLOAT64_C(  -740.85), SIMDE_FLOAT64_C(   827.67), SIMDE_FLOAT64_C(  -361.38), SIMDE_FLOAT64_C(   519.56) },
         UINT8_MAX,
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    33.17),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   138.29),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   735.82),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -836.35) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   561.16), SIMDE_FLOAT64_C(  -179.93),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -805.58),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   487.25) },
         UINT8_MAX },
    /* 11 */
    { { SIMDE_FLOAT64_C(  -262.09), SIMDE_FLOAT64_C(  -670.88), SIMDE_FLOAT64_C(   621.35), SIMDE_FLOAT64_C(    -2.95),
        SIMDE_FLOAT64_C(  -557.29), SIMDE_FLOAT64_C(  -740.02), SIMDE_FLOAT64_C(   830.53), SIMDE_FLOAT64_C(   992.05) },
      { SIMDE_FLOAT64_C(  -262.09), SIMDE_FLOAT64_C(     8.98), SIMDE_FLOAT64_C(   621.35), SIMDE_FLOAT64_C(  -224.71),
        SIMDE_FLOAT64_C(  -557.29), SIMDE_FLOAT64_C(   638.96), SIMDE_FLOAT64_C(   830.53), SIMDE_FLOAT64_C(  -774.57) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -991.44),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   852.02),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   988.10),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -425.43) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -164.26), SIMDE_FLOAT64_C(   527.96),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -934.62),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -196.72) },
      UINT8_C(  0) },
    /* 12 */
    { { SIMDE_FLOAT64_C(   989.03), SIMDE_FLOAT64_C(  -147.72), SIMDE_FLOAT64_C(   800.34), SIMDE_FLOAT64_C(  -568.26),
        SIMDE_FLOAT64_C(   112.25), SIMDE_FLOAT64_C(   630.86), SIMDE_FLOAT64_C(  -576.21), SIMDE_FLOAT64_C(    81.88) },
      { SIMDE_FLOAT64_C(   989.03), SIMDE_FLOAT64_C(   929.01), SIMDE_FLOAT64_C(   800.34), SIMDE_FLOAT64_C(   476.49),
        SIMDE_FLOAT64_C(   112.25), SIMDE_FLOAT64_C(  -847.97), SIMDE_FLOAT64_C(  -576.21), SIMDE_FLOAT64_C(  -759.91) },
      UINT8_C(170),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    65.64),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   904.96),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -892.22),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -397.01) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -992.51), SIMDE_FLOAT64_C(  -653.33),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   667.39),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   656.43) },
      UINT8_C(130) },
    /* 13 */
    { { SIMDE_FLOAT64_C(  -570.13), SIMDE_FLOAT64_C(   612.52), SIMDE_FLOAT64_C(  -911.84), SIMDE_FLOAT64_C(   542.12),
        SIMDE_FLOAT64_C(   243.38), SIMDE_FLOAT64_C(  -488.05), SIMDE_FLOAT64_C(  -376.00), SIMDE_FLOAT64_C(   883.22) },
      { SIMDE_FLOAT64_C(  -570.13), SIMDE_FLOAT64_C(  -518.83), SIMDE_FLOAT64_C(  -911.84), SIMDE_FLOAT64_C(  -991.08),
        SIMDE_FLOAT64_C(   243.38), SIMDE_FLOAT64_C(    61.64), SIMDE_FLOAT64_C(  -376.00), SIMDE_FLOAT64_C(  -206.22) },
      UINT8_C(223),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -658.88),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   181.01),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   178.27),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   392.42) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -869.33), SIMDE_FLOAT64_C(  -598.68),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   708.27),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -861.86) },
      UINT8_C(130) },
    /* 14 */
    { { SIMDE_FLOAT64_C(   826.02), SIMDE_FLOAT64_C(  -402.26), SIMDE_FLOAT64_C(   680.26), SIMDE_FLOAT64_C(    69.41),
        SIMDE_FLOAT64_C(   109.69), SIMDE_FLOAT64_C(  -695.73), SIMDE_FLOAT64_C(   -47.37), SIMDE_FLOAT64_C(   550.65) },
      { SIMDE_FLOAT64_C(   826.02), SIMDE_FLOAT64_C(  -687.65), SIMDE_FLOAT64_C(   680.26), SIMDE_FLOAT64_C(   418.64),
        SIMDE_FLOAT64_C(   109.69), SIMDE_FLOAT64_C(   808.58), SIMDE_FLOAT64_C(   -47.37), SIMDE_FLOAT64_C(   501.25) },
      UINT8_C(130),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   -88.84),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -401.40),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -533.73),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   275.18) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -607.65), SIMDE_FLOAT64_C(  -871.67),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -394.15),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -568.12) },
      UINT8_C(130) },
    /* 15 */
    { { SIMDE_FLOAT64_C(   235.64), SIMDE_FLOAT64_C(  -876.38), SIMDE_FLOAT64_C(   501.28), SIMDE_FLOAT64_C(  -654.66),
        SIMDE_FLOAT64_C(  -572.12), SIMDE_FLOAT64_C(  -546.08), SIMDE_FLOAT64_C(   895.99), SIMDE_FLOAT64_C(   213.32) },
      { SIMDE_FLOAT64_C(   235.64), SIMDE_FLOAT64_C(   455.56), SIMDE_FLOAT64_C(   501.28), SIMDE_FLOAT64_C(  -859.76),
        SIMDE_FLOAT64_C(  -572.12), SIMDE_FLOAT64_C(  -155.61), SIMDE_FLOAT64_C(   895.99), SIMDE_FLOAT64_C(   413.84) },
         UINT8_MAX,
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -676.24),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   844.98),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -996.54),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   386.98) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   248.49), SIMDE_FLOAT64_C(   692.19),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   886.39),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   122.04) },
         UINT8_MAX },
    /* 16 */
    { { SIMDE_FLOAT64_C(  -740.84), SIMDE_FLOAT64_C(   -65.18), SIMDE_FLOAT64_C(   467.37), SIMDE_FLOAT64_C(  -312.96),
        SIMDE_FLOAT64_C(   388.73), SIMDE_FLOAT64_C(   363.36), SIMDE_FLOAT64_C(   900.37), SIMDE_FLOAT64_C(  -845.00) },
      { SIMDE_FLOAT64_C(  -740.84), SIMDE_FLOAT64_C(  -467.67), SIMDE_FLOAT64_C(   467.37), SIMDE_FLOAT64_C(  -916.94),
        SIMDE_FLOAT64_C(   388.73), SIMDE_FLOAT64_C(   936.73), SIMDE_FLOAT64_C(   900.37), SIMDE_FLOAT64_C(   132.27) },
      UINT8_C( 85),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   509.34),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    50.53),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -902.60),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   908.60) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -870.31), SIMDE_FLOAT64_C(   -89.74),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -734.77),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -475.61) },
      UINT8_C(  0) },
    /* 17 */
    { { SIMDE_FLOAT64_C(   278.61), SIMDE_FLOAT64_C(   821.69), SIMDE_FLOAT64_C(   211.44), SIMDE_FLOAT64_C(  -332.65),
        SIMDE_FLOAT64_C(   185.06), SIMDE_FLOAT64_C(   111.80), SIMDE_FLOAT64_C(  -177.66), SIMDE_FLOAT64_C(  -996.02) },
      { SIMDE_FLOAT64_C(   278.61), SIMDE_FLOAT64_C(   117.58), SIMDE_FLOAT64_C(   211.44), SIMDE_FLOAT64_C(    20.85),
        SIMDE_FLOAT64_C(   185.06), SIMDE_FLOAT64_C(   583.95), SIMDE_FLOAT64_C(  -177.66), SIMDE_FLOAT64_C(   314.81) },
      UINT8_C(168),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   130.36),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -393.91),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -197.17),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -426.35) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   424.95), SIMDE_FLOAT64_C(   805.93),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   768.74),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    47.35) },
      UINT8_C(130) },
    /* 18 */
    { { SIMDE_FLOAT64_C(   981.94), SIMDE_FLOAT64_C(   -66.41), SIMDE_FLOAT64_C(   714.70), SIMDE_FLOAT64_C(   167.00),
        SIMDE_FLOAT64_C(  -954.60), SIMDE_FLOAT64_C(  -462.95), SIMDE_FLOAT64_C(   170.98), SIMDE_FLOAT64_C(   689.53) },
      { SIMDE_FLOAT64_C(   981.94), SIMDE_FLOAT64_C(   258.02), SIMDE_FLOAT64_C(   714.70), SIMDE_FLOAT64_C(  -291.06),
        SIMDE_FLOAT64_C(  -954.60), SIMDE_FLOAT64_C(  -136.50), SIMDE_FLOAT64_C(   170.98), SIMDE_FLOAT64_C(   935.26) },
      UINT8_C(247),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -610.92),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   221.63),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    56.03),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -875.56) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   601.22), SIMDE_FLOAT64_C(   322.20),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   761.47),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   743.41) },
      UINT8_C(130) },
    /* 19 */
    { { SIMDE_FLOAT64_C(   977.95), SIMDE_FLOAT64_C(    11.77), SIMDE_FLOAT64_C(   -89.59), SIMDE_FLOAT64_C(  -976.66),
        SIMDE_FLOAT64_C(   548.82), SIMDE_FLOAT64_C(  -918.61), SIMDE_FLOAT64_C(   712.87), SIMDE_FLOAT64_C(   203.44) },
      { SIMDE_FLOAT64_C(   977.95), SIMDE_FLOAT64_C(  -576.74), SIMDE_FLOAT64_C(   -89.59), SIMDE_FLOAT64_C(  -818.62),
        SIMDE_FLOAT64_C(   548.82), SIMDE_FLOAT64_C(   936.13), SIMDE_FLOAT64_C(   712.87), SIMDE_FLOAT64_C(   280.62) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -342.02),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   517.13),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -702.47),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -805.01) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   -36.23), SIMDE_FLOAT64_C(  -555.29),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -991.87),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   986.07) },
      UINT8_C(125) },
    /* 20 */
    { { SIMDE_FLOAT64_C(  -246.45), SIMDE_FLOAT64_C(  -685.96), SIMDE_FLOAT64_C(  -990.58), SIMDE_FLOAT64_C(  -697.63),
        SIMDE_FLOAT64_C(  -604.57), SIMDE_FLOAT64_C(   722.29), SIMDE_FLOAT64_C(   505.81), SIMDE_FLOAT64_C(   734.84) },
      { SIMDE_FLOAT64_C(  -246.45), SIMDE_FLOAT64_C(   418.19), SIMDE_FLOAT64_C(  -990.58), SIMDE_FLOAT64_C(   432.30),
        SIMDE_FLOAT64_C(  -604.57), SIMDE_FLOAT64_C(  -967.13), SIMDE_FLOAT64_C(   505.81), SIMDE_FLOAT64_C(   679.54) },
      UINT8_C(170),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   215.18),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -595.13),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -161.76),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   411.47) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    44.57), SIMDE_FLOAT64_C(    71.68),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -213.66),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   539.89) },
         UINT8_MAX },
    /* 21 */
    { { SIMDE_FLOAT64_C(  -210.65), SIMDE_FLOAT64_C(   805.62), SIMDE_FLOAT64_C(   842.26), SIMDE_FLOAT64_C(   184.78),
        SIMDE_FLOAT64_C(   527.91), SIMDE_FLOAT64_C(   348.07), SIMDE_FLOAT64_C(   -80.37), SIMDE_FLOAT64_C(   673.46) },
      { SIMDE_FLOAT64_C(  -210.65), SIMDE_FLOAT64_C(  -164.15), SIMDE_FLOAT64_C(   842.26), SIMDE_FLOAT64_C(  -879.41),
        SIMDE_FLOAT64_C(   527.91), SIMDE_FLOAT64_C(   818.69), SIMDE_FLOAT64_C(   -80.37), SIMDE_FLOAT64_C(   559.58) },
      UINT8_C(223),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    -3.20),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -453.41),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -435.70),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   637.04) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    29.74), SIMDE_FLOAT64_C(   447.17),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   505.05),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -705.60) },
      UINT8_C(253) },
    /* 22 */
    { { SIMDE_FLOAT64_C(    49.00), SIMDE_FLOAT64_C(   777.36), SIMDE_FLOAT64_C(   479.18), SIMDE_FLOAT64_C(  -423.08),
        SIMDE_FLOAT64_C(   125.43), SIMDE_FLOAT64_C(  -601.20), SIMDE_FLOAT64_C(  -749.62), SIMDE_FLOAT64_C(   891.69) },
      { SIMDE_FLOAT64_C(    49.00), SIMDE_FLOAT64_C(   356.15), SIMDE_FLOAT64_C(   479.18), SIMDE_FLOAT64_C(  -896.62),
        SIMDE_FLOAT64_C(   125.43), SIMDE_FLOAT64_C(   812.41), SIMDE_FLOAT64_C(  -749.62), SIMDE_FLOAT64_C(  -791.29) },
      UINT8_C(138),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   627.40),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -355.76),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -286.64),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   800.57) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   728.46), SIMDE_FLOAT64_C(  -804.21),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   -28.16),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -979.15) },
      UINT8_C(253) },
    /* 23 */
    { { SIMDE_FLOAT64_C(   908.25), SIMDE_FLOAT64_C(    21.72), SIMDE_FLOAT64_C(  -402.23), SIMDE_FLOAT64_C(    33.68),
        SIMDE_FLOAT64_C(   420.53), SIMDE_FLOAT64_C(  -151.85), SIMDE_FLOAT64_C(   -74.63), SIMDE_FLOAT64_C(  -344.82) },
      { SIMDE_FLOAT64_C(   908.25), SIMDE_FLOAT64_C(   -62.34), SIMDE_FLOAT64_C(  -402.23), SIMDE_FLOAT64_C(   379.13),
        SIMDE_FLOAT64_C(   420.53), SIMDE_FLOAT64_C(  -578.48), SIMDE_FLOAT64_C(   -74.63), SIMDE_FLOAT64_C(   559.27) },
         UINT8_MAX,
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -656.85),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -759.38),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -515.20),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -200.39) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -763.03), SIMDE_FLOAT64_C(  -952.25),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -632.14),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -723.88) },
      UINT8_C(130) },
    /* 24 */
    { { SIMDE_FLOAT64_C(  -387.99), SIMDE_FLOAT64_C(   803.72), SIMDE_FLOAT64_C(   309.80), SIMDE_FLOAT64_C(  -967.46),
        SIMDE_FLOAT64_C(  -348.14), SIMDE_FLOAT64_C(  -764.83), SIMDE_FLOAT64_C(  -312.28), SIMDE_FLOAT64_C(  -143.85) },
      { SIMDE_FLOAT64_C(  -387.99), SIMDE_FLOAT64_C(   446.28), SIMDE_FLOAT64_C(   309.80), SIMDE_FLOAT64_C(   922.89),
        SIMDE_FLOAT64_C(  -348.14), SIMDE_FLOAT64_C(   823.13), SIMDE_FLOAT64_C(  -312.28), SIMDE_FLOAT64_C(   916.71) },
      UINT8_C( 85),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   685.69),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   222.80),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -801.48),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -616.25) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -929.85), SIMDE_FLOAT64_C(   568.86),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   660.44),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -727.55) },
      UINT8_C(125) },
    /* 25 */
    { { SIMDE_FLOAT64_C(  -421.48), SIMDE_FLOAT64_C(  -610.73), SIMDE_FLOAT64_C(  -695.01), SIMDE_FLOAT64_C(   230.38),
        SIMDE_FLOAT64_C(  -375.56), SIMDE_FLOAT64_C(    -7.29), SIMDE_FLOAT64_C(  -913.46), SIMDE_FLOAT64_C(   797.27) },
      { SIMDE_FLOAT64_C(  -421.48), SIMDE_FLOAT64_C(  -678.18), SIMDE_FLOAT64_C(  -695.01), SIMDE_FLOAT64_C(  -693.21),
        SIMDE_FLOAT64_C(  -375.56), SIMDE_FLOAT64_C(   202.33), SIMDE_FLOAT64_C(  -913.46), SIMDE_FLOAT64_C(  -688.76) },
      UINT8_C( 32),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -619.17),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    58.51),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   556.44),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    14.85) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    11.12), SIMDE_FLOAT64_C(   818.21),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   785.93),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -635.54) },
      UINT8_C(127) },
    /* 26 */
    { { SIMDE_FLOAT64_C(   286.95), SIMDE_FLOAT64_C(   864.48), SIMDE_FLOAT64_C(   594.84), SIMDE_FLOAT64_C(   911.40),
        SIMDE_FLOAT64_C(  -142.81), SIMDE_FLOAT64_C(   681.38), SIMDE_FLOAT64_C(   708.67), SIMDE_FLOAT64_C(   296.18) },
      { SIMDE_FLOAT64_C(   286.95), SIMDE_FLOAT64_C(   428.83), SIMDE_FLOAT64_C(   594.84), SIMDE_FLOAT64_C(  -851.84),
        SIMDE_FLOAT64_C(  -142.81), SIMDE_FLOAT64_C(   826.47), SIMDE_FLOAT64_C(   708.67), SIMDE_FLOAT64_C(   519.19) },
      UINT8_C(245),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   993.44),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -213.34),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -980.04),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   176.47) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   619.71), SIMDE_FLOAT64_C(  -536.50),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -482.60),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   804.35) },
      UINT8_C(253) },
    /* 27 */
    { { SIMDE_FLOAT64_C(   887.47), SIMDE_FLOAT64_C(  -223.69), SIMDE_FLOAT64_C(   715.75), SIMDE_FLOAT64_C(  -255.34),
        SIMDE_FLOAT64_C(  -542.31), SIMDE_FLOAT64_C(   424.41), SIMDE_FLOAT64_C(  -959.16), SIMDE_FLOAT64_C(  -539.11) },
      { SIMDE_FLOAT64_C(   887.47), SIMDE_FLOAT64_C(   643.81), SIMDE_FLOAT64_C(   715.75), SIMDE_FLOAT64_C(   484.42),
        SIMDE_FLOAT64_C(  -542.31), SIMDE_FLOAT64_C(    68.45), SIMDE_FLOAT64_C(  -959.16), SIMDE_FLOAT64_C(   677.59) },
      UINT8_C(  0),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   581.30),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -388.24),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -734.24),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -367.65) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -748.27), SIMDE_FLOAT64_C(   449.37),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -725.28),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -837.81) },
      UINT8_C(  0) },
    /* 28 */
    { { SIMDE_FLOAT64_C(   407.16), SIMDE_FLOAT64_C(   922.96), SIMDE_FLOAT64_C(   -93.15), SIMDE_FLOAT64_C(   864.85),
        SIMDE_FLOAT64_C(   347.37), SIMDE_FLOAT64_C(   -52.31), SIMDE_FLOAT64_C(  -674.26), SIMDE_FLOAT64_C(  -799.38) },
      { SIMDE_FLOAT64_C(   407.16), SIMDE_FLOAT64_C(   -65.20), SIMDE_FLOAT64_C(   -93.15), SIMDE_FLOAT64_C(  -938.22),
        SIMDE_FLOAT64_C(   347.37), SIMDE_FLOAT64_C(  -311.36), SIMDE_FLOAT64_C(  -674.26), SIMDE_FLOAT64_C(    65.14) },
      UINT8_C(170),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -796.39),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   871.21),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   465.13),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -645.16) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   952.93), SIMDE_FLOAT64_C(  -242.30),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -416.22),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   990.93) },
      UINT8_C(130) },
    /* 29 */
    { { SIMDE_FLOAT64_C(  -112.12), SIMDE_FLOAT64_C(    60.62), SIMDE_FLOAT64_C(   855.79), SIMDE_FLOAT64_C(  -764.75),
        SIMDE_FLOAT64_C(  -991.70), SIMDE_FLOAT64_C(  -818.47), SIMDE_FLOAT64_C(  -564.13), SIMDE_FLOAT64_C(  -400.20) },
      { SIMDE_FLOAT64_C(  -112.12), SIMDE_FLOAT64_C(  -879.09), SIMDE_FLOAT64_C(   855.79), SIMDE_FLOAT64_C(   119.57),
        SIMDE_FLOAT64_C(  -991.70), SIMDE_FLOAT64_C(  -599.07), SIMDE_FLOAT64_C(  -564.13), SIMDE_FLOAT64_C(    79.50) },
      UINT8_C( 87),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   861.61),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    73.91),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   454.28),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    43.60) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -813.55), SIMDE_FLOAT64_C(    35.18),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   151.37),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -960.75) },
      UINT8_C(130) },
    /* 30 */
    { { SIMDE_FLOAT64_C(  -750.44), SIMDE_FLOAT64_C(   837.71), SIMDE_FLOAT64_C(  -725.50), SIMDE_FLOAT64_C(  -742.14),
        SIMDE_FLOAT64_C(  -980.76), SIMDE_FLOAT64_C(  -289.63), SIMDE_FLOAT64_C(  -142.34), SIMDE_FLOAT64_C(   135.57) },
      { SIMDE_FLOAT64_C(  -750.44), SIMDE_FLOAT64_C(   519.23), SIMDE_FLOAT64_C(  -725.50), SIMDE_FLOAT64_C(   640.84),
        SIMDE_FLOAT64_C(  -980.76), SIMDE_FLOAT64_C(  -560.15), SIMDE_FLOAT64_C(  -142.34), SIMDE_FLOAT64_C(  -475.29) },
      UINT8_C(162),
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   671.06),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   628.20),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(    27.37),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -467.45) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   706.97), SIMDE_FLOAT64_C(  -476.46),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(  -104.08),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   145.48) },
      UINT8_C(  0) },
    /* 31 */
    { { SIMDE_FLOAT64_C(  -656.83), SIMDE_FLOAT64_C(  -321.05), SIMDE_FLOAT64_C(   403.35), SIMDE_FLOAT64_C(  -637.59),
        SIMDE_FLOAT64_C(   389.32), SIMDE_FLOAT64_C(  -738.99), SIMDE_FLOAT64_C(   497.98), SIMDE_FLOAT64_C(  -779.39) },
      { SIMDE_FLOAT64_C(  -656.83), SIMDE_FLOAT64_C(   753.12), SIMDE_FLOAT64_C(   403.35), SIMDE_FLOAT64_C(   700.40),
        SIMDE_FLOAT64_C(   389.32), SIMDE_FLOAT64_C(  -418.20), SIMDE_FLOAT64_C(   497.98), SIMDE_FLOAT64_C(  -505.57) },
         UINT8_MAX,
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   823.74),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   378.20),
                    SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -205.58),             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -935.06) },
      {             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(  -565.71), SIMDE_FLOAT64_C(  -569.86),             SIMDE_MATH_NAN,
        SIMDE_FLOAT64_C(   -60.24),             SIMDE_MATH_NAN,             SIMDE_MATH_NAN, SIMDE_FLOAT64_C(   282.93) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d b = simde_mm512_loadu_pd(test_vec[i].b);
    simde__mmask8 r = simde_mm512_cmp_pd_mask(a, b, HEDLEY_STATIC_CAST(int, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  const simde__m512d nans = simde_mm512_set1_pd(SIMDE_MATH_NAN);
  for (int i = 0 ; i < 32 ; i++) {
    simde__m512d a, b;
    simde__mmask8 r;

    fprintf(stdout, "    /* %d */\n", i);
    a = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_mm512_mask_blend_pd(85, b, a);
    SIMDE_CONSTIFY_32_(simde_mm512_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f64x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_MIDDLE);

    a = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    b = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    a = simde_mm512_mask_blend_pd(85, a, nans);
    b = simde_mm512_mask_blend_pd(105, b, nans);
    SIMDE_CONSTIFY_32_(simde_mm512_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), i, a, b);
    simde_test_x86_write_f64x8(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_f64x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

#else

/* To avoid a warning about expr < 0 always evaluating to false
 * (-Wtype-limits) because there are no functions to test. */

static int
test_simde_dummy (SIMDE_MUNIT_TEST_ARGS) {
  return 0;
}

#endif /* !defined(SIMDE_NATIVE_ALIASES_TESTING */

SIMDE_TEST_FUNC_LIST_BEGIN
  #if !defined(SIMDE_NATIVE_ALIASES_TESTING)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_ps_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_pd_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi64_mask)

    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_ps_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_pd_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi64_mask)

    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_ps_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_pd_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi64_mask)

  #else
    SIMDE_TEST_FUNC_LIST_ENTRY(dummy)
  #endif
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
