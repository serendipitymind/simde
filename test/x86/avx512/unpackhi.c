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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 */

#define SIMDE_TEST_X86_AVX512_INSN unpackhi

#include <test/x86/avx512/test-avx512.h>
#include <simde/x86/avx512/set.h>
#include <simde/x86/avx512/unpackhi.h>

static int
test_simde_mm512_unpackhi_epi8 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { { -INT8_C(  38), -INT8_C( 108),  INT8_C(  78), -INT8_C(  75),  INT8_C(  17), -INT8_C( 126), -INT8_C(  86), -INT8_C( 124),
        -INT8_C(   8),  INT8_C(  83), -INT8_C( 101), -INT8_C(   3),  INT8_C(   9),  INT8_C( 117),  INT8_C(  11),  INT8_C(  58),
        -INT8_C(  88),  INT8_C(  72), -INT8_C( 104), -INT8_C(   2),  INT8_C(  22),  INT8_C(  10), -INT8_C(  57), -INT8_C(  70),
         INT8_C( 115), -INT8_C(   6), -INT8_C(  90), -INT8_C( 111), -INT8_C(  44),  INT8_C(  11), -INT8_C(  33), -INT8_C(  82),
        -INT8_C(  97),  INT8_C(  45),  INT8_C(  99), -INT8_C(  80), -INT8_C(  80),  INT8_C(  13),  INT8_C(  52), -INT8_C(  88),
         INT8_C(  96), -INT8_C(  49), -INT8_C(  91),  INT8_C( 105),  INT8_C(  68), -INT8_C(  80), -INT8_C(  92), -INT8_C(  19),
        -INT8_C(   8),  INT8_C(  60), -INT8_C(  21),  INT8_C(  14),  INT8_C(  70), -INT8_C(  78), -INT8_C(  56), -INT8_C(  70),
        -INT8_C(  84),  INT8_C( 111),  INT8_C(  75),      INT8_MIN,  INT8_C( 122),  INT8_C(  42),  INT8_C(  46),  INT8_C(  25) },
      {  INT8_C(  88), -INT8_C( 111), -INT8_C(  55),  INT8_C(   8), -INT8_C(  97), -INT8_C(   3), -INT8_C(  80), -INT8_C(   1),
        -INT8_C(  52),  INT8_C(  85),  INT8_C( 105),  INT8_C(  16),  INT8_C(   6),  INT8_C(  13), -INT8_C(   3), -INT8_C(   2),
         INT8_C(  73), -INT8_C(  24),  INT8_C(  13), -INT8_C( 112), -INT8_C( 102), -INT8_C(  43),  INT8_C(  74),  INT8_C(  70),
         INT8_C(  68), -INT8_C( 107), -INT8_C(  58), -INT8_C(  66), -INT8_C(  64), -INT8_C(  12), -INT8_C(  41),  INT8_C(  24),
        -INT8_C( 122), -INT8_C(  96),  INT8_C(  32),  INT8_C(  37), -INT8_C(  99), -INT8_C(  48),  INT8_C(  36),  INT8_C( 105),
         INT8_C(  37), -INT8_C( 115),  INT8_C( 122),  INT8_C(  43), -INT8_C( 102),  INT8_C( 119),  INT8_C(  42), -INT8_C(  28),
         INT8_C(  96),  INT8_C(  55),  INT8_C( 116), -INT8_C(   6),  INT8_C(  12), -INT8_C(  66),  INT8_C(  65),  INT8_C(  81),
         INT8_C(  83),  INT8_C(   7),  INT8_C(  15),  INT8_C(  19), -INT8_C(   4), -INT8_C(  25),  INT8_C(  43), -INT8_C( 126) },
      { -INT8_C(   8), -INT8_C(  52),  INT8_C(  83),  INT8_C(  85), -INT8_C( 101),  INT8_C( 105), -INT8_C(   3),  INT8_C(  16),
         INT8_C(   9),  INT8_C(   6),  INT8_C( 117),  INT8_C(  13),  INT8_C(  11), -INT8_C(   3),  INT8_C(  58), -INT8_C(   2),
         INT8_C( 115),  INT8_C(  68), -INT8_C(   6), -INT8_C( 107), -INT8_C(  90), -INT8_C(  58), -INT8_C( 111), -INT8_C(  66),
        -INT8_C(  44), -INT8_C(  64),  INT8_C(  11), -INT8_C(  12), -INT8_C(  33), -INT8_C(  41), -INT8_C(  82),  INT8_C(  24),
         INT8_C(  96),  INT8_C(  37), -INT8_C(  49), -INT8_C( 115), -INT8_C(  91),  INT8_C( 122),  INT8_C( 105),  INT8_C(  43),
         INT8_C(  68), -INT8_C( 102), -INT8_C(  80),  INT8_C( 119), -INT8_C(  92),  INT8_C(  42), -INT8_C(  19), -INT8_C(  28),
        -INT8_C(  84),  INT8_C(  83),  INT8_C( 111),  INT8_C(   7),  INT8_C(  75),  INT8_C(  15),      INT8_MIN,  INT8_C(  19),
         INT8_C( 122), -INT8_C(   4),  INT8_C(  42), -INT8_C(  25),  INT8_C(  46),  INT8_C(  43),  INT8_C(  25), -INT8_C( 126) } },
    { { -INT8_C( 121),  INT8_C(  75), -INT8_C(  89),  INT8_C(  37),  INT8_C(  27), -INT8_C(  53), -INT8_C( 114),  INT8_C(  65),
         INT8_C(  89),  INT8_C(   8),  INT8_C( 108), -INT8_C(  13),      INT8_MIN, -INT8_C( 106), -INT8_C(  41), -INT8_C(  32),
        -INT8_C(  51),  INT8_C(  75), -INT8_C(  38), -INT8_C(  38),  INT8_C(   9),  INT8_C(  27),  INT8_C(  43),  INT8_C(  93),
         INT8_C(  35),  INT8_C(  58),  INT8_C( 112),  INT8_C(  31),  INT8_C(  33), -INT8_C( 100), -INT8_C(  95), -INT8_C(  87),
        -INT8_C(  25),  INT8_C(  72), -INT8_C(  50),  INT8_C(   3),  INT8_C(  19),  INT8_C(  92),  INT8_C(  68),  INT8_C( 108),
         INT8_C( 101), -INT8_C(  80),  INT8_C(  96), -INT8_C(  27),  INT8_C(  71),  INT8_C(  55), -INT8_C(  59),  INT8_C(  20),
        -INT8_C( 125), -INT8_C(  97), -INT8_C(  18), -INT8_C( 116), -INT8_C(  69),  INT8_C(  25), -INT8_C(  23), -INT8_C(  34),
         INT8_C(  84),  INT8_C(  90), -INT8_C(   3),  INT8_C( 117), -INT8_C(  10), -INT8_C(  98),  INT8_C(  30), -INT8_C(  35) },
      { -INT8_C(  26), -INT8_C(  20), -INT8_C(  32), -INT8_C(   7),  INT8_C(  73),  INT8_C(  36),  INT8_C( 102), -INT8_C(  82),
        -INT8_C(  43), -INT8_C(  58), -INT8_C( 109),  INT8_C(  28), -INT8_C(   3),  INT8_C(  88),  INT8_C(  48),      INT8_MIN,
        -INT8_C(   9),  INT8_C(  31),  INT8_C(  13), -INT8_C(  78),  INT8_C(  56), -INT8_C(  10), -INT8_C( 112), -INT8_C( 116),
         INT8_C(  80), -INT8_C( 115),  INT8_C(   2),  INT8_C(  70),  INT8_C(  43),  INT8_C(  32),  INT8_C(  36),  INT8_C(  17),
         INT8_C(  13),  INT8_C(   4),  INT8_C(  11),  INT8_C(  86),  INT8_C(  41),  INT8_C( 113),  INT8_C(   4), -INT8_C(   2),
         INT8_C(  55), -INT8_C( 105),  INT8_C(  26),  INT8_C(  52), -INT8_C(  17),  INT8_C(  74), -INT8_C(  75), -INT8_C(  26),
         INT8_C( 105), -INT8_C(  62), -INT8_C( 103), -INT8_C(  94), -INT8_C(  72),  INT8_C(  41),  INT8_C(  46),  INT8_C(   9),
        -INT8_C(  73),  INT8_C(  48),  INT8_C(  79), -INT8_C(  30),  INT8_C(  81),  INT8_C( 115), -INT8_C(  12),  INT8_C(  94) },
      {  INT8_C(  89), -INT8_C(  43),  INT8_C(   8), -INT8_C(  58),  INT8_C( 108), -INT8_C( 109), -INT8_C(  13),  INT8_C(  28),
             INT8_MIN, -INT8_C(   3), -INT8_C( 106),  INT8_C(  88), -INT8_C(  41),  INT8_C(  48), -INT8_C(  32),      INT8_MIN,
         INT8_C(  35),  INT8_C(  80),  INT8_C(  58), -INT8_C( 115),  INT8_C( 112),  INT8_C(   2),  INT8_C(  31),  INT8_C(  70),
         INT8_C(  33),  INT8_C(  43), -INT8_C( 100),  INT8_C(  32), -INT8_C(  95),  INT8_C(  36), -INT8_C(  87),  INT8_C(  17),
         INT8_C( 101),  INT8_C(  55), -INT8_C(  80), -INT8_C( 105),  INT8_C(  96),  INT8_C(  26), -INT8_C(  27),  INT8_C(  52),
         INT8_C(  71), -INT8_C(  17),  INT8_C(  55),  INT8_C(  74), -INT8_C(  59), -INT8_C(  75),  INT8_C(  20), -INT8_C(  26),
         INT8_C(  84), -INT8_C(  73),  INT8_C(  90),  INT8_C(  48), -INT8_C(   3),  INT8_C(  79),  INT8_C( 117), -INT8_C(  30),
        -INT8_C(  10),  INT8_C(  81), -INT8_C(  98),  INT8_C( 115),  INT8_C(  30), -INT8_C(  12), -INT8_C(  35),  INT8_C(  94) } },
    { {  INT8_C( 120), -INT8_C(   1), -INT8_C(  76), -INT8_C(  95),  INT8_C( 112), -INT8_C(  72), -INT8_C(  97), -INT8_C(  89),
         INT8_C(  79), -INT8_C(  71), -INT8_C(  37),  INT8_C(  62),  INT8_C(   3), -INT8_C( 112),  INT8_C(  36),  INT8_C( 109),
         INT8_C(  82), -INT8_C(  67),  INT8_C(  15),  INT8_C(  11), -INT8_C(  25),  INT8_C(  61),  INT8_C(  20), -INT8_C(  98),
         INT8_C( 110),  INT8_C(  99),      INT8_MIN, -INT8_C(  65), -INT8_C(  41),  INT8_C( 116),  INT8_C(  29),  INT8_C(  79),
         INT8_C( 115), -INT8_C(  47), -INT8_C(  16), -INT8_C(  29), -INT8_C( 119), -INT8_C( 113), -INT8_C( 118), -INT8_C(  40),
         INT8_C(  72),  INT8_C( 102),  INT8_C(  22),  INT8_C(  75), -INT8_C(  10),  INT8_C(  58), -INT8_C(  72),  INT8_C(  73),
        -INT8_C(   8), -INT8_C(  57),  INT8_C(  84), -INT8_C(  33),  INT8_C(   5),  INT8_C( 104),  INT8_C( 125),  INT8_C( 115),
        -INT8_C(  53), -INT8_C(   3),  INT8_C(  50), -INT8_C(  94),  INT8_C( 114),  INT8_C(  79), -INT8_C(  15), -INT8_C(  27) },
      {  INT8_C(  32), -INT8_C(  31), -INT8_C(  55), -INT8_C(  87),  INT8_C( 112),  INT8_C(  83), -INT8_C( 127), -INT8_C(  72),
        -INT8_C(  71), -INT8_C( 105),  INT8_C(   4), -INT8_C(  80), -INT8_C(  47), -INT8_C(  68), -INT8_C(   7), -INT8_C(  55),
        -INT8_C( 124),  INT8_C(  77), -INT8_C(  88), -INT8_C( 119), -INT8_C(  75),  INT8_C(  37), -INT8_C(   4),      INT8_MIN,
         INT8_C(  35),  INT8_C(  46),  INT8_C(  35), -INT8_C( 107),  INT8_C( 125),  INT8_C(  20),  INT8_C( 122), -INT8_C(  99),
        -INT8_C(  10),  INT8_C(  67),  INT8_C(  70),  INT8_C( 102), -INT8_C( 105), -INT8_C(  57),  INT8_C(  31),  INT8_C(  80),
         INT8_C(  94),  INT8_C(  35),  INT8_C(   0),  INT8_C(  47), -INT8_C(  33), -INT8_C(   7), -INT8_C(   7),  INT8_C(  99),
         INT8_C(  70), -INT8_C(  95), -INT8_C(  20), -INT8_C(   5), -INT8_C(  57), -INT8_C(  24),  INT8_C( 124), -INT8_C(  22),
         INT8_C(  22), -INT8_C(  97),      INT8_MAX, -INT8_C( 109), -INT8_C(  77), -INT8_C(   7),  INT8_C(  48), -INT8_C(  87) },
      {  INT8_C(  79), -INT8_C(  71), -INT8_C(  71), -INT8_C( 105), -INT8_C(  37),  INT8_C(   4),  INT8_C(  62), -INT8_C(  80),
         INT8_C(   3), -INT8_C(  47), -INT8_C( 112), -INT8_C(  68),  INT8_C(  36), -INT8_C(   7),  INT8_C( 109), -INT8_C(  55),
         INT8_C( 110),  INT8_C(  35),  INT8_C(  99),  INT8_C(  46),      INT8_MIN,  INT8_C(  35), -INT8_C(  65), -INT8_C( 107),
        -INT8_C(  41),  INT8_C( 125),  INT8_C( 116),  INT8_C(  20),  INT8_C(  29),  INT8_C( 122),  INT8_C(  79), -INT8_C(  99),
         INT8_C(  72),  INT8_C(  94),  INT8_C( 102),  INT8_C(  35),  INT8_C(  22),  INT8_C(   0),  INT8_C(  75),  INT8_C(  47),
        -INT8_C(  10), -INT8_C(  33),  INT8_C(  58), -INT8_C(   7), -INT8_C(  72), -INT8_C(   7),  INT8_C(  73),  INT8_C(  99),
        -INT8_C(  53),  INT8_C(  22), -INT8_C(   3), -INT8_C(  97),  INT8_C(  50),      INT8_MAX, -INT8_C(  94), -INT8_C( 109),
         INT8_C( 114), -INT8_C(  77),  INT8_C(  79), -INT8_C(   7), -INT8_C(  15),  INT8_C(  48), -INT8_C(  27), -INT8_C(  87) } },
    { {  INT8_C(  61),  INT8_C( 118),  INT8_C(  16), -INT8_C(  44),  INT8_C(  61),  INT8_C(  47),  INT8_C(  36), -INT8_C( 101),
         INT8_C(  82),  INT8_C(  37), -INT8_C(  53),  INT8_C(  49),  INT8_C(  30), -INT8_C(  60), -INT8_C( 107),  INT8_C( 101),
         INT8_C( 101), -INT8_C( 127),  INT8_C(  96),  INT8_C(  44),  INT8_C( 106), -INT8_C(  36),  INT8_C(  22),      INT8_MIN,
         INT8_C( 123), -INT8_C( 107),  INT8_C(  20),  INT8_C(  47), -INT8_C( 113),  INT8_C(  68), -INT8_C(  40), -INT8_C(  52),
        -INT8_C(  69), -INT8_C(  24), -INT8_C(  96), -INT8_C(   8),  INT8_C(  23), -INT8_C(  60), -INT8_C( 108),  INT8_C( 105),
        -INT8_C(  23),  INT8_C(  95), -INT8_C( 101),  INT8_C(   8),  INT8_C(  35),  INT8_C(  48),  INT8_C( 109), -INT8_C( 120),
        -INT8_C(  79), -INT8_C(  51), -INT8_C(  75),  INT8_C(  27), -INT8_C(  86), -INT8_C(  53), -INT8_C( 100),  INT8_C(  37),
         INT8_C(  97), -INT8_C(  80),  INT8_C(  84), -INT8_C(  16), -INT8_C(  12),  INT8_C(  45), -INT8_C(  68), -INT8_C(  81) },
      {  INT8_C(  21),  INT8_C(  92), -INT8_C(  88),  INT8_C(  45),  INT8_C(  32),  INT8_C(  60), -INT8_C( 106),  INT8_C(  10),
        -INT8_C( 101),  INT8_C(  49),  INT8_C(  18), -INT8_C(  66),  INT8_C(  97),      INT8_MAX,  INT8_C(  70),  INT8_C(  19),
         INT8_C(  76), -INT8_C(   5),  INT8_C(  46), -INT8_C(  10), -INT8_C(  57), -INT8_C(  54),  INT8_C(  28),  INT8_C(  40),
         INT8_C( 122),  INT8_C( 112),  INT8_C(  24),  INT8_C( 111), -INT8_C(  99), -INT8_C(  44),  INT8_C(  30), -INT8_C(  77),
         INT8_C(  48), -INT8_C(  58), -INT8_C(  32),  INT8_C(  80),  INT8_C(   2),  INT8_C( 118),  INT8_C(  90), -INT8_C(  99),
        -INT8_C(  88),  INT8_C( 108),  INT8_C(  91),  INT8_C(   9), -INT8_C(  21), -INT8_C(  94),  INT8_C(  28),  INT8_C(  56),
        -INT8_C(  99),  INT8_C(  75),  INT8_C(  46),  INT8_C( 100),  INT8_C(  21),  INT8_C(  74), -INT8_C( 116), -INT8_C( 112),
        -INT8_C(  69), -INT8_C(  92), -INT8_C(   1),  INT8_C(  88),  INT8_C( 120),  INT8_C(  29),  INT8_C(  11), -INT8_C(  88) },
      {  INT8_C(  82), -INT8_C( 101),  INT8_C(  37),  INT8_C(  49), -INT8_C(  53),  INT8_C(  18),  INT8_C(  49), -INT8_C(  66),
         INT8_C(  30),  INT8_C(  97), -INT8_C(  60),      INT8_MAX, -INT8_C( 107),  INT8_C(  70),  INT8_C( 101),  INT8_C(  19),
         INT8_C( 123),  INT8_C( 122), -INT8_C( 107),  INT8_C( 112),  INT8_C(  20),  INT8_C(  24),  INT8_C(  47),  INT8_C( 111),
        -INT8_C( 113), -INT8_C(  99),  INT8_C(  68), -INT8_C(  44), -INT8_C(  40),  INT8_C(  30), -INT8_C(  52), -INT8_C(  77),
        -INT8_C(  23), -INT8_C(  88),  INT8_C(  95),  INT8_C( 108), -INT8_C( 101),  INT8_C(  91),  INT8_C(   8),  INT8_C(   9),
         INT8_C(  35), -INT8_C(  21),  INT8_C(  48), -INT8_C(  94),  INT8_C( 109),  INT8_C(  28), -INT8_C( 120),  INT8_C(  56),
         INT8_C(  97), -INT8_C(  69), -INT8_C(  80), -INT8_C(  92),  INT8_C(  84), -INT8_C(   1), -INT8_C(  16),  INT8_C(  88),
        -INT8_C(  12),  INT8_C( 120),  INT8_C(  45),  INT8_C(  29), -INT8_C(  68),  INT8_C(  11), -INT8_C(  81), -INT8_C(  88) } },
    { { -INT8_C(  28), -INT8_C(  21), -INT8_C(   7), -INT8_C(  26),  INT8_C(  98),  INT8_C(  83), -INT8_C( 124),  INT8_C(  10),
        -INT8_C(  64), -INT8_C(  33),  INT8_C(  19), -INT8_C(  85), -INT8_C( 127),  INT8_C(  48), -INT8_C(  29),  INT8_C(  31),
         INT8_C( 123),  INT8_C(  18), -INT8_C( 125), -INT8_C( 112),  INT8_C(  92),  INT8_C(  16),  INT8_C(  32),  INT8_C(  23),
        -INT8_C(  76),  INT8_C(  31),  INT8_C( 112),  INT8_C(  45),  INT8_C(  61),  INT8_C( 123), -INT8_C(  43),  INT8_C(  33),
         INT8_C( 103), -INT8_C(  50),  INT8_C(   7), -INT8_C(  55),  INT8_C(  34), -INT8_C( 117), -INT8_C(  45), -INT8_C(  30),
         INT8_C( 107), -INT8_C(  26), -INT8_C( 115), -INT8_C(  20),  INT8_C(  22),  INT8_C( 113),  INT8_C(  11), -INT8_C( 111),
        -INT8_C( 125), -INT8_C( 113),  INT8_C(  34), -INT8_C(  33), -INT8_C(  97),  INT8_C(  66), -INT8_C(   9),  INT8_C(  83),
         INT8_C(  98),  INT8_C( 103),      INT8_MIN, -INT8_C(  97), -INT8_C(  30),  INT8_C(  86), -INT8_C(  64),  INT8_C(  73) },
      {  INT8_C(  36), -INT8_C(  57),  INT8_C(  18),  INT8_C(  70),  INT8_C(  83), -INT8_C(  27),  INT8_C(  40), -INT8_C(  66),
        -INT8_C(  52), -INT8_C(  74), -INT8_C(  86), -INT8_C(  30),  INT8_C(  39), -INT8_C(  74),  INT8_C( 116), -INT8_C(  86),
         INT8_C(  69), -INT8_C( 106), -INT8_C( 119), -INT8_C(  28), -INT8_C(  40),      INT8_MIN,  INT8_C(  55),  INT8_C(  58),
        -INT8_C(  25), -INT8_C(  72), -INT8_C(  39), -INT8_C(  54),  INT8_C(  14), -INT8_C( 103),  INT8_C(  19),  INT8_C(  50),
         INT8_C(  97),  INT8_C(  38),  INT8_C( 121), -INT8_C(  76),  INT8_C(  11), -INT8_C(  95),  INT8_C( 114), -INT8_C(  41),
         INT8_C(  87),  INT8_C(  28), -INT8_C(  70),  INT8_C( 126), -INT8_C(  46),  INT8_C(  46),  INT8_C(  40),  INT8_C(  23),
        -INT8_C(  60), -INT8_C(  78), -INT8_C(   5), -INT8_C( 100),  INT8_C(  50),  INT8_C(  51), -INT8_C(  41),  INT8_C(  26),
        -INT8_C(  21), -INT8_C(  80), -INT8_C(  28), -INT8_C(   7),  INT8_C(  74), -INT8_C(   9),  INT8_C(  43), -INT8_C(  85) },
      { -INT8_C(  64), -INT8_C(  52), -INT8_C(  33), -INT8_C(  74),  INT8_C(  19), -INT8_C(  86), -INT8_C(  85), -INT8_C(  30),
        -INT8_C( 127),  INT8_C(  39),  INT8_C(  48), -INT8_C(  74), -INT8_C(  29),  INT8_C( 116),  INT8_C(  31), -INT8_C(  86),
        -INT8_C(  76), -INT8_C(  25),  INT8_C(  31), -INT8_C(  72),  INT8_C( 112), -INT8_C(  39),  INT8_C(  45), -INT8_C(  54),
         INT8_C(  61),  INT8_C(  14),  INT8_C( 123), -INT8_C( 103), -INT8_C(  43),  INT8_C(  19),  INT8_C(  33),  INT8_C(  50),
         INT8_C( 107),  INT8_C(  87), -INT8_C(  26),  INT8_C(  28), -INT8_C( 115), -INT8_C(  70), -INT8_C(  20),  INT8_C( 126),
         INT8_C(  22), -INT8_C(  46),  INT8_C( 113),  INT8_C(  46),  INT8_C(  11),  INT8_C(  40), -INT8_C( 111),  INT8_C(  23),
         INT8_C(  98), -INT8_C(  21),  INT8_C( 103), -INT8_C(  80),      INT8_MIN, -INT8_C(  28), -INT8_C(  97), -INT8_C(   7),
        -INT8_C(  30),  INT8_C(  74),  INT8_C(  86), -INT8_C(   9), -INT8_C(  64),  INT8_C(  43),  INT8_C(  73), -INT8_C(  85) } },
    { {  INT8_C(  29), -INT8_C(  92),  INT8_C(  95),  INT8_C(  41),  INT8_C(  70), -INT8_C(  47),  INT8_C(   0), -INT8_C(  99),
        -INT8_C(  19), -INT8_C(  70),  INT8_C(  28), -INT8_C(  64), -INT8_C(  24),  INT8_C(  68), -INT8_C(  41), -INT8_C(  84),
        -INT8_C(  10), -INT8_C(  45),  INT8_C(  73),  INT8_C(  41),  INT8_C(   6),  INT8_C(  32),  INT8_C(  67), -INT8_C(  15),
        -INT8_C(  48),  INT8_C(  39), -INT8_C(  22),  INT8_C(  26),  INT8_C(  30),  INT8_C(  21), -INT8_C(  59),  INT8_C(  60),
        -INT8_C(  70),  INT8_C(  36),  INT8_C( 101),  INT8_C(   0), -INT8_C(  11),  INT8_C( 101), -INT8_C(  99), -INT8_C(  29),
         INT8_C(  32), -INT8_C(  71), -INT8_C(  93),  INT8_C(   8), -INT8_C(   2),  INT8_C( 122), -INT8_C(  75), -INT8_C(  12),
         INT8_C(  77), -INT8_C(   2),  INT8_C(  29),  INT8_C(  83),  INT8_C(  30),  INT8_C(  96),  INT8_C(  68), -INT8_C(  18),
        -INT8_C( 121),  INT8_C(  46),  INT8_C(   9), -INT8_C(  90),  INT8_C(  68), -INT8_C(  50), -INT8_C(  30), -INT8_C(   2) },
      { -INT8_C(  13),  INT8_C(  71), -INT8_C(   2), -INT8_C(  24), -INT8_C(  84), -INT8_C( 101), -INT8_C(  53), -INT8_C(  52),
         INT8_C(  85),  INT8_C( 110), -INT8_C(  43),  INT8_C(  83), -INT8_C(  23), -INT8_C( 118),  INT8_C(  71),  INT8_C(  54),
        -INT8_C( 120),  INT8_C( 101), -INT8_C( 118), -INT8_C(  90), -INT8_C(  59), -INT8_C(  50), -INT8_C( 108),  INT8_C(  77),
        -INT8_C(   3), -INT8_C(  99), -INT8_C(  13),  INT8_C(  65),  INT8_C( 108), -INT8_C(  43),  INT8_C(  63),  INT8_C(  95),
         INT8_C(  28),  INT8_C(  61),  INT8_C(  71), -INT8_C(  56), -INT8_C(  40),  INT8_C(  19), -INT8_C( 107),  INT8_C(  45),
        -INT8_C( 127),  INT8_C( 106),      INT8_MIN,  INT8_C( 106), -INT8_C(  12), -INT8_C(  56), -INT8_C(  95),  INT8_C( 124),
         INT8_C(  45),  INT8_C(  43),  INT8_C(  34), -INT8_C(  14), -INT8_C(   7), -INT8_C(  74),  INT8_C(  63), -INT8_C(  10),
         INT8_C(  84),  INT8_C(  50),  INT8_C(  55), -INT8_C(  64),  INT8_C(   7),  INT8_C( 118),  INT8_C(  31),  INT8_C(  35) },
      { -INT8_C(  19),  INT8_C(  85), -INT8_C(  70),  INT8_C( 110),  INT8_C(  28), -INT8_C(  43), -INT8_C(  64),  INT8_C(  83),
        -INT8_C(  24), -INT8_C(  23),  INT8_C(  68), -INT8_C( 118), -INT8_C(  41),  INT8_C(  71), -INT8_C(  84),  INT8_C(  54),
        -INT8_C(  48), -INT8_C(   3),  INT8_C(  39), -INT8_C(  99), -INT8_C(  22), -INT8_C(  13),  INT8_C(  26),  INT8_C(  65),
         INT8_C(  30),  INT8_C( 108),  INT8_C(  21), -INT8_C(  43), -INT8_C(  59),  INT8_C(  63),  INT8_C(  60),  INT8_C(  95),
         INT8_C(  32), -INT8_C( 127), -INT8_C(  71),  INT8_C( 106), -INT8_C(  93),      INT8_MIN,  INT8_C(   8),  INT8_C( 106),
        -INT8_C(   2), -INT8_C(  12),  INT8_C( 122), -INT8_C(  56), -INT8_C(  75), -INT8_C(  95), -INT8_C(  12),  INT8_C( 124),
        -INT8_C( 121),  INT8_C(  84),  INT8_C(  46),  INT8_C(  50),  INT8_C(   9),  INT8_C(  55), -INT8_C(  90), -INT8_C(  64),
         INT8_C(  68),  INT8_C(   7), -INT8_C(  50),  INT8_C( 118), -INT8_C(  30),  INT8_C(  31), -INT8_C(   2),  INT8_C(  35) } },
    { { -INT8_C(  77),  INT8_C( 102), -INT8_C(  20), -INT8_C( 116),  INT8_C( 121), -INT8_C( 127), -INT8_C(  71), -INT8_C(   5),
        -INT8_C(  21),  INT8_C(  58),  INT8_C( 101), -INT8_C(  33),  INT8_C(   2),  INT8_C(   6),  INT8_C(  91),  INT8_C(  47),
         INT8_C(  49),  INT8_C( 125),  INT8_C(  33),  INT8_C(  43),  INT8_C(  51),  INT8_C(  97),  INT8_C(  33), -INT8_C( 121),
        -INT8_C( 109),  INT8_C(  89),  INT8_C(  71), -INT8_C( 101), -INT8_C(  49),  INT8_C( 102), -INT8_C(  66), -INT8_C( 125),
        -INT8_C(  51), -INT8_C(  86),  INT8_C(  15),  INT8_C(  70),  INT8_C(  43), -INT8_C(  56),  INT8_C(  65),  INT8_C(  22),
         INT8_C(   2), -INT8_C(  89), -INT8_C(  11),  INT8_C(   4), -INT8_C(  83),  INT8_C(  80),  INT8_C(  51), -INT8_C(  33),
        -INT8_C(  51),  INT8_C(  85),  INT8_C(  10),  INT8_C(   1), -INT8_C(  74),  INT8_C(  43), -INT8_C( 120),  INT8_C(  73),
        -INT8_C( 124), -INT8_C(  48), -INT8_C(  28),  INT8_C(  84),  INT8_C(  54), -INT8_C(  93), -INT8_C(  41),  INT8_C(   3) },
      {  INT8_C(  77), -INT8_C(  26),  INT8_C(  74),  INT8_C( 121), -INT8_C(  82), -INT8_C( 117), -INT8_C( 113), -INT8_C(  79),
         INT8_C(  50), -INT8_C( 123), -INT8_C(  75), -INT8_C(  32), -INT8_C(  43), -INT8_C(  23), -INT8_C(  65), -INT8_C(  93),
         INT8_C(  62), -INT8_C(  55), -INT8_C(  92), -INT8_C(  12), -INT8_C(  12),  INT8_C(  44),  INT8_C(  61),  INT8_C( 121),
        -INT8_C(   4),  INT8_C(  34), -INT8_C(  51),  INT8_C(  51), -INT8_C(  59), -INT8_C(  92),  INT8_C(  54),  INT8_C(  18),
        -INT8_C( 118),      INT8_MIN, -INT8_C( 117),  INT8_C(  56),  INT8_C(  12),  INT8_C(  27), -INT8_C(  23),  INT8_C(  62),
        -INT8_C(  96), -INT8_C(  97),  INT8_C(  30),  INT8_C( 117), -INT8_C( 120), -INT8_C(  35),  INT8_C(  24), -INT8_C(  58),
        -INT8_C(  90), -INT8_C(  68), -INT8_C(  70), -INT8_C( 101), -INT8_C(  23), -INT8_C(   9),  INT8_C(  20), -INT8_C(  27),
         INT8_C(  25), -INT8_C(  31),  INT8_C(  24), -INT8_C(  34), -INT8_C( 123),  INT8_C(  79), -INT8_C(  15),  INT8_C(  15) },
      { -INT8_C(  21),  INT8_C(  50),  INT8_C(  58), -INT8_C( 123),  INT8_C( 101), -INT8_C(  75), -INT8_C(  33), -INT8_C(  32),
         INT8_C(   2), -INT8_C(  43),  INT8_C(   6), -INT8_C(  23),  INT8_C(  91), -INT8_C(  65),  INT8_C(  47), -INT8_C(  93),
        -INT8_C( 109), -INT8_C(   4),  INT8_C(  89),  INT8_C(  34),  INT8_C(  71), -INT8_C(  51), -INT8_C( 101),  INT8_C(  51),
        -INT8_C(  49), -INT8_C(  59),  INT8_C( 102), -INT8_C(  92), -INT8_C(  66),  INT8_C(  54), -INT8_C( 125),  INT8_C(  18),
         INT8_C(   2), -INT8_C(  96), -INT8_C(  89), -INT8_C(  97), -INT8_C(  11),  INT8_C(  30),  INT8_C(   4),  INT8_C( 117),
        -INT8_C(  83), -INT8_C( 120),  INT8_C(  80), -INT8_C(  35),  INT8_C(  51),  INT8_C(  24), -INT8_C(  33), -INT8_C(  58),
        -INT8_C( 124),  INT8_C(  25), -INT8_C(  48), -INT8_C(  31), -INT8_C(  28),  INT8_C(  24),  INT8_C(  84), -INT8_C(  34),
         INT8_C(  54), -INT8_C( 123), -INT8_C(  93),  INT8_C(  79), -INT8_C(  41), -INT8_C(  15),  INT8_C(   3),  INT8_C(  15) } },
    { { -INT8_C(  49),  INT8_C( 124),  INT8_C(  71), -INT8_C(  37), -INT8_C( 105),  INT8_C(  49),  INT8_C(  26),  INT8_C(  55),
        -INT8_C(  48),  INT8_C(  56), -INT8_C(  83),  INT8_C(  88),  INT8_C(  22), -INT8_C(  59),  INT8_C(  30), -INT8_C(  68),
        -INT8_C( 126), -INT8_C(  40),  INT8_C(  87),  INT8_C( 107), -INT8_C(  49),  INT8_C( 107),  INT8_C(  80), -INT8_C(  23),
         INT8_C(  76),  INT8_C( 105), -INT8_C(  57), -INT8_C(  47), -INT8_C(  72), -INT8_C(  72), -INT8_C(  32), -INT8_C( 121),
         INT8_C(  53),  INT8_C(  40),  INT8_C(  99), -INT8_C(  52),  INT8_C(  89),  INT8_C( 125),  INT8_C(   4),  INT8_C(  41),
        -INT8_C(  75), -INT8_C(  79), -INT8_C( 127), -INT8_C(  53),  INT8_C( 118), -INT8_C(  97), -INT8_C( 120), -INT8_C(   8),
         INT8_C( 119), -INT8_C(  33),  INT8_C(  99),  INT8_C(  70),  INT8_C(  75), -INT8_C(  76),  INT8_C(  47), -INT8_C( 105),
         INT8_C(  29), -INT8_C(   9),  INT8_C( 105), -INT8_C(  43), -INT8_C(  81),  INT8_C(  73),  INT8_C(  92), -INT8_C(  28) },
      {  INT8_C( 113), -INT8_C(  65), -INT8_C(  79), -INT8_C(  54),  INT8_C(  60), -INT8_C(  75), -INT8_C(  13), -INT8_C(  14),
         INT8_C( 102),  INT8_C( 116), -INT8_C(  67), -INT8_C(  36),  INT8_C(  19),  INT8_C(  69), -INT8_C(  43), -INT8_C( 118),
         INT8_C(  37),  INT8_C(  56), -INT8_C(  47),  INT8_C( 112), -INT8_C(  20),  INT8_C(   0),  INT8_C(   7),  INT8_C(   9),
        -INT8_C(   9),  INT8_C( 112), -INT8_C(  34), -INT8_C(  89), -INT8_C(  70),  INT8_C(  59), -INT8_C( 117),  INT8_C(  43),
        -INT8_C(   6),  INT8_C(  60), -INT8_C(  10),  INT8_C(  55), -INT8_C(  15), -INT8_C(  23),  INT8_C(  41),  INT8_C(  87),
         INT8_C(  94), -INT8_C(  26),  INT8_C(  52),  INT8_C( 113),  INT8_C(  44),  INT8_C(   9), -INT8_C(   4),  INT8_C(  81),
         INT8_C(  65), -INT8_C(  51), -INT8_C(  63),  INT8_C(  46), -INT8_C(  51), -INT8_C(  56),  INT8_C(  55), -INT8_C(  59),
         INT8_C(  57),  INT8_C(  22),  INT8_C( 108), -INT8_C(  13),  INT8_C(  81), -INT8_C(   9),  INT8_C(  30),  INT8_C(  75) },
      { -INT8_C(  48),  INT8_C( 102),  INT8_C(  56),  INT8_C( 116), -INT8_C(  83), -INT8_C(  67),  INT8_C(  88), -INT8_C(  36),
         INT8_C(  22),  INT8_C(  19), -INT8_C(  59),  INT8_C(  69),  INT8_C(  30), -INT8_C(  43), -INT8_C(  68), -INT8_C( 118),
         INT8_C(  76), -INT8_C(   9),  INT8_C( 105),  INT8_C( 112), -INT8_C(  57), -INT8_C(  34), -INT8_C(  47), -INT8_C(  89),
        -INT8_C(  72), -INT8_C(  70), -INT8_C(  72),  INT8_C(  59), -INT8_C(  32), -INT8_C( 117), -INT8_C( 121),  INT8_C(  43),
        -INT8_C(  75),  INT8_C(  94), -INT8_C(  79), -INT8_C(  26), -INT8_C( 127),  INT8_C(  52), -INT8_C(  53),  INT8_C( 113),
         INT8_C( 118),  INT8_C(  44), -INT8_C(  97),  INT8_C(   9), -INT8_C( 120), -INT8_C(   4), -INT8_C(   8),  INT8_C(  81),
         INT8_C(  29),  INT8_C(  57), -INT8_C(   9),  INT8_C(  22),  INT8_C( 105),  INT8_C( 108), -INT8_C(  43), -INT8_C(  13),
        -INT8_C(  81),  INT8_C(  81),  INT8_C(  73), -INT8_C(   9),  INT8_C(  92),  INT8_C(  30), -INT8_C(  28),  INT8_C(  75) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi8(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi8(test_vec[i].b);
    simde__m512i r = simde_mm512_unpackhi_epi8(a, b);
    simde_test_x86_assert_equal_i8x64(r, simde_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_unpackhi_epi16 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 31862), -INT16_C( 28008), -INT16_C( 17358), -INT16_C(  9367), -INT16_C(  8035),  INT16_C( 28789),  INT16_C(  9734),  INT16_C( 32189),
        -INT16_C( 29888),  INT16_C( 14371),  INT16_C( 13788), -INT16_C( 27397),  INT16_C( 24497), -INT16_C(  8132), -INT16_C( 27140), -INT16_C( 30909),
        -INT16_C(  9447),  INT16_C( 19225), -INT16_C( 32105),  INT16_C( 13350), -INT16_C( 25502),  INT16_C( 26788),  INT16_C( 25026),  INT16_C(   741),
         INT16_C(  2541), -INT16_C( 14021),  INT16_C( 13886), -INT16_C(  4003), -INT16_C( 26219), -INT16_C( 28208),  INT16_C(  5167),  INT16_C( 18456) },
      {  INT16_C( 12783), -INT16_C( 30829), -INT16_C( 17997),  INT16_C(  5563),  INT16_C( 24661),  INT16_C(  6270),  INT16_C( 25537), -INT16_C( 20966),
         INT16_C( 21868), -INT16_C( 21641), -INT16_C( 10869),  INT16_C(  8347),  INT16_C( 27502), -INT16_C( 25166), -INT16_C( 13697),  INT16_C( 28645),
         INT16_C( 30972), -INT16_C( 20490), -INT16_C( 20174), -INT16_C( 30779),  INT16_C( 17169), -INT16_C( 11361), -INT16_C( 17754),  INT16_C(  4993),
        -INT16_C(  1777), -INT16_C( 25666),  INT16_C( 22990),  INT16_C( 15547),  INT16_C( 28100),  INT16_C( 17626), -INT16_C( 16584),  INT16_C( 13491) },
      { -INT16_C(  8035),  INT16_C( 24661),  INT16_C( 28789),  INT16_C(  6270),  INT16_C(  9734),  INT16_C( 25537),  INT16_C( 32189), -INT16_C( 20966),
         INT16_C( 24497),  INT16_C( 27502), -INT16_C(  8132), -INT16_C( 25166), -INT16_C( 27140), -INT16_C( 13697), -INT16_C( 30909),  INT16_C( 28645),
        -INT16_C( 25502),  INT16_C( 17169),  INT16_C( 26788), -INT16_C( 11361),  INT16_C( 25026), -INT16_C( 17754),  INT16_C(   741),  INT16_C(  4993),
        -INT16_C( 26219),  INT16_C( 28100), -INT16_C( 28208),  INT16_C( 17626),  INT16_C(  5167), -INT16_C( 16584),  INT16_C( 18456),  INT16_C( 13491) } },
    { { -INT16_C( 22216),  INT16_C( 27363), -INT16_C( 22438),  INT16_C( 27889), -INT16_C( 28181), -INT16_C( 28097), -INT16_C( 16309),  INT16_C( 23205),
         INT16_C( 25529), -INT16_C( 30731), -INT16_C( 20036), -INT16_C( 32572), -INT16_C( 25058),  INT16_C( 22212),  INT16_C( 30557), -INT16_C( 27254),
         INT16_C( 28192),  INT16_C( 31743), -INT16_C(  3818),  INT16_C(   743),  INT16_C(  9858), -INT16_C( 12908),  INT16_C( 14822), -INT16_C( 24537),
         INT16_C(  7580),  INT16_C( 22567), -INT16_C(  5170), -INT16_C(  4904), -INT16_C( 25207), -INT16_C(  6333), -INT16_C( 13036),  INT16_C( 13692) },
      {  INT16_C( 31803),  INT16_C( 21168), -INT16_C( 26771), -INT16_C(  4268), -INT16_C(  5955), -INT16_C( 23620), -INT16_C(  7391), -INT16_C( 17085),
         INT16_C( 27392), -INT16_C( 12779), -INT16_C(  4778), -INT16_C(  8005), -INT16_C(   374), -INT16_C( 24633),  INT16_C( 17355),  INT16_C(  2004),
        -INT16_C( 31553),  INT16_C( 11353), -INT16_C( 21221), -INT16_C( 10213), -INT16_C( 10347), -INT16_C( 18821), -INT16_C( 16453), -INT16_C( 17549),
        -INT16_C( 30678), -INT16_C( 32630),  INT16_C( 17781),  INT16_C(    96),  INT16_C( 10051),  INT16_C(  3743),  INT16_C( 29547),  INT16_C( 10773) },
      { -INT16_C( 28181), -INT16_C(  5955), -INT16_C( 28097), -INT16_C( 23620), -INT16_C( 16309), -INT16_C(  7391),  INT16_C( 23205), -INT16_C( 17085),
        -INT16_C( 25058), -INT16_C(   374),  INT16_C( 22212), -INT16_C( 24633),  INT16_C( 30557),  INT16_C( 17355), -INT16_C( 27254),  INT16_C(  2004),
         INT16_C(  9858), -INT16_C( 10347), -INT16_C( 12908), -INT16_C( 18821),  INT16_C( 14822), -INT16_C( 16453), -INT16_C( 24537), -INT16_C( 17549),
        -INT16_C( 25207),  INT16_C( 10051), -INT16_C(  6333),  INT16_C(  3743), -INT16_C( 13036),  INT16_C( 29547),  INT16_C( 13692),  INT16_C( 10773) } },
    { {  INT16_C( 28407),  INT16_C(  4695),  INT16_C( 29211), -INT16_C( 20246),  INT16_C( 25930),  INT16_C(  1382), -INT16_C(  9948),  INT16_C( 20160),
         INT16_C( 19041), -INT16_C( 10289),  INT16_C( 12175), -INT16_C( 11561),  INT16_C( 30295), -INT16_C( 15647), -INT16_C(  2327), -INT16_C(  7956),
         INT16_C( 17253), -INT16_C( 32526), -INT16_C(  9034),  INT16_C(    49), -INT16_C( 26815),  INT16_C( 26117), -INT16_C( 14991), -INT16_C( 11596),
        -INT16_C( 31984), -INT16_C( 24663), -INT16_C( 32589),  INT16_C(  2674),  INT16_C( 21494), -INT16_C(  8244), -INT16_C( 18359), -INT16_C( 20801) },
      { -INT16_C( 19972), -INT16_C( 19921),  INT16_C( 24717), -INT16_C( 12366), -INT16_C( 18441),  INT16_C( 26677), -INT16_C(  5764), -INT16_C( 29637),
        -INT16_C(  7059),  INT16_C(  8236), -INT16_C( 24987),  INT16_C( 23338), -INT16_C(  2319),  INT16_C( 14907), -INT16_C(  1362), -INT16_C( 21783),
         INT16_C(  6316),  INT16_C( 14684),  INT16_C(  3704),  INT16_C( 28424),  INT16_C( 15813),  INT16_C( 17112),  INT16_C(  4903), -INT16_C( 27442),
        -INT16_C(  1289),  INT16_C( 23732), -INT16_C(  8552), -INT16_C( 30280), -INT16_C(  3116), -INT16_C( 32060), -INT16_C( 21011), -INT16_C( 26323) },
      {  INT16_C( 25930), -INT16_C( 18441),  INT16_C(  1382),  INT16_C( 26677), -INT16_C(  9948), -INT16_C(  5764),  INT16_C( 20160), -INT16_C( 29637),
         INT16_C( 30295), -INT16_C(  2319), -INT16_C( 15647),  INT16_C( 14907), -INT16_C(  2327), -INT16_C(  1362), -INT16_C(  7956), -INT16_C( 21783),
        -INT16_C( 26815),  INT16_C( 15813),  INT16_C( 26117),  INT16_C( 17112), -INT16_C( 14991),  INT16_C(  4903), -INT16_C( 11596), -INT16_C( 27442),
         INT16_C( 21494), -INT16_C(  3116), -INT16_C(  8244), -INT16_C( 32060), -INT16_C( 18359), -INT16_C( 21011), -INT16_C( 20801), -INT16_C( 26323) } },
    { { -INT16_C( 30267),  INT16_C( 15827), -INT16_C(  9320),  INT16_C( 23980), -INT16_C( 31719),  INT16_C( 16543),  INT16_C( 28311), -INT16_C( 28716),
        -INT16_C( 30616),  INT16_C(   491), -INT16_C( 23706),  INT16_C( 14986),  INT16_C( 20118), -INT16_C( 31556), -INT16_C(  5637), -INT16_C( 16355),
        -INT16_C(  3981),  INT16_C(  3069), -INT16_C( 21812), -INT16_C(  6808),  INT16_C(  2094), -INT16_C( 14811), -INT16_C(  1674), -INT16_C(  8619),
         INT16_C( 16513), -INT16_C(  6177),  INT16_C( 27364),  INT16_C( 31265), -INT16_C(  8776), -INT16_C( 19202),  INT16_C(  7367),  INT16_C( 14964) },
      {  INT16_C( 29196), -INT16_C( 10171), -INT16_C( 21220),  INT16_C( 19133), -INT16_C(  7499),  INT16_C( 11024),  INT16_C( 26075),  INT16_C( 23562),
        -INT16_C(  5722), -INT16_C( 30141),  INT16_C( 25683),  INT16_C(  3076),  INT16_C(   834),  INT16_C(  2496),  INT16_C( 13343),  INT16_C( 11075),
        -INT16_C( 30554), -INT16_C( 15868), -INT16_C( 16075), -INT16_C(  5363),  INT16_C(  7588),  INT16_C( 32534),  INT16_C(  8323),  INT16_C( 10716),
         INT16_C(  7946),  INT16_C( 23987), -INT16_C( 18556), -INT16_C( 14743),  INT16_C( 10682), -INT16_C(  9777),  INT16_C(  4702),  INT16_C(  1029) },
      { -INT16_C( 31719), -INT16_C(  7499),  INT16_C( 16543),  INT16_C( 11024),  INT16_C( 28311),  INT16_C( 26075), -INT16_C( 28716),  INT16_C( 23562),
         INT16_C( 20118),  INT16_C(   834), -INT16_C( 31556),  INT16_C(  2496), -INT16_C(  5637),  INT16_C( 13343), -INT16_C( 16355),  INT16_C( 11075),
         INT16_C(  2094),  INT16_C(  7588), -INT16_C( 14811),  INT16_C( 32534), -INT16_C(  1674),  INT16_C(  8323), -INT16_C(  8619),  INT16_C( 10716),
        -INT16_C(  8776),  INT16_C( 10682), -INT16_C( 19202), -INT16_C(  9777),  INT16_C(  7367),  INT16_C(  4702),  INT16_C( 14964),  INT16_C(  1029) } },
    { {  INT16_C(  2458), -INT16_C( 12345), -INT16_C( 11062),  INT16_C( 28346), -INT16_C( 11791),  INT16_C( 29934), -INT16_C( 13583), -INT16_C(  1123),
         INT16_C( 20713),  INT16_C( 27993), -INT16_C( 15864), -INT16_C( 15821),  INT16_C(   748),  INT16_C( 19100), -INT16_C( 24300), -INT16_C( 20914),
         INT16_C(  5546),  INT16_C( 29822),  INT16_C( 14569), -INT16_C(  9245), -INT16_C( 12023), -INT16_C(  1201), -INT16_C(  4709), -INT16_C( 31498),
         INT16_C( 20285),  INT16_C( 17906),  INT16_C(  9490), -INT16_C(   504), -INT16_C( 23512),  INT16_C( 15432), -INT16_C( 27067), -INT16_C(  4117) },
      {  INT16_C( 27052), -INT16_C( 27293),  INT16_C( 18081), -INT16_C( 21648), -INT16_C( 16361), -INT16_C( 19802), -INT16_C( 25427), -INT16_C(  5577),
         INT16_C( 10732), -INT16_C(   464),  INT16_C( 14414),  INT16_C( 30460),  INT16_C( 17628),  INT16_C(  8627), -INT16_C( 24870), -INT16_C( 31216),
         INT16_C( 29447), -INT16_C( 22500), -INT16_C( 29510), -INT16_C( 11949), -INT16_C(  1716), -INT16_C(  1660), -INT16_C( 17514), -INT16_C( 32028),
         INT16_C(  5348),  INT16_C( 12928),  INT16_C( 31820),  INT16_C( 10409),  INT16_C( 23744), -INT16_C( 26039),  INT16_C( 23034),  INT16_C(   289) },
      { -INT16_C( 11791), -INT16_C( 16361),  INT16_C( 29934), -INT16_C( 19802), -INT16_C( 13583), -INT16_C( 25427), -INT16_C(  1123), -INT16_C(  5577),
         INT16_C(   748),  INT16_C( 17628),  INT16_C( 19100),  INT16_C(  8627), -INT16_C( 24300), -INT16_C( 24870), -INT16_C( 20914), -INT16_C( 31216),
        -INT16_C( 12023), -INT16_C(  1716), -INT16_C(  1201), -INT16_C(  1660), -INT16_C(  4709), -INT16_C( 17514), -INT16_C( 31498), -INT16_C( 32028),
        -INT16_C( 23512),  INT16_C( 23744),  INT16_C( 15432), -INT16_C( 26039), -INT16_C( 27067),  INT16_C( 23034), -INT16_C(  4117),  INT16_C(   289) } },
    { {  INT16_C( 15820), -INT16_C( 31063), -INT16_C(   567),  INT16_C(  5720), -INT16_C(  8970), -INT16_C( 29681), -INT16_C(  3177),  INT16_C( 31502),
        -INT16_C( 29177),  INT16_C( 21421),  INT16_C( 22026), -INT16_C( 13701), -INT16_C( 15182), -INT16_C( 21403), -INT16_C( 31203), -INT16_C(  5459),
         INT16_C( 22467), -INT16_C( 29584), -INT16_C( 14252),  INT16_C( 19106), -INT16_C( 19804),  INT16_C( 15319), -INT16_C(  6747), -INT16_C( 21066),
         INT16_C( 25716),  INT16_C( 32256),  INT16_C( 31930),  INT16_C( 27977), -INT16_C( 20928),  INT16_C( 24089), -INT16_C( 14540), -INT16_C(  2232) },
      { -INT16_C( 18402),  INT16_C( 29315),  INT16_C(  9857),  INT16_C(  9660), -INT16_C( 27688),  INT16_C( 32097),  INT16_C(  6009), -INT16_C(  4822),
         INT16_C( 11131),  INT16_C( 13931), -INT16_C( 19289), -INT16_C(  6237), -INT16_C( 17310), -INT16_C( 27067), -INT16_C( 29309), -INT16_C( 24179),
         INT16_C(  4422), -INT16_C( 14573), -INT16_C( 12233),  INT16_C(  4076),  INT16_C( 19811), -INT16_C(  9076), -INT16_C( 18587), -INT16_C(  7991),
         INT16_C( 13794), -INT16_C( 30442), -INT16_C( 17943),  INT16_C( 19568), -INT16_C( 18826), -INT16_C(  1566),  INT16_C( 28739), -INT16_C( 30309) },
      { -INT16_C(  8970), -INT16_C( 27688), -INT16_C( 29681),  INT16_C( 32097), -INT16_C(  3177),  INT16_C(  6009),  INT16_C( 31502), -INT16_C(  4822),
        -INT16_C( 15182), -INT16_C( 17310), -INT16_C( 21403), -INT16_C( 27067), -INT16_C( 31203), -INT16_C( 29309), -INT16_C(  5459), -INT16_C( 24179),
        -INT16_C( 19804),  INT16_C( 19811),  INT16_C( 15319), -INT16_C(  9076), -INT16_C(  6747), -INT16_C( 18587), -INT16_C( 21066), -INT16_C(  7991),
        -INT16_C( 20928), -INT16_C( 18826),  INT16_C( 24089), -INT16_C(  1566), -INT16_C( 14540),  INT16_C( 28739), -INT16_C(  2232), -INT16_C( 30309) } },
    { { -INT16_C( 20863), -INT16_C( 18352),  INT16_C( 15742), -INT16_C(  7481),  INT16_C( 21386), -INT16_C(  4162), -INT16_C( 30710), -INT16_C(  4912),
        -INT16_C(  6467), -INT16_C( 22923), -INT16_C(  6496),  INT16_C(  5874), -INT16_C( 10852), -INT16_C(  8433), -INT16_C( 21947), -INT16_C( 14743),
        -INT16_C( 18087), -INT16_C( 10370),  INT16_C( 17910), -INT16_C( 32327),  INT16_C( 30872), -INT16_C( 23696),  INT16_C( 16384), -INT16_C( 17009),
         INT16_C(  1319), -INT16_C( 14493),  INT16_C( 22251), -INT16_C( 30755), -INT16_C(  5077),  INT16_C( 28774), -INT16_C( 12393), -INT16_C(  4042) },
      { -INT16_C( 19319),  INT16_C( 32711), -INT16_C( 32263), -INT16_C( 28416),  INT16_C( 29177), -INT16_C(  1740), -INT16_C( 15183), -INT16_C( 10058),
         INT16_C(  6601), -INT16_C( 19297),  INT16_C( 31855), -INT16_C( 26053), -INT16_C( 24215),  INT16_C(    10),  INT16_C( 16497), -INT16_C(  1296),
        -INT16_C( 18444), -INT16_C(  4743),  INT16_C( 31288),  INT16_C( 12671), -INT16_C( 19477), -INT16_C( 25558), -INT16_C(  8073),  INT16_C( 16501),
         INT16_C(  5370),  INT16_C( 27124),  INT16_C( 12177), -INT16_C(  1532),  INT16_C(  3793),  INT16_C( 17146), -INT16_C(  5553),  INT16_C( 17212) },
      {  INT16_C( 21386),  INT16_C( 29177), -INT16_C(  4162), -INT16_C(  1740), -INT16_C( 30710), -INT16_C( 15183), -INT16_C(  4912), -INT16_C( 10058),
        -INT16_C( 10852), -INT16_C( 24215), -INT16_C(  8433),  INT16_C(    10), -INT16_C( 21947),  INT16_C( 16497), -INT16_C( 14743), -INT16_C(  1296),
         INT16_C( 30872), -INT16_C( 19477), -INT16_C( 23696), -INT16_C( 25558),  INT16_C( 16384), -INT16_C(  8073), -INT16_C( 17009),  INT16_C( 16501),
        -INT16_C(  5077),  INT16_C(  3793),  INT16_C( 28774),  INT16_C( 17146), -INT16_C( 12393), -INT16_C(  5553), -INT16_C(  4042),  INT16_C( 17212) } },
    { { -INT16_C( 19039), -INT16_C(  9679), -INT16_C( 20433),  INT16_C(  6667),  INT16_C( 13923), -INT16_C(  9289),  INT16_C( 11286),  INT16_C(  4123),
         INT16_C(  4160), -INT16_C( 11910),  INT16_C( 32319),  INT16_C(  4299), -INT16_C( 14964), -INT16_C(  9390), -INT16_C( 29009),  INT16_C( 20767),
         INT16_C( 20548),  INT16_C( 29483),  INT16_C( 13824),  INT16_C( 25486),  INT16_C( 17772), -INT16_C( 31938),  INT16_C( 23153), -INT16_C( 20077),
         INT16_C(  3434), -INT16_C( 22141),  INT16_C( 20107),  INT16_C(  6330),  INT16_C(  3092), -INT16_C( 15373),  INT16_C(  4763), -INT16_C(  8428) },
      {  INT16_C( 16226),  INT16_C( 25170), -INT16_C(  8074), -INT16_C(  7482),  INT16_C(  1061), -INT16_C( 27035), -INT16_C(  1698), -INT16_C( 14264),
        -INT16_C( 13562), -INT16_C( 28046),  INT16_C( 11289),  INT16_C( 11690), -INT16_C( 25288), -INT16_C( 11279),  INT16_C(  1456),  INT16_C(  4786),
         INT16_C(  1349), -INT16_C( 17547),  INT16_C( 15333),  INT16_C(  2973),  INT16_C(   831), -INT16_C( 24927), -INT16_C(  5636),  INT16_C(   614),
        -INT16_C( 10060), -INT16_C( 12652),  INT16_C( 15876),  INT16_C( 15867), -INT16_C(  4900), -INT16_C( 29680), -INT16_C( 15374),  INT16_C( 14238) },
      {  INT16_C( 13923),  INT16_C(  1061), -INT16_C(  9289), -INT16_C( 27035),  INT16_C( 11286), -INT16_C(  1698),  INT16_C(  4123), -INT16_C( 14264),
        -INT16_C( 14964), -INT16_C( 25288), -INT16_C(  9390), -INT16_C( 11279), -INT16_C( 29009),  INT16_C(  1456),  INT16_C( 20767),  INT16_C(  4786),
         INT16_C( 17772),  INT16_C(   831), -INT16_C( 31938), -INT16_C( 24927),  INT16_C( 23153), -INT16_C(  5636), -INT16_C( 20077),  INT16_C(   614),
         INT16_C(  3092), -INT16_C(  4900), -INT16_C( 15373), -INT16_C( 29680),  INT16_C(  4763), -INT16_C( 15374), -INT16_C(  8428),  INT16_C( 14238) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi16(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi16(test_vec[i].b);
    simde__m512i r = simde_mm512_unpackhi_epi16(a, b);
    simde_test_x86_assert_equal_i16x32(r, simde_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_unpackhi_epi32 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  2071539538), -INT32_C(  1292562566), -INT32_C(   411127984), -INT32_C(  1435371435),  INT32_C(   903717735), -INT32_C(  1003795278), -INT32_C(   469164256), -INT32_C(   618004435),
         INT32_C(  1247785168),  INT32_C(   285038015),  INT32_C(  1509391108), -INT32_C(   637245069), -INT32_C(  1106255604),  INT32_C(  1300380205), -INT32_C(  2043573415),  INT32_C(  1533107083) },
      { -INT32_C(   878264308),  INT32_C(   450601749), -INT32_C(  1854680290),  INT32_C(  1215002428), -INT32_C(  2063173032),  INT32_C(   231901364), -INT32_C(  1617755373),  INT32_C(  1811608671),
        -INT32_C(   902389580),  INT32_C(  1642336835),  INT32_C(   569530341),  INT32_C(   661216719), -INT32_C(  1951567913),  INT32_C(   177766391), -INT32_C(   492229757), -INT32_C(   733109217) },
      { -INT32_C(   411127984), -INT32_C(  1854680290), -INT32_C(  1435371435),  INT32_C(  1215002428), -INT32_C(   469164256), -INT32_C(  1617755373), -INT32_C(   618004435),  INT32_C(  1811608671),
         INT32_C(  1509391108),  INT32_C(   569530341), -INT32_C(   637245069),  INT32_C(   661216719), -INT32_C(  2043573415), -INT32_C(   492229757),  INT32_C(  1533107083), -INT32_C(   733109217) } },
    { { -INT32_C(  2002877628),  INT32_C(  2062123669), -INT32_C(  1466180391),  INT32_C(   298845241),  INT32_C(  1788640627),  INT32_C(  2138387964), -INT32_C(  2141118880),  INT32_C(   122990274),
        -INT32_C(   946867662),  INT32_C(  1296201844), -INT32_C(  1896424108),  INT32_C(  1436534498),  INT32_C(  1069562691), -INT32_C(   775998096),  INT32_C(   374415443),  INT32_C(    18720207) },
      {  INT32_C(   197700759),  INT32_C(  2035812900), -INT32_C(   905490712),  INT32_C(  1461691924),  INT32_C(  1385684961),  INT32_C(  1747146260),  INT32_C(  1165915254), -INT32_C(  1337550055),
         INT32_C(  1807421255),  INT32_C(    31724313),  INT32_C(  1993075554),  INT32_C(  1942940561), -INT32_C(   540711478),  INT32_C(   826796219),  INT32_C(  1970783580), -INT32_C(  1490698912) },
      { -INT32_C(  1466180391), -INT32_C(   905490712),  INT32_C(   298845241),  INT32_C(  1461691924), -INT32_C(  2141118880),  INT32_C(  1165915254),  INT32_C(   122990274), -INT32_C(  1337550055),
        -INT32_C(  1896424108),  INT32_C(  1993075554),  INT32_C(  1436534498),  INT32_C(  1942940561),  INT32_C(   374415443),  INT32_C(  1970783580),  INT32_C(    18720207), -INT32_C(  1490698912) } },
    { { -INT32_C(   434970420),  INT32_C(  1441265651),  INT32_C(  1959572450),  INT32_C(  1760008862), -INT32_C(  1169707777), -INT32_C(   252997996), -INT32_C(  1285201325), -INT32_C(   329610720),
         INT32_C(  1574071658),  INT32_C(  1186183780),  INT32_C(   196771693),  INT32_C(   410231065), -INT32_C(   506283187), -INT32_C(  1647198903),  INT32_C(  1078998560),  INT32_C(   707570624) },
      {  INT32_C(  2089352984),  INT32_C(   650263481), -INT32_C(   751731270),  INT32_C(  1827382302), -INT32_C(  1454523041), -INT32_C(  1706680454),  INT32_C(   383424085),  INT32_C(  1497368385),
        -INT32_C(  1076508666), -INT32_C(  1109026813),  INT32_C(   865080853),  INT32_C(   446659514), -INT32_C(  1279005384),  INT32_C(  1632438540), -INT32_C(   512284513),  INT32_C(   893040687) },
      {  INT32_C(  1959572450), -INT32_C(   751731270),  INT32_C(  1760008862),  INT32_C(  1827382302), -INT32_C(  1285201325),  INT32_C(   383424085), -INT32_C(   329610720),  INT32_C(  1497368385),
         INT32_C(   196771693),  INT32_C(   865080853),  INT32_C(   410231065),  INT32_C(   446659514),  INT32_C(  1078998560), -INT32_C(   512284513),  INT32_C(   707570624),  INT32_C(   893040687) } },
    { { -INT32_C(  2064379776), -INT32_C(  1119757912), -INT32_C(  1443835153), -INT32_C(  2034003891), -INT32_C(  1992718723),  INT32_C(   803899023), -INT32_C(   586128722), -INT32_C(  1710077414),
         INT32_C(    35522138), -INT32_C(   826318625),  INT32_C(  2138615858), -INT32_C(  1123730624),  INT32_C(  1363558082),  INT32_C(  1937780933), -INT32_C(  1403940718),  INT32_C(   893871067) },
      {  INT32_C(  1228432746), -INT32_C(   149358651), -INT32_C(   411594585), -INT32_C(  1918599989), -INT32_C(  2132809029), -INT32_C(  1376559333), -INT32_C(   883276560),  INT32_C(   302031272),
        -INT32_C(   866437114), -INT32_C(   675056848), -INT32_C(   809551357),  INT32_C(  1918657463),  INT32_C(  1777482574), -INT32_C(  1961433701), -INT32_C(   766087126),  INT32_C(   417617425) },
      { -INT32_C(  1443835153), -INT32_C(   411594585), -INT32_C(  2034003891), -INT32_C(  1918599989), -INT32_C(   586128722), -INT32_C(   883276560), -INT32_C(  1710077414),  INT32_C(   302031272),
         INT32_C(  2138615858), -INT32_C(   809551357), -INT32_C(  1123730624),  INT32_C(  1918657463), -INT32_C(  1403940718), -INT32_C(   766087126),  INT32_C(   893871067),  INT32_C(   417617425) } },
    { { -INT32_C(  1075560561), -INT32_C(  1231640653), -INT32_C(  1719314974),  INT32_C(   118219449), -INT32_C(  1200554723),  INT32_C(   205752034),  INT32_C(   148806135),  INT32_C(  2132853488),
        -INT32_C(  1254226942), -INT32_C(  1905535828), -INT32_C(   483921622), -INT32_C(   253087021),  INT32_C(   296311343), -INT32_C(   685839136),  INT32_C(  1994456198), -INT32_C(  1040908097) },
      { -INT32_C(  1317653755),  INT32_C(   826269959), -INT32_C(  1525389614), -INT32_C(   946405736),  INT32_C(   970473304), -INT32_C(  1324288469), -INT32_C(  1306005261), -INT32_C(   160228111),
         INT32_C(  1470622031), -INT32_C(  1651972406), -INT32_C(   465396404), -INT32_C(   190064485),  INT32_C(  1127056151),  INT32_C(  1844723066),  INT32_C(   522132526), -INT32_C(  2011852232) },
      { -INT32_C(  1719314974), -INT32_C(  1525389614),  INT32_C(   118219449), -INT32_C(   946405736),  INT32_C(   148806135), -INT32_C(  1306005261),  INT32_C(  2132853488), -INT32_C(   160228111),
        -INT32_C(   483921622), -INT32_C(   465396404), -INT32_C(   253087021), -INT32_C(   190064485),  INT32_C(  1994456198),  INT32_C(   522132526), -INT32_C(  1040908097), -INT32_C(  2011852232) } },
    { {  INT32_C(  1172290683), -INT32_C(   287152222), -INT32_C(  1596840700),  INT32_C(   362053117),  INT32_C(  2052636928),  INT32_C(   753356030), -INT32_C(  1588918680),  INT32_C(   321478808),
        -INT32_C(  1101461476),  INT32_C(  1957444463),  INT32_C(  1578401376), -INT32_C(    59528965),  INT32_C(  1735838569), -INT32_C(  2137760233), -INT32_C(    48111772),  INT32_C(  1544571456) },
      { -INT32_C(  1055168174),  INT32_C(    87410597),  INT32_C(  1097025862),  INT32_C(  1514002161), -INT32_C(  1178422111),  INT32_C(  1983469074),  INT32_C(  1970494005), -INT32_C(   153975644),
        -INT32_C(  1833439763), -INT32_C(    90640972),  INT32_C(   675085110),  INT32_C(  1937930706),  INT32_C(  1059865645), -INT32_C(   793352806),  INT32_C(  1665477055), -INT32_C(  1688660051) },
      { -INT32_C(  1596840700),  INT32_C(  1097025862),  INT32_C(   362053117),  INT32_C(  1514002161), -INT32_C(  1588918680),  INT32_C(  1970494005),  INT32_C(   321478808), -INT32_C(   153975644),
         INT32_C(  1578401376),  INT32_C(   675085110), -INT32_C(    59528965),  INT32_C(  1937930706), -INT32_C(    48111772),  INT32_C(  1665477055),  INT32_C(  1544571456), -INT32_C(  1688660051) } },
    { { -INT32_C(  1188228860),  INT32_C(   884196862), -INT32_C(  1822625855), -INT32_C(  1777934487), -INT32_C(  1093258461),  INT32_C(  1485737112),  INT32_C(  1673253813), -INT32_C(   268560917),
         INT32_C(   615000870), -INT32_C(  1302831887), -INT32_C(  1270500021), -INT32_C(  1219802220),  INT32_C(   393552254),  INT32_C(  1651442605), -INT32_C(  1027265833),  INT32_C(  1706148671) },
      { -INT32_C(   527869201),  INT32_C(    26403510),  INT32_C(   733403031), -INT32_C(  1579024094), -INT32_C(   810002398),  INT32_C(   842082139), -INT32_C(  1846216879), -INT32_C(  1443453254),
        -INT32_C(  1232502784), -INT32_C(   105440414),  INT32_C(   354708978), -INT32_C(  1867118994), -INT32_C(  1168150946), -INT32_C(   420703851),  INT32_C(  1115152776), -INT32_C(  2014548345) },
      { -INT32_C(  1822625855),  INT32_C(   733403031), -INT32_C(  1777934487), -INT32_C(  1579024094),  INT32_C(  1673253813), -INT32_C(  1846216879), -INT32_C(   268560917), -INT32_C(  1443453254),
        -INT32_C(  1270500021),  INT32_C(   354708978), -INT32_C(  1219802220), -INT32_C(  1867118994), -INT32_C(  1027265833),  INT32_C(  1115152776),  INT32_C(  1706148671), -INT32_C(  2014548345) } },
    { {  INT32_C(  1346205166), -INT32_C(  2092305263), -INT32_C(   795316894), -INT32_C(   765374861),  INT32_C(  1368178876), -INT32_C(   650610607), -INT32_C(   534991015),  INT32_C(   191301661),
         INT32_C(   240886909), -INT32_C(    74275687), -INT32_C(  2050282991),  INT32_C(   894905465),  INT32_C(  1049093101), -INT32_C(  1256669349), -INT32_C(  1936378770), -INT32_C(  1181221572) },
      {  INT32_C(   986248097), -INT32_C(  1456121193), -INT32_C(    47316604),  INT32_C(   439584045), -INT32_C(  1017529752), -INT32_C(   411537031), -INT32_C(   512553307), -INT32_C(  1399190773),
        -INT32_C(  1779997954),  INT32_C(  1094589628),  INT32_C(  1262382109),  INT32_C(  1499820529), -INT32_C(  1541554645), -INT32_C(   728984273), -INT32_C(  1363804253),  INT32_C(   140201994) },
      { -INT32_C(   795316894), -INT32_C(    47316604), -INT32_C(   765374861),  INT32_C(   439584045), -INT32_C(   534991015), -INT32_C(   512553307),  INT32_C(   191301661), -INT32_C(  1399190773),
        -INT32_C(  2050282991),  INT32_C(  1262382109),  INT32_C(   894905465),  INT32_C(  1499820529), -INT32_C(  1936378770), -INT32_C(  1363804253), -INT32_C(  1181221572),  INT32_C(   140201994) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi32(test_vec[i].b);
    simde__m512i r = simde_mm512_unpackhi_epi32(a, b);
    simde_test_x86_assert_equal_i32x16(r, simde_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_mask_unpackhi_epi32 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const simde__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   845938351),  INT32_C(   181822741), -INT32_C(   904073692),  INT32_C(  1797241754), -INT32_C(  1981644741), -INT32_C(   725414939), -INT32_C(   676171526), -INT32_C(  1255219612),
        -INT32_C(   192757025),  INT32_C(  1291737128), -INT32_C(   518579385),  INT32_C(   306984662), -INT32_C(  1667551562),  INT32_C(   863002169),  INT32_C(   839525070), -INT32_C(   538494976) },
      UINT16_C(27131),
      { -INT32_C(   759028780), -INT32_C(  2031285905),  INT32_C(   918406378),  INT32_C(  1902408662), -INT32_C(  2133877489), -INT32_C(   626876976), -INT32_C(  1240226865),  INT32_C(  1444875906),
        -INT32_C(  1607868111), -INT32_C(   735635734), -INT32_C(  1341463334),  INT32_C(  1730244183), -INT32_C(   605556725),  INT32_C(  1555466637), -INT32_C(  1357723348),  INT32_C(   134558423) },
      { -INT32_C(    22532589),  INT32_C(   533909317),  INT32_C(   198237619),  INT32_C(  1467150667),  INT32_C(  1865570786),  INT32_C(   265087202), -INT32_C(  2000757071),  INT32_C(   613466896),
         INT32_C(   924989938), -INT32_C(  1135152120),  INT32_C(   499590865), -INT32_C(    93046504),  INT32_C(  1953080978),  INT32_C(  1082340751),  INT32_C(   617169172), -INT32_C(   146253563) },
      {  INT32_C(   918406378),  INT32_C(   198237619), -INT32_C(   904073692),  INT32_C(  1467150667), -INT32_C(  1240226865), -INT32_C(  2000757071),  INT32_C(  1444875906),  INT32_C(   613466896),
        -INT32_C(  1341463334),  INT32_C(  1291737128), -INT32_C(   518579385), -INT32_C(    93046504), -INT32_C(  1667551562),  INT32_C(   617169172),  INT32_C(   134558423), -INT32_C(   538494976) } },
    { { -INT32_C(  1691456878),  INT32_C(   811041887), -INT32_C(  1018356053), -INT32_C(   373440169), -INT32_C(   144890264),  INT32_C(  1882710364),  INT32_C(   664010786), -INT32_C(   333521574),
        -INT32_C(  1501082553),  INT32_C(  2094522065),  INT32_C(  1396647164),  INT32_C(  1312619750),  INT32_C(  2118490658), -INT32_C(  1645314949), -INT32_C(   674921603), -INT32_C(  1480334496) },
      UINT16_C(19248),
      {  INT32_C(   623444302), -INT32_C(  1136056707), -INT32_C(  1229443207), -INT32_C(  1034888579),  INT32_C(  1195363161),  INT32_C(   768261224),  INT32_C(  1443965587),  INT32_C(   547439058),
        -INT32_C(  1085945022),  INT32_C(  1786482417),  INT32_C(   991966142),  INT32_C(  1727950861), -INT32_C(  1532084933), -INT32_C(  1932429063),  INT32_C(  1977803427),  INT32_C(  1704297507) },
      {  INT32_C(  1076157007),  INT32_C(   648716136), -INT32_C(   530396206),  INT32_C(  2001100859), -INT32_C(  1776552803),  INT32_C(   270724205), -INT32_C(   226163505), -INT32_C(   682091896),
         INT32_C(  1545043188), -INT32_C(   293354980), -INT32_C(   925964915), -INT32_C(   499182267),  INT32_C(  1987598857),  INT32_C(   377920071),  INT32_C(   654838686),  INT32_C(   436101157) },
      { -INT32_C(  1691456878),  INT32_C(   811041887), -INT32_C(  1018356053), -INT32_C(   373440169),  INT32_C(  1443965587), -INT32_C(   226163505),  INT32_C(   664010786), -INT32_C(   333521574),
         INT32_C(   991966142), -INT32_C(   925964915),  INT32_C(  1396647164), -INT32_C(   499182267),  INT32_C(  2118490658), -INT32_C(  1645314949),  INT32_C(  1704297507), -INT32_C(  1480334496) } },
    { { -INT32_C(   109701411),  INT32_C(  1709701592),  INT32_C(   590198494), -INT32_C(   737841717),  INT32_C(   239828423), -INT32_C(  1255878377),  INT32_C(    47984093),  INT32_C(  1780276109),
        -INT32_C(   916221199),  INT32_C(  1764641675), -INT32_C(   863216895), -INT32_C(  1868525112),  INT32_C(   631171854), -INT32_C(  1696939075),  INT32_C(  2107422704), -INT32_C(  2081900398) },
      UINT16_C(19274),
      {  INT32_C(  2056705356), -INT32_C(   891971778),  INT32_C(    73113187), -INT32_C(   856725202),  INT32_C(  1771023502), -INT32_C(   484409530), -INT32_C(   442780931), -INT32_C(  2127501771),
        -INT32_C(    84162629), -INT32_C(  1027288994), -INT32_C(  1664737170),  INT32_C(   409580937), -INT32_C(  1467877278),  INT32_C(  1972085112), -INT32_C(  2007357613), -INT32_C(   938898931) },
      { -INT32_C(  1329462191),  INT32_C(  1131579348),  INT32_C(   819935399),  INT32_C(  1330137325), -INT32_C(  1174877887), -INT32_C(  1121025174), -INT32_C(  1220179798),  INT32_C(  1686130194),
         INT32_C(   655639122),  INT32_C(  1886029513), -INT32_C(  1415493186), -INT32_C(   738465390),  INT32_C(   495776691),  INT32_C(   568048246),  INT32_C(  1423450178), -INT32_C(  1044883345) },
      { -INT32_C(   109701411),  INT32_C(   819935399),  INT32_C(   590198494),  INT32_C(  1330137325),  INT32_C(   239828423), -INT32_C(  1255878377), -INT32_C(  2127501771),  INT32_C(  1780276109),
        -INT32_C(  1664737170), -INT32_C(  1415493186), -INT32_C(   863216895), -INT32_C(   738465390),  INT32_C(   631171854), -INT32_C(  1696939075), -INT32_C(   938898931), -INT32_C(  2081900398) } },
    { {  INT32_C(  1692978331),  INT32_C(   282415698),  INT32_C(   784103068),  INT32_C(   302102367),  INT32_C(   540052906), -INT32_C(  1992225977), -INT32_C(  1696785877),  INT32_C(   224171634),
        -INT32_C(  1250802590),  INT32_C(   868566935),  INT32_C(   476152253), -INT32_C(   500211144),  INT32_C(   906190831), -INT32_C(  1782627222), -INT32_C(   785343394), -INT32_C(  1780577230) },
      UINT16_C(20688),
      {  INT32_C(   261580618), -INT32_C(    74361702), -INT32_C(  1604466319), -INT32_C(  1342223188),  INT32_C(  1106471298),  INT32_C(   786322174),  INT32_C(    28971043), -INT32_C(   279802971),
        -INT32_C(  1912608270), -INT32_C(  1350004674),  INT32_C(    89122393), -INT32_C(  1246474702), -INT32_C(  1242126153),  INT32_C(   501535994), -INT32_C(  1977704731),  INT32_C(   477786153) },
      { -INT32_C(  1733723814),  INT32_C(  1648832777),  INT32_C(  1248368151), -INT32_C(  1677779740), -INT32_C(  1101990460), -INT32_C(  1344588598), -INT32_C(    63309101), -INT32_C(  1005014166),
         INT32_C(   895271212),  INT32_C(   177775603),  INT32_C(   492044345), -INT32_C(   508013796),  INT32_C(   329189705),  INT32_C(   281180989), -INT32_C(   552731787), -INT32_C(   609999441) },
      {  INT32_C(  1692978331),  INT32_C(   282415698),  INT32_C(   784103068),  INT32_C(   302102367),  INT32_C(    28971043), -INT32_C(  1992225977), -INT32_C(   279802971), -INT32_C(  1005014166),
        -INT32_C(  1250802590),  INT32_C(   868566935),  INT32_C(   476152253), -INT32_C(   500211144), -INT32_C(  1977704731), -INT32_C(  1782627222),  INT32_C(   477786153), -INT32_C(  1780577230) } },
    { { -INT32_C(   636419865), -INT32_C(   572216924), -INT32_C(   973391447), -INT32_C(   710495348), -INT32_C(    85440836),  INT32_C(   906668737),  INT32_C(  1427445670),  INT32_C(   607172925),
         INT32_C(  1593721274), -INT32_C(  1824791830), -INT32_C(  1470548453), -INT32_C(  1501691926),  INT32_C(   111175237), -INT32_C(  1237538032), -INT32_C(    16035134), -INT32_C(   987546613) },
      UINT16_C( 8573),
      {  INT32_C(  1594124323),  INT32_C(  1419059195),  INT32_C(  1163165639), -INT32_C(   961832666), -INT32_C(   596526176),  INT32_C(  2100180082),  INT32_C(  1471756851),  INT32_C(   595081215),
        -INT32_C(  1702724449),  INT32_C(  1676613532), -INT32_C(  1113046121),  INT32_C(  2088981468), -INT32_C(  2124810993),  INT32_C(  1560184617), -INT32_C(  1045186367), -INT32_C(  1897648913) },
      {  INT32_C(  1143498408),  INT32_C(   346560381),  INT32_C(   936464474), -INT32_C(  1280092764),  INT32_C(  1932790858),  INT32_C(  1439642260), -INT32_C(   652835862),  INT32_C(  1466432175),
        -INT32_C(   560164768),  INT32_C(    49431719),  INT32_C(   943309972),  INT32_C(  1676405785), -INT32_C(  1915346951),  INT32_C(  1021486418), -INT32_C(   686360280),  INT32_C(  1412398579) },
      {  INT32_C(  1163165639), -INT32_C(   572216924), -INT32_C(   961832666), -INT32_C(  1280092764),  INT32_C(  1471756851), -INT32_C(   652835862),  INT32_C(   595081215),  INT32_C(   607172925),
        -INT32_C(  1113046121), -INT32_C(  1824791830), -INT32_C(  1470548453), -INT32_C(  1501691926),  INT32_C(   111175237), -INT32_C(   686360280), -INT32_C(    16035134), -INT32_C(   987546613) } },
    { { -INT32_C(  1254962419), -INT32_C(  1548278769),  INT32_C(    31256808), -INT32_C(   714815524),  INT32_C(   962738919),  INT32_C(   141903328),  INT32_C(   836799294),  INT32_C(   377818889),
        -INT32_C(   372525094), -INT32_C(   997424420),  INT32_C(  1338402930),  INT32_C(   388246064),  INT32_C(  1162905445),  INT32_C(   172869068),  INT32_C(  1513827665),  INT32_C(   376488252) },
      UINT16_C(15480),
      { -INT32_C(  1933683457), -INT32_C(   537644775), -INT32_C(  1542904704), -INT32_C(  1960088005),  INT32_C(    22149043),  INT32_C(  1026466305),  INT32_C(  1828613116), -INT32_C(  2119666047),
        -INT32_C(   468883509),  INT32_C(   415433112),  INT32_C(  1639762982), -INT32_C(   286398405), -INT32_C(   504414497), -INT32_C(   602005792),  INT32_C(   172498057),  INT32_C(  1569452434) },
      { -INT32_C(   264136616), -INT32_C(  1073216358),  INT32_C(   203539665), -INT32_C(  1929769300),  INT32_C(   745400908), -INT32_C(  1861711096),  INT32_C(   966480295), -INT32_C(  1701370046),
         INT32_C(  1502271679), -INT32_C(  1374055715),  INT32_C(    62536534), -INT32_C(  1768967095), -INT32_C(  1497170786),  INT32_C(   775408519),  INT32_C(  1583862556), -INT32_C(  1174864134) },
      { -INT32_C(  1254962419), -INT32_C(  1548278769),  INT32_C(    31256808), -INT32_C(  1929769300),  INT32_C(  1828613116),  INT32_C(   966480295), -INT32_C(  2119666047),  INT32_C(   377818889),
        -INT32_C(   372525094), -INT32_C(   997424420), -INT32_C(   286398405), -INT32_C(  1768967095),  INT32_C(   172498057),  INT32_C(  1583862556),  INT32_C(  1513827665),  INT32_C(   376488252) } },
    { { -INT32_C(  1273789737),  INT32_C(  1784818708), -INT32_C(  1318249369),  INT32_C(  1866988752),  INT32_C(  2132085240), -INT32_C(   257077804),  INT32_C(   441324832), -INT32_C(   355252717),
        -INT32_C(   593566008),  INT32_C(  2051473427), -INT32_C(   315902948), -INT32_C(  1453559119),  INT32_C(  1344893308), -INT32_C(   566110530), -INT32_C(      487445), -INT32_C(  1628844842) },
      UINT16_C(34994),
      { -INT32_C(  1031223941),  INT32_C(  1802937664), -INT32_C(   287430766), -INT32_C(    94414127), -INT32_C(   338682198), -INT32_C(   176440068), -INT32_C(  1530834501),  INT32_C(  1781298159),
         INT32_C(  2032973113), -INT32_C(   320560294), -INT32_C(  1680162102), -INT32_C(   946521828),  INT32_C(  1420977751), -INT32_C(   582406878),  INT32_C(  1853950590), -INT32_C(  1227313539) },
      { -INT32_C(  1120991901),  INT32_C(  1923683496), -INT32_C(   217152554),  INT32_C(   347775932),  INT32_C(   728263689),  INT32_C(   403222938),  INT32_C(   948341435), -INT32_C(  1678811336),
         INT32_C(   207101540),  INT32_C(   159252786),  INT32_C(  1090292868),  INT32_C(   961852976), -INT32_C(  1134183390),  INT32_C(   685075821),  INT32_C(   794909687),  INT32_C(   516575418) },
      { -INT32_C(  1273789737), -INT32_C(   217152554), -INT32_C(  1318249369),  INT32_C(  1866988752), -INT32_C(  1530834501),  INT32_C(   948341435),  INT32_C(   441324832), -INT32_C(  1678811336),
        -INT32_C(   593566008),  INT32_C(  2051473427), -INT32_C(   315902948),  INT32_C(   961852976),  INT32_C(  1344893308), -INT32_C(   566110530), -INT32_C(      487445),  INT32_C(   516575418) } },
    { { -INT32_C(  1591074194), -INT32_C(  1481987805),  INT32_C(  1709745717),  INT32_C(  2124364892),  INT32_C(  1715143929),  INT32_C(  1770983537),  INT32_C(   647557227), -INT32_C(  1371249856),
        -INT32_C(  1454411899),  INT32_C(  1297152280), -INT32_C(    72140641),  INT32_C(  1853510261), -INT32_C(   942361258),  INT32_C(   808477637), -INT32_C(  1823028909), -INT32_C(  1321034964) },
      UINT16_C(37130),
      { -INT32_C(  1416945062),  INT32_C(   585312879), -INT32_C(  1602987994), -INT32_C(  1688876346), -INT32_C(  1023534446), -INT32_C(  1584704950),  INT32_C(   658290661), -INT32_C(  1011267991),
        -INT32_C(   680639384), -INT32_C(  1795534226),  INT32_C(  1899261610), -INT32_C(   888370887), -INT32_C(   292746589),  INT32_C(  1116674396),  INT32_C(   963234768),  INT32_C(  2046632465) },
      { -INT32_C(   716149914),  INT32_C(  1751730877), -INT32_C(   220619079), -INT32_C(   876747481),  INT32_C(  1287211759),  INT32_C(   881739875),  INT32_C(   611186451), -INT32_C(  2137167334),
        -INT32_C(  1823085098), -INT32_C(   235159752), -INT32_C(  2065443747), -INT32_C(  1454399303),  INT32_C(  1324681450),  INT32_C(  1669497680), -INT32_C(  1803030662),  INT32_C(   806691930) },
      { -INT32_C(  1591074194), -INT32_C(   220619079),  INT32_C(  1709745717), -INT32_C(   876747481),  INT32_C(  1715143929),  INT32_C(  1770983537),  INT32_C(   647557227), -INT32_C(  1371249856),
         INT32_C(  1899261610),  INT32_C(  1297152280), -INT32_C(    72140641),  INT32_C(  1853510261),  INT32_C(   963234768),  INT32_C(   808477637), -INT32_C(  1823028909),  INT32_C(   806691930) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i src = simde_mm512_loadu_epi32(test_vec[i].src);
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi32(test_vec[i].b);
    simde__m512i r = simde_mm512_mask_unpackhi_epi32(src, test_vec[i].k, a, b);
    simde_test_x86_assert_equal_i32x16(r, simde_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_maskz_unpackhi_epi32 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(51483),
      {  INT32_C(  1016400561),  INT32_C(   499657637),  INT32_C(   914546290),  INT32_C(   167035570),  INT32_C(  2082096633),  INT32_C(  1908618834), -INT32_C(  2117789817),  INT32_C(  2001461702),
        -INT32_C(  1850482453),  INT32_C(  2108586763),  INT32_C(   196292697), -INT32_C(   350902286),  INT32_C(  1063727085), -INT32_C(   307221915),  INT32_C(   141456962),  INT32_C(  1115732311) },
      { -INT32_C(  1529662567),  INT32_C(   136413615), -INT32_C(  1542138446),  INT32_C(  1771055484), -INT32_C(  1112934568),  INT32_C(  1688885538),  INT32_C(   644618703),  INT32_C(  1818881234),
        -INT32_C(   821019616),  INT32_C(  1893151422), -INT32_C(  2095781113),  INT32_C(  1827513364), -INT32_C(  1104505188), -INT32_C(  1088236305), -INT32_C(  1058697491), -INT32_C(  1691595141) },
      {  INT32_C(   914546290), -INT32_C(  1542138446),  INT32_C(           0),  INT32_C(  1771055484), -INT32_C(  2117789817),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(   196292697),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1827513364),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1115732311), -INT32_C(  1691595141) } },
    { UINT16_C(15499),
      {  INT32_C(  1097746794), -INT32_C(   852658759), -INT32_C(   428719623), -INT32_C(   662958418), -INT32_C(   290690100),  INT32_C(   276601386), -INT32_C(  2040596646), -INT32_C(    54335086),
        -INT32_C(   348245710), -INT32_C(  1615238234),  INT32_C(  1518676908),  INT32_C(    87163193), -INT32_C(  1745625235), -INT32_C(   744001671), -INT32_C(   111606169),  INT32_C(   586554351) },
      { -INT32_C(   217238452),  INT32_C(  1267910303),  INT32_C(   715528433), -INT32_C(  2043684839),  INT32_C(   807281335), -INT32_C(   117193326), -INT32_C(  1141744437), -INT32_C(   992089992),
        -INT32_C(  1145574884), -INT32_C(  1576580431),  INT32_C(  2077076834),  INT32_C(  1023605893), -INT32_C(  1318248417), -INT32_C(  1314229787),  INT32_C(  1164745933), -INT32_C(  1593161339) },
      { -INT32_C(   428719623),  INT32_C(   715528433),  INT32_C(           0), -INT32_C(  2043684839),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   992089992),
         INT32_C(           0),  INT32_C(           0),  INT32_C(    87163193),  INT32_C(  1023605893), -INT32_C(   111606169),  INT32_C(  1164745933),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(49459),
      {  INT32_C(  1678500957),  INT32_C(  1410428295), -INT32_C(   347040023),  INT32_C(  1091268563), -INT32_C(   894242784),  INT32_C(   224886689), -INT32_C(   816386875), -INT32_C(   359626099),
        -INT32_C(   179397522), -INT32_C(   230072567), -INT32_C(   908223754),  INT32_C(   705357833),  INT32_C(  2062859481),  INT32_C(    25713468), -INT32_C(   707731897), -INT32_C(   675323800) },
      {  INT32_C(   114036476),  INT32_C(  1677203053), -INT32_C(  1188178256), -INT32_C(  1746716738),  INT32_C(   806541556),  INT32_C(  2066848307), -INT32_C(   514850440),  INT32_C(  1589120865),
        -INT32_C(  1956346851),  INT32_C(  1257135258), -INT32_C(   251389134),  INT32_C(  1200154451), -INT32_C(   210265409), -INT32_C(  1385256908),  INT32_C(   177127081), -INT32_C(   345487667) },
      { -INT32_C(   347040023), -INT32_C(  1188178256),  INT32_C(           0),  INT32_C(           0), -INT32_C(   816386875), -INT32_C(   514850440),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   908223754),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   675323800), -INT32_C(   345487667) } },
    { UINT16_C(52426),
      {  INT32_C(  1680434550), -INT32_C(  1283433553), -INT32_C(   711208116), -INT32_C(  1854973414), -INT32_C(  1153850291), -INT32_C(   545660335),  INT32_C(  1428506604), -INT32_C(  1474171086),
         INT32_C(    67914324), -INT32_C(   222851930),  INT32_C(  2043105887), -INT32_C(    99993683),  INT32_C(   750142427),  INT32_C(   302722853), -INT32_C(  1469632394),  INT32_C(  1951434783) },
      {  INT32_C(  2037931475),  INT32_C(  1231761385),  INT32_C(   801256322),  INT32_C(  1160367466),  INT32_C(   913432336), -INT32_C(  2058847217), -INT32_C(   886132820),  INT32_C(   171933239),
        -INT32_C(   981223461),  INT32_C(  1762586599), -INT32_C(  1936142302), -INT32_C(  1361985123), -INT32_C(  1327218015),  INT32_C(  1781869758),  INT32_C(   305488859), -INT32_C(  1122142750) },
      {  INT32_C(           0),  INT32_C(   801256322),  INT32_C(           0),  INT32_C(  1160367466),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1474171086),  INT32_C(   171933239),
         INT32_C(           0),  INT32_C(           0), -INT32_C(    99993683), -INT32_C(  1361985123),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1951434783), -INT32_C(  1122142750) } },
    { UINT16_C(41004),
      { -INT32_C(  1869671550),  INT32_C(   341946748),  INT32_C(   248970813), -INT32_C(  1873774676),  INT32_C(  1555828263), -INT32_C(  1363110024), -INT32_C(   953965910), -INT32_C(   496545953),
        -INT32_C(   546113693), -INT32_C(   420162648),  INT32_C(  2129971922),  INT32_C(  1745831233), -INT32_C(   859518125),  INT32_C(   226133091), -INT32_C(  2032886490),  INT32_C(  1332231148) },
      { -INT32_C(   617620942), -INT32_C(  2134826066),  INT32_C(   788444653),  INT32_C(  1318456826),  INT32_C(  1008360153),  INT32_C(    88708319),  INT32_C(   495656241), -INT32_C(  1955728552),
         INT32_C(  2087099598), -INT32_C(  1392760897), -INT32_C(   673514788), -INT32_C(   517640184), -INT32_C(  1457700918),  INT32_C(    78603987), -INT32_C(   618579325), -INT32_C(    60387794) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1873774676),  INT32_C(  1318456826),  INT32_C(           0),  INT32_C(   495656241),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   618579325),  INT32_C(           0), -INT32_C(    60387794) } },
    { UINT16_C(52266),
      {  INT32_C(  1962142072),  INT32_C(  1869533333), -INT32_C(   857770329),  INT32_C(  1946921559),  INT32_C(    47898195),  INT32_C(    54353378), -INT32_C(  1617794247), -INT32_C(   546522009),
         INT32_C(   995319718), -INT32_C(   693386961), -INT32_C(  1885173192), -INT32_C(  2012959436),  INT32_C(  1837817483),  INT32_C(  1970390844), -INT32_C(  1726741710),  INT32_C(  1702396095) },
      {  INT32_C(   262261728), -INT32_C(   974762867),  INT32_C(   190089430), -INT32_C(  1047308234),  INT32_C(  1932467511),  INT32_C(   401121509),  INT32_C(  1655831715),  INT32_C(  1573398909),
        -INT32_C(  2106824203), -INT32_C(  1958260043),  INT32_C(   295082971),  INT32_C(   735193588),  INT32_C(   748552519),  INT32_C(  1145276065), -INT32_C(     5770110),  INT32_C(   324824862) },
      {  INT32_C(           0),  INT32_C(   190089430),  INT32_C(           0), -INT32_C(  1047308234),  INT32_C(           0),  INT32_C(  1655831715),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0), -INT32_C(  2012959436),  INT32_C(   735193588),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1702396095),  INT32_C(   324824862) } },
    { UINT16_C(51672),
      { -INT32_C(   602174059), -INT32_C(  1351092712), -INT32_C(   623351033),  INT32_C(   886775702), -INT32_C(  1883603637),  INT32_C(  1753431489), -INT32_C(  1713921732),  INT32_C(  1231204276),
         INT32_C(  1428520252),  INT32_C(  2063900020), -INT32_C(  1621763064),  INT32_C(  1205023228),  INT32_C(  1876332206),  INT32_C(   131619531), -INT32_C(  1331646469), -INT32_C(  1678179745) },
      { -INT32_C(   185589888), -INT32_C(   999295812), -INT32_C(   849099311), -INT32_C(  1542178826), -INT32_C(  1877742651),  INT32_C(  1083697989), -INT32_C(    84920165), -INT32_C(  1164580294),
        -INT32_C(   978418167),  INT32_C(  1267277434), -INT32_C(   652677661), -INT32_C(   377672412),  INT32_C(  1568247832),  INT32_C(   396169340), -INT32_C(  2096001464), -INT32_C(  2126666120) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1542178826), -INT32_C(  1713921732),  INT32_C(           0),  INT32_C(  1231204276), -INT32_C(  1164580294),
        -INT32_C(  1621763064),  INT32_C(           0),  INT32_C(           0), -INT32_C(   377672412),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1678179745), -INT32_C(  2126666120) } },
    { UINT16_C(60460),
      { -INT32_C(   821385402),  INT32_C(   196931058),  INT32_C(  1127801030),  INT32_C(  1121145033),  INT32_C(  1246973869),  INT32_C(  2010684262),  INT32_C(  1545490462),  INT32_C(   390613713),
        -INT32_C(   454601999),  INT32_C(    99590975),  INT32_C(  1296574340),  INT32_C(   613423991), -INT32_C(   781261973), -INT32_C(  1656141954),  INT32_C(  1777952663), -INT32_C(  1551875663) },
      { -INT32_C(   762878061), -INT32_C(  1881704949),  INT32_C(   349970333), -INT32_C(  1523028934), -INT32_C(   831019441), -INT32_C(  2056535827), -INT32_C(   655465433),  INT32_C(   947613349),
        -INT32_C(   519437610),  INT32_C(   359719288),  INT32_C(   975784960),  INT32_C(   148922809), -INT32_C(   170502392),  INT32_C(  1048199447),  INT32_C(  1242982565), -INT32_C(  1383951657) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1121145033), -INT32_C(  1523028934),  INT32_C(           0), -INT32_C(   655465433),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(   613423991),  INT32_C(   148922809),  INT32_C(           0),  INT32_C(  1242982565), -INT32_C(  1551875663), -INT32_C(  1383951657) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi32(test_vec[i].b);
    simde__m512i r = simde_mm512_maskz_unpackhi_epi32(test_vec[i].k, a, b);
    simde_test_x86_assert_equal_i32x16(r, simde_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_unpackhi_epi64 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 5674973837736279222),  INT64_C( 1704843706748687788), -INT64_C( 2673855454977685508),  INT64_C( 7390821164380328012),
        -INT64_C( 5519861701786044300), -INT64_C( 8880554723753151143), -INT64_C( 1373376201226907275),  INT64_C( 3598748551275955674) },
      { -INT64_C( 8104881503415520309),  INT64_C( 3417945947432930147), -INT64_C(  605674648809090292),  INT64_C( 8604534647075985787),
        -INT64_C( 4401197301858979547),  INT64_C( 3207437185291396506),  INT64_C( 8899798961286072686), -INT64_C( 7139017439677512502) },
      {  INT64_C( 1704843706748687788),  INT64_C( 3417945947432930147),  INT64_C( 7390821164380328012),  INT64_C( 8604534647075985787),
        -INT64_C( 8880554723753151143),  INT64_C( 3207437185291396506),  INT64_C( 3598748551275955674), -INT64_C( 7139017439677512502) } },
    { { -INT64_C( 6993840614198987057),  INT64_C( 2762244475499748360),  INT64_C( 7546683454867451199),  INT64_C( 6578294010817703736),
         INT64_C( 3360944426133832193),  INT64_C( 5544964737207866902),  INT64_C( 1287825918278739417), -INT64_C( 8530650538818159179) },
      { -INT64_C( 5286887723346483380),  INT64_C( 9016900953806534857), -INT64_C( 6267087682875839204),  INT64_C( 2063926056140392082),
         INT64_C( 3388730460615947184),  INT64_C( 2254375981441977128), -INT64_C( 6525123138314551595), -INT64_C( 7508235156986044703) },
      {  INT64_C( 2762244475499748360),  INT64_C( 9016900953806534857),  INT64_C( 6578294010817703736),  INT64_C( 2063926056140392082),
         INT64_C( 5544964737207866902),  INT64_C( 2254375981441977128), -INT64_C( 8530650538818159179), -INT64_C( 7508235156986044703) } },
    { { -INT64_C( 5463729958601401194),  INT64_C( 3358500432824512889), -INT64_C( 6070735980921901304),  INT64_C( 8810495403391484103),
        -INT64_C( 7653970804518539898),  INT64_C( 1899300861932292308), -INT64_C( 8948099700948053413), -INT64_C( 2390154280872890762) },
      { -INT64_C( 4515780095567115612),  INT64_C( 7867024461783786753), -INT64_C( 3397574104711994469),  INT64_C( 2795044881987931026),
         INT64_C( 8428269494163847753), -INT64_C( 3954524210633120353), -INT64_C( 6321935581393735400), -INT64_C( 5355805335178023676) },
      {  INT64_C( 3358500432824512889),  INT64_C( 7867024461783786753),  INT64_C( 8810495403391484103),  INT64_C( 2795044881987931026),
         INT64_C( 1899300861932292308), -INT64_C( 3954524210633120353), -INT64_C( 2390154280872890762), -INT64_C( 5355805335178023676) } },
    { {  INT64_C( 7781906569962950934),  INT64_C( 6034324575617844679), -INT64_C( 2690891062124880405), -INT64_C( 6010321335520052362),
         INT64_C( 3158206844560114334),  INT64_C( 1533438474208735417), -INT64_C( 8400130205098308421),  INT64_C( 8591747521593593337) },
      {  INT64_C( 6923639730661847686), -INT64_C( 3537969947851082330), -INT64_C( 5456166221627982681),  INT64_C( 4533858979574756506),
         INT64_C( 2880401063509806323), -INT64_C( 8046561543942812302),  INT64_C( 1965984521606968538),  INT64_C(   17867119812119624) },
      {  INT64_C( 6034324575617844679), -INT64_C( 3537969947851082330), -INT64_C( 6010321335520052362),  INT64_C( 4533858979574756506),
         INT64_C( 1533438474208735417), -INT64_C( 8046561543942812302),  INT64_C( 8591747521593593337),  INT64_C(   17867119812119624) } },
    { {  INT64_C( 1796817116571723846), -INT64_C( 6964269516958341152), -INT64_C( 9060261440426629722), -INT64_C( 7920210737626885624),
         INT64_C( 2515339843687001898),  INT64_C( 2131122330677249327),  INT64_C( 5450911646426122218),  INT64_C( 1528821352979572825) },
      {  INT64_C(  687588226762117413), -INT64_C( 6564117700413655725),  INT64_C(  929550991278680920), -INT64_C( 2047191976181086315),
        -INT64_C( 7179347720035140742),  INT64_C( 4875232268415031882),  INT64_C( 8230545872734023800),  INT64_C( 5669620523642120716) },
      { -INT64_C( 6964269516958341152), -INT64_C( 6564117700413655725), -INT64_C( 7920210737626885624), -INT64_C( 2047191976181086315),
         INT64_C( 2131122330677249327),  INT64_C( 4875232268415031882),  INT64_C( 1528821352979572825),  INT64_C( 5669620523642120716) } },
    { {  INT64_C( 1795247188014725380),  INT64_C( 1551707346594254681),  INT64_C( 4072793568706586421),  INT64_C(  397637285874727010),
         INT64_C( 9119855377055426218), -INT64_C( 6323814244269445748),  INT64_C( 6419671222899670707),  INT64_C( 8484240513725045432) },
      { -INT64_C( 2676125203499773639), -INT64_C( 6969530513325121493), -INT64_C(  428522275801791810), -INT64_C( 1661421789014418256),
        -INT64_C( 8598740181978784125), -INT64_C( 6102819577427219005),  INT64_C( 1273231586017463407),  INT64_C( 6026518686306491914) },
      {  INT64_C( 1551707346594254681), -INT64_C( 6969530513325121493),  INT64_C(  397637285874727010), -INT64_C( 1661421789014418256),
        -INT64_C( 6323814244269445748), -INT64_C( 6102819577427219005),  INT64_C( 8484240513725045432),  INT64_C( 6026518686306491914) } },
    { { -INT64_C( 3584356179152410532),  INT64_C( 4906051486565099388), -INT64_C( 1366101536430197353),  INT64_C( 8224496603040749312),
        -INT64_C( 7923120276673706191), -INT64_C( 1446479750494385965), -INT64_C( 8910389985721130105),  INT64_C( 8578925485341605510) },
      {  INT64_C( 8676113926111639998), -INT64_C(  178157761981226169), -INT64_C( 2163206275760428047), -INT64_C( 8900366555989324099),
         INT64_C( 3882596483972280260),  INT64_C(  114375668867858979),  INT64_C( 8790870050429468099), -INT64_C( 3815965023055020865) },
      {  INT64_C( 4906051486565099388), -INT64_C(  178157761981226169),  INT64_C( 8224496603040749312), -INT64_C( 8900366555989324099),
        -INT64_C( 1446479750494385965),  INT64_C(  114375668867858979),  INT64_C( 8578925485341605510), -INT64_C( 3815965023055020865) } },
    { {  INT64_C( 5064883971355337511),  INT64_C( 3975118758575282968), -INT64_C( 3339045211638226573),  INT64_C( 5017679719933880016),
         INT64_C(  229020489171523268), -INT64_C(  762702069681615424), -INT64_C( 2028082101023817737),  INT64_C( 6221543192684802548) },
      {  INT64_C( 1522616162351690034), -INT64_C( 7282123080597942920), -INT64_C( 2924385492342205069), -INT64_C( 5505579620542896977),
        -INT64_C( 5089926017148922127), -INT64_C( 3783407034755209920),  INT64_C( 5724845474327962296), -INT64_C( 8352723087043503136) },
      {  INT64_C( 3975118758575282968), -INT64_C( 7282123080597942920),  INT64_C( 5017679719933880016), -INT64_C( 5505579620542896977),
        -INT64_C(  762702069681615424), -INT64_C( 3783407034755209920),  INT64_C( 6221543192684802548), -INT64_C( 8352723087043503136) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi64(test_vec[i].b);
    simde__m512i r = simde_mm512_unpackhi_epi64(a, b);
    simde_test_x86_assert_equal_i64x8(r, simde_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_mask_unpackhi_epi64 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const simde__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 7069803644757198776), -INT64_C( 4615162892095733676), -INT64_C(  592481269516063786), -INT64_C( 1951860547106282793),
        -INT64_C( 4776033971357979552),  INT64_C( 5002361671084263902), -INT64_C( 4358438887271442656),  INT64_C( 2173316391371875352) },
      UINT8_C(151),
      { -INT64_C(  709390414091619285),  INT64_C( 1573069356133635006),  INT64_C(  606233209243590810),  INT64_C(  128886772494413063),
         INT64_C( 5007284282810091865),  INT64_C( 2853255272115790405),  INT64_C( 7429215763538876973),  INT64_C( 2190546868698210446) },
      {  INT64_C( 7491494037575544876), -INT64_C( 1605388421232398332), -INT64_C( 6997445859961327587),  INT64_C(  925324662345530868),
        -INT64_C( 6655275033583256551),  INT64_C( 3336302082892533414),  INT64_C( 5678071947195074077),  INT64_C( 5907279296854478606) },
      {  INT64_C( 1573069356133635006), -INT64_C( 1605388421232398332),  INT64_C(  128886772494413063), -INT64_C( 1951860547106282793),
         INT64_C( 2853255272115790405),  INT64_C( 5002361671084263902), -INT64_C( 4358438887271442656),  INT64_C( 5907279296854478606) } },
    { {  INT64_C( 1485397804821887067), -INT64_C( 2594421111010111041),  INT64_C( 5310541361681377801), -INT64_C(  851970217608758620),
        -INT64_C( 6094342348518540311), -INT64_C( 5694949449454066851),  INT64_C( 8766475294402150436), -INT64_C(  500319502935400576) },
      UINT8_C( 61),
      { -INT64_C( 6780779749143129604), -INT64_C(  581535033743376305), -INT64_C( 3219152390828831732),  INT64_C( 8600428855462947994),
        -INT64_C( 6301129233037032402), -INT64_C( 2750750709133486620), -INT64_C( 9191284002811764566),  INT64_C( 6799862801798878315) },
      {  INT64_C( 2629214055428676731), -INT64_C(  768756215722218067),  INT64_C( 7771248126770479220), -INT64_C( 1592112153495505320),
        -INT64_C( 5229179405259165430), -INT64_C( 5231346292924190938),  INT64_C( 6192989337208027527), -INT64_C(  380336530459311520) },
      { -INT64_C(  581535033743376305), -INT64_C( 2594421111010111041),  INT64_C( 8600428855462947994), -INT64_C( 1592112153495505320),
        -INT64_C( 2750750709133486620), -INT64_C( 5231346292924190938),  INT64_C( 8766475294402150436), -INT64_C(  500319502935400576) } },
    { { -INT64_C( 3046769058687715850), -INT64_C(  712503947079194334),  INT64_C( 8645343478219091111), -INT64_C( 7810088800309880655),
        -INT64_C( 5334713237796298637), -INT64_C( 7811151688720826829),  INT64_C( 3958718521792730723), -INT64_C( 6468731249084647927) },
      UINT8_C( 84),
      { -INT64_C(  122831116445431603), -INT64_C( 7038706715525976393), -INT64_C( 1013211252603499851), -INT64_C( 6991614324525197775),
         INT64_C( 3770077475849560211),  INT64_C( 8004712632414256368), -INT64_C( 4257513995879761346),  INT64_C( 7875214485939097308) },
      { -INT64_C(  940388094190813119), -INT64_C( 4073364536899791537), -INT64_C( 1219725549161624134),  INT64_C( 2233757779520879544),
         INT64_C( 2439491752443101158),  INT64_C( 9166411585423033573),  INT64_C( 6373121923404920175), -INT64_C( 2785500613664174757) },
      { -INT64_C( 3046769058687715850), -INT64_C(  712503947079194334), -INT64_C( 6991614324525197775), -INT64_C( 7810088800309880655),
         INT64_C( 8004712632414256368), -INT64_C( 7811151688720826829),  INT64_C( 7875214485939097308), -INT64_C( 6468731249084647927) } },
    { { -INT64_C( 1613524498428840792), -INT64_C( 2943791305104767519),  INT64_C( 3349213380578243459), -INT64_C(  674204802864421672),
        -INT64_C( 1954830502096509198),  INT64_C( 5192843631966959569),  INT64_C(  911105038106651042), -INT64_C( 4932148811623101623) },
      UINT8_C(226),
      { -INT64_C( 3973360625745808899), -INT64_C( 8239945592549338029), -INT64_C( 3859920510259160257), -INT64_C( 8141295836826761818),
         INT64_C(  529019795796519679), -INT64_C(  488165872602782584),  INT64_C( 8093966657315011530),  INT64_C( 3725204992451339920) },
      {  INT64_C( 5132725608845101952), -INT64_C( 4633873112000936449), -INT64_C( 1485453012652994776),  INT64_C(  858709281465037954),
        -INT64_C( 4102644257746080502),  INT64_C( 6373210507922310079), -INT64_C( 2790634902909175040),  INT64_C( 2104723718819304352) },
      { -INT64_C( 1613524498428840792), -INT64_C( 4633873112000936449),  INT64_C( 3349213380578243459), -INT64_C(  674204802864421672),
        -INT64_C( 1954830502096509198),  INT64_C( 6373210507922310079),  INT64_C( 3725204992451339920),  INT64_C( 2104723718819304352) } },
    { { -INT64_C( 5928997778794056664),  INT64_C( 7530813971257964538),  INT64_C( 1783153292805852302), -INT64_C(  897772364895137957),
         INT64_C( 7100377162194317805),  INT64_C( 5521993398515264210), -INT64_C( 4076337739640773412),  INT64_C( 1584632822595170368) },
      UINT8_C( 48),
      {  INT64_C( 7227990955471204833), -INT64_C( 8182076068310447992), -INT64_C( 1887076183540250465), -INT64_C( 5123086641714570505),
         INT64_C(  284433193656156180),  INT64_C( 2360746577768354570),  INT64_C( 8714605466890518230),  INT64_C( 3812602887049051481) },
      { -INT64_C( 6191124988141888106), -INT64_C( 7824543943707611192), -INT64_C( 1127994923696646048),  INT64_C( 3810056961364428683),
         INT64_C( 2265972895572730663),  INT64_C( 6006062423011250441),  INT64_C( 1257221751015995370), -INT64_C( 4258543097014807789) },
      { -INT64_C( 5928997778794056664),  INT64_C( 7530813971257964538),  INT64_C( 1783153292805852302), -INT64_C(  897772364895137957),
         INT64_C( 2360746577768354570),  INT64_C( 6006062423011250441), -INT64_C( 4076337739640773412),  INT64_C( 1584632822595170368) } },
    { { -INT64_C( 8018660757471482816), -INT64_C( 2623996657626898353),  INT64_C( 7511669188945889447), -INT64_C( 2755077514440606878),
        -INT64_C( 7187137741158527189), -INT64_C( 1146883776033417144), -INT64_C( 6631450114154204612),  INT64_C( 4451638983883100758) },
      UINT8_C(181),
      { -INT64_C( 3915423366592073928),  INT64_C( 7722832208524927870), -INT64_C( 3976589726295671265), -INT64_C( 4751834691028730125),
         INT64_C( 3386928682753790214),  INT64_C( 3679808221115060908),  INT64_C(  111320118149889548), -INT64_C( 9043112067202390724) },
      {  INT64_C( 1538746084056735866), -INT64_C( 3092621876665812020),  INT64_C( 8906468869616943498), -INT64_C( 2844659382430600057),
        -INT64_C( 4261426724832330147), -INT64_C( 1335396026422093863), -INT64_C( 5173475773515493766), -INT64_C( 8338061567367286508) },
      {  INT64_C( 7722832208524927870), -INT64_C( 2623996657626898353), -INT64_C( 4751834691028730125), -INT64_C( 2755077514440606878),
         INT64_C( 3679808221115060908), -INT64_C( 1335396026422093863), -INT64_C( 6631450114154204612), -INT64_C( 8338061567367286508) } },
    { { -INT64_C( 8918201848107479680),  INT64_C( 1375178542360748229), -INT64_C( 3560939127798195285),  INT64_C( 1309291379511284081),
         INT64_C( 8216191539417349846),  INT64_C( 8382237705590418343), -INT64_C( 8010525915789385909),  INT64_C( 8420424661542573110) },
      UINT8_C(217),
      {  INT64_C( 2797852451683038566), -INT64_C( 8823286804088155246),  INT64_C( 8849349408075322035), -INT64_C( 2695249943931086748),
        -INT64_C( 6367005662361805439), -INT64_C( 6279857409327787283),  INT64_C( 1650870196476411400),  INT64_C( 1150508682674766249) },
      { -INT64_C( 4231361816949948789), -INT64_C( 5881996165970841408),  INT64_C( 6501517465147514737),  INT64_C(  623026708646661896),
         INT64_C( 2236863357412454509), -INT64_C( 8614485485033282529), -INT64_C( 6855070614581177902),  INT64_C( 3493346093460509941) },
      { -INT64_C( 8823286804088155246),  INT64_C( 1375178542360748229), -INT64_C( 3560939127798195285),  INT64_C(  623026708646661896),
        -INT64_C( 6279857409327787283),  INT64_C( 8382237705590418343),  INT64_C( 1150508682674766249),  INT64_C( 3493346093460509941) } },
    { {  INT64_C( 3497748872912249356),  INT64_C( 7296790800698269604), -INT64_C( 3948876406478803207),  INT64_C( 5768466005105535001),
        -INT64_C( 6406294586842709033),  INT64_C( 8042484957435792805),  INT64_C( 7147267899479065747), -INT64_C(  635204218856117349) },
      UINT8_C(208),
      {  INT64_C(  348038273292296170),  INT64_C( 3248759616153717980),  INT64_C( 1781131515178417385), -INT64_C( 5821130368954558951),
         INT64_C( 7065219299381480208),  INT64_C( 6382999729258858175), -INT64_C(  127741380596069515), -INT64_C( 5461286742673493055) },
      { -INT64_C( 8750555892677929817), -INT64_C( 2299788635459980723),  INT64_C( 8534106568873118093),  INT64_C( 1731689519062570160),
        -INT64_C( 2840693246252432570), -INT64_C( 8029168252790406841), -INT64_C( 9196506772007709645), -INT64_C( 3039968717940958139) },
      {  INT64_C( 3497748872912249356),  INT64_C( 7296790800698269604), -INT64_C( 3948876406478803207),  INT64_C( 5768466005105535001),
         INT64_C( 6382999729258858175),  INT64_C( 8042484957435792805), -INT64_C( 5461286742673493055), -INT64_C( 3039968717940958139) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i src = simde_mm512_loadu_epi64(test_vec[i].src);
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi64(test_vec[i].b);
    simde__m512i r = simde_mm512_mask_unpackhi_epi64(src, test_vec[i].k, a, b);
    simde_test_x86_assert_equal_i64x8(r, simde_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_maskz_unpackhi_epi64 (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(153),
      { -INT64_C( 8112061316095643358),  INT64_C( 6935063002730107714), -INT64_C( 8212087054093038310), -INT64_C( 2607746116477873009),
         INT64_C( 3144832950630399454), -INT64_C( 5137730184067198396),  INT64_C( 4921647922986935914), -INT64_C( 8765048687182594872) },
      { -INT64_C( 7702493116525469689), -INT64_C( 9014066700485133679),  INT64_C( 1259476586114192513),  INT64_C( 5114663378355418315),
         INT64_C( 7382897138880386447),  INT64_C( 8711789030675930443), -INT64_C( 8941289340094502767), -INT64_C(  314705051237516003) },
      {  INT64_C( 6935063002730107714),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 5114663378355418315),
        -INT64_C( 5137730184067198396),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  314705051237516003) } },
    { UINT8_C(135),
      { -INT64_C( 5875859059498033961), -INT64_C( 1917092265710921897), -INT64_C( 6480081549246251315), -INT64_C( 1761174571067478505),
         INT64_C(  281901227435879401),  INT64_C( 8934645429188398939), -INT64_C( 3293675335527130245),  INT64_C( 6045918023986601201) },
      { -INT64_C( 3096937802851485867), -INT64_C( 7768449762043438073),  INT64_C( 1446924468239245447), -INT64_C( 5937155702654636322),
        -INT64_C( 6394068444791232269), -INT64_C( 6611974618886688368), -INT64_C( 3070053105838688980),  INT64_C( 7633521611942754793) },
      { -INT64_C( 1917092265710921897), -INT64_C( 7768449762043438073), -INT64_C( 1761174571067478505),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 7633521611942754793) } },
    { UINT8_C(224),
      {  INT64_C( 5540033489920907166),  INT64_C( 4556251704725245059),  INT64_C( 7626860197896948725), -INT64_C( 6359002367070889431),
        -INT64_C( 1546745235567460320), -INT64_C(  761050062070877593), -INT64_C( 2658604054839187776),  INT64_C( 7171837558290971095) },
      { -INT64_C( 4391248934469730115), -INT64_C(  140064324810700776), -INT64_C( 8603624947397496053), -INT64_C( 8185309017209743934),
         INT64_C( 5012719882948773489), -INT64_C(  862882530471279158),  INT64_C( 4582777096366109792),  INT64_C( 7205948465858413216) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C(  862882530471279158),  INT64_C( 7171837558290971095),  INT64_C( 7205948465858413216) } },
    { UINT8_C(177),
      {  INT64_C( 5061802720489189811),  INT64_C(  165828678393022336), -INT64_C(  377810925094215978), -INT64_C( 1847230989765627135),
        -INT64_C( 2410944925150790782), -INT64_C( 5390499936422085918),  INT64_C( 8099610054905737334),  INT64_C( 2874557315520681163) },
      { -INT64_C( 4904149681540743243),  INT64_C( 5278813408176356271),  INT64_C( 8983307628280486349), -INT64_C(  630400309860317069),
         INT64_C( 5706899244913896349),  INT64_C(  491489641323306421),  INT64_C( 1974245321395571000), -INT64_C( 2671070819006554187) },
      {  INT64_C(  165828678393022336),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C( 5390499936422085918),  INT64_C(  491489641323306421),  INT64_C(                   0), -INT64_C( 2671070819006554187) } },
    { UINT8_C( 30),
      { -INT64_C( 6942565444498861832), -INT64_C( 8340139830770323222),  INT64_C( 7981591915461356065), -INT64_C( 6767355895724917629),
        -INT64_C( 3749461226211191851), -INT64_C( 3315751378116882324),  INT64_C(  233243253361619007),  INT64_C( 4732736115633153889) },
      { -INT64_C( 1473080054470065438), -INT64_C( 5268272069701115295), -INT64_C( 3960141369283656825), -INT64_C(  840903116746062937),
        -INT64_C( 2292537272392225083),  INT64_C( 7481840039537838503), -INT64_C( 4626753622671206248), -INT64_C( 1693030055064461445) },
      {  INT64_C(                   0), -INT64_C( 5268272069701115295), -INT64_C( 6767355895724917629), -INT64_C(  840903116746062937),
        -INT64_C( 3315751378116882324),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 20),
      { -INT64_C( 2007351597602755528),  INT64_C( 7985773938750014054),  INT64_C( 4507335351563900057), -INT64_C( 3147372254168278505),
        -INT64_C( 4560890390205241739),  INT64_C( 4485249486166915513),  INT64_C( 2414171336890893475), -INT64_C( 6960398803146907145) },
      { -INT64_C( 1607182324373407336),  INT64_C( 2197746152424123805), -INT64_C( 6708426467054661701), -INT64_C( 8030870343051295038),
        -INT64_C( 4796557763825959850), -INT64_C( 2775486076469499679),  INT64_C( 9195090110878505805), -INT64_C( 4299820068894879541) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 3147372254168278505),  INT64_C(                   0),
         INT64_C( 4485249486166915513),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(244),
      { -INT64_C( 2095632079641097515), -INT64_C( 7089832664592529712), -INT64_C(   15524902365195400), -INT64_C( 5534588735990232252),
        -INT64_C( 1794783307867863429),  INT64_C( 7659445327702071277), -INT64_C( 4426703642843175580), -INT64_C( 6247284131365912901) },
      { -INT64_C( 7100499098819753354), -INT64_C( 2348460646870504691), -INT64_C( 4937067789841936324),  INT64_C( 2511218222881405697),
        -INT64_C( 4973339936969148243),  INT64_C( 1957695701676279760), -INT64_C( 6724628912605389327),  INT64_C( 5951978209226654412) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 5534588735990232252),  INT64_C(                   0),
         INT64_C( 7659445327702071277),  INT64_C( 1957695701676279760), -INT64_C( 6247284131365912901),  INT64_C( 5951978209226654412) } },
    { UINT8_C(100),
      { -INT64_C(  990848356323315639),  INT64_C( 6095881269656193925),  INT64_C( 4169656188011907520),  INT64_C( 4069620111938311530),
        -INT64_C( 2486296676270633759), -INT64_C(  397131255520211342),  INT64_C( 8436624239100720037),  INT64_C(  388723695290217278) },
      {  INT64_C( 7261790460150090898),  INT64_C(  833741416274152285), -INT64_C( 4107569250562498946),  INT64_C( 2528587095903275758),
         INT64_C( 1566080278840911972), -INT64_C( 3345543044703434720),  INT64_C( 7145622119506409948), -INT64_C( 2767306435414581789) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4069620111938311530),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 3345543044703434720),  INT64_C(  388723695290217278),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi64(test_vec[i].b);
    simde__m512i r = simde_mm512_maskz_unpackhi_epi64(test_vec[i].k, a, b);
    simde_test_x86_assert_equal_i64x8(r, simde_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_simde_mm512_unpackhi_ps (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float32 a[16];
    const simde_float32 b[16];
    const simde_float32 r[16];
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(   501.25), SIMDE_FLOAT32_C(   354.44), SIMDE_FLOAT32_C(  -760.98), SIMDE_FLOAT32_C(   367.33),
        SIMDE_FLOAT32_C(   563.03), SIMDE_FLOAT32_C(   427.95), SIMDE_FLOAT32_C(   773.44), SIMDE_FLOAT32_C(   713.02),
        SIMDE_FLOAT32_C(   119.86), SIMDE_FLOAT32_C(   -12.94), SIMDE_FLOAT32_C(   867.61), SIMDE_FLOAT32_C(  -313.75),
        SIMDE_FLOAT32_C(   147.55), SIMDE_FLOAT32_C(  -416.41), SIMDE_FLOAT32_C(  -877.86), SIMDE_FLOAT32_C(  -808.32) },
      { SIMDE_FLOAT32_C(  -972.10), SIMDE_FLOAT32_C(   298.29), SIMDE_FLOAT32_C(   288.84), SIMDE_FLOAT32_C(   810.71),
        SIMDE_FLOAT32_C(   626.39), SIMDE_FLOAT32_C(  -563.97), SIMDE_FLOAT32_C(   -14.08), SIMDE_FLOAT32_C(  -444.31),
        SIMDE_FLOAT32_C(   614.59), SIMDE_FLOAT32_C(    98.84), SIMDE_FLOAT32_C(   586.04), SIMDE_FLOAT32_C(   326.60),
        SIMDE_FLOAT32_C(  -948.81), SIMDE_FLOAT32_C(  -681.79), SIMDE_FLOAT32_C(   949.12), SIMDE_FLOAT32_C(   552.44) },
      { SIMDE_FLOAT32_C(  -760.98), SIMDE_FLOAT32_C(   288.84), SIMDE_FLOAT32_C(   367.33), SIMDE_FLOAT32_C(   810.71),
        SIMDE_FLOAT32_C(   773.44), SIMDE_FLOAT32_C(   -14.08), SIMDE_FLOAT32_C(   713.02), SIMDE_FLOAT32_C(  -444.31),
        SIMDE_FLOAT32_C(   867.61), SIMDE_FLOAT32_C(   586.04), SIMDE_FLOAT32_C(  -313.75), SIMDE_FLOAT32_C(   326.60),
        SIMDE_FLOAT32_C(  -877.86), SIMDE_FLOAT32_C(   949.12), SIMDE_FLOAT32_C(  -808.32), SIMDE_FLOAT32_C(   552.44) } },
    { { SIMDE_FLOAT32_C(   672.66), SIMDE_FLOAT32_C(  -811.86), SIMDE_FLOAT32_C(   -80.22), SIMDE_FLOAT32_C(   235.69),
        SIMDE_FLOAT32_C(   616.10), SIMDE_FLOAT32_C(  -306.79), SIMDE_FLOAT32_C(   -51.30), SIMDE_FLOAT32_C(  -264.04),
        SIMDE_FLOAT32_C(   680.27), SIMDE_FLOAT32_C(  -183.69), SIMDE_FLOAT32_C(   422.21), SIMDE_FLOAT32_C(  -172.18),
        SIMDE_FLOAT32_C(   399.90), SIMDE_FLOAT32_C(   544.35), SIMDE_FLOAT32_C(    19.50), SIMDE_FLOAT32_C(   427.80) },
      { SIMDE_FLOAT32_C(  -157.35), SIMDE_FLOAT32_C(  -691.65), SIMDE_FLOAT32_C(   238.51), SIMDE_FLOAT32_C(  -530.97),
        SIMDE_FLOAT32_C(  -255.62), SIMDE_FLOAT32_C(  -775.57), SIMDE_FLOAT32_C(    24.72), SIMDE_FLOAT32_C(  -641.03),
        SIMDE_FLOAT32_C(   323.27), SIMDE_FLOAT32_C(  -389.24), SIMDE_FLOAT32_C(   685.56), SIMDE_FLOAT32_C(   374.47),
        SIMDE_FLOAT32_C(   -71.03), SIMDE_FLOAT32_C(   634.69), SIMDE_FLOAT32_C(   -73.09), SIMDE_FLOAT32_C(  -398.37) },
      { SIMDE_FLOAT32_C(   -80.22), SIMDE_FLOAT32_C(   238.51), SIMDE_FLOAT32_C(   235.69), SIMDE_FLOAT32_C(  -530.97),
        SIMDE_FLOAT32_C(   -51.30), SIMDE_FLOAT32_C(    24.72), SIMDE_FLOAT32_C(  -264.04), SIMDE_FLOAT32_C(  -641.03),
        SIMDE_FLOAT32_C(   422.21), SIMDE_FLOAT32_C(   685.56), SIMDE_FLOAT32_C(  -172.18), SIMDE_FLOAT32_C(   374.47),
        SIMDE_FLOAT32_C(    19.50), SIMDE_FLOAT32_C(   -73.09), SIMDE_FLOAT32_C(   427.80), SIMDE_FLOAT32_C(  -398.37) } },
    { { SIMDE_FLOAT32_C(   822.83), SIMDE_FLOAT32_C(   846.69), SIMDE_FLOAT32_C(   837.31), SIMDE_FLOAT32_C(   438.93),
        SIMDE_FLOAT32_C(  -460.10), SIMDE_FLOAT32_C(  -213.99), SIMDE_FLOAT32_C(  -825.11), SIMDE_FLOAT32_C(  -779.83),
        SIMDE_FLOAT32_C(   602.32), SIMDE_FLOAT32_C(   597.10), SIMDE_FLOAT32_C(    47.99), SIMDE_FLOAT32_C(     2.23),
        SIMDE_FLOAT32_C(   141.46), SIMDE_FLOAT32_C(  -932.50), SIMDE_FLOAT32_C(  -569.97), SIMDE_FLOAT32_C(   984.10) },
      { SIMDE_FLOAT32_C(  -624.16), SIMDE_FLOAT32_C(   668.54), SIMDE_FLOAT32_C(  -546.86), SIMDE_FLOAT32_C(   120.22),
        SIMDE_FLOAT32_C(   892.97), SIMDE_FLOAT32_C(   477.86), SIMDE_FLOAT32_C(   479.19), SIMDE_FLOAT32_C(   216.25),
        SIMDE_FLOAT32_C(  -911.39), SIMDE_FLOAT32_C(   164.75), SIMDE_FLOAT32_C(  -409.28), SIMDE_FLOAT32_C(    17.58),
        SIMDE_FLOAT32_C(  -200.56), SIMDE_FLOAT32_C(   517.63), SIMDE_FLOAT32_C(   619.21), SIMDE_FLOAT32_C(  -377.73) },
      { SIMDE_FLOAT32_C(   837.31), SIMDE_FLOAT32_C(  -546.86), SIMDE_FLOAT32_C(   438.93), SIMDE_FLOAT32_C(   120.22),
        SIMDE_FLOAT32_C(  -825.11), SIMDE_FLOAT32_C(   479.19), SIMDE_FLOAT32_C(  -779.83), SIMDE_FLOAT32_C(   216.25),
        SIMDE_FLOAT32_C(    47.99), SIMDE_FLOAT32_C(  -409.28), SIMDE_FLOAT32_C(     2.23), SIMDE_FLOAT32_C(    17.58),
        SIMDE_FLOAT32_C(  -569.97), SIMDE_FLOAT32_C(   619.21), SIMDE_FLOAT32_C(   984.10), SIMDE_FLOAT32_C(  -377.73) } },
    { { SIMDE_FLOAT32_C(   364.32), SIMDE_FLOAT32_C(   456.52), SIMDE_FLOAT32_C(  -938.80), SIMDE_FLOAT32_C(   904.22),
        SIMDE_FLOAT32_C(  -757.47), SIMDE_FLOAT32_C(  -763.91), SIMDE_FLOAT32_C(  -875.61), SIMDE_FLOAT32_C(   844.86),
        SIMDE_FLOAT32_C(   833.20), SIMDE_FLOAT32_C(   172.39), SIMDE_FLOAT32_C(  -152.92), SIMDE_FLOAT32_C(   -25.35),
        SIMDE_FLOAT32_C(   239.88), SIMDE_FLOAT32_C(   277.11), SIMDE_FLOAT32_C(   -41.24), SIMDE_FLOAT32_C(   615.73) },
      { SIMDE_FLOAT32_C(   -54.35), SIMDE_FLOAT32_C(   411.89), SIMDE_FLOAT32_C(  -264.06), SIMDE_FLOAT32_C(  -161.38),
        SIMDE_FLOAT32_C(  -110.25), SIMDE_FLOAT32_C(  -784.87), SIMDE_FLOAT32_C(  -945.13), SIMDE_FLOAT32_C(   -21.64),
        SIMDE_FLOAT32_C(   379.88), SIMDE_FLOAT32_C(  -354.42), SIMDE_FLOAT32_C(   995.95), SIMDE_FLOAT32_C(  -820.68),
        SIMDE_FLOAT32_C(  -836.79), SIMDE_FLOAT32_C(   615.15), SIMDE_FLOAT32_C(  -198.41), SIMDE_FLOAT32_C(   527.53) },
      { SIMDE_FLOAT32_C(  -938.80), SIMDE_FLOAT32_C(  -264.06), SIMDE_FLOAT32_C(   904.22), SIMDE_FLOAT32_C(  -161.38),
        SIMDE_FLOAT32_C(  -875.61), SIMDE_FLOAT32_C(  -945.13), SIMDE_FLOAT32_C(   844.86), SIMDE_FLOAT32_C(   -21.64),
        SIMDE_FLOAT32_C(  -152.92), SIMDE_FLOAT32_C(   995.95), SIMDE_FLOAT32_C(   -25.35), SIMDE_FLOAT32_C(  -820.68),
        SIMDE_FLOAT32_C(   -41.24), SIMDE_FLOAT32_C(  -198.41), SIMDE_FLOAT32_C(   615.73), SIMDE_FLOAT32_C(   527.53) } },
    { { SIMDE_FLOAT32_C(    71.67), SIMDE_FLOAT32_C(  -137.20), SIMDE_FLOAT32_C(   431.75), SIMDE_FLOAT32_C(   314.20),
        SIMDE_FLOAT32_C(    98.89), SIMDE_FLOAT32_C(   556.14), SIMDE_FLOAT32_C(   159.06), SIMDE_FLOAT32_C(   -67.91),
        SIMDE_FLOAT32_C(  -271.47), SIMDE_FLOAT32_C(  -993.86), SIMDE_FLOAT32_C(   906.74), SIMDE_FLOAT32_C(   968.41),
        SIMDE_FLOAT32_C(   283.25), SIMDE_FLOAT32_C(  -134.50), SIMDE_FLOAT32_C(   584.13), SIMDE_FLOAT32_C(  -771.10) },
      { SIMDE_FLOAT32_C(  -722.60), SIMDE_FLOAT32_C(  -679.92), SIMDE_FLOAT32_C(    67.52), SIMDE_FLOAT32_C(   167.15),
        SIMDE_FLOAT32_C(  -464.79), SIMDE_FLOAT32_C(   122.39), SIMDE_FLOAT32_C(  -854.49), SIMDE_FLOAT32_C(   915.09),
        SIMDE_FLOAT32_C(   767.97), SIMDE_FLOAT32_C(  -858.54), SIMDE_FLOAT32_C(  -905.59), SIMDE_FLOAT32_C(   931.19),
        SIMDE_FLOAT32_C(   756.61), SIMDE_FLOAT32_C(  -104.00), SIMDE_FLOAT32_C(   458.72), SIMDE_FLOAT32_C(  -171.72) },
      { SIMDE_FLOAT32_C(   431.75), SIMDE_FLOAT32_C(    67.52), SIMDE_FLOAT32_C(   314.20), SIMDE_FLOAT32_C(   167.15),
        SIMDE_FLOAT32_C(   159.06), SIMDE_FLOAT32_C(  -854.49), SIMDE_FLOAT32_C(   -67.91), SIMDE_FLOAT32_C(   915.09),
        SIMDE_FLOAT32_C(   906.74), SIMDE_FLOAT32_C(  -905.59), SIMDE_FLOAT32_C(   968.41), SIMDE_FLOAT32_C(   931.19),
        SIMDE_FLOAT32_C(   584.13), SIMDE_FLOAT32_C(   458.72), SIMDE_FLOAT32_C(  -771.10), SIMDE_FLOAT32_C(  -171.72) } },
    { { SIMDE_FLOAT32_C(   758.80), SIMDE_FLOAT32_C(  -109.54), SIMDE_FLOAT32_C(  -857.52), SIMDE_FLOAT32_C(  -142.31),
        SIMDE_FLOAT32_C(  -553.40), SIMDE_FLOAT32_C(   301.54), SIMDE_FLOAT32_C(   789.78), SIMDE_FLOAT32_C(   175.13),
        SIMDE_FLOAT32_C(   307.68), SIMDE_FLOAT32_C(   696.52), SIMDE_FLOAT32_C(   143.54), SIMDE_FLOAT32_C(  -409.06),
        SIMDE_FLOAT32_C(  -437.98), SIMDE_FLOAT32_C(  -272.33), SIMDE_FLOAT32_C(  -180.16), SIMDE_FLOAT32_C(  -160.59) },
      { SIMDE_FLOAT32_C(    47.75), SIMDE_FLOAT32_C(   887.36), SIMDE_FLOAT32_C(  -993.44), SIMDE_FLOAT32_C(   582.95),
        SIMDE_FLOAT32_C(     9.75), SIMDE_FLOAT32_C(  -847.93), SIMDE_FLOAT32_C(   498.04), SIMDE_FLOAT32_C(  -222.27),
        SIMDE_FLOAT32_C(  -706.47), SIMDE_FLOAT32_C(   592.44), SIMDE_FLOAT32_C(  -291.09), SIMDE_FLOAT32_C(  -949.86),
        SIMDE_FLOAT32_C(  -511.56), SIMDE_FLOAT32_C(  -832.37), SIMDE_FLOAT32_C(  -121.58), SIMDE_FLOAT32_C(  -752.76) },
      { SIMDE_FLOAT32_C(  -857.52), SIMDE_FLOAT32_C(  -993.44), SIMDE_FLOAT32_C(  -142.31), SIMDE_FLOAT32_C(   582.95),
        SIMDE_FLOAT32_C(   789.78), SIMDE_FLOAT32_C(   498.04), SIMDE_FLOAT32_C(   175.13), SIMDE_FLOAT32_C(  -222.27),
        SIMDE_FLOAT32_C(   143.54), SIMDE_FLOAT32_C(  -291.09), SIMDE_FLOAT32_C(  -409.06), SIMDE_FLOAT32_C(  -949.86),
        SIMDE_FLOAT32_C(  -180.16), SIMDE_FLOAT32_C(  -121.58), SIMDE_FLOAT32_C(  -160.59), SIMDE_FLOAT32_C(  -752.76) } },
    { { SIMDE_FLOAT32_C(    58.09), SIMDE_FLOAT32_C(    20.91), SIMDE_FLOAT32_C(   104.93), SIMDE_FLOAT32_C(   504.70),
        SIMDE_FLOAT32_C(  -677.55), SIMDE_FLOAT32_C(  -105.30), SIMDE_FLOAT32_C(  -320.17), SIMDE_FLOAT32_C(   630.13),
        SIMDE_FLOAT32_C(  -408.78), SIMDE_FLOAT32_C(   823.36), SIMDE_FLOAT32_C(  -778.93), SIMDE_FLOAT32_C(   153.24),
        SIMDE_FLOAT32_C(  -448.97), SIMDE_FLOAT32_C(    40.91), SIMDE_FLOAT32_C(   992.65), SIMDE_FLOAT32_C(   598.78) },
      { SIMDE_FLOAT32_C(   -71.73), SIMDE_FLOAT32_C(   999.22), SIMDE_FLOAT32_C(   181.73), SIMDE_FLOAT32_C(   938.02),
        SIMDE_FLOAT32_C(  -848.71), SIMDE_FLOAT32_C(  -320.23), SIMDE_FLOAT32_C(  -284.25), SIMDE_FLOAT32_C(  -555.18),
        SIMDE_FLOAT32_C(  -727.79), SIMDE_FLOAT32_C(   424.66), SIMDE_FLOAT32_C(  -505.04), SIMDE_FLOAT32_C(  -239.35),
        SIMDE_FLOAT32_C(   592.29), SIMDE_FLOAT32_C(   373.38), SIMDE_FLOAT32_C(     7.89), SIMDE_FLOAT32_C(  -349.61) },
      { SIMDE_FLOAT32_C(   104.93), SIMDE_FLOAT32_C(   181.73), SIMDE_FLOAT32_C(   504.70), SIMDE_FLOAT32_C(   938.02),
        SIMDE_FLOAT32_C(  -320.17), SIMDE_FLOAT32_C(  -284.25), SIMDE_FLOAT32_C(   630.13), SIMDE_FLOAT32_C(  -555.18),
        SIMDE_FLOAT32_C(  -778.93), SIMDE_FLOAT32_C(  -505.04), SIMDE_FLOAT32_C(   153.24), SIMDE_FLOAT32_C(  -239.35),
        SIMDE_FLOAT32_C(   992.65), SIMDE_FLOAT32_C(     7.89), SIMDE_FLOAT32_C(   598.78), SIMDE_FLOAT32_C(  -349.61) } },
    { { SIMDE_FLOAT32_C(  -605.71), SIMDE_FLOAT32_C(  -887.18), SIMDE_FLOAT32_C(  -844.92), SIMDE_FLOAT32_C(  -283.26),
        SIMDE_FLOAT32_C(     7.52), SIMDE_FLOAT32_C(  -165.09), SIMDE_FLOAT32_C(  -653.13), SIMDE_FLOAT32_C(   598.74),
        SIMDE_FLOAT32_C(  -341.73), SIMDE_FLOAT32_C(  -432.06), SIMDE_FLOAT32_C(  -248.02), SIMDE_FLOAT32_C(   209.30),
        SIMDE_FLOAT32_C(   608.85), SIMDE_FLOAT32_C(  -255.36), SIMDE_FLOAT32_C(  -191.92), SIMDE_FLOAT32_C(  -462.88) },
      { SIMDE_FLOAT32_C(  -256.15), SIMDE_FLOAT32_C(   989.81), SIMDE_FLOAT32_C(  -524.85), SIMDE_FLOAT32_C(  -104.86),
        SIMDE_FLOAT32_C(  -330.42), SIMDE_FLOAT32_C(   190.90), SIMDE_FLOAT32_C(   339.96), SIMDE_FLOAT32_C(   -58.21),
        SIMDE_FLOAT32_C(  -384.44), SIMDE_FLOAT32_C(   834.93), SIMDE_FLOAT32_C(   702.44), SIMDE_FLOAT32_C(  -792.14),
        SIMDE_FLOAT32_C(   208.31), SIMDE_FLOAT32_C(  -289.67), SIMDE_FLOAT32_C(  -141.76), SIMDE_FLOAT32_C(   602.60) },
      { SIMDE_FLOAT32_C(  -844.92), SIMDE_FLOAT32_C(  -524.85), SIMDE_FLOAT32_C(  -283.26), SIMDE_FLOAT32_C(  -104.86),
        SIMDE_FLOAT32_C(  -653.13), SIMDE_FLOAT32_C(   339.96), SIMDE_FLOAT32_C(   598.74), SIMDE_FLOAT32_C(   -58.21),
        SIMDE_FLOAT32_C(  -248.02), SIMDE_FLOAT32_C(   702.44), SIMDE_FLOAT32_C(   209.30), SIMDE_FLOAT32_C(  -792.14),
        SIMDE_FLOAT32_C(  -191.92), SIMDE_FLOAT32_C(  -141.76), SIMDE_FLOAT32_C(  -462.88), SIMDE_FLOAT32_C(   602.60) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 b = simde_mm512_loadu_ps(test_vec[i].b);
    simde__m512 r = simde_mm512_unpackhi_ps(a, b);
    simde_test_x86_assert_equal_f32x16(r, simde_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_simde_mm512_mask_unpackhi_ps (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float32 src[16];
    const simde__mmask8 k;
    const simde_float32 a[16];
    const simde_float32 b[16];
    const simde_float32 r[16];
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(  -291.23), SIMDE_FLOAT32_C(  -139.21), SIMDE_FLOAT32_C(  -441.81), SIMDE_FLOAT32_C(   845.97),
        SIMDE_FLOAT32_C(   226.47), SIMDE_FLOAT32_C(   261.10), SIMDE_FLOAT32_C(  -120.23), SIMDE_FLOAT32_C(  -359.39),
        SIMDE_FLOAT32_C(   746.86), SIMDE_FLOAT32_C(   235.27), SIMDE_FLOAT32_C(  -137.88), SIMDE_FLOAT32_C(  -417.19),
        SIMDE_FLOAT32_C(  -188.37), SIMDE_FLOAT32_C(  -129.27), SIMDE_FLOAT32_C(   402.91), SIMDE_FLOAT32_C(   826.61) },
      UINT8_C(205),
      { SIMDE_FLOAT32_C(   236.89), SIMDE_FLOAT32_C(  -388.23), SIMDE_FLOAT32_C(   941.34), SIMDE_FLOAT32_C(  -208.36),
        SIMDE_FLOAT32_C(  -307.76), SIMDE_FLOAT32_C(  -934.29), SIMDE_FLOAT32_C(  -828.65), SIMDE_FLOAT32_C(  -292.89),
        SIMDE_FLOAT32_C(   823.73), SIMDE_FLOAT32_C(  -702.66), SIMDE_FLOAT32_C(  -158.41), SIMDE_FLOAT32_C(  -940.13),
        SIMDE_FLOAT32_C(   882.43), SIMDE_FLOAT32_C(   -63.81), SIMDE_FLOAT32_C(  -231.36), SIMDE_FLOAT32_C(  -256.78) },
      { SIMDE_FLOAT32_C(   494.38), SIMDE_FLOAT32_C(  -385.40), SIMDE_FLOAT32_C(   969.69), SIMDE_FLOAT32_C(  -244.52),
        SIMDE_FLOAT32_C(   494.37), SIMDE_FLOAT32_C(  -389.71), SIMDE_FLOAT32_C(  -497.66), SIMDE_FLOAT32_C(  -270.36),
        SIMDE_FLOAT32_C(   472.42), SIMDE_FLOAT32_C(    85.15), SIMDE_FLOAT32_C(   541.28), SIMDE_FLOAT32_C(  -656.85),
        SIMDE_FLOAT32_C(  -511.95), SIMDE_FLOAT32_C(   367.89), SIMDE_FLOAT32_C(   -73.71), SIMDE_FLOAT32_C(   724.95) },
      { SIMDE_FLOAT32_C(   941.34), SIMDE_FLOAT32_C(  -139.21), SIMDE_FLOAT32_C(  -208.36), SIMDE_FLOAT32_C(  -244.52),
        SIMDE_FLOAT32_C(   226.47), SIMDE_FLOAT32_C(   261.10), SIMDE_FLOAT32_C(  -292.89), SIMDE_FLOAT32_C(  -270.36),
        SIMDE_FLOAT32_C(   746.86), SIMDE_FLOAT32_C(   235.27), SIMDE_FLOAT32_C(  -137.88), SIMDE_FLOAT32_C(  -417.19),
        SIMDE_FLOAT32_C(  -188.37), SIMDE_FLOAT32_C(  -129.27), SIMDE_FLOAT32_C(   402.91), SIMDE_FLOAT32_C(   826.61) } },
    { { SIMDE_FLOAT32_C(   979.66), SIMDE_FLOAT32_C(  -132.37), SIMDE_FLOAT32_C(  -483.42), SIMDE_FLOAT32_C(  -328.09),
        SIMDE_FLOAT32_C(   -66.66), SIMDE_FLOAT32_C(  -312.07), SIMDE_FLOAT32_C(   379.02), SIMDE_FLOAT32_C(  -242.93),
        SIMDE_FLOAT32_C(   -14.73), SIMDE_FLOAT32_C(  -779.39), SIMDE_FLOAT32_C(  -183.06), SIMDE_FLOAT32_C(  -132.30),
        SIMDE_FLOAT32_C(   156.80), SIMDE_FLOAT32_C(   585.58), SIMDE_FLOAT32_C(   610.92), SIMDE_FLOAT32_C(  -348.82) },
      UINT8_C(248),
      { SIMDE_FLOAT32_C(   580.60), SIMDE_FLOAT32_C(   406.66), SIMDE_FLOAT32_C(   694.56), SIMDE_FLOAT32_C(  -809.10),
        SIMDE_FLOAT32_C(   908.99), SIMDE_FLOAT32_C(  -575.80), SIMDE_FLOAT32_C(   663.32), SIMDE_FLOAT32_C(    -5.86),
        SIMDE_FLOAT32_C(   965.47), SIMDE_FLOAT32_C(  -993.54), SIMDE_FLOAT32_C(   482.19), SIMDE_FLOAT32_C(   333.36),
        SIMDE_FLOAT32_C(   -67.24), SIMDE_FLOAT32_C(   207.14), SIMDE_FLOAT32_C(   313.03), SIMDE_FLOAT32_C(   800.39) },
      { SIMDE_FLOAT32_C(   723.72), SIMDE_FLOAT32_C(   984.93), SIMDE_FLOAT32_C(  -266.27), SIMDE_FLOAT32_C(  -588.35),
        SIMDE_FLOAT32_C(   363.95), SIMDE_FLOAT32_C(   490.80), SIMDE_FLOAT32_C(   396.93), SIMDE_FLOAT32_C(   584.57),
        SIMDE_FLOAT32_C(  -692.25), SIMDE_FLOAT32_C(  -735.37), SIMDE_FLOAT32_C(  -258.63), SIMDE_FLOAT32_C(   893.33),
        SIMDE_FLOAT32_C(   875.54), SIMDE_FLOAT32_C(   392.54), SIMDE_FLOAT32_C(  -906.49), SIMDE_FLOAT32_C(   456.15) },
      { SIMDE_FLOAT32_C(   979.66), SIMDE_FLOAT32_C(  -132.37), SIMDE_FLOAT32_C(  -483.42), SIMDE_FLOAT32_C(  -588.35),
        SIMDE_FLOAT32_C(   663.32), SIMDE_FLOAT32_C(   396.93), SIMDE_FLOAT32_C(    -5.86), SIMDE_FLOAT32_C(   584.57),
        SIMDE_FLOAT32_C(   -14.73), SIMDE_FLOAT32_C(  -779.39), SIMDE_FLOAT32_C(  -183.06), SIMDE_FLOAT32_C(  -132.30),
        SIMDE_FLOAT32_C(   156.80), SIMDE_FLOAT32_C(   585.58), SIMDE_FLOAT32_C(   610.92), SIMDE_FLOAT32_C(  -348.82) } },
    { { SIMDE_FLOAT32_C(  -200.80), SIMDE_FLOAT32_C(   788.07), SIMDE_FLOAT32_C(   647.05), SIMDE_FLOAT32_C(  -291.81),
        SIMDE_FLOAT32_C(  -787.73), SIMDE_FLOAT32_C(   310.36), SIMDE_FLOAT32_C(   702.33), SIMDE_FLOAT32_C(  -822.26),
        SIMDE_FLOAT32_C(   316.83), SIMDE_FLOAT32_C(   184.52), SIMDE_FLOAT32_C(   511.11), SIMDE_FLOAT32_C(  -750.41),
        SIMDE_FLOAT32_C(  -608.34), SIMDE_FLOAT32_C(  -175.86), SIMDE_FLOAT32_C(  -950.03), SIMDE_FLOAT32_C(  -884.62) },
      UINT8_C(108),
      { SIMDE_FLOAT32_C(  -216.30), SIMDE_FLOAT32_C(  -472.96), SIMDE_FLOAT32_C(  -826.97), SIMDE_FLOAT32_C(  -725.50),
        SIMDE_FLOAT32_C(   923.97), SIMDE_FLOAT32_C(   757.59), SIMDE_FLOAT32_C(  -417.75), SIMDE_FLOAT32_C(  -811.41),
        SIMDE_FLOAT32_C(  -501.04), SIMDE_FLOAT32_C(  -524.42), SIMDE_FLOAT32_C(  -935.86), SIMDE_FLOAT32_C(   891.50),
        SIMDE_FLOAT32_C(  -430.91), SIMDE_FLOAT32_C(   520.29), SIMDE_FLOAT32_C(  -309.30), SIMDE_FLOAT32_C(  -642.84) },
      { SIMDE_FLOAT32_C(   167.33), SIMDE_FLOAT32_C(   398.89), SIMDE_FLOAT32_C(  -430.57), SIMDE_FLOAT32_C(  -522.30),
        SIMDE_FLOAT32_C(   101.22), SIMDE_FLOAT32_C(  -252.83), SIMDE_FLOAT32_C(   794.53), SIMDE_FLOAT32_C(  -714.26),
        SIMDE_FLOAT32_C(  -741.72), SIMDE_FLOAT32_C(  -955.89), SIMDE_FLOAT32_C(  -322.60), SIMDE_FLOAT32_C(    82.42),
        SIMDE_FLOAT32_C(  -905.92), SIMDE_FLOAT32_C(  -207.21), SIMDE_FLOAT32_C(   891.49), SIMDE_FLOAT32_C(  -122.21) },
      { SIMDE_FLOAT32_C(  -200.80), SIMDE_FLOAT32_C(   788.07), SIMDE_FLOAT32_C(  -725.50), SIMDE_FLOAT32_C(  -522.30),
        SIMDE_FLOAT32_C(  -787.73), SIMDE_FLOAT32_C(   794.53), SIMDE_FLOAT32_C(  -811.41), SIMDE_FLOAT32_C(  -822.26),
        SIMDE_FLOAT32_C(   316.83), SIMDE_FLOAT32_C(   184.52), SIMDE_FLOAT32_C(   511.11), SIMDE_FLOAT32_C(  -750.41),
        SIMDE_FLOAT32_C(  -608.34), SIMDE_FLOAT32_C(  -175.86), SIMDE_FLOAT32_C(  -950.03), SIMDE_FLOAT32_C(  -884.62) } },
    { { SIMDE_FLOAT32_C(   319.83), SIMDE_FLOAT32_C(  -935.48), SIMDE_FLOAT32_C(   152.29), SIMDE_FLOAT32_C(   243.79),
        SIMDE_FLOAT32_C(   822.11), SIMDE_FLOAT32_C(   734.54), SIMDE_FLOAT32_C(   432.39), SIMDE_FLOAT32_C(  -678.93),
        SIMDE_FLOAT32_C(  -789.89), SIMDE_FLOAT32_C(   496.52), SIMDE_FLOAT32_C(  -787.43), SIMDE_FLOAT32_C(  -220.80),
        SIMDE_FLOAT32_C(    16.81), SIMDE_FLOAT32_C(   -96.74), SIMDE_FLOAT32_C(   136.36), SIMDE_FLOAT32_C(  -815.85) },
      UINT8_C(254),
      { SIMDE_FLOAT32_C(   705.79), SIMDE_FLOAT32_C(  -338.16), SIMDE_FLOAT32_C(   403.37), SIMDE_FLOAT32_C(  -547.04),
        SIMDE_FLOAT32_C(  -543.63), SIMDE_FLOAT32_C(   689.11), SIMDE_FLOAT32_C(  -288.75), SIMDE_FLOAT32_C(  -499.52),
        SIMDE_FLOAT32_C(  -633.49), SIMDE_FLOAT32_C(   793.66), SIMDE_FLOAT32_C(  -405.44), SIMDE_FLOAT32_C(   159.30),
        SIMDE_FLOAT32_C(   685.15), SIMDE_FLOAT32_C(   472.35), SIMDE_FLOAT32_C(  -520.87), SIMDE_FLOAT32_C(   749.67) },
      { SIMDE_FLOAT32_C(  -375.36), SIMDE_FLOAT32_C(   722.92), SIMDE_FLOAT32_C(   571.78), SIMDE_FLOAT32_C(  -640.83),
        SIMDE_FLOAT32_C(   155.31), SIMDE_FLOAT32_C(   892.85), SIMDE_FLOAT32_C(  -430.72), SIMDE_FLOAT32_C(  -348.16),
        SIMDE_FLOAT32_C(  -894.59), SIMDE_FLOAT32_C(   348.48), SIMDE_FLOAT32_C(   668.65), SIMDE_FLOAT32_C(     8.68),
        SIMDE_FLOAT32_C(  -515.16), SIMDE_FLOAT32_C(   852.79), SIMDE_FLOAT32_C(   310.83), SIMDE_FLOAT32_C(  -809.37) },
      { SIMDE_FLOAT32_C(   319.83), SIMDE_FLOAT32_C(   571.78), SIMDE_FLOAT32_C(  -547.04), SIMDE_FLOAT32_C(  -640.83),
        SIMDE_FLOAT32_C(  -288.75), SIMDE_FLOAT32_C(  -430.72), SIMDE_FLOAT32_C(  -499.52), SIMDE_FLOAT32_C(  -348.16),
        SIMDE_FLOAT32_C(  -789.89), SIMDE_FLOAT32_C(   496.52), SIMDE_FLOAT32_C(  -787.43), SIMDE_FLOAT32_C(  -220.80),
        SIMDE_FLOAT32_C(    16.81), SIMDE_FLOAT32_C(   -96.74), SIMDE_FLOAT32_C(   136.36), SIMDE_FLOAT32_C(  -815.85) } },
    { { SIMDE_FLOAT32_C(  -485.36), SIMDE_FLOAT32_C(  -285.81), SIMDE_FLOAT32_C(  -356.40), SIMDE_FLOAT32_C(   -29.00),
        SIMDE_FLOAT32_C(  -596.70), SIMDE_FLOAT32_C(   354.84), SIMDE_FLOAT32_C(   471.48), SIMDE_FLOAT32_C(  -230.19),
        SIMDE_FLOAT32_C(   148.51), SIMDE_FLOAT32_C(  -933.96), SIMDE_FLOAT32_C(   929.12), SIMDE_FLOAT32_C(  -166.34),
        SIMDE_FLOAT32_C(   538.39), SIMDE_FLOAT32_C(  -591.75), SIMDE_FLOAT32_C(  -416.67), SIMDE_FLOAT32_C(  -836.97) },
      UINT8_C(155),
      { SIMDE_FLOAT32_C(  -844.89), SIMDE_FLOAT32_C(  -477.80), SIMDE_FLOAT32_C(   286.48), SIMDE_FLOAT32_C(  -952.04),
        SIMDE_FLOAT32_C(    91.48), SIMDE_FLOAT32_C(   938.32), SIMDE_FLOAT32_C(  -846.63), SIMDE_FLOAT32_C(  -560.04),
        SIMDE_FLOAT32_C(   606.97), SIMDE_FLOAT32_C(   162.05), SIMDE_FLOAT32_C(   -75.19), SIMDE_FLOAT32_C(   459.76),
        SIMDE_FLOAT32_C(  -527.12), SIMDE_FLOAT32_C(   115.44), SIMDE_FLOAT32_C(   974.40), SIMDE_FLOAT32_C(   187.07) },
      { SIMDE_FLOAT32_C(   759.04), SIMDE_FLOAT32_C(   -54.60), SIMDE_FLOAT32_C(   590.37), SIMDE_FLOAT32_C(   113.88),
        SIMDE_FLOAT32_C(  -583.12), SIMDE_FLOAT32_C(  -639.81), SIMDE_FLOAT32_C(  -737.62), SIMDE_FLOAT32_C(  -517.08),
        SIMDE_FLOAT32_C(  -710.70), SIMDE_FLOAT32_C(    96.05), SIMDE_FLOAT32_C(  -978.69), SIMDE_FLOAT32_C(  -302.45),
        SIMDE_FLOAT32_C(   679.38), SIMDE_FLOAT32_C(  -815.66), SIMDE_FLOAT32_C(  -171.28), SIMDE_FLOAT32_C(   834.49) },
      { SIMDE_FLOAT32_C(   286.48), SIMDE_FLOAT32_C(   590.37), SIMDE_FLOAT32_C(  -356.40), SIMDE_FLOAT32_C(   113.88),
        SIMDE_FLOAT32_C(  -846.63), SIMDE_FLOAT32_C(   354.84), SIMDE_FLOAT32_C(   471.48), SIMDE_FLOAT32_C(  -517.08),
        SIMDE_FLOAT32_C(   148.51), SIMDE_FLOAT32_C(  -933.96), SIMDE_FLOAT32_C(   929.12), SIMDE_FLOAT32_C(  -166.34),
        SIMDE_FLOAT32_C(   538.39), SIMDE_FLOAT32_C(  -591.75), SIMDE_FLOAT32_C(  -416.67), SIMDE_FLOAT32_C(  -836.97) } },
    { { SIMDE_FLOAT32_C(  -293.47), SIMDE_FLOAT32_C(  -884.79), SIMDE_FLOAT32_C(   882.45), SIMDE_FLOAT32_C(   798.01),
        SIMDE_FLOAT32_C(  -946.47), SIMDE_FLOAT32_C(  -964.17), SIMDE_FLOAT32_C(  -762.02), SIMDE_FLOAT32_C(   660.49),
        SIMDE_FLOAT32_C(   197.88), SIMDE_FLOAT32_C(   162.78), SIMDE_FLOAT32_C(   120.25), SIMDE_FLOAT32_C(   670.75),
        SIMDE_FLOAT32_C(  -721.78), SIMDE_FLOAT32_C(    94.65), SIMDE_FLOAT32_C(  -142.18), SIMDE_FLOAT32_C(  -962.74) },
      UINT8_C(128),
      { SIMDE_FLOAT32_C(  -551.81), SIMDE_FLOAT32_C(   151.14), SIMDE_FLOAT32_C(  -543.07), SIMDE_FLOAT32_C(  -191.62),
        SIMDE_FLOAT32_C(   413.52), SIMDE_FLOAT32_C(   -60.15), SIMDE_FLOAT32_C(    97.68), SIMDE_FLOAT32_C(  -490.43),
        SIMDE_FLOAT32_C(   -38.84), SIMDE_FLOAT32_C(   795.24), SIMDE_FLOAT32_C(  -811.05), SIMDE_FLOAT32_C(   145.50),
        SIMDE_FLOAT32_C(  -376.04), SIMDE_FLOAT32_C(  -976.56), SIMDE_FLOAT32_C(   852.03), SIMDE_FLOAT32_C(  -260.84) },
      { SIMDE_FLOAT32_C(   905.89), SIMDE_FLOAT32_C(   650.04), SIMDE_FLOAT32_C(  -207.31), SIMDE_FLOAT32_C(   941.71),
        SIMDE_FLOAT32_C(   888.02), SIMDE_FLOAT32_C(  -546.82), SIMDE_FLOAT32_C(   139.59), SIMDE_FLOAT32_C(    50.80),
        SIMDE_FLOAT32_C(   573.44), SIMDE_FLOAT32_C(  -189.66), SIMDE_FLOAT32_C(   329.03), SIMDE_FLOAT32_C(  -331.91),
        SIMDE_FLOAT32_C(   668.17), SIMDE_FLOAT32_C(   366.29), SIMDE_FLOAT32_C(  -291.87), SIMDE_FLOAT32_C(  -883.64) },
      { SIMDE_FLOAT32_C(  -293.47), SIMDE_FLOAT32_C(  -884.79), SIMDE_FLOAT32_C(   882.45), SIMDE_FLOAT32_C(   798.01),
        SIMDE_FLOAT32_C(  -946.47), SIMDE_FLOAT32_C(  -964.17), SIMDE_FLOAT32_C(  -762.02), SIMDE_FLOAT32_C(    50.80),
        SIMDE_FLOAT32_C(   197.88), SIMDE_FLOAT32_C(   162.78), SIMDE_FLOAT32_C(   120.25), SIMDE_FLOAT32_C(   670.75),
        SIMDE_FLOAT32_C(  -721.78), SIMDE_FLOAT32_C(    94.65), SIMDE_FLOAT32_C(  -142.18), SIMDE_FLOAT32_C(  -962.74) } },
    { { SIMDE_FLOAT32_C(  -482.57), SIMDE_FLOAT32_C(   165.06), SIMDE_FLOAT32_C(   -75.26), SIMDE_FLOAT32_C(   930.95),
        SIMDE_FLOAT32_C(  -895.09), SIMDE_FLOAT32_C(  -977.57), SIMDE_FLOAT32_C(  -559.48), SIMDE_FLOAT32_C(    66.06),
        SIMDE_FLOAT32_C(   817.66), SIMDE_FLOAT32_C(  -370.53), SIMDE_FLOAT32_C(  -788.44), SIMDE_FLOAT32_C(  -558.38),
        SIMDE_FLOAT32_C(  -347.09), SIMDE_FLOAT32_C(  -936.41), SIMDE_FLOAT32_C(   180.79), SIMDE_FLOAT32_C(  -441.20) },
      UINT8_C(109),
      { SIMDE_FLOAT32_C(   973.48), SIMDE_FLOAT32_C(  -499.49), SIMDE_FLOAT32_C(   601.65), SIMDE_FLOAT32_C(  -573.34),
        SIMDE_FLOAT32_C(   640.10), SIMDE_FLOAT32_C(  -347.55), SIMDE_FLOAT32_C(  -999.90), SIMDE_FLOAT32_C(  -549.55),
        SIMDE_FLOAT32_C(   981.48), SIMDE_FLOAT32_C(  -331.82), SIMDE_FLOAT32_C(  -881.39), SIMDE_FLOAT32_C(   347.77),
        SIMDE_FLOAT32_C(   376.32), SIMDE_FLOAT32_C(  -765.03), SIMDE_FLOAT32_C(   865.20), SIMDE_FLOAT32_C(  -458.63) },
      { SIMDE_FLOAT32_C(   159.72), SIMDE_FLOAT32_C(   796.15), SIMDE_FLOAT32_C(  -353.72), SIMDE_FLOAT32_C(   182.14),
        SIMDE_FLOAT32_C(  -763.33), SIMDE_FLOAT32_C(   712.34), SIMDE_FLOAT32_C(    -0.19), SIMDE_FLOAT32_C(  -133.87),
        SIMDE_FLOAT32_C(   923.90), SIMDE_FLOAT32_C(   441.43), SIMDE_FLOAT32_C(   519.04), SIMDE_FLOAT32_C(   987.49),
        SIMDE_FLOAT32_C(  -377.78), SIMDE_FLOAT32_C(  -922.16), SIMDE_FLOAT32_C(   701.12), SIMDE_FLOAT32_C(  -404.31) },
      { SIMDE_FLOAT32_C(   601.65), SIMDE_FLOAT32_C(   165.06), SIMDE_FLOAT32_C(  -573.34), SIMDE_FLOAT32_C(   182.14),
        SIMDE_FLOAT32_C(  -895.09), SIMDE_FLOAT32_C(    -0.19), SIMDE_FLOAT32_C(  -549.55), SIMDE_FLOAT32_C(    66.06),
        SIMDE_FLOAT32_C(   817.66), SIMDE_FLOAT32_C(  -370.53), SIMDE_FLOAT32_C(  -788.44), SIMDE_FLOAT32_C(  -558.38),
        SIMDE_FLOAT32_C(  -347.09), SIMDE_FLOAT32_C(  -936.41), SIMDE_FLOAT32_C(   180.79), SIMDE_FLOAT32_C(  -441.20) } },
    { { SIMDE_FLOAT32_C(  -421.65), SIMDE_FLOAT32_C(   302.77), SIMDE_FLOAT32_C(    22.35), SIMDE_FLOAT32_C(  -781.55),
        SIMDE_FLOAT32_C(   955.22), SIMDE_FLOAT32_C(    22.45), SIMDE_FLOAT32_C(  -331.11), SIMDE_FLOAT32_C(   936.70),
        SIMDE_FLOAT32_C(   690.63), SIMDE_FLOAT32_C(  -212.49), SIMDE_FLOAT32_C(   284.46), SIMDE_FLOAT32_C(    66.95),
        SIMDE_FLOAT32_C(    22.48), SIMDE_FLOAT32_C(   149.66), SIMDE_FLOAT32_C(   608.33), SIMDE_FLOAT32_C(  -817.80) },
      UINT8_C(168),
      { SIMDE_FLOAT32_C(  -745.39), SIMDE_FLOAT32_C(   364.34), SIMDE_FLOAT32_C(   182.47), SIMDE_FLOAT32_C(   966.95),
        SIMDE_FLOAT32_C(  -635.85), SIMDE_FLOAT32_C(  -951.39), SIMDE_FLOAT32_C(   890.85), SIMDE_FLOAT32_C(   805.58),
        SIMDE_FLOAT32_C(   567.65), SIMDE_FLOAT32_C(   878.34), SIMDE_FLOAT32_C(  -572.21), SIMDE_FLOAT32_C(   645.49),
        SIMDE_FLOAT32_C(   579.46), SIMDE_FLOAT32_C(    23.49), SIMDE_FLOAT32_C(  -776.17), SIMDE_FLOAT32_C(  -117.78) },
      { SIMDE_FLOAT32_C(  -954.16), SIMDE_FLOAT32_C(  -557.72), SIMDE_FLOAT32_C(  -162.56), SIMDE_FLOAT32_C(    68.29),
        SIMDE_FLOAT32_C(   111.17), SIMDE_FLOAT32_C(  -225.86), SIMDE_FLOAT32_C(  -241.07), SIMDE_FLOAT32_C(   898.68),
        SIMDE_FLOAT32_C(  -941.40), SIMDE_FLOAT32_C(   825.88), SIMDE_FLOAT32_C(   -78.84), SIMDE_FLOAT32_C(   208.26),
        SIMDE_FLOAT32_C(   434.20), SIMDE_FLOAT32_C(   103.36), SIMDE_FLOAT32_C(  -845.93), SIMDE_FLOAT32_C(   688.81) },
      { SIMDE_FLOAT32_C(  -421.65), SIMDE_FLOAT32_C(   302.77), SIMDE_FLOAT32_C(    22.35), SIMDE_FLOAT32_C(    68.29),
        SIMDE_FLOAT32_C(   955.22), SIMDE_FLOAT32_C(  -241.07), SIMDE_FLOAT32_C(  -331.11), SIMDE_FLOAT32_C(   898.68),
        SIMDE_FLOAT32_C(   690.63), SIMDE_FLOAT32_C(  -212.49), SIMDE_FLOAT32_C(   284.46), SIMDE_FLOAT32_C(    66.95),
        SIMDE_FLOAT32_C(    22.48), SIMDE_FLOAT32_C(   149.66), SIMDE_FLOAT32_C(   608.33), SIMDE_FLOAT32_C(  -817.80) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 src = simde_mm512_loadu_ps(test_vec[i].src);
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 b = simde_mm512_loadu_ps(test_vec[i].b);
    simde__m512 r = simde_mm512_mask_unpackhi_ps(src, test_vec[i].k, a, b);
    simde_test_x86_assert_equal_f32x16(r, simde_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_simde_mm512_maskz_unpackhi_ps (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde__mmask16 k;
    const simde_float32 a[16];
    const simde_float32 b[16];
    const simde_float32 r[16];
  } test_vec[] = {
    { UINT16_C(52453),
      { SIMDE_FLOAT32_C(  -753.67), SIMDE_FLOAT32_C(  -929.72), SIMDE_FLOAT32_C(   958.55), SIMDE_FLOAT32_C(  -635.03),
        SIMDE_FLOAT32_C(   813.44), SIMDE_FLOAT32_C(  -572.90), SIMDE_FLOAT32_C(  -350.63), SIMDE_FLOAT32_C(  -428.42),
        SIMDE_FLOAT32_C(  -919.26), SIMDE_FLOAT32_C(  -171.52), SIMDE_FLOAT32_C(   935.83), SIMDE_FLOAT32_C(   125.81),
        SIMDE_FLOAT32_C(   572.34), SIMDE_FLOAT32_C(  -746.26), SIMDE_FLOAT32_C(   391.38), SIMDE_FLOAT32_C(  -448.91) },
      { SIMDE_FLOAT32_C(   293.95), SIMDE_FLOAT32_C(  -500.41), SIMDE_FLOAT32_C(  -983.39), SIMDE_FLOAT32_C(   919.70),
        SIMDE_FLOAT32_C(  -282.02), SIMDE_FLOAT32_C(   527.32), SIMDE_FLOAT32_C(   354.40), SIMDE_FLOAT32_C(   182.05),
        SIMDE_FLOAT32_C(  -816.83), SIMDE_FLOAT32_C(  -162.74), SIMDE_FLOAT32_C(   223.66), SIMDE_FLOAT32_C(   371.74),
        SIMDE_FLOAT32_C(  -962.96), SIMDE_FLOAT32_C(  -571.14), SIMDE_FLOAT32_C(  -839.06), SIMDE_FLOAT32_C(  -716.63) },
      { SIMDE_FLOAT32_C(   958.55), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -635.03), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   354.40), SIMDE_FLOAT32_C(  -428.42), SIMDE_FLOAT32_C(   182.05),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   125.81), SIMDE_FLOAT32_C(   371.74),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -448.91), SIMDE_FLOAT32_C(  -716.63) } },
    { UINT16_C(19907),
      { SIMDE_FLOAT32_C(  -351.66), SIMDE_FLOAT32_C(  -687.42), SIMDE_FLOAT32_C(  -453.41), SIMDE_FLOAT32_C(   297.70),
        SIMDE_FLOAT32_C(  -115.84), SIMDE_FLOAT32_C(  -372.67), SIMDE_FLOAT32_C(  -873.82), SIMDE_FLOAT32_C(  -180.01),
        SIMDE_FLOAT32_C(   753.15), SIMDE_FLOAT32_C(   698.52), SIMDE_FLOAT32_C(    73.73), SIMDE_FLOAT32_C(   144.52),
        SIMDE_FLOAT32_C(  -750.38), SIMDE_FLOAT32_C(  -632.32), SIMDE_FLOAT32_C(   644.11), SIMDE_FLOAT32_C(  -733.77) },
      { SIMDE_FLOAT32_C(  -712.62), SIMDE_FLOAT32_C(  -637.91), SIMDE_FLOAT32_C(   793.55), SIMDE_FLOAT32_C(   641.79),
        SIMDE_FLOAT32_C(   544.14), SIMDE_FLOAT32_C(   976.71), SIMDE_FLOAT32_C(  -520.96), SIMDE_FLOAT32_C(  -232.20),
        SIMDE_FLOAT32_C(   348.45), SIMDE_FLOAT32_C(  -483.91), SIMDE_FLOAT32_C(   196.65), SIMDE_FLOAT32_C(   509.39),
        SIMDE_FLOAT32_C(  -200.54), SIMDE_FLOAT32_C(   695.80), SIMDE_FLOAT32_C(   628.88), SIMDE_FLOAT32_C(   447.80) },
      { SIMDE_FLOAT32_C(  -453.41), SIMDE_FLOAT32_C(   793.55), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -180.01), SIMDE_FLOAT32_C(  -232.20),
        SIMDE_FLOAT32_C(    73.73), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   144.52), SIMDE_FLOAT32_C(   509.39),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -733.77), SIMDE_FLOAT32_C(     0.00) } },
    { UINT16_C(11533),
      { SIMDE_FLOAT32_C(  -254.50), SIMDE_FLOAT32_C(  -107.46), SIMDE_FLOAT32_C(  -197.20), SIMDE_FLOAT32_C(  -128.31),
        SIMDE_FLOAT32_C(   712.53), SIMDE_FLOAT32_C(  -444.05), SIMDE_FLOAT32_C(  -429.79), SIMDE_FLOAT32_C(  -213.73),
        SIMDE_FLOAT32_C(   700.47), SIMDE_FLOAT32_C(  -180.18), SIMDE_FLOAT32_C(   153.95), SIMDE_FLOAT32_C(   344.58),
        SIMDE_FLOAT32_C(    86.05), SIMDE_FLOAT32_C(   441.33), SIMDE_FLOAT32_C(   706.67), SIMDE_FLOAT32_C(  -120.40) },
      { SIMDE_FLOAT32_C(    83.12), SIMDE_FLOAT32_C(   250.82), SIMDE_FLOAT32_C(  -143.69), SIMDE_FLOAT32_C(   562.16),
        SIMDE_FLOAT32_C(  -981.39), SIMDE_FLOAT32_C(  -795.24), SIMDE_FLOAT32_C(  -921.75), SIMDE_FLOAT32_C(   215.27),
        SIMDE_FLOAT32_C(   714.16), SIMDE_FLOAT32_C(  -122.29), SIMDE_FLOAT32_C(   -88.94), SIMDE_FLOAT32_C(   343.04),
        SIMDE_FLOAT32_C(  -674.50), SIMDE_FLOAT32_C(   -80.56), SIMDE_FLOAT32_C(   518.51), SIMDE_FLOAT32_C(    71.00) },
      { SIMDE_FLOAT32_C(  -197.20), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -128.31), SIMDE_FLOAT32_C(   562.16),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(   153.95), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   344.58), SIMDE_FLOAT32_C(   343.04),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   518.51), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00) } },
    { UINT16_C(60827),
      { SIMDE_FLOAT32_C(   942.69), SIMDE_FLOAT32_C(   524.51), SIMDE_FLOAT32_C(  -122.74), SIMDE_FLOAT32_C(  -487.10),
        SIMDE_FLOAT32_C(  -689.22), SIMDE_FLOAT32_C(  -422.26), SIMDE_FLOAT32_C(   332.72), SIMDE_FLOAT32_C(   464.73),
        SIMDE_FLOAT32_C(   922.32), SIMDE_FLOAT32_C(  -581.23), SIMDE_FLOAT32_C(   -93.94), SIMDE_FLOAT32_C(   629.00),
        SIMDE_FLOAT32_C(   298.37), SIMDE_FLOAT32_C(   989.17), SIMDE_FLOAT32_C(  -120.19), SIMDE_FLOAT32_C(  -845.32) },
      { SIMDE_FLOAT32_C(   551.34), SIMDE_FLOAT32_C(  -101.58), SIMDE_FLOAT32_C(  -640.56), SIMDE_FLOAT32_C(   629.58),
        SIMDE_FLOAT32_C(  -886.31), SIMDE_FLOAT32_C(  -926.40), SIMDE_FLOAT32_C(  -492.71), SIMDE_FLOAT32_C(    24.75),
        SIMDE_FLOAT32_C(   416.64), SIMDE_FLOAT32_C(  -167.21), SIMDE_FLOAT32_C(   944.19), SIMDE_FLOAT32_C(   -64.86),
        SIMDE_FLOAT32_C(   903.79), SIMDE_FLOAT32_C(   756.16), SIMDE_FLOAT32_C(   256.46), SIMDE_FLOAT32_C(   846.48) },
      { SIMDE_FLOAT32_C(  -122.74), SIMDE_FLOAT32_C(  -640.56), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   629.58),
        SIMDE_FLOAT32_C(   332.72), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(    24.75),
        SIMDE_FLOAT32_C(   -93.94), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   629.00), SIMDE_FLOAT32_C(   -64.86),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   256.46), SIMDE_FLOAT32_C(  -845.32), SIMDE_FLOAT32_C(   846.48) } },
    { UINT16_C( 9611),
      { SIMDE_FLOAT32_C(  -640.62), SIMDE_FLOAT32_C(   591.45), SIMDE_FLOAT32_C(  -288.54), SIMDE_FLOAT32_C(   692.10),
        SIMDE_FLOAT32_C(    56.18), SIMDE_FLOAT32_C(  -366.22), SIMDE_FLOAT32_C(  -889.14), SIMDE_FLOAT32_C(   962.24),
        SIMDE_FLOAT32_C(  -737.23), SIMDE_FLOAT32_C(   409.23), SIMDE_FLOAT32_C(   951.41), SIMDE_FLOAT32_C(   142.58),
        SIMDE_FLOAT32_C(   563.90), SIMDE_FLOAT32_C(   502.75), SIMDE_FLOAT32_C(  -959.00), SIMDE_FLOAT32_C(   923.35) },
      { SIMDE_FLOAT32_C(   132.33), SIMDE_FLOAT32_C(  -845.31), SIMDE_FLOAT32_C(   996.94), SIMDE_FLOAT32_C(   639.62),
        SIMDE_FLOAT32_C(   179.44), SIMDE_FLOAT32_C(   413.58), SIMDE_FLOAT32_C(  -527.59), SIMDE_FLOAT32_C(   123.63),
        SIMDE_FLOAT32_C(  -651.28), SIMDE_FLOAT32_C(  -623.79), SIMDE_FLOAT32_C(  -120.21), SIMDE_FLOAT32_C(   605.18),
        SIMDE_FLOAT32_C(  -777.31), SIMDE_FLOAT32_C(  -839.53), SIMDE_FLOAT32_C(   738.90), SIMDE_FLOAT32_C(  -417.93) },
      { SIMDE_FLOAT32_C(  -288.54), SIMDE_FLOAT32_C(   996.94), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   639.62),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   123.63),
        SIMDE_FLOAT32_C(   951.41), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   142.58), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   738.90), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00) } },
    { UINT16_C(47427),
      { SIMDE_FLOAT32_C(  -725.84), SIMDE_FLOAT32_C(  -191.90), SIMDE_FLOAT32_C(    84.13), SIMDE_FLOAT32_C(  -614.97),
        SIMDE_FLOAT32_C(  -229.66), SIMDE_FLOAT32_C(   346.90), SIMDE_FLOAT32_C(   794.26), SIMDE_FLOAT32_C(  -278.25),
        SIMDE_FLOAT32_C(  -510.51), SIMDE_FLOAT32_C(   358.16), SIMDE_FLOAT32_C(  -775.50), SIMDE_FLOAT32_C(  -469.51),
        SIMDE_FLOAT32_C(   281.51), SIMDE_FLOAT32_C(   356.83), SIMDE_FLOAT32_C(  -314.82), SIMDE_FLOAT32_C(   278.45) },
      { SIMDE_FLOAT32_C(    -3.55), SIMDE_FLOAT32_C(   864.62), SIMDE_FLOAT32_C(  -307.97), SIMDE_FLOAT32_C(   468.87),
        SIMDE_FLOAT32_C(   -11.75), SIMDE_FLOAT32_C(    40.75), SIMDE_FLOAT32_C(   845.07), SIMDE_FLOAT32_C(   868.04),
        SIMDE_FLOAT32_C(  -354.07), SIMDE_FLOAT32_C(  -932.24), SIMDE_FLOAT32_C(  -971.49), SIMDE_FLOAT32_C(  -615.17),
        SIMDE_FLOAT32_C(  -350.17), SIMDE_FLOAT32_C(   780.43), SIMDE_FLOAT32_C(  -164.81), SIMDE_FLOAT32_C(   -76.00) },
      { SIMDE_FLOAT32_C(    84.13), SIMDE_FLOAT32_C(  -307.97), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -278.25), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(  -775.50), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -615.17),
        SIMDE_FLOAT32_C(  -314.82), SIMDE_FLOAT32_C(  -164.81), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   -76.00) } },
    { UINT16_C(61115),
      { SIMDE_FLOAT32_C(   309.02), SIMDE_FLOAT32_C(   358.87), SIMDE_FLOAT32_C(   266.23), SIMDE_FLOAT32_C(   103.28),
        SIMDE_FLOAT32_C(  -919.38), SIMDE_FLOAT32_C(   755.71), SIMDE_FLOAT32_C(  -538.56), SIMDE_FLOAT32_C(  -694.88),
        SIMDE_FLOAT32_C(  -713.79), SIMDE_FLOAT32_C(   742.95), SIMDE_FLOAT32_C(   661.95), SIMDE_FLOAT32_C(   -28.61),
        SIMDE_FLOAT32_C(    21.40), SIMDE_FLOAT32_C(  -341.60), SIMDE_FLOAT32_C(  -163.99), SIMDE_FLOAT32_C(   713.43) },
      { SIMDE_FLOAT32_C(  -872.73), SIMDE_FLOAT32_C(   824.26), SIMDE_FLOAT32_C(  -245.82), SIMDE_FLOAT32_C(   972.34),
        SIMDE_FLOAT32_C(   692.31), SIMDE_FLOAT32_C(   400.12), SIMDE_FLOAT32_C(  -959.90), SIMDE_FLOAT32_C(   720.81),
        SIMDE_FLOAT32_C(   784.95), SIMDE_FLOAT32_C(  -310.06), SIMDE_FLOAT32_C(   501.24), SIMDE_FLOAT32_C(  -379.86),
        SIMDE_FLOAT32_C(   613.93), SIMDE_FLOAT32_C(  -910.23), SIMDE_FLOAT32_C(  -460.54), SIMDE_FLOAT32_C(   -77.04) },
      { SIMDE_FLOAT32_C(   266.23), SIMDE_FLOAT32_C(  -245.82), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   972.34),
        SIMDE_FLOAT32_C(  -538.56), SIMDE_FLOAT32_C(  -959.90), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   720.81),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   501.24), SIMDE_FLOAT32_C(   -28.61), SIMDE_FLOAT32_C(  -379.86),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -460.54), SIMDE_FLOAT32_C(   713.43), SIMDE_FLOAT32_C(   -77.04) } },
    { UINT16_C(43528),
      { SIMDE_FLOAT32_C(  -973.76), SIMDE_FLOAT32_C(   529.26), SIMDE_FLOAT32_C(   561.40), SIMDE_FLOAT32_C(  -512.32),
        SIMDE_FLOAT32_C(   834.38), SIMDE_FLOAT32_C(   847.61), SIMDE_FLOAT32_C(  -769.38), SIMDE_FLOAT32_C(   496.33),
        SIMDE_FLOAT32_C(  -181.01), SIMDE_FLOAT32_C(   252.02), SIMDE_FLOAT32_C(  -845.27), SIMDE_FLOAT32_C(   655.01),
        SIMDE_FLOAT32_C(   -34.55), SIMDE_FLOAT32_C(  -718.00), SIMDE_FLOAT32_C(   479.27), SIMDE_FLOAT32_C(   719.63) },
      { SIMDE_FLOAT32_C(  -745.66), SIMDE_FLOAT32_C(   171.58), SIMDE_FLOAT32_C(   119.74), SIMDE_FLOAT32_C(  -705.55),
        SIMDE_FLOAT32_C(  -107.61), SIMDE_FLOAT32_C(   -95.31), SIMDE_FLOAT32_C(   -15.62), SIMDE_FLOAT32_C(  -606.37),
        SIMDE_FLOAT32_C(   524.83), SIMDE_FLOAT32_C(  -401.68), SIMDE_FLOAT32_C(  -516.59), SIMDE_FLOAT32_C(  -935.71),
        SIMDE_FLOAT32_C(   521.28), SIMDE_FLOAT32_C(   932.05), SIMDE_FLOAT32_C(   869.98), SIMDE_FLOAT32_C(   547.51) },
      { SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -705.55),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(     0.00),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -516.59), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(  -935.71),
        SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   869.98), SIMDE_FLOAT32_C(     0.00), SIMDE_FLOAT32_C(   547.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 b = simde_mm512_loadu_ps(test_vec[i].b);
    simde__m512 r = simde_mm512_maskz_unpackhi_ps(test_vec[i].k, a, b);
    simde_test_x86_assert_equal_f32x16(r, simde_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_simde_mm512_unpackhi_pd (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float64 a[8];
    const simde_float64 b[8];
    const simde_float64 r[8];
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(  -303.04), SIMDE_FLOAT64_C(   484.86), SIMDE_FLOAT64_C(  -578.12), SIMDE_FLOAT64_C(   269.18),
        SIMDE_FLOAT64_C(  -655.06), SIMDE_FLOAT64_C(  -192.80), SIMDE_FLOAT64_C(  -504.95), SIMDE_FLOAT64_C(   -13.86) },
      { SIMDE_FLOAT64_C(  -659.88), SIMDE_FLOAT64_C(  -876.52), SIMDE_FLOAT64_C(   331.70), SIMDE_FLOAT64_C(   855.30),
        SIMDE_FLOAT64_C(  -350.13), SIMDE_FLOAT64_C(  -147.74), SIMDE_FLOAT64_C(   998.52), SIMDE_FLOAT64_C(   390.85) },
      { SIMDE_FLOAT64_C(   484.86), SIMDE_FLOAT64_C(  -876.52), SIMDE_FLOAT64_C(   269.18), SIMDE_FLOAT64_C(   855.30),
        SIMDE_FLOAT64_C(  -192.80), SIMDE_FLOAT64_C(  -147.74), SIMDE_FLOAT64_C(   -13.86), SIMDE_FLOAT64_C(   390.85) } },
    { { SIMDE_FLOAT64_C(   -87.98), SIMDE_FLOAT64_C(   370.18), SIMDE_FLOAT64_C(  -919.77), SIMDE_FLOAT64_C(   771.23),
        SIMDE_FLOAT64_C(    18.30), SIMDE_FLOAT64_C(   191.55), SIMDE_FLOAT64_C(  -358.05), SIMDE_FLOAT64_C(   800.62) },
      { SIMDE_FLOAT64_C(   678.98), SIMDE_FLOAT64_C(   604.48), SIMDE_FLOAT64_C(  -562.21), SIMDE_FLOAT64_C(  -868.39),
        SIMDE_FLOAT64_C(   766.01), SIMDE_FLOAT64_C(  -121.09), SIMDE_FLOAT64_C(   212.43), SIMDE_FLOAT64_C(  -537.02) },
      { SIMDE_FLOAT64_C(   370.18), SIMDE_FLOAT64_C(   604.48), SIMDE_FLOAT64_C(   771.23), SIMDE_FLOAT64_C(  -868.39),
        SIMDE_FLOAT64_C(   191.55), SIMDE_FLOAT64_C(  -121.09), SIMDE_FLOAT64_C(   800.62), SIMDE_FLOAT64_C(  -537.02) } },
    { { SIMDE_FLOAT64_C(  -636.22), SIMDE_FLOAT64_C(   634.32), SIMDE_FLOAT64_C(   732.16), SIMDE_FLOAT64_C(  -291.28),
        SIMDE_FLOAT64_C(  -558.48), SIMDE_FLOAT64_C(  -772.79), SIMDE_FLOAT64_C(   694.86), SIMDE_FLOAT64_C(  -218.36) },
      { SIMDE_FLOAT64_C(  -649.31), SIMDE_FLOAT64_C(    26.55), SIMDE_FLOAT64_C(  -363.06), SIMDE_FLOAT64_C(     0.56),
        SIMDE_FLOAT64_C(   878.82), SIMDE_FLOAT64_C(  -364.54), SIMDE_FLOAT64_C(  -608.59), SIMDE_FLOAT64_C(  -209.17) },
      { SIMDE_FLOAT64_C(   634.32), SIMDE_FLOAT64_C(    26.55), SIMDE_FLOAT64_C(  -291.28), SIMDE_FLOAT64_C(     0.56),
        SIMDE_FLOAT64_C(  -772.79), SIMDE_FLOAT64_C(  -364.54), SIMDE_FLOAT64_C(  -218.36), SIMDE_FLOAT64_C(  -209.17) } },
    { { SIMDE_FLOAT64_C(  -994.36), SIMDE_FLOAT64_C(  -528.36), SIMDE_FLOAT64_C(  -437.93), SIMDE_FLOAT64_C(    23.94),
        SIMDE_FLOAT64_C(   663.19), SIMDE_FLOAT64_C(   204.02), SIMDE_FLOAT64_C(  -175.44), SIMDE_FLOAT64_C(   342.17) },
      { SIMDE_FLOAT64_C(  -191.50), SIMDE_FLOAT64_C(   262.35), SIMDE_FLOAT64_C(   473.78), SIMDE_FLOAT64_C(  -425.48),
        SIMDE_FLOAT64_C(  -858.74), SIMDE_FLOAT64_C(  -313.78), SIMDE_FLOAT64_C(    37.50), SIMDE_FLOAT64_C(  -494.96) },
      { SIMDE_FLOAT64_C(  -528.36), SIMDE_FLOAT64_C(   262.35), SIMDE_FLOAT64_C(    23.94), SIMDE_FLOAT64_C(  -425.48),
        SIMDE_FLOAT64_C(   204.02), SIMDE_FLOAT64_C(  -313.78), SIMDE_FLOAT64_C(   342.17), SIMDE_FLOAT64_C(  -494.96) } },
    { { SIMDE_FLOAT64_C(  -679.47), SIMDE_FLOAT64_C(  -230.35), SIMDE_FLOAT64_C(   213.75), SIMDE_FLOAT64_C(  -237.95),
        SIMDE_FLOAT64_C(    -3.14), SIMDE_FLOAT64_C(   -91.39), SIMDE_FLOAT64_C(   543.69), SIMDE_FLOAT64_C(   347.54) },
      { SIMDE_FLOAT64_C(   935.16), SIMDE_FLOAT64_C(  -819.37), SIMDE_FLOAT64_C(  -651.90), SIMDE_FLOAT64_C(   813.98),
        SIMDE_FLOAT64_C(  -183.91), SIMDE_FLOAT64_C(  -260.49), SIMDE_FLOAT64_C(  -395.18), SIMDE_FLOAT64_C(  -178.27) },
      { SIMDE_FLOAT64_C(  -230.35), SIMDE_FLOAT64_C(  -819.37), SIMDE_FLOAT64_C(  -237.95), SIMDE_FLOAT64_C(   813.98),
        SIMDE_FLOAT64_C(   -91.39), SIMDE_FLOAT64_C(  -260.49), SIMDE_FLOAT64_C(   347.54), SIMDE_FLOAT64_C(  -178.27) } },
    { { SIMDE_FLOAT64_C(   211.15), SIMDE_FLOAT64_C(   166.89), SIMDE_FLOAT64_C(   845.67), SIMDE_FLOAT64_C(  -125.66),
        SIMDE_FLOAT64_C(  -629.09), SIMDE_FLOAT64_C(  -329.77), SIMDE_FLOAT64_C(  -783.49), SIMDE_FLOAT64_C(   179.41) },
      { SIMDE_FLOAT64_C(   932.58), SIMDE_FLOAT64_C(   690.29), SIMDE_FLOAT64_C(   753.93), SIMDE_FLOAT64_C(  -926.16),
        SIMDE_FLOAT64_C(  -623.49), SIMDE_FLOAT64_C(  -208.57), SIMDE_FLOAT64_C(  -421.12), SIMDE_FLOAT64_C(  -302.96) },
      { SIMDE_FLOAT64_C(   166.89), SIMDE_FLOAT64_C(   690.29), SIMDE_FLOAT64_C(  -125.66), SIMDE_FLOAT64_C(  -926.16),
        SIMDE_FLOAT64_C(  -329.77), SIMDE_FLOAT64_C(  -208.57), SIMDE_FLOAT64_C(   179.41), SIMDE_FLOAT64_C(  -302.96) } },
    { { SIMDE_FLOAT64_C(   561.08), SIMDE_FLOAT64_C(   792.63), SIMDE_FLOAT64_C(   459.09), SIMDE_FLOAT64_C(  -442.06),
        SIMDE_FLOAT64_C(  -298.76), SIMDE_FLOAT64_C(     2.78), SIMDE_FLOAT64_C(   905.48), SIMDE_FLOAT64_C(  -363.60) },
      { SIMDE_FLOAT64_C(   183.42), SIMDE_FLOAT64_C(  -746.41), SIMDE_FLOAT64_C(  -549.61), SIMDE_FLOAT64_C(   999.50),
        SIMDE_FLOAT64_C(    -6.90), SIMDE_FLOAT64_C(    55.20), SIMDE_FLOAT64_C(  -178.77), SIMDE_FLOAT64_C(  -795.75) },
      { SIMDE_FLOAT64_C(   792.63), SIMDE_FLOAT64_C(  -746.41), SIMDE_FLOAT64_C(  -442.06), SIMDE_FLOAT64_C(   999.50),
        SIMDE_FLOAT64_C(     2.78), SIMDE_FLOAT64_C(    55.20), SIMDE_FLOAT64_C(  -363.60), SIMDE_FLOAT64_C(  -795.75) } },
    { { SIMDE_FLOAT64_C(  -777.91), SIMDE_FLOAT64_C(  -333.10), SIMDE_FLOAT64_C(    78.59), SIMDE_FLOAT64_C(  -407.00),
        SIMDE_FLOAT64_C(   337.13), SIMDE_FLOAT64_C(   295.09), SIMDE_FLOAT64_C(   772.42), SIMDE_FLOAT64_C(   269.71) },
      { SIMDE_FLOAT64_C(   -14.62), SIMDE_FLOAT64_C(   526.35), SIMDE_FLOAT64_C(   343.55), SIMDE_FLOAT64_C(   361.89),
        SIMDE_FLOAT64_C(  -682.22), SIMDE_FLOAT64_C(   922.43), SIMDE_FLOAT64_C(  -941.07), SIMDE_FLOAT64_C(   878.86) },
      { SIMDE_FLOAT64_C(  -333.10), SIMDE_FLOAT64_C(   526.35), SIMDE_FLOAT64_C(  -407.00), SIMDE_FLOAT64_C(   361.89),
        SIMDE_FLOAT64_C(   295.09), SIMDE_FLOAT64_C(   922.43), SIMDE_FLOAT64_C(   269.71), SIMDE_FLOAT64_C(   878.86) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d b = simde_mm512_loadu_pd(test_vec[i].b);
    simde__m512d r = simde_mm512_unpackhi_pd(a, b);
    simde_test_x86_assert_equal_f64x8(r, simde_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_simde_mm512_mask_unpackhi_pd (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float64 src[8];
    const simde__mmask8 k;
    const simde_float64 a[8];
    const simde_float64 b[8];
    const simde_float64 r[8];
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(   622.29), SIMDE_FLOAT64_C(  -234.57), SIMDE_FLOAT64_C(   242.01), SIMDE_FLOAT64_C(   603.16),
        SIMDE_FLOAT64_C(  -763.13), SIMDE_FLOAT64_C(  -189.71), SIMDE_FLOAT64_C(  -905.64), SIMDE_FLOAT64_C(  -228.31) },
      UINT8_C(111),
      { SIMDE_FLOAT64_C(  -284.33), SIMDE_FLOAT64_C(    76.05), SIMDE_FLOAT64_C(   485.40), SIMDE_FLOAT64_C(   792.35),
        SIMDE_FLOAT64_C(   520.35), SIMDE_FLOAT64_C(  -375.72), SIMDE_FLOAT64_C(   317.52), SIMDE_FLOAT64_C(  -414.64) },
      { SIMDE_FLOAT64_C(  -632.01), SIMDE_FLOAT64_C(  -915.99), SIMDE_FLOAT64_C(    85.58), SIMDE_FLOAT64_C(  -240.99),
        SIMDE_FLOAT64_C(    54.79), SIMDE_FLOAT64_C(   838.88), SIMDE_FLOAT64_C(   324.71), SIMDE_FLOAT64_C(   651.03) },
      { SIMDE_FLOAT64_C(    76.05), SIMDE_FLOAT64_C(  -915.99), SIMDE_FLOAT64_C(   792.35), SIMDE_FLOAT64_C(  -240.99),
        SIMDE_FLOAT64_C(  -763.13), SIMDE_FLOAT64_C(   838.88), SIMDE_FLOAT64_C(  -414.64), SIMDE_FLOAT64_C(  -228.31) } },
    { { SIMDE_FLOAT64_C(   707.10), SIMDE_FLOAT64_C(   330.16), SIMDE_FLOAT64_C(  -750.77), SIMDE_FLOAT64_C(  -538.69),
        SIMDE_FLOAT64_C(   746.68), SIMDE_FLOAT64_C(   -52.02), SIMDE_FLOAT64_C(  -916.39), SIMDE_FLOAT64_C(  -487.89) },
      UINT8_C(246),
      { SIMDE_FLOAT64_C(   686.77), SIMDE_FLOAT64_C(  -251.02), SIMDE_FLOAT64_C(     0.28), SIMDE_FLOAT64_C(   781.13),
        SIMDE_FLOAT64_C(   520.67), SIMDE_FLOAT64_C(  -181.67), SIMDE_FLOAT64_C(  -503.21), SIMDE_FLOAT64_C(  -403.28) },
      { SIMDE_FLOAT64_C(  -696.27), SIMDE_FLOAT64_C(  -710.85), SIMDE_FLOAT64_C(  -882.93), SIMDE_FLOAT64_C(   -71.98),
        SIMDE_FLOAT64_C(   606.67), SIMDE_FLOAT64_C(  -297.57), SIMDE_FLOAT64_C(   296.01), SIMDE_FLOAT64_C(   690.68) },
      { SIMDE_FLOAT64_C(   707.10), SIMDE_FLOAT64_C(  -710.85), SIMDE_FLOAT64_C(   781.13), SIMDE_FLOAT64_C(  -538.69),
        SIMDE_FLOAT64_C(  -181.67), SIMDE_FLOAT64_C(  -297.57), SIMDE_FLOAT64_C(  -403.28), SIMDE_FLOAT64_C(   690.68) } },
    { { SIMDE_FLOAT64_C(   788.01), SIMDE_FLOAT64_C(  -944.98), SIMDE_FLOAT64_C(  -254.53), SIMDE_FLOAT64_C(   626.89),
        SIMDE_FLOAT64_C(   379.74), SIMDE_FLOAT64_C(  -603.50), SIMDE_FLOAT64_C(   333.99), SIMDE_FLOAT64_C(  -290.11) },
      UINT8_C( 94),
      { SIMDE_FLOAT64_C(   795.30), SIMDE_FLOAT64_C(  -543.43), SIMDE_FLOAT64_C(   593.70), SIMDE_FLOAT64_C(   878.91),
        SIMDE_FLOAT64_C(   -31.32), SIMDE_FLOAT64_C(   783.69), SIMDE_FLOAT64_C(   565.68), SIMDE_FLOAT64_C(   717.66) },
      { SIMDE_FLOAT64_C(  -216.03), SIMDE_FLOAT64_C(   346.80), SIMDE_FLOAT64_C(   238.33), SIMDE_FLOAT64_C(   602.30),
        SIMDE_FLOAT64_C(   843.60), SIMDE_FLOAT64_C(   835.06), SIMDE_FLOAT64_C(   906.03), SIMDE_FLOAT64_C(  -867.26) },
      { SIMDE_FLOAT64_C(   788.01), SIMDE_FLOAT64_C(   346.80), SIMDE_FLOAT64_C(   878.91), SIMDE_FLOAT64_C(   602.30),
        SIMDE_FLOAT64_C(   783.69), SIMDE_FLOAT64_C(  -603.50), SIMDE_FLOAT64_C(   717.66), SIMDE_FLOAT64_C(  -290.11) } },
    { { SIMDE_FLOAT64_C(   952.12), SIMDE_FLOAT64_C(  -165.95), SIMDE_FLOAT64_C(   739.41), SIMDE_FLOAT64_C(  -345.45),
        SIMDE_FLOAT64_C(  -869.94), SIMDE_FLOAT64_C(   430.09), SIMDE_FLOAT64_C(  -557.44), SIMDE_FLOAT64_C(  -814.92) },
      UINT8_C(172),
      { SIMDE_FLOAT64_C(  -930.55), SIMDE_FLOAT64_C(   564.82), SIMDE_FLOAT64_C(  -427.95), SIMDE_FLOAT64_C(   403.44),
        SIMDE_FLOAT64_C(  -725.29), SIMDE_FLOAT64_C(   217.77), SIMDE_FLOAT64_C(   198.74), SIMDE_FLOAT64_C(  -268.72) },
      { SIMDE_FLOAT64_C(  -188.52), SIMDE_FLOAT64_C(    77.65), SIMDE_FLOAT64_C(   699.95), SIMDE_FLOAT64_C(  -404.83),
        SIMDE_FLOAT64_C(  -356.67), SIMDE_FLOAT64_C(   417.61), SIMDE_FLOAT64_C(   379.14), SIMDE_FLOAT64_C(   990.13) },
      { SIMDE_FLOAT64_C(   952.12), SIMDE_FLOAT64_C(  -165.95), SIMDE_FLOAT64_C(   403.44), SIMDE_FLOAT64_C(  -404.83),
        SIMDE_FLOAT64_C(  -869.94), SIMDE_FLOAT64_C(   417.61), SIMDE_FLOAT64_C(  -557.44), SIMDE_FLOAT64_C(   990.13) } },
    { { SIMDE_FLOAT64_C(  -344.05), SIMDE_FLOAT64_C(   -18.56), SIMDE_FLOAT64_C(   833.73), SIMDE_FLOAT64_C(  -509.00),
        SIMDE_FLOAT64_C(  -112.53), SIMDE_FLOAT64_C(   966.47), SIMDE_FLOAT64_C(  -556.87), SIMDE_FLOAT64_C(   721.52) },
      UINT8_C(219),
      { SIMDE_FLOAT64_C(    97.68), SIMDE_FLOAT64_C(   851.58), SIMDE_FLOAT64_C(   135.98), SIMDE_FLOAT64_C(   540.24),
        SIMDE_FLOAT64_C(  -963.34), SIMDE_FLOAT64_C(   311.54), SIMDE_FLOAT64_C(   609.69), SIMDE_FLOAT64_C(   601.48) },
      { SIMDE_FLOAT64_C(   883.59), SIMDE_FLOAT64_C(    13.12), SIMDE_FLOAT64_C(   876.19), SIMDE_FLOAT64_C(   101.36),
        SIMDE_FLOAT64_C(  -788.13), SIMDE_FLOAT64_C(  -392.54), SIMDE_FLOAT64_C(   912.84), SIMDE_FLOAT64_C(   289.52) },
      { SIMDE_FLOAT64_C(   851.58), SIMDE_FLOAT64_C(    13.12), SIMDE_FLOAT64_C(   833.73), SIMDE_FLOAT64_C(   101.36),
        SIMDE_FLOAT64_C(   311.54), SIMDE_FLOAT64_C(   966.47), SIMDE_FLOAT64_C(   601.48), SIMDE_FLOAT64_C(   289.52) } },
    { { SIMDE_FLOAT64_C(  -692.58), SIMDE_FLOAT64_C(  -491.99), SIMDE_FLOAT64_C(   932.85), SIMDE_FLOAT64_C(   725.03),
        SIMDE_FLOAT64_C(   887.14), SIMDE_FLOAT64_C(   922.98), SIMDE_FLOAT64_C(  -619.02), SIMDE_FLOAT64_C(  -131.42) },
      UINT8_C(109),
      { SIMDE_FLOAT64_C(  -128.02), SIMDE_FLOAT64_C(   756.05), SIMDE_FLOAT64_C(   723.19), SIMDE_FLOAT64_C(   315.11),
        SIMDE_FLOAT64_C(   477.57), SIMDE_FLOAT64_C(   429.08), SIMDE_FLOAT64_C(  -587.21), SIMDE_FLOAT64_C(   329.15) },
      { SIMDE_FLOAT64_C(  -434.95), SIMDE_FLOAT64_C(   953.03), SIMDE_FLOAT64_C(   365.82), SIMDE_FLOAT64_C(   876.59),
        SIMDE_FLOAT64_C(   562.71), SIMDE_FLOAT64_C(   -32.71), SIMDE_FLOAT64_C(   760.18), SIMDE_FLOAT64_C(  -424.16) },
      { SIMDE_FLOAT64_C(   756.05), SIMDE_FLOAT64_C(  -491.99), SIMDE_FLOAT64_C(   315.11), SIMDE_FLOAT64_C(   876.59),
        SIMDE_FLOAT64_C(   887.14), SIMDE_FLOAT64_C(   -32.71), SIMDE_FLOAT64_C(   329.15), SIMDE_FLOAT64_C(  -131.42) } },
    { { SIMDE_FLOAT64_C(  -156.52), SIMDE_FLOAT64_C(  -138.46), SIMDE_FLOAT64_C(  -212.30), SIMDE_FLOAT64_C(   450.94),
        SIMDE_FLOAT64_C(  -225.62), SIMDE_FLOAT64_C(  -922.78), SIMDE_FLOAT64_C(   758.36), SIMDE_FLOAT64_C(   282.39) },
      UINT8_C(216),
      { SIMDE_FLOAT64_C(   483.39), SIMDE_FLOAT64_C(   169.53), SIMDE_FLOAT64_C(   933.05), SIMDE_FLOAT64_C(   864.37),
        SIMDE_FLOAT64_C(  -961.89), SIMDE_FLOAT64_C(   689.77), SIMDE_FLOAT64_C(  -263.65), SIMDE_FLOAT64_C(   794.16) },
      { SIMDE_FLOAT64_C(   412.95), SIMDE_FLOAT64_C(  -948.54), SIMDE_FLOAT64_C(   271.73), SIMDE_FLOAT64_C(  -157.97),
        SIMDE_FLOAT64_C(  -535.75), SIMDE_FLOAT64_C(  -399.12), SIMDE_FLOAT64_C(   407.08), SIMDE_FLOAT64_C(  -582.72) },
      { SIMDE_FLOAT64_C(  -156.52), SIMDE_FLOAT64_C(  -138.46), SIMDE_FLOAT64_C(  -212.30), SIMDE_FLOAT64_C(  -157.97),
        SIMDE_FLOAT64_C(   689.77), SIMDE_FLOAT64_C(  -922.78), SIMDE_FLOAT64_C(   794.16), SIMDE_FLOAT64_C(  -582.72) } },
    { { SIMDE_FLOAT64_C(   966.70), SIMDE_FLOAT64_C(   283.67), SIMDE_FLOAT64_C(   979.99), SIMDE_FLOAT64_C(   -66.01),
        SIMDE_FLOAT64_C(    43.85), SIMDE_FLOAT64_C(  -444.18), SIMDE_FLOAT64_C(   777.47), SIMDE_FLOAT64_C(   905.40) },
      UINT8_C( 28),
      { SIMDE_FLOAT64_C(   228.41), SIMDE_FLOAT64_C(  -320.22), SIMDE_FLOAT64_C(   420.75), SIMDE_FLOAT64_C(   -13.23),
        SIMDE_FLOAT64_C(   962.17), SIMDE_FLOAT64_C(   430.81), SIMDE_FLOAT64_C(  -529.84), SIMDE_FLOAT64_C(   131.70) },
      { SIMDE_FLOAT64_C(   363.87), SIMDE_FLOAT64_C(  -665.47), SIMDE_FLOAT64_C(   169.81), SIMDE_FLOAT64_C(    53.63),
        SIMDE_FLOAT64_C(    70.89), SIMDE_FLOAT64_C(   -36.03), SIMDE_FLOAT64_C(  -533.41), SIMDE_FLOAT64_C(   122.35) },
      { SIMDE_FLOAT64_C(   966.70), SIMDE_FLOAT64_C(   283.67), SIMDE_FLOAT64_C(   -13.23), SIMDE_FLOAT64_C(    53.63),
        SIMDE_FLOAT64_C(   430.81), SIMDE_FLOAT64_C(  -444.18), SIMDE_FLOAT64_C(   777.47), SIMDE_FLOAT64_C(   905.40) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d src = simde_mm512_loadu_pd(test_vec[i].src);
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d b = simde_mm512_loadu_pd(test_vec[i].b);
    simde__m512d r = simde_mm512_mask_unpackhi_pd(src, test_vec[i].k, a, b);
    simde_test_x86_assert_equal_f64x8(r, simde_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_simde_mm512_maskz_unpackhi_pd (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde__mmask8 k;
    const simde_float64 a[8];
    const simde_float64 b[8];
    const simde_float64 r[8];
  } test_vec[] = {
    { UINT8_C(242),
      { SIMDE_FLOAT64_C(  -414.51), SIMDE_FLOAT64_C(   -12.29), SIMDE_FLOAT64_C(   160.05), SIMDE_FLOAT64_C(   653.04),
        SIMDE_FLOAT64_C(   -28.81), SIMDE_FLOAT64_C(  -415.64), SIMDE_FLOAT64_C(   403.48), SIMDE_FLOAT64_C(   868.73) },
      { SIMDE_FLOAT64_C(  -715.00), SIMDE_FLOAT64_C(  -297.46), SIMDE_FLOAT64_C(  -191.42), SIMDE_FLOAT64_C(   464.99),
        SIMDE_FLOAT64_C(    12.91), SIMDE_FLOAT64_C(   240.45), SIMDE_FLOAT64_C(   671.85), SIMDE_FLOAT64_C(   163.80) },
      { SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(  -297.46), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00),
        SIMDE_FLOAT64_C(  -415.64), SIMDE_FLOAT64_C(   240.45), SIMDE_FLOAT64_C(   868.73), SIMDE_FLOAT64_C(   163.80) } },
    { UINT8_C( 85),
      { SIMDE_FLOAT64_C(   676.07), SIMDE_FLOAT64_C(   842.78), SIMDE_FLOAT64_C(  -171.45), SIMDE_FLOAT64_C(   899.43),
        SIMDE_FLOAT64_C(  -868.32), SIMDE_FLOAT64_C(   497.70), SIMDE_FLOAT64_C(  -644.76), SIMDE_FLOAT64_C(   -32.60) },
      { SIMDE_FLOAT64_C(  -827.94), SIMDE_FLOAT64_C(  -663.55), SIMDE_FLOAT64_C(   469.25), SIMDE_FLOAT64_C(   386.02),
        SIMDE_FLOAT64_C(  -683.55), SIMDE_FLOAT64_C(    95.06), SIMDE_FLOAT64_C(   971.52), SIMDE_FLOAT64_C(   304.16) },
      { SIMDE_FLOAT64_C(   842.78), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   899.43), SIMDE_FLOAT64_C(     0.00),
        SIMDE_FLOAT64_C(   497.70), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   -32.60), SIMDE_FLOAT64_C(     0.00) } },
    { UINT8_C(113),
      { SIMDE_FLOAT64_C(   624.56), SIMDE_FLOAT64_C(  -724.64), SIMDE_FLOAT64_C(  -160.53), SIMDE_FLOAT64_C(    28.03),
        SIMDE_FLOAT64_C(  -855.91), SIMDE_FLOAT64_C(   124.47), SIMDE_FLOAT64_C(   730.57), SIMDE_FLOAT64_C(   -47.33) },
      { SIMDE_FLOAT64_C(  -410.54), SIMDE_FLOAT64_C(  -256.52), SIMDE_FLOAT64_C(  -806.88), SIMDE_FLOAT64_C(  -738.68),
        SIMDE_FLOAT64_C(   907.28), SIMDE_FLOAT64_C(  -267.36), SIMDE_FLOAT64_C(   937.39), SIMDE_FLOAT64_C(   750.06) },
      { SIMDE_FLOAT64_C(  -724.64), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00),
        SIMDE_FLOAT64_C(   124.47), SIMDE_FLOAT64_C(  -267.36), SIMDE_FLOAT64_C(   -47.33), SIMDE_FLOAT64_C(     0.00) } },
    { UINT8_C(176),
      { SIMDE_FLOAT64_C(   836.82), SIMDE_FLOAT64_C(   881.74), SIMDE_FLOAT64_C(    58.89), SIMDE_FLOAT64_C(  -807.94),
        SIMDE_FLOAT64_C(  -150.85), SIMDE_FLOAT64_C(   230.95), SIMDE_FLOAT64_C(  -471.49), SIMDE_FLOAT64_C(  -681.61) },
      { SIMDE_FLOAT64_C(  -383.03), SIMDE_FLOAT64_C(  -155.04), SIMDE_FLOAT64_C(   413.45), SIMDE_FLOAT64_C(  -411.51),
        SIMDE_FLOAT64_C(  -850.88), SIMDE_FLOAT64_C(   668.57), SIMDE_FLOAT64_C(  -786.95), SIMDE_FLOAT64_C(  -575.52) },
      { SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00),
        SIMDE_FLOAT64_C(   230.95), SIMDE_FLOAT64_C(   668.57), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(  -575.52) } },
    { UINT8_C(127),
      { SIMDE_FLOAT64_C(   241.08), SIMDE_FLOAT64_C(  -431.43), SIMDE_FLOAT64_C(   632.51), SIMDE_FLOAT64_C(   -28.35),
        SIMDE_FLOAT64_C(   521.24), SIMDE_FLOAT64_C(  -778.03), SIMDE_FLOAT64_C(   715.13), SIMDE_FLOAT64_C(   714.36) },
      { SIMDE_FLOAT64_C(  -516.71), SIMDE_FLOAT64_C(   622.41), SIMDE_FLOAT64_C(  -553.00), SIMDE_FLOAT64_C(  -579.33),
        SIMDE_FLOAT64_C(   372.48), SIMDE_FLOAT64_C(  -991.81), SIMDE_FLOAT64_C(  -742.51), SIMDE_FLOAT64_C(   254.22) },
      { SIMDE_FLOAT64_C(  -431.43), SIMDE_FLOAT64_C(   622.41), SIMDE_FLOAT64_C(   -28.35), SIMDE_FLOAT64_C(  -579.33),
        SIMDE_FLOAT64_C(  -778.03), SIMDE_FLOAT64_C(  -991.81), SIMDE_FLOAT64_C(   714.36), SIMDE_FLOAT64_C(     0.00) } },
    { UINT8_C(235),
      { SIMDE_FLOAT64_C(  -550.44), SIMDE_FLOAT64_C(  -896.63), SIMDE_FLOAT64_C(  -701.96), SIMDE_FLOAT64_C(   -21.93),
        SIMDE_FLOAT64_C(  -578.24), SIMDE_FLOAT64_C(   -84.99), SIMDE_FLOAT64_C(   823.02), SIMDE_FLOAT64_C(   835.22) },
      { SIMDE_FLOAT64_C(   503.50), SIMDE_FLOAT64_C(   972.15), SIMDE_FLOAT64_C(   503.78), SIMDE_FLOAT64_C(   716.55),
        SIMDE_FLOAT64_C(  -603.37), SIMDE_FLOAT64_C(  -988.18), SIMDE_FLOAT64_C(   -42.37), SIMDE_FLOAT64_C(   -34.81) },
      { SIMDE_FLOAT64_C(  -896.63), SIMDE_FLOAT64_C(   972.15), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   716.55),
        SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(  -988.18), SIMDE_FLOAT64_C(   835.22), SIMDE_FLOAT64_C(   -34.81) } },
    { UINT8_C( 35),
      { SIMDE_FLOAT64_C(   929.28), SIMDE_FLOAT64_C(  -513.57), SIMDE_FLOAT64_C(   866.30), SIMDE_FLOAT64_C(   644.41),
        SIMDE_FLOAT64_C(  -799.21), SIMDE_FLOAT64_C(  -650.41), SIMDE_FLOAT64_C(   266.82), SIMDE_FLOAT64_C(  -352.22) },
      { SIMDE_FLOAT64_C(  -229.74), SIMDE_FLOAT64_C(  -360.70), SIMDE_FLOAT64_C(  -344.02), SIMDE_FLOAT64_C(    27.76),
        SIMDE_FLOAT64_C(   893.52), SIMDE_FLOAT64_C(   723.06), SIMDE_FLOAT64_C(   477.32), SIMDE_FLOAT64_C(   996.89) },
      { SIMDE_FLOAT64_C(  -513.57), SIMDE_FLOAT64_C(  -360.70), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00),
        SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   723.06), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00) } },
    { UINT8_C(169),
      { SIMDE_FLOAT64_C(  -544.62), SIMDE_FLOAT64_C(  -581.34), SIMDE_FLOAT64_C(   -63.90), SIMDE_FLOAT64_C(  -721.59),
        SIMDE_FLOAT64_C(  -746.13), SIMDE_FLOAT64_C(  -560.39), SIMDE_FLOAT64_C(  -749.45), SIMDE_FLOAT64_C(   757.66) },
      { SIMDE_FLOAT64_C(  -843.85), SIMDE_FLOAT64_C(  -352.82), SIMDE_FLOAT64_C(   769.48), SIMDE_FLOAT64_C(   113.78),
        SIMDE_FLOAT64_C(   612.37), SIMDE_FLOAT64_C(   413.81), SIMDE_FLOAT64_C(    43.06), SIMDE_FLOAT64_C(  -901.20) },
      { SIMDE_FLOAT64_C(  -581.34), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   113.78),
        SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(   413.81), SIMDE_FLOAT64_C(     0.00), SIMDE_FLOAT64_C(  -901.20) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d b = simde_mm512_loadu_pd(test_vec[i].b);
    simde__m512d r = simde_mm512_maskz_unpackhi_pd(test_vec[i].k, a, b);
    simde_test_x86_assert_equal_f64x8(r, simde_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

SIMDE_TEST_FUNC_LIST_BEGIN
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_epi8)

  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_epi16)

  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_unpackhi_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_maskz_unpackhi_epi32)

  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_epi64)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_unpackhi_epi64)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_maskz_unpackhi_epi64)

  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_ps)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_unpackhi_ps)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_maskz_unpackhi_ps)

  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_unpackhi_pd)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_unpackhi_pd)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_maskz_unpackhi_pd)
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
