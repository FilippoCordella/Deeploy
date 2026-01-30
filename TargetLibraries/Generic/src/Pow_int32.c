/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"


void Pow_int32_uint32_int32(const int32_t *__restrict__ data_in,
                            const uint32_t *__restrict__ exponent,
                            int32_t *__restrict__ data_out,
                            int32_t size)
{
    for (int i = 0; i < size; i++) {
        uint32_t exp = exponent[i];
        int32_t base = data_in[i];
        int32_t result = 1;

        while (exp) {
            if (exp & 1u) {
                int64_t intermediate = (int64_t)result * (int64_t)base;
                intermediate = CLAMP(intermediate, (int64_t)INT32_MIN, (int64_t)INT32_MAX);
                result = (int32_t)intermediate;
            }
            exp >>= 1u;

            if (exp) {
                int64_t intermediate = (int64_t)base * (int64_t)base;
                intermediate = CLAMP(intermediate, (int64_t)INT32_MIN, (int64_t)INT32_MAX);
                base = (int32_t)intermediate;
            }
            if ((result == INT32_MAX || result == INT32_MIN) && base >= 0) {
              break;
            }
        }
        data_out[i] = result;
    }
}

void Pow_int32_scalar_uint32_int32(const int32_t *__restrict__ data_in,
                                   uint32_t exponent,
                                   int32_t *__restrict__ data_out,
                                   int32_t size) {
  for (int32_t i = 0; i < size; i++) {
    uint32_t exp = exponent;
    int32_t base = data_in[i];
    int32_t result = 1;

    while (exp) {
      if (exp & 1u) {
        int64_t intermediate = (int64_t)result * (int64_t)base;
        intermediate = CLAMP(intermediate, (int64_t)INT32_MIN, (int64_t)INT32_MAX);
        result = (int32_t)intermediate;
      }
      exp >>= 1u;

      if (exp) {
        int64_t intermediate = (int64_t)base * (int64_t)base;
        intermediate = CLAMP(intermediate, (int64_t)INT32_MIN, (int64_t)INT32_MAX);
        base = (int32_t)intermediate;
      }
      if ((result == INT32_MAX || result == INT32_MIN) && base >= 0) {
        break;
      }
    }
    data_out[i] = result;
  }
}