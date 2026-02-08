/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"


void Pow_int32_uint32_int32(const int32_t *__restrict__ data_in,
                            const int32_t *__restrict__ exponent,
                            int32_t *__restrict__ data_out,
                            int32_t size, int32_t exponent_offset){
  for (int i = 0; i < size; i++) {
    uint32_t exp = (uint32_t)(exponent[i] + exponent_offset);
    int32_t base = data_in[i];
    int32_t result = 1;
    int overflow = 0;
    while (exp) {
        if (exp & 1u) {
            int64_t intermediate = (int64_t)result * (int64_t)base;
            if (intermediate < (int64_t)INT32_MIN || intermediate > (int64_t)INT32_MAX) {
                overflow = 1;
                break;
            }
            result = (int32_t)intermediate;
        }
        exp >>= 1u;
        if (exp) {
            int64_t intermediate = (int64_t)base * (int64_t)base;
            if (intermediate < (int64_t)INT32_MIN || intermediate > (int64_t)INT32_MAX) {
                overflow = 1;
                break;
            }
            base = (int32_t)intermediate;
        }
    }
    data_out[i] = overflow ? 0 : result;
  }
}

void Pow_int32_scalar_uint32_int32(const int32_t *__restrict__ data_in,
                                   int32_t exponent,
                                   int32_t *__restrict__ data_out,
                                   int32_t size, int32_t exponent_offset) {
  uint32_t exp_const = (uint32_t)(exponent + exponent_offset);
  for (int i = 0; i < size; i++) {
    uint32_t exp = exp_const;
    int32_t base = data_in[i];
    int32_t result = 1;
    int overflow = 0;
    while (exp) {
        if (exp & 1u) {
            int64_t intermediate = (int64_t)result * (int64_t)base;
            if (intermediate < (int64_t)INT32_MIN || intermediate > (int64_t)INT32_MAX) {
                overflow = 1;
                break;
            }
            result = (int32_t)intermediate;
        }
        exp >>= 1u;
        if (exp) {
            int64_t intermediate = (int64_t)base * (int64_t)base;
            if (intermediate < (int64_t)INT32_MIN || intermediate > (int64_t)INT32_MAX) {
                overflow = 1;
                break;
            }
            base = (int32_t)intermediate;
        }
    }
    data_out[i] = overflow ? 0 : result;
  }
}