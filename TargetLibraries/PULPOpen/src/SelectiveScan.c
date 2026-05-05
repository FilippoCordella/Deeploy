/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

static inline int32_t _ssm_sat_i32_i64(int64_t x) {
  if (x > (int64_t)INT32_MAX)
    return INT32_MAX;
  if (x < (int64_t)INT32_MIN)
    return INT32_MIN;
  return (int32_t)x;
}

// Symmetric rounding right-shift
static inline int64_t _ssm_round_shift_i64(int64_t x, int s) {
  if (s <= 0)
    return x;
  const int64_t half = (int64_t)1 << (s - 1);
  if (x >= 0) {
    return (x + half) >> s;
  } else {
    return -(((-x) + half) >> s);
  }
}

void PULP_SelectiveScan_i8_i8(
    const int8_t *__restrict__ x, const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
    const int32_t *__restrict__ C, const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip, int8_t *__restrict__ y,
    int32_t *__restrict__ h_buffer, uint32_t B_size, uint32_t L,
    uint32_t D_inner, uint32_t N) {
  const int8_t core_id = pi_core_id();
  const int8_t log2Core = LOG2(NUM_CORES);

  // Parallelize over D_inner.
  const uint32_t D_chunk =
      (D_inner >> log2Core) + ((D_inner & (NUM_CORES - 1)) != 0);
  const uint32_t D_start = MIN(core_id * D_chunk, D_inner);
  const uint32_t D_end = MIN(D_start + D_chunk, D_inner);

  for (uint32_t b = 0; b < B_size; b++) {
    for (uint32_t d = D_start; d < D_end; d++) {
      int32_t *h_row = h_buffer + (b * D_inner + d) * N;
      for (uint32_t n = 0; n < N; n++) {
        h_row[n] = 0;
      }
    }
  }

  pi_cl_team_barrier();

  if (D_start >= D_end) {
    pi_cl_team_barrier();
    return;
  }

  for (uint32_t b = 0; b < B_size; b++) {
    for (uint32_t t = 0; t < L; t++) {
      const int32_t *B_row = B + (b * L + t) * N;
      const int32_t *C_row = C + (b * L + t) * N;

      for (uint32_t d = D_start; d < D_end; d++) {
        const int32_t x_val = (int32_t)x[(b * L + t) * D_inner + d];
        const int32_t z_val = (int32_t)z[(b * L + t) * D_inner + d];
        const int32_t dt_val = (int32_t)dt[(b * L + t) * D_inner + d]; // Q8.8
        int32_t *h_row = h_buffer + (b * D_inner + d) * N;
        const int32_t *A_row = A + d * N;

        int64_t y_acc = 0;

        for (uint32_t n = 0; n < N; n++) {
          // Q15 discretization: dA = exp(dt*A), dB = dt*B.
          const int32_t dt_A = (int32_t)(((int64_t)dt_val * A_row[n]) >> 8);

          // Promote Q15 -> Q20, clip to [-20*Q20, 0], round to LUT step.
          const int64_t exp_arg_q20 = (int64_t)dt_A
                                      << (20 - SSM_WIDE_FRAC_BITS);
          int64_t exp_clip = exp_arg_q20;
          if (exp_clip < -(int64_t)SSM_EXP_LUT_RANGE_Q20)
            exp_clip = -(int64_t)SSM_EXP_LUT_RANGE_Q20;
          if (exp_clip > 0)
            exp_clip = 0;
          int64_t exp_idx_i64 = (exp_clip + (int64_t)SSM_EXP_LUT_RANGE_Q20 +
                                 (SSM_EXP_LUT_STEP_Q20 >> 1)) >>
                                SSM_EXP_LUT_STEP_LOG2;
          if (exp_idx_i64 < 0)
            exp_idx_i64 = 0;
          if (exp_idx_i64 > (int64_t)(SSM_EXP_LUT_ENTRIES - 1))
            exp_idx_i64 = (int64_t)(SSM_EXP_LUT_ENTRIES - 1);
          int32_t dA_q15 =
              (int32_t)SelectiveScan_exp_lut_qwide[(uint32_t)exp_idx_i64];
          if (exp_arg_q20 >= 0)
            dA_q15 = (int32_t)((1 << SSM_WIDE_FRAC_BITS) - 1);

          const int32_t dB_q15 = (int32_t)(((int64_t)dt_val * B_row[n]) >> 8);

          // Recurrence: h = sat_i32((dA*h)>>15 + dB*x).
          int64_t h_acc = ((int64_t)dA_q15 * h_row[n]) >> SSM_WIDE_FRAC_BITS;
          h_acc += (int64_t)dB_q15 * (int64_t)x_val;
          h_row[n] = _ssm_sat_i32_i64(h_acc);

          // Output projection: y_acc += (h*C) >> 15.
          y_acc +=
              ((int64_t)h_row[n] * (int64_t)C_row[n]) >> SSM_WIDE_FRAC_BITS;
        }

        // Skip connection.
        y_acc += (int64_t)D_skip[d] * (int64_t)x_val;

        // SiLU gate from LUT.
        const int64_t gate_q13 =
            (int64_t)SelectiveScan_gate_lut_q13[z_val + 128];
        const int64_t y_gated = _ssm_round_shift_i64(y_acc * gate_q13, 13);

        // Output requantize to int8.
        int64_t y_out =
            _ssm_round_shift_i64(y_gated * (int64_t)SSM_OUTPUT_MUL_Q40, 40);
        if (y_out > 127)
          y_out = 127;
        if (y_out < -128)
          y_out = -128;
        y[(b * L + t) * D_inner + d] = (int8_t)y_out;
      }
    }
  }

  pi_cl_team_barrier();
}
