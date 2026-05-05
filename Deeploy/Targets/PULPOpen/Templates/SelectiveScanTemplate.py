# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULPSelectiveScanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # Hidden state h_buffer: int32 [B, D_inner, N] → 4 bytes/elem
        h_buffer_size = (operatorRepresentation['batch_size'] * operatorRepresentation['d_inner'] *
                         operatorRepresentation['d_state'] * 4)
        h_buffer_name = operatorRepresentation['nodeName'] + "_h_buffer"
        return [(h_buffer_name, h_buffer_size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        h_buffer_name, h_buffer_dim = PULPSelectiveScanTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(h_buffer_name, h_buffer_dim)

        operatorRepresentation['h_buffer'] = h_buffer_name
        return ctxt, operatorRepresentation, [h_buffer_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        return ctxt, operatorRepresentation, []


referenceTemplate = PULPSelectiveScanTemplate("""
// SelectiveScan (Name: ${nodeName}, Op: ${nodeOp})
PULP_SelectiveScan_i8_i8(
    (const int8_t  *) ${x},
    (const int8_t  *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int32_t *) ${A},
    (const int32_t *) ${D_skip},
    (int8_t        *) ${y},
    (int32_t       *) ${h_buffer},
    ${batch_size},
    ${seq_len},
    ${d_inner},
    ${d_state}
);
""")
