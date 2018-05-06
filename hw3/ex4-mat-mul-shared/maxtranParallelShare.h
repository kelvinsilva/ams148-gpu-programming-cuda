#ifndef MAT_TRAN_PAR_SH_H_
#define MAT_TRAN_PAR_SH_H_

#include <stdio.h>
#include <cuda.h>

__global__ void matTran(int result_row_size, int result_col_size, float* result, int input_row_size, int input_col_size, float* matrix);

void matTranSharedParallel(float* matrix, float *result, int row, int col);

#endif
