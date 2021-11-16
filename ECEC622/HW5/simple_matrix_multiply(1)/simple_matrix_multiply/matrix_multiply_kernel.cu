/* Device code for matrix multiplication: P = M * N 
    Author: Naga Kandasamy
 */
#ifndef _MATRIX_MULTIPLY_KERNEL_H_
#define _MATRIX_MULTIPLY_KERNEL_H_

#include "matrix.h"

__global__ void matrix_multiply(float *P, float *M, float *N)
{
	/* Obtain thread location within the block */
	int tx, ty;
    tx = threadIdx.x;
	ty = threadIdx.y;

    int k;
    float M_element, N_element;
	float P_temp = 0;
	for (k = 0; k < WM; ++k) {
		M_element = M[ty * WM + k];
		N_element = N[k * WN + tx];
		P_temp += M_element * N_element;
	}

	P[ty * WN + tx] = P_temp;
}

#endif /* _MATRIX_MULTIPY_KERNEL_H_ */
