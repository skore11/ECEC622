/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void blur_filter_kernel (const float *in, float *out, int size)
{
	
	int i,j;
    int curr_row, curr_col;
    float blur_value;
    int num_neighbors;

    /* Obtain thread index within the thread block */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	
	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	
	/* Find position in matrix */
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;
	
	
    blur_value = 0.0;
    num_neighbors = 0;
    for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
        for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
			/* Accumulate values of neighbors while checking for 
			 * boundary conditions */
			curr_row = row_number + i;
			curr_col = column_number + j;
			if ((curr_row > -1) && (curr_row < size) &&\
					(curr_col > -1) && (curr_col < size)) {
				blur_value += in[curr_row * size + curr_col];
				num_neighbors += 1;
			}
        }
    }

     /* Write averaged blurred value out */
    out[row_number * size + column_number] = (float)(blur_value/num_neighbors);
	
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
