#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(const float *A_d, float *gpu_naive_d, const float *B_d, float *new_x, int num_rows, int num_columns, double *ssd_d)
{

    int threadY = threadIdx.y; 	  
    int blockY = blockIdx.y;
		  
    int row = blockDim.y * blockY + threadY; /* Obtain row number. */
	float sum;
    int i;
    if (row < num_rows) {
        sum = 0.0;
        for (i = 0; i < num_columns; i++) {		
			if (row != i)			
				sum += A_d[num_columns * row + i] * gpu_naive_d[i];
        }
		  
        new_x[row] = (B_d[row] - sum)/A_d[row * num_columns + row];
    }
	
	*ssd_d += (new_x[row] - gpu_naive_d[row]) * (new_x[row] - gpu_naive_d[row]);
    
}

__global__ void jacobi_iteration_kernel_optimized(const float *A_d, float *gpu_opt_d, const float *B_d, float *new_x_opt, int num_rows, int num_columns, double *ssd_d)
{

	/* Declare shared memory for the thread block */
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];

	/* Calculate thread index, block index and position in matrix */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;

    if (row < num_rows) {
        float sum = 0.0;

        for (int i = 0; i < num_columns; i += TILE_SIZE) {
            /* Bring TILE_SIZE elements per row of A into shared memory */
            aTile[threadY][threadX] = A_d[num_columns * row + i + threadX]; 		
            
            /* Bring TILE_SIZE elements of the vector X into shared memory */
            if(threadY == 0) 
                xTile[threadX] = gpu_opt_d[i + threadX]; 
		
            /* Barrier sync to ensure that shared memory has been populated */
            __syncthreads();

            /* Compute partial sum for the current tile */
            int k;
            if (threadX == 0) {
                for (k = 0; k < TILE_SIZE; k += 1) 
                    sum += aTile[threadY][k] * xTile[k]; 		      
            }
            __syncthreads();
        }

	new_x_opt[row] = (B_d[row] - sum)/A_d[row * num_columns + row];
    }
	
	*ssd_d += (new_x_opt[row] - gpu_opt_d[row]) * (new_x_opt[row] - gpu_opt_d[row]);
}

