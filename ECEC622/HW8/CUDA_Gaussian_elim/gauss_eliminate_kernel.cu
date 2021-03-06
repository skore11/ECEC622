 /* Device code. */
#include "gauss_eliminate.h"

__global__ void division__kernel(float *U, float* current_row, int k, int offset)
{
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	
	pivot = U[k*width + k];
	
	while(tid < n)
	{
		elem = U[k*width + k + 1 + tid];
		elem = elem/pivot;
		tid = tid + stride;
	}
	
	if(blockIdx.x == 0 && threadIdx.x == 0)
	{
		U[k*width + k] = 1;
	}
}

__global__ void elimination_kernel(float *U, float *current_row, int k, int offset)
{
	row = k + 1 + blockIdx.x;
	
	float alpha, a;
	while(row < n)
	{
		{
			x = A[row * width + k + 1 + tid];
			alpha = A[row * width + k];
			a = A[k * width + k + 1 + tid];
			x = x - alpha*a;
			
			tid = tid + blockDim.x;
		}
		_syncthreads()
		
		if(threadIdx.x == 0 && alpha == 0)
		{
			row = row + gridDim.x;
		}
	}
}
