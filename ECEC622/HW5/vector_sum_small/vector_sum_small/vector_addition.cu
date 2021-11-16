/* Host side code that calls a GPU kernel to perform vector addition on the GPU using a single thread block.
   
   We restrict the size of the vector to be up to 1024 elements which is the maximum thread block size on this 
   GPU.

    Author: Naga Kandasamy
    Date modified: May 3, 2020
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define NUM_ELEMENTS 1024

/* Include the kernel code during the compiler preprocessing step */
#include "vector_addition_kernel.cu"

void run_test(void);
void compute_on_device(float *, float *, float *, int);
extern "C" void compute_gold(float *, float *, float *, int);

int main(int argc, char **argv) 
{
    run_test();
	
    exit(EXIT_SUCCESS);
}

/* Perform vector addition on the CPU and the GPU */
void run_test(void) 
{                                                        
    int num_elements = NUM_ELEMENTS;
	float diff;
	int i; 

    /* Allocate memory on the CPU for input vectors A and B, and output vector C */
    int vector_length = sizeof(float) * num_elements;
	float *A = (float *)malloc(vector_length);
	float *B = (float *)malloc(vector_length);
	float *gold_result = (float *)malloc(vector_length);            /* Result vector computed on CPU */
	float *gpu_result = (float *)malloc(vector_length);             /* Result vector computed on GPU */
	
	/* Initialize the input data to be integer values between 0 and 5 */ 
	for (i = 0; i < num_elements; i++) {
		A[i] = floorf(5 * (rand() / (float)RAND_MAX));
     	B[i] = floorf(5 * (rand() / (float)RAND_MAX));
	}

	/* Compute reference solution on CPU */
	compute_gold(A, B, gold_result, num_elements);
    
	/* Compute result vector on GPU */ 
	compute_on_device(A, B, gpu_result, num_elements);

	/* Compute differences between CPU and GPU results */
    diff = 0.0;
    for (i = 0; i < num_elements; i++)
		diff += fabsf(gold_result[i] - gpu_result[i]);

	printf("Difference between the CPU and GPU result = %f\n", diff);
   
	/* Cleanup memory */
	free((void *)A);
	free((void *)B);
	free((void *)gold_result);
	free((void *)gpu_result);
	
	return;
}

/* Vector addition on GPU */
void compute_on_device(float *A_on_host, float *B_on_host, float *gpu_result, int num_elements)
{                                                                                                 
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 

	/* Allocate space on GPU for vectors A and B, and copy contents of vectors to GPU */
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	/* Allocate space for result vector on GPU */
	cudaMalloc((void**)&C_on_device, num_elements * sizeof(float));
	
 	/* Set up execution grid on the GPU */
	dim3 thread_block(num_elements, 1, 1);          /* Set number of threads in thread block */
	dim3 grid(1,1);

	vector_addition_kernel<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements);                                                                                                       
	/* Copy result vector back from GPU */ 
	cudaMemcpy(gpu_result, C_on_device, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free memory on GPU */
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);
}
     
