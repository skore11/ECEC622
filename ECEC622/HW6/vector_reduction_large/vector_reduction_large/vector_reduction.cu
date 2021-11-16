/* Host side code. 
   Reduction of arbitrary sized vectors using atomics. 
   Also shows the use of pinned memory to map a portion of the CPU address space to the GPU's address space.

   Author: Naga Kandasamy
   Date created: February 14, 2017
   Date modified: May 17, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* Include kernel code */
#include "vector_reduction_kernel.cu"

void run_test(int);
double compute_on_device(float *, int);
void check_for_error(const char *);
extern "C" double compute_gold(float *, int);

int main(int argc, char **argv) 
{
    if (argc < 2) {
		printf("Usage: %s num-elements\n", argv[0]);
        printf("num-elements: Number of elements to be reduced\n");
		exit(EXIT_FAILURE);	
	}

	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	exit(EXIT_SUCCESS);
}

/* Perform reduction on the CPU and the GPU and compare results for correctness */ 
void run_test(int num_elements) 
{
    struct timeval start, stop;	

	/* Allocate memory on the CPU for the input vector */
	int vector_size = sizeof(float) * num_elements;
	float *A = (float *)malloc(vector_size);
		
	/* Randomly generate input data to be values between -.5 and +.5 */	
	printf("\nCreating a random vector with %d elements\n", num_elements);
	srand(time(NULL));
    int i;
	for (i = 0; i < num_elements; i++)
		A[i] = rand()/(float)RAND_MAX - .5;
		
	/* Reduce vector on CPU */
	printf("\nReducing the vector with %d elements on the CPU\n", num_elements);
	gettimeofday(&start, NULL);
	double reference = compute_gold(A, num_elements);
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    printf("Answer = %f\n", reference);

	/* Compute the result vector on the GPU */ 
	printf("\nReducing the vector with %d elements on the GPU\n", num_elements);
    gettimeofday(&start, NULL);
	double gpu_result = compute_on_device(A, num_elements);
    gettimeofday (&stop, NULL);
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("Answer = %f\n", gpu_result);
	
	/* Cleanup memory */
	free(A);
	exit(EXIT_SUCCESS);
}

double compute_on_device(float *A_on_host, int num_elements)
{
	float *A_on_device = NULL;
	double *result_on_device = NULL;
    struct timeval start, stop;	

    gettimeofday(&start, NULL);
	
    /* Allocate space on GPU for vector and copy contents over */
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	/* Allocate space for result on GPU and initialize */
	cudaMalloc((void**)&result_on_device, sizeof(double));
	cudaMemset(result_on_device, 0.0f, sizeof(double));

	/* Allocate space for the lock on GPU and initialize it */
	int *mutex_on_device = NULL;
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
	cudaMemset(mutex_on_device, 0, sizeof(int));

    gettimeofday(&stop, NULL);
    printf("Data transfer time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

 	/* Set up execution grid on GPU */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); 
	dim3 grid(NUM_BLOCKS,1);
	
	/* Launch kernel */
    gettimeofday(&start, NULL);
	vector_reduction_kernel<<<grid, thread_block>>>(A_on_device, result_on_device, num_elements, mutex_on_device);
	cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);
    printf("Kernel execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	check_for_error("KERNEL FAILURE");

	double sum;
	cudaMemcpy(&sum, result_on_device, sizeof(double), cudaMemcpyDeviceToHost);

	/* Free memory */
	cudaFree(A_on_device);
	cudaFree(result_on_device);
	cudaFree(mutex_on_device);

	return sum;
}

void check_for_error (const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
