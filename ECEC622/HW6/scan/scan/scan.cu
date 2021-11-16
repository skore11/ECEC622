/* Host code for parallel scan. 
 * Author: Naga Kandasamy
 * Date created: November 27, 2015
 * Date modified: May 17, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define NUM_ELEMENTS 1024

/* Include kernel */
#include "scan_kernel.cu"

void run_test(int);
extern "C" void compute_gold(float *, float *, int);
void check_CUDA_error(const char *);
int check_results(float *, float *, int, float);

int main(int argc, char **argv) 
{
    int num_elements = NUM_ELEMENTS;
    run_test(num_elements);

    exit(EXIT_SUCCESS);
}

void run_test(int num_elements) 
{ 
    int mem_size = sizeof(float) * num_elements;
    int shared_mem_size = sizeof(float) * num_elements;

    /* Memory on host to store input data */
    float* h_data = (float *)malloc(mem_size);
      
    /* Initialize input data to be integer values between 0 and 10 */
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; ++i)
        h_data[i] = floorf(10 * (rand()/(float)RAND_MAX));
    
    /* Compute reference solution */
    float *reference = (float *)malloc(mem_size);  
    compute_gold(reference, h_data, num_elements);

    /* Allocate memory on device for input and output arrays */
    float *d_in;
    float *d_out;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    /* Copy input array to device */
    cudaMemcpy(d_in, h_data, mem_size, cudaMemcpyHostToDevice);

    /* Set up execution grid.
     * Note: this implementation only supports a single thread-block worth of data.
     */
    dim3  grid(1, 1);
    dim3 threads(NUM_ELEMENTS, 1, 1);
 
    printf("\nRunning parallel prefix sum (scan) of %d elements\n", num_elements);
    scan_kernel<<< grid, threads, 2 * shared_mem_size >>>(d_out, d_in, num_elements);
    cudaDeviceSynchronize();
    check_CUDA_error("KERNEL EXECUTION FAILURE");
  
    /* Copy result from device to host */
    cudaMemcpy(h_data, d_out, mem_size, cudaMemcpyDeviceToHost);
        
    float epsilon = 0.0f;
    int result = check_results(reference, h_data, num_elements, epsilon);
    printf("TEST %s\n", (0 == result) ? "PASSED" : "FAILED");

    /* cleanup memory */
    free(h_data);
    free(reference);
    cudaFree(d_in);
    cudaFree(d_out);

    exit(EXIT_SUCCESS);
}

void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


int check_results(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int check = 0;
    int i;
    for (i = 0; i < num_elements; i++)
        if ((reference[i] - gpu_result[i]) > threshold) {
            check = -1;
            break;
        }

    return check;
}
