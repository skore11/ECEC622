/* Host code for stream compaction. 
 * The input stream is filtered to remove all values <= 0 in it. 
 * The output stream contains only positive value > 0.
 * 
 * Author: Naga Kandasamy
 * Date created: May 12, 2019
 * Date modified: May 30, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* Uncomment the line below if you want debug information */
// #define DEBUG

#define NUM_ELEMENTS 1024
#define MIN_NUMBER -10
#define MAX_NUMBER 10

/* Include kernel */
#include "compact_kernel.cu"

void run_test(int);
extern "C" void compute_gold(int *, int *, int *);
void check_CUDA_error(const char *);
int check_results(int *, int *, int);
void print_elements(int *, int);
void print_scanned_elements(int *, int);
int rand_int(int, int);

int main(int argc, char **argv) 
{
    int num_elements = NUM_ELEMENTS;
    run_test(num_elements);

    exit(EXIT_SUCCESS);
}

void run_test(int num_elements) 
{ 
    /* Memory on host to store input data */
    int mem_size = sizeof(int) * num_elements;
    int *h_data = (int *)malloc(mem_size);
      
    /* Initialize input data to be random values between [-0.5, +0.5] */
    printf("\nGenerating input stream of %d elements\n", num_elements);
    srand(time (NULL));
    int i;
    for (i = 0; i < num_elements; ++i)
        h_data[i] = rand_int(MIN_NUMBER, MAX_NUMBER);

#ifdef DEBUG
    printf("\nOriginal stream\n");
    print_elements(h_data, num_elements);
#endif
    
    /* Compute reference solution */
    printf("\nCompacting stream on CPU\n");
    int *reference = (int *)malloc(mem_size); 
    int h_new_n = num_elements;
    compute_gold(reference, h_data, &h_new_n);

#ifdef DEBUG
    print_elements(reference, h_new_n);
#endif
    printf("Number of elements in compacted stream = %d\n", h_new_n);

    /* Allocate memory on device for input and output arrays */
    printf("\nCompacting stream on GPU\n");
    int *d_in, *d_out;
    cudaMalloc((void **)&d_in, mem_size);
    cudaMalloc((void **)&d_out, mem_size);

    /* Copy input array to device */
    cudaMemcpy(d_in, h_data, mem_size, cudaMemcpyHostToDevice);
    
    /* Allocate memory on host and on device for the scanned flag array */
    int *h_flag, *d_flag;
    h_flag = (int *)malloc(num_elements * sizeof(int));
    cudaMalloc((void **)&d_flag, num_elements * sizeof(int));
    
    /* Allocate memory on device for integer. 
     * It stores the number of elements in the compacted stream. 
    */
    int *d_new_n;
    cudaMalloc((void **)&d_new_n, sizeof(int));

    /* Set up execution grid.
     * Note: this implementation only supports a single thread-block worth of data.
     */
    dim3  grid(1, 1);
    dim3 threads(NUM_ELEMENTS, 1, 1);
 
    compact_kernel<<<grid, threads>>>(d_out, d_in, num_elements, d_new_n, d_flag);
    cudaDeviceSynchronize();
    check_CUDA_error("KERNEL EXECUTION FAILURE");
  
    /* Copy results from device to host */
    cudaMemcpy(&h_new_n, d_new_n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag, d_flag, num_elements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_out, h_new_n * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    print_scanned_elements(h_flag, num_elements);
    print_elements(h_data, h_new_n);
#endif
    printf("Number of elements in compacted stream = %d\n", h_new_n);
        
    int result = check_results(reference, h_data, h_new_n);
    printf("\nTEST %s\n", (0 == result) ? "PASSED" : "FAILED");

    /* Cleanup memory */
    free(h_data);
    free(reference);
    cudaFree(d_new_n);
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

/* Return random integer between [min, max] */
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Check GPU and CPU results. Return 0 on success, -1 otherwise */
int check_results(int *reference, int *gpu_result, int n)
{
    int check = 0;
    int i;

    for (i = 0; i < n; i++)
        if (reference[i] != gpu_result[i]) {
            check = -1;
            break;
        }

    return check;
}

void print_elements(int *in, int num_elements)
{
    int i;
    for (i = 0; i < num_elements; i++)
        printf ("%0.2f ", in[i]);
    
    printf ("\n");
}

void print_scanned_elements(int *in, int num_elements)
{
    int i;
    for (i = 0; i < num_elements; i++)
        printf ("%d ", in[i]);

    printf ("\n");
}
