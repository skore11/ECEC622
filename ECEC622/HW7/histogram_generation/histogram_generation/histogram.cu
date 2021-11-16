/* Histogram generation on the GPU. 
 * Host-side code.
	
 * Author: Naga Kandasamy
 * Date modified: May 17, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define THREAD_BLOCK_SIZE 256 
#define NUM_BLOCKS 40 
#define HISTOGRAM_SIZE 256 /* Histogram has 256 bins */

#include "histogram_kernel.cu"

void run_test(int);
void compute_on_device(int *, int *, int, int);
void check_for_error(const char *);
extern "C" void compute_gold(int *, int *, int, int);
void check_histogram(int *, int, int);
void print_histogram(int *, int , int);

int main(int argc, char **argv) 
{
	if (argc < 2) {
		printf("Usage: %s num-elements\n", argv[0]);
		exit(EXIT_SUCCESS);	
	}

	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	
    exit(EXIT_SUCCESS);
}

void run_test(int num_elements) 
{
	float diff;
	int i; 

    /* Allocate and initialize space to store histograms generated by the CPU and the GPU */
	int *histogram_on_cpu = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 
    memset(histogram_on_cpu, 0, sizeof(int) * HISTOGRAM_SIZE);    
    
    int *histogram_on_gpu = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 
    memset(histogram_on_gpu, 0, sizeof(int) * HISTOGRAM_SIZE);    

	/* Generate input data to be integer values between 0 and (HISTOGRAM_SIZE - 1) */
    printf("\nGenerating input data\n");
	int size = sizeof(int) * num_elements;
	int *input_data = (int *)malloc (size);
    for(i = 0; i < num_elements; i++)
        input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));

	printf("\nGenerating histrgram on CPU\n");
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

	compute_gold(input_data, histogram_on_cpu, num_elements, HISTOGRAM_SIZE);

    gettimeofday(&stop, NULL);
	printf("Eexcution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

	check_histogram(histogram_on_cpu, num_elements, HISTOGRAM_SIZE);
    
    printf("\nGenerating histrgram on device\n");
	compute_on_device(input_data, histogram_on_gpu, num_elements, HISTOGRAM_SIZE);
	check_histogram(histogram_on_gpu, num_elements, HISTOGRAM_SIZE);

	/* Compute the differences between the CPU and GPU results. */
	diff = 0.0;
    for(i = 0; i < HISTOGRAM_SIZE; i++)
		diff += abs(histogram_on_cpu[i] - histogram_on_gpu[i]);

	printf("Difference between CPU and device results = %f\n", diff);
   
	/* cleanup memory. */
	free((void *)input_data);
	free((void *)histogram_on_cpu);
	free((void *)histogram_on_gpu);

	exit(EXIT_SUCCESS);
}

void compute_on_device(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    int *input_data_on_device = NULL;
	int *histogram_on_device = NULL;

	/* Allocate space on GPU for input data */
	cudaMalloc((void**)&input_data_on_device, num_elements * sizeof(int));
	cudaMemcpy(input_data_on_device, input_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);

	/* Allocate space on GPU for histogram and initialize contents to zero */
	cudaMalloc((void**)&histogram_on_device, histogram_size * sizeof(int));
	cudaMemset(histogram_on_device, 0, histogram_size * sizeof(int));

 	/* Set up the execution grid on GPU */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
	dim3 grid(NUM_BLOCKS,1);
	
    struct timeval start, stop;	
	gettimeofday(&start, NULL);
    
    printf("Using global memory to generate histrogram\n");    
	histogram_kernel_slow<<<grid, thread_block>>>(input_data_on_device, histogram_on_device, num_elements, histogram_size); 
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);
	printf("Execution time = %f \n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	check_for_error("KERNEL FAILURE");

	printf("Using shared memory to generate histogram\n");
    gettimeofday(&start, NULL);
    cudaMemset(histogram_on_device, 0, histogram_size * sizeof(int)); /* Reset histogram */
    histogram_kernel_fast<<<grid, thread_block>>>(input_data_on_device, histogram_on_device, num_elements, histogram_size); 
	cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);
	printf("Eexecution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	check_for_error("KERNEL FAILURE");

	/* Copy result back from GPU */ 
	cudaMemcpy(histogram, histogram_on_device, histogram_size * sizeof(int), cudaMemcpyDeviceToHost);
	print_histogram(histogram, histogram_size, num_elements);
	/* Free memory */
	cudaFree(input_data_on_device);
	cudaFree(histogram_on_device);
}

/* Check correctness of result: sum of histogram bins must equal number of input elements */
void check_histogram(int *histogram, int num_elements, int histogram_size)
{
	int sum = 0;
    int i;
	for (i = 0; i < histogram_size; i++)
		sum += histogram[i];

	printf("Number of histogram entries = %d. \n", sum);
	if (sum == num_elements)
		printf("Histogram generated successfully. \n");
	else
		printf("Error generating histogram. \n");
	
    printf("\n");
}


/* Check for errors during kernel execution */
void check_for_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
