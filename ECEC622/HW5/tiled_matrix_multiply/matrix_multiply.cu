/* Host code for matrix multiplication: C = A * B.
 * 
 * Author: Naga Kandasamy
 * Date created: February 17, 2015
 * Date modified: May 5, 2020
 * 
 * Notes: use provided Makefile
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Include the kernel code */
#include "matrix_multiply_kernel.cu"

extern "C" void compute_gold_naive(matrix_t, matrix_t, matrix_t);
extern "C" void compute_gold_blocked(matrix_t, matrix_t, matrix_t);
matrix_t allocate_matrix_on_device(matrix_t);
matrix_t allocate_matrix(int, int, int);
void copy_matrix_to_device(matrix_t, matrix_t);
void copy_matrix_from_device(matrix_t, matrix_t);
void free_matrix_on_device(matrix_t *);
void free_matrix_on_host(matrix_t *);
void matrix_multiply_on_device(matrix_t, matrix_t, matrix_t);
void check_CUDA_error(const char *);
int check_results(float *, float *, int, float);

int main(int argc, char **argv) 
{
    srand(time(NULL));
    
    matrix_t M, N;
    M = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);       /* Create and populate the matrices */
	N  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);

    struct timeval start, stop;	
    fprintf(stderr, "\nMultiplying %d x %d matrices on CPU using naive version\n", MATRIX_SIZE, MATRIX_SIZE);
	gettimeofday(&start, NULL);

    matrix_t P_reference_naive;
    P_reference_naive = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);  
    compute_gold_naive(M, N, P_reference_naive);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                              (stop.tv_usec - start.tv_usec)/(float)1000000));

    fprintf(stderr, "\nMultiplying %d x %d matrices on CPU using blocked version\n", MATRIX_SIZE, MATRIX_SIZE);
	gettimeofday(&start, NULL);

    matrix_t P_reference_blocked;
    P_reference_blocked = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);  
    compute_gold_blocked(M, N, P_reference_blocked);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                              (stop.tv_usec - start.tv_usec)/(float)1000000));


    fprintf(stderr, "\nMultiplying %d x %d matrices on the GPU\n", MATRIX_SIZE, MATRIX_SIZE);
    
    matrix_t P_device;
    P_device = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    matrix_multiply_on_device(M, N, P_device); 

    /* Check if the device result matches reference solution */
    int num_elements = M.height * M.width;
    float eps = 1e-6;
	int status = check_results (P_reference_blocked.elements, P_device.elements, num_elements, eps);
	fprintf(stderr, "TEST %s\n", (1 == status) ? "PASSED" : "FAILED");
	
	/* Free matrices on host */
	free_matrix_on_host(&M);
	free_matrix_on_host(&N);
    free_matrix_on_host(&P_reference_naive);
    free_matrix_on_host(&P_reference_blocked);
	free_matrix_on_host(&P_device);

	exit(EXIT_SUCCESS);
}

/* Multiply matrices on device */
void matrix_multiply_on_device(matrix_t M, matrix_t N, matrix_t P)
{
    /* Allocate memory and copy matrices to the device */
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    matrix_t Md = allocate_matrix_on_device(M);
	copy_matrix_to_device(Md, M);
	
    matrix_t Nd = allocate_matrix_on_device(N);
	copy_matrix_to_device(Nd, N);

    matrix_t Pd = allocate_matrix_on_device(P);

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Data transfer time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                                  (stop.tv_usec - start.tv_usec)/(float)1000000));


	fprintf(stderr, "Nd size = %d\n", Nd.width);
    /* Set up the execution grid */
    dim3 threads(TILE_SIZE, TILE_SIZE);                     
    fprintf(stderr, "Setting up a %d x %d grid of thread blocks\n", (Pd.width + TILE_SIZE - 1)/TILE_SIZE,\\
            (Pd.height + TILE_SIZE - 1)/TILE_SIZE);
	dim3 grid((Pd.width + TILE_SIZE - 1)/TILE_SIZE, (Pd.height + TILE_SIZE - 1)/TILE_SIZE);

	gettimeofday(&start, NULL);

	/* Launch kernel */
	matrix_multiply_kernel<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaDeviceSynchronize(); /* Kernel execution is asynchronous; force CPU to wait here */

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Kernel execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                                     (stop.tv_usec - start.tv_usec)/(float)1000000));

    check_CUDA_error("Error in kernel");

    copy_matrix_from_device(P, Pd);                        

    free_matrix_on_device(&Md);                                  
	free_matrix_on_device(&Nd);
	free_matrix_on_device(&Pd);
}

/* Allocate memory on device for matrix */
matrix_t allocate_matrix_on_device(matrix_t M)                        
{
	matrix_t Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	
    cudaMalloc((void**)&Mdevice.elements, size);
    if (Mdevice.elements == NULL) {
        fprintf(stderr, "CudaMalloc error\n");
        exit(EXIT_FAILURE);
    }

	return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
   */
matrix_t allocate_matrix(int height, int width, int init)
{
	matrix_t M;
	M.width = width;
	M.height = height;
	
    int size = M.width * M.height;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    int i;
	for (i = 0; i < M.height * M.width; i++)
		M.elements[i] = (init == 0) ? (0.0f) : floor((3 *(rand()/(float)RAND_MAX)));
	
	return M;
}	

/* Copy matrix from host memory to device memory */
void copy_matrix_to_device(matrix_t Mdevice, matrix_t Mhost)      
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

/* Copy matrix from device memory to host memory */
void copy_matrix_from_device(matrix_t Mhost, matrix_t Mdevice)   
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

/* Free matrix on device */
void free_matrix_on_device(matrix_t  *M)                              
{
	cudaFree(M->elements);
	M->elements = NULL;
}

/* Free matrix on host */
void free_matrix_on_host(matrix_t *M)
{
	free(M->elements);
	M->elements = NULL;
}

/* Check for errors during kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
}

/* Check the correctness of reference and device results */
int check_results(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int check_mark = 1;
    float max_diff = 0.0;
    int i;

    for (i = 0; i < num_elements; i++)
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold)
            check_mark = 0;
        
    for (i = 0; i < num_elements; i++)
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > max_diff)
            max_diff = fabsf((reference[i] - gpu_result[i])/reference[i]);
        
    fprintf(stderr, "Max diff = %f\n", max_diff); 

    return check_mark;
}
