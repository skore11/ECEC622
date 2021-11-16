/* Matrix multiplication: C = A * B.
 * Host code.

 * Modified: Naga Kandasamy
 * Date modified: May 5, 2020
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/* Include CUDA kernel during preprocessing step */
#include "matrix_multiply_kernel.cu"

extern "C" void compute_gold(float *, float *, float *, int, int, int);
matrix_t allocate_matrix_on_device(matrix_t);
matrix_t allocate_matrix(int, int, int);
void copy_matrix_to_device(matrix_t, matrix_t);
void copy_matrix_from_device(matrix_t, matrix_t);
void matrix_multiply_on_device(matrix_t, matrix_t, matrix_t);
int check_results(float *, float *, int, float);

int main(int argc, char** argv) 
{    
    /* Allocate and populate square matrices */
    matrix_t  M, N, P;
	unsigned int num_elements = WP * HP;
	srand(time(NULL));
    M  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); 
    N  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);
    P  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);

    /* Multiply using CPU */
    fprintf(stderr, "Multiplying two %d x %d matrices on CPU\n", M.height, M.width);
	matrix_t reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	compute_gold(M.elements, N.elements, reference.elements, HM, WM, WN);

    /* Multiply matrices on the device */
    fprintf(stderr, "Multiplying two %d x %d matrices on GPU\n", M.height, M.width);
    matrix_multiply_on_device(M, N, P);         	
		
	/* Check if device result is same as reference solution */
    fprintf(stderr, "Checking GPU result for correctness\n");
    float eps = 1e-6;
	int status = check_results(reference.elements, P.elements, num_elements, eps);
	fprintf(stderr, "TEST %s\n", (1 == status) ? "PASSED" : "FAILED");
	
    free(M.elements);                           /* Free host matrices */
    free(N.elements); 
	free(P.elements); 
	
    exit(EXIT_SUCCESS);
}

void matrix_multiply_on_device(matrix_t M, matrix_t N, matrix_t P)
{
    matrix_t d_M = allocate_matrix_on_device(M);  /* Allocate device memory */
	matrix_t d_N = allocate_matrix_on_device(N);
	matrix_t d_P = allocate_matrix_on_device(P);

	copy_matrix_to_device(d_M, M); 	            /* Copy matrices to device memory */
	copy_matrix_to_device(d_N, N);

	dim3 threads(MATRIX_SIZE, MATRIX_SIZE);     /* Set up execution grid */
	dim3 grid(d_M.width/threads.x, d_N.height/threads.y);

	/* Launch kernel */
	matrix_multiply<<<grid, threads>>>(d_P.elements, d_M.elements, d_N.elements);

	cudaError_t err = cudaGetLastError(); 	    /* Check for error */
	if (cudaSuccess != err) {
		fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	copy_matrix_from_device(P, d_P); 	        /* Copy result from device to host */

    cudaFree(d_M.elements);                     /* Free GPU memory */
	cudaFree(d_N.elements);
	cudaFree(d_P.elements);
}

/* Allocate matrix on device */
matrix_t allocate_matrix_on_device (matrix_t M)    
{
	matrix_t Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a matrix of dimensions height * width
   If init == 0, initialize to all zeros.  
   If init == 1, perform random initialization.
   */
matrix_t allocate_matrix(int height, int width, int init)
{
	matrix_t M;
	M.width = width; 
    M.height = height;
	int size = M.width * M.height;
	M.elements = (float *)malloc(size * sizeof(float));

    int i;
	for (i = 0; i < M.height * M.width; i++)
		M.elements[i] = (init == 0) ? (0.0f) : (rand()/(float)RAND_MAX);
	
	return M;
}	

/* Copy from host to device */
void copy_matrix_to_device(matrix_t Mdevice, matrix_t Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

/* Copy from device to host */
void copy_matrix_from_device(matrix_t Mhost, matrix_t Mdevice)    
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

/* Check results */
int check_results(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int i;
    int check_mark = 1;
    for (i = 0; i < num_elements; i++)
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold) {
            check_mark = 0;
            break;
        }

    return check_mark;
}
