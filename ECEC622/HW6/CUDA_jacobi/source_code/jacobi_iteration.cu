/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: February 23, 2021
 *
 * Student name(s); Abishek S Kumar
 * Date modified: 02/26/2021
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	struct timeval start, stop;	
	
	
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
    display_jacobi_solution(A, gpu_opt_solution_x, B); 
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, 
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
    struct timeval start1, stop1;	
	gettimeofday(&start1, NULL);

    /* Allocate memory and copy matrices to the device */
	//int size = A.num_rows * A.num_columns *sizeof(float);	
	
	matrix_t A_d = allocate_matrix_on_device(A);
	copy_matrix_to_device(A_d, A);

	matrix_t B_d = allocate_matrix_on_device(B);
	copy_matrix_to_device(B_d, B);	
	
	matrix_t gpu_naive_d = allocate_matrix_on_device(gpu_naive_sol_x);
	copy_matrix_to_device(gpu_naive_d, B);
	
	matrix_t gpu_opt_d = allocate_matrix_on_device(gpu_opt_sol_x);
	copy_matrix_to_device(gpu_opt_d, B);
	
	matrix_t new_x = allocate_matrix_on_device(gpu_naive_sol_x);
	matrix_t new_x_opt = allocate_matrix_on_device(gpu_opt_sol_x);
	//copy_matrix_to_device(new_x, B);
	check_CUDA_error("Error in mem allocation of input matrices");
	
	int done = 0;
	int done2 = 0;
	//double sum;
    double ssd = 0.0;
	double mse = 0.0;
    int num_iter = 0;
	int size = sizeof(double);
	//double ssd_d = 0.0;
	//double ssd_d_opt = 0.0 ;
	double *ssd_d, *ssd_d_opt;
	//double *ssd_d = 0.0;
	//double *ssd_d_opt = 0.0 ;
	//fprintf(stderr, "ssd on device naive = %f\n", ssd_d);
	//fprintf(stderr, "ssd on device opt = %f\n", ssd_d_opt);
	//cudaMalloc((double **) &ssd_d, sizeof(double));
	//cudaMalloc((double **) &ssd_d_opt, sizeof(double));
	cudaMalloc((double **) &ssd_d, size);
	cudaMalloc((double **) &ssd_d_opt, size);
	//cudaMemset(&ssd_d, 0, 1);
	//cudaMemset(&ssd_d_opt, 0, 1);
	//cudaMemcpy(&ssd_d, &ssd, 0*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(&ssd_d_opt, &ssd, 0*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ssd_d, &ssd, size, cudaMemcpyHostToDevice);
	cudaMemcpy(ssd_d_opt, &ssd, size, cudaMemcpyHostToDevice);
	check_CUDA_error("Error in SSd allocation");
	
	gettimeofday(&stop1, NULL);
	fprintf(stderr, "Data transfer time = %fs\n", (float)(stop1.tv_sec - start1.tv_sec +\
                                                  (stop1.tv_usec - start1.tv_usec)/(float)1000000));
												  
    
    /* Set up execution configuration for the naive kernel and launch it */
	dim3 threads(1, THREAD_BLOCK_SIZE, 1);
	dim3 grid(1, (gpu_naive_d.num_rows + THREAD_BLOCK_SIZE - 1)/THREAD_BLOCK_SIZE);	
	
	while (!done) {
	//fprintf(stderr, "while loop begun here\n"); 
    jacobi_iteration_kernel_naive<<<grid, threads>>>(A_d.elements, gpu_naive_d.elements, B_d.elements, new_x.elements, A_d.num_rows, A_d.num_columns, ssd_d);
    cudaDeviceSynchronize();
	
	matrix_t temp = gpu_naive_d;
	gpu_naive_d = new_x;
	new_x = temp;
	cudaMemcpy(&ssd, ssd_d, size, cudaMemcpyDeviceToHost);
	num_iter++;
    mse = sqrt(ssd); /* Mean squared error. */
    //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        
		
    if ((mse <= THRESHOLD) || num_iter == 10000)
        done = 1;
	}
    gettimeofday(&stop1, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop1.tv_sec - start1.tv_sec +\
                (stop1.tv_usec - start1.tv_usec)/(float)1000000));

    check_CUDA_error("Error in kernel");

	/* Set up execution configuration for optimized kernel that uses shared memory. 
       Memory accesses made by threads are coalesced. 
     */
    fprintf(stderr, "\nUsing optimized version when matrix A is stored in row major form on device\n");
    gettimeofday(&start1, NULL);

	threads.x = threads.y = TILE_SIZE;
	grid.x = 1;
    grid.y = (gpu_opt_d.num_rows + TILE_SIZE - 1)/TILE_SIZE;
	
	int num_iter2 = 0;
	while (!done2) {
    jacobi_iteration_kernel_optimized<<< grid, threads >>>(A_d.elements, gpu_opt_d.elements, B_d.elements, new_x_opt.elements, A_d.num_rows, A_d.num_columns, ssd_d_opt);
	cudaDeviceSynchronize();
    
	matrix_t temp = gpu_opt_d;
	gpu_opt_d = new_x_opt;
	new_x_opt = temp;
	cudaMemcpy(&ssd, ssd_d_opt, size, cudaMemcpyDeviceToHost);
	num_iter2++;
    mse = sqrt(ssd); /* Mean squared error. */
    //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter2, mse); 
        
		
    if ((mse <= THRESHOLD) || num_iter == 10000)
        done2 = 1;
	}

    gettimeofday(&stop1, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop1.tv_sec - start1.tv_sec +\
                (stop1.tv_usec - start1.tv_usec)/(float)1000000));
	
	check_CUDA_error("Error in kernel");

	/* Copy result from the device */
	copy_matrix_from_device(gpu_naive_sol_x, gpu_naive_d); 
	copy_matrix_from_device(gpu_opt_sol_x, gpu_opt_d); 
	
	/* Free memory on device */
	cudaFree(&A_d);
	cudaFree(&B_d);
	cudaFree(&gpu_naive_d);
	cudaFree(&gpu_opt_d);
	cudaFree(&new_x);
	cudaFree(&new_x_opt);
	cudaFree(&ssd_d);
	cudaFree(&ssd_d_opt);
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

