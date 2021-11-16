/* Reference code for solving the equation by jacobi iteration method */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "jacobi_solver.h"

void compute_using_omp(const matrix_t A, matrix_t x, const matrix_t B, int max_iter, int num_threads)
{
    int i, j;

    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      
    
	//Pointers for copying values to and from the x and x_new buffers
	float *prev;
	float *curr;
	float temp;
	
    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++)
        x.elements[i] = B.elements[i];

    /* Perform Jacobi iteration. */
    int done = 0;
	double sum;
    double ssd, mse;
    int num_iter = 0;
    
	
	#pragma omp parallel private(i,j, sum, prev, curr, temp) shared(ssd, num_rows, num_cols, num_iter) num_threads(num_threads)
   
   while (!done) {

	#pragma omp for schedule(dynamic)
        for (i = 0; i < num_rows; i++) {
            sum = 0.0;
            for (j = 0; j < num_cols; j++) {
                if (i != j)
					//here is where we see the new values shown in the x buffer; it has become prev values
                    sum += A.elements[i * num_cols + j] * x.elements[j];
            }
           
            /* Update values for the unknowns for the current row. */
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
        }

        /* Note: you can optimize the above nested loops by removing the branch 
         * statement within the j loop. The rewritten code is as follows: 
         *
         * for (i = 0; i < num_rows; i++){
         *      double sum = -A.elements[i * num_cols + i] * ref_x.elements[i];
         *      for (j = 0; j < num_cols; j++)
         *          sum += A.elements[i * num_cols + j] * ref_x.elements[j];
         * }
         *
         * new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
         */

        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
		
		#pragma omp for reduction(+:ssd)
        for (i = 0; i < num_rows; i++) {
            ssd += (new_x.elements[i] - x.elements[i]) * (new_x.elements[i] - x.elements[i]);
            //x.elements[i] = new_x.elements[i];
			prev = &x.elements[i];
			curr = &new_x.elements[i];
			//printf("the previous value : %f\n", prev);
			//printf("the current value : %f\n", curr);
			temp = *curr;
			//printf("temp after copying the current value : %f\n", temp);
			*prev = temp;
			//printf("temp after copying to prev value : %f\n", prev);
			x.elements[i] = temp;
        }
		#pragma omp barrier
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        
		
        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
    }
	#pragma omp master
    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");

    free(new_x.elements);
}
    


