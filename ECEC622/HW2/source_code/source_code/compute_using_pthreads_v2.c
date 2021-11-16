/* p-thread code for solving the equation by jacobi iteration method using chunking*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include "jacobi_solver.h"

typedef struct thread_data_s {
    int tid;            /* Thread identifier */
    int num_threads;    /* Number of threads in the worker pool */
    int chunk_size;     /* Size of data to be processed by thread */
	int iter;			/*max iterations allowed*/
    matrix_t A;       /* The A matrix. */
    matrix_t X_sol;       /* The prev solution matrix. */
	matrix_t X_new_sol;   /* The curr solution matrix*/
    matrix_t B;       /* The result matrix, B */
	double *partial_ssd ;/* every thread has a partial SSD calculated for the chunk*/ 
	pthread_mutex_t *mutex_for_SSD_sum; /* Location of lock variable protecting SSD sum */
	float *prev; /*pointer to the previous x_sol*/
	float *curr; /*pointer to the current x_sol being calculated*/
} thread_data_t;

  void *jacobi_v2(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
	
	float temp;
	int tid = thread_data->tid;
	int max_iter = thread_data->iter;
	int stride = thread_data->num_threads;
    int i;
    //int num_rows = thread_data->A.num_rows;
	//double sum;
	
	/* Allocate n x 1 matrix to hold iteration values.*/
    //matrix_t new_sol_x_v2 = allocate_matrix(num_rows, 1, 0);
	
	/* Initialize current jacobi solution. */
    // for (i = 0; i < num_rows; i++)
        // thread_data->X_sol.elements[i] = thread_data->B.elements[i];
	
	int done = 0;
    double ssd, mse;
    int num_iter = 0;

	while (!done) {
		
		 while (tid < thread_data->B.num_rows) {
			double sum = 0.0;
			for (i = 0; i < thread_data->A.num_columns; i++) 
				sum += thread_data->A.elements[tid * thread_data->A.num_columns + i] * thread_data->X_sol.elements[i];

			thread_data->B.elements[tid] = sum;
			tid += stride;
			//fprintf(stderr,"%f", sum);
			ssd = 0.0;
			for (i = thread_data->tid * thread_data->chunk_size; i < (thread_data->tid + 1) * thread_data->chunk_size; i++) {
					//ssd += (new_sol_x_v2.elements[i] - thread_data->X_sol.elements[i]) * (new_sol_x_v2.elements[i] - thread_data->X_sol.elements[i]);
					//thread_data->X_sol.elements[i] = new_sol_x_v2.elements[i];
					fprintf(stderr,"check here before lock\n");
					pthread_mutex_lock(thread_data->mutex_for_SSD_sum);
					ssd = (thread_data->X_new_sol.elements[i] - thread_data->X_sol.elements[i]) * (thread_data->X_new_sol.elements[i] - thread_data->X_sol.elements[i]);
					*(thread_data->partial_ssd) += ssd;
					thread_data->prev = &thread_data->X_sol.elements[i] ;
					thread_data->curr = &thread_data->X_new_sol.elements[i];
					
					//Use the temp variable to switch from new to old buffer
					temp = *(thread_data->curr);
					thread_data->X_sol.elements[i] = temp;
					*(thread_data->prev) = temp;
					thread_data->X_new_sol.elements[i] = *(thread_data->prev);
					pthread_mutex_unlock(thread_data->mutex_for_SSD_sum);
					}
					num_iter++;
					mse = sqrt(ssd); /* Mean squared error. */
					fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
				}
		
		if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
	}

    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
    //free((void *)thread_data);
	//free(new_sol_x_v2.elements);
    pthread_exit(NULL);
} 


void compute_using_pthreads_v2(const matrix_t A, matrix_t old_values_x, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{
	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
	pthread_attr_t attributes;      /* Thread attributes */
	pthread_attr_init(&attributes);
	
	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	
	pthread_mutex_t mutex_for_SSD_sum;                                                  /* Lock for the shared variable sum */
    pthread_mutex_init(&mutex_for_SSD_sum, NULL); 
	
	double SSD_sum;
	float prev;
	float curr;
	
    int i;
	fprintf(stderr, "no. of rows in B: %d\n", B.num_rows);
	//thread_data_t *thread_data;	
	int chunk_size = (int)floor(B.num_rows/(float)num_threads);
    fprintf(stderr, "chunk size from init: %d\n", chunk_size);
    /* Fork point: create worker threads */
    
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].chunk_size = chunk_size;
		thread_data[i].iter = max_iter;
        thread_data[i].A = A;
        thread_data[i].X_sol = B;
        thread_data[i].X_new_sol = mt_sol_x_v2;
		thread_data[i].B = B;
		thread_data[i].prev = &prev;
		thread_data[i].curr = &curr;
 		thread_data[i].mutex_for_SSD_sum = &mutex_for_SSD_sum;
		thread_data[i].partial_ssd = &SSD_sum;
        } 
		
	 
	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, jacobi_v2, (void *)&thread_data[i]);

	/* Join point: wait for the workers to finish */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	/* Free dynamically allocated data structures */
	free((void *)thread_data);
 
/* Display statistic related to the Jacobi solution */

}
