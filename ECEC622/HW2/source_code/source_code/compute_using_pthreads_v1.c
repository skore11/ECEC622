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


  

   void *jacobi_v1(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
	
	
	
	float temp;
    int i, j;
	int max_iter = thread_data->iter;
	//double sum;
	//printf(stderr,"testing to see if A has columns: %d", num_cols);
	
	//might need to allocate only for a chunk size
	// if(thread_data->tid == 0)
	// {
		// fprintf(stderr, "id : %d \n", thread_data->tid);
		// fprintf(stderr, "no. of threads: %d \n", thread_data->num_threads);
		// fprintf(stderr, "chunk size: %d \n", thread_data->chunk_size);
		// print_matrix(thread_data->X_sol);
		// fprintf(stderr, "printing A matrix \n");
		// print_matrix(thread_data->A);
	// }
	
	/* Initialize current jacobi solution. */
	//here as well the num_rows might need to be chunked out
	// if (thread_data->tid < (thread_data->num_threads - 1))
	// {
		// for (i = thread_data->tid * thread_data->chunk_size; i < (thread_data->tid + 1) * thread_data->chunk_size; i++) 
		// {
			// thread_data->X_sol.elements[i] = thread_data->B.elements[i];
			// print_matrix(thread_data->X_sol);
		// }
	// }
	// else
	// {
		// for (i = thread_data->tid * thread_data->chunk_size; i < thread_data->A.num_rows; i++) 
		// {
			// thread_data->X_sol.elements[i] = thread_data->B.elements[i];
		// }
	// }
	
	int done = 0;
    float ssd, mse;
    int num_iter = 0;

	while (!done) {
		fprintf(stderr,"not done yet\n");
		printf("Thread number %d is processing for iteration %d\n", thread_data->tid, num_iter);
		if (thread_data->tid < (thread_data->num_threads - 1)) { /* Threads 0 through n - 2 process chunk_size output elements */
			for (i = thread_data->tid * thread_data->chunk_size; i < (thread_data->tid + 1) * thread_data->chunk_size; i++) {
				double sum = 0.0;
				for (j = 0; j < thread_data->A.num_columns; j++) {
					if (i != j)
					{
						
						//fprintf(stderr,"%f", thread_data->A.elements[i * thread_data->A.num_columns + j]);
						fprintf(stderr,"i - check chunk here- value %d \n", i);
						fprintf(stderr,"j value %d \n", j);
						fprintf(stderr,"X sol trial for %f \n", thread_data->X_sol.elements[j]);
						
					sum += thread_data->A.elements[i * thread_data->A.num_columns + j] * thread_data->X_sol.elements[j]; 
					//fprintf(stderr,"%f", sum);
					}
				}
				
				fprintf(stderr,"sum calc %f\n", sum);
				thread_data->X_new_sol.elements[i] = (thread_data->B.elements[i] - sum)/thread_data->A.elements[i * thread_data->A.num_columns + i];
				ssd = 0.0; 
				//mutex lock here
				fprintf(stderr,"check here before lock\n");
				 /* Accumulate partial sums into the shared variable */ 
				pthread_mutex_lock(thread_data->mutex_for_SSD_sum);
				ssd = (thread_data->X_new_sol.elements[i] - thread_data->X_sol.elements[i]) * (thread_data->X_new_sol.elements[i] - thread_data->X_sol.elements[i]);
				*(thread_data->partial_ssd) += ssd;
				thread_data->prev = &thread_data->X_sol.elements[i] ;
				thread_data->curr = &thread_data->X_new_sol.elements[i];
				fprintf(stderr,"check here during lock\n");
				//Use the temp variable to switch from new to old buffer
				temp = *(thread_data->curr);
				thread_data->X_sol.elements[i] = temp;
				*(thread_data->prev) = temp;
				thread_data->X_new_sol.elements[i] = *(thread_data->prev);
				pthread_mutex_unlock(thread_data->mutex_for_SSD_sum);
				
				
				fprintf(stderr,"check here after lock\n");
			}
		
						
		// fprintf(stderr,"check here before sync\n");

		/* Wait here for all threads to catch up before starting the next iteration. */
        //barrier_sync(&barrier, thread_data->tid, thread_data->num_threads); 
		
		// fprintf(stderr,"check here after sync\n");

        // //barrier sync here
        // num_iter++;
        // mse = sqrt(ssd); /* Mean squared error. */
        // fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        			
		}
		else { /* Last thread may have to process more than chunk_size output elements */
			for (i = thread_data->tid * thread_data->chunk_size; i < thread_data->A.num_rows; i++) {
				double sum = 0.0;
				for (j = 0; j < thread_data->A.num_columns; j++) {
					if(i != j)
					{
					fprintf(stderr,"i - check chunk here- value %d \n", i);
					fprintf(stderr,"j value %d \n", j);
					fprintf(stderr,"X sol trial for %f \n", thread_data->X_sol.elements[j]);
					sum += thread_data->A.elements[i * thread_data->A.num_columns + j] * thread_data->X_sol.elements[j];
					}
	 				
				}
				
				fprintf(stderr,"sum calc %f\n", sum);
				thread_data->X_new_sol.elements[i] = (thread_data->B.elements[i] - sum)/thread_data->A.elements[i * thread_data->A.num_columns + i];
				ssd = 0.0; 
				//mutex lock here
				 /* Accumulate partial sums into the shared variable */ 
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
				
				// barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);
			}
		
        

		}
		
		printf("Thread %d is at synchronization barrier\n", thread_data->tid);
		//barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);
        num_iter++;
		// if (thread_data->tid == thread_data->num_threads - 1)
		// {
			// fprintf(stderr, "partial ssd %f\n", *(thread_data->partial_ssd));
			// double mse2 = sqrt(*(thread_data->partial_ssd));
			// if ((mse2 <= THRESHOLD) || (num_iter == max_iter))
			// fprintf(stderr, "Iteration: %d. MSE2 = %f\n", num_iter, mse2);
            // done = 1;
		// }
		//fprintf(stderr, "partial ssd %f\n", *(thread_data->partial_ssd));
		fprintf(stderr, "global ssd %f\n", ssd);
        mse = sqrt(ssd); /* Mean squared error. */
		//double mse2 = sqrt(*(thread_data->partial_ssd));
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse);
		//fprintf(stderr, "Iteration: %d. MSE2 = %f\n", num_iter, mse2);
		
		//barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);
		if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
	}
    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");

    pthread_exit(NULL);
}

void compute_using_pthreads_v1(const matrix_t A, matrix_t old_values_x, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
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
        thread_data[i].X_new_sol = mt_sol_x_v1;
		thread_data[i].B = B;
		thread_data[i].prev = &prev;
		thread_data[i].curr = &curr;
 		thread_data[i].mutex_for_SSD_sum = &mutex_for_SSD_sum;
		thread_data[i].partial_ssd = &SSD_sum;
        } 
		
	 
	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, jacobi_v1, (void *)&thread_data[i]);

	/* Join point: wait for the workers to finish */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	/* Free dynamically allocated data structures */
	free((void *)thread_data);
    }




