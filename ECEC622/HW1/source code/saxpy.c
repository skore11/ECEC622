/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 14, 2020
 * Date modified: January 19, 2021 
 *
 * Student names: Abishek S Kumar
 * Date: 01/26/2021
 *
 * */

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

 /* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
	int tid;                        /* The thread ID */
	int num_threads;                /* Number of threads in the worker pool */
	int num_elements;               /* Number of elements in the vector */
	float a;				/* Pointer to scalar a */
	float *x;                /* Pointer to x */
	float *y;                /* Pointer to y */
	//float *y2;                /* Pointer to y2 */
	//float *y3;                /* Pointer to y3 */
	int offset;                     /* Starting offset for each thread within the vectors */
	int chunk_size;                 /* Chunk size */
	//double *partial_sum;            /* Starting address of the partial_sum array */
} thread_data_t;

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
void *saxpy_chunk(void *);
void *saxpy_stride(void *);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    fprintf(stderr, "\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");
 
    if (check_results(y1, y3, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference solution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

void *saxpy_chunk(void *args)
{
	/* Typecast argument as a pointer to the thread_data_t structure */
	thread_data_t *thread_data = (thread_data_t *)args;

	
	//double partial_sum = 0.0;
	if (thread_data->tid < (thread_data->num_threads - 1)) {
		for (int i = thread_data->offset; i < (thread_data->offset + thread_data->chunk_size); i++)
			thread_data->y[i] = (thread_data->a * thread_data->x[i])  +  thread_data->y[i];
	}
	else { /* This takes care of the number of elements that the final thread must process */
		for (int i = thread_data->offset; i < thread_data->num_elements; i++)
			thread_data->y[i] = (thread_data->a * thread_data->x[i]) + thread_data->y[i];
	}

	/* Store partial sum into the partial_sum array */
	//thread_data->partial_sum[thread_data->tid] = partial_sum;

	pthread_exit(NULL);
}

/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
	pthread_attr_t attributes;      /* Thread attributes */
	pthread_attr_init(&attributes); /* Initialize thread attributes to default values */

	/* Fork point: allocate memory on heap for required data structures and create worker threads */
	int i;
	int chunk_size = (int)floor((float)num_elements / (float)num_threads); /* Compute the chunk size */
	//double *partial_sum = (double *)malloc(sizeof(double) * num_threads); /* Allocate memory to store partial sums returned by threads */

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	for (i = 0; i < num_threads; i++) {
		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].num_elements = num_elements;
		thread_data[i].a = a;
		thread_data[i].x = x;
		thread_data[i].y = y;
		thread_data[i].offset = i * chunk_size; //actual index offset for each thread stored here
		thread_data[i].chunk_size = chunk_size; //assign chunk size for each thread
		//thread_data[i].partial_sum = partial_sum;
	}

	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, saxpy_chunk, (void *)&thread_data[i]);

	/* Join point: wait for the workers to finish */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	/* Free dynamically allocated data structures */
	free((void *)thread_data);

}

void *saxpy_stride(void *args)
{
	thread_data_t *thread_data = (thread_data_t *)args; /* Typecast argument to pointer to thread_data_t structure */
	int offset = thread_data->tid;
	int stride = thread_data->num_threads;
	//double partial_sum = 0.0;

	while (offset < thread_data->num_elements) {
		thread_data->y[offset] = (thread_data->a * thread_data->x[offset]) + thread_data->y[offset];
		offset += stride;
	}

	pthread_exit(NULL);
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));    /* Data structure to store thread IDs */
	pthread_attr_t attributes;                                                      /* Thread attributes */
	pthread_attr_init(&attributes);                                                /* Initialize thread attributes to default values */

																				   /* Fork point: Allocate memory for required data structures and create the worker threads */
	int i;

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	for (i = 0; i < num_threads; i++) {
		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].num_elements = num_elements;
		thread_data[i].a = a;
		thread_data[i].x = x;
		thread_data[i].y = y;
	}

	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, saxpy_stride, (void *)&thread_data[i]);

	/* Join point: Wait for the workers to finish */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	/* Free data structures */
	free((void *)thread_data);

}

/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}



