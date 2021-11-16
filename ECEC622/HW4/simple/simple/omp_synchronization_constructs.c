/* This code computes the vector dot product A.*B. 
 *
 * It illustrates the use of various synchronization constructs available in omp.
 *
 * Compile as follows: gcc -fopenmp omp_synchronization_constructs.c -o omp_synchronization_constructs -std=c99 -O3 -Wall
 *
 * Author: Naga Kandasamy
 * Date created: April 24, 2011 
 * Date modified: April 26, 2020
 * */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Function prototypes */
double compute_gold(float *, float *, int);
double compute_using_openmp_v1(float *, float *, int, int);
double compute_using_openmp_v2(float *, float *, int, int);
double compute_using_openmp_v3(float *, float *, int, int);
double compute_using_openmp_v4(float *, float *, int, int);
double compute_using_openmp_v5(float *, float *, int, int);

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s num-threads num-elements\n", argv[0]);
        fprintf(stderr, "num-threads: Number of threads to create\n");
        fprintf(stderr, "num-elements: Number of elements\n");
        exit(EXIT_FAILURE);
    }
 
    int thread_count = atoi(argv[1]);
    int num_elements = atoi(argv[2]); 
  
    /* Create vectors A and B and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Creating two random vectors with %d elements each\n", num_elements);
    float *vector_a = (float *)malloc(sizeof(float) * num_elements);
    float *vector_b = (float *)malloc(sizeof(float) * num_elements);
    if ((vector_a == NULL) || (vector_b == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));		/* Seed random number generator */
    int i;
    for (i = 0; i < num_elements; i++) {
        vector_a[i] = (rand() / (float)RAND_MAX) - 0.5;
        vector_b[i] = (rand() / (float)RAND_MAX) - 0.5;
    }

    /* Compute dot product using the reference, single-threaded solution */
    fprintf(stderr, "\nDot product of vectors with %d elements using the single-threaded implementation\n", num_elements);
    struct timeval start, stop;
    
    gettimeofday(&start, NULL);
    double reference = compute_gold(vector_a, vector_b, num_elements);
    gettimeofday(&stop, NULL);
    
    fprintf(stderr, "Reference solution = %f\n", reference);
    fprintf(stderr, "Execution time = %fs\n",
	  (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
 
    double omp_result; 
    /* Compute with omp using the critical construct */ 
    fprintf(stderr, "\nDot product of vectors with %d elements using omp with the critical construct\n", num_elements);
    
    gettimeofday(&start, NULL);
    omp_result = compute_using_openmp_v1(vector_a, vector_b, thread_count, num_elements);
    gettimeofday(&stop, NULL);
    
    fprintf(stderr, "Omp solution = %f\n", omp_result);
    fprintf(stderr, "Execution time = %fs\n", 
            (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
  
    /* Compute with omp using the atomic construct */ 
    fprintf(stderr, "\nDot product of vectors with %d elements using omp with the atomic construct\n", num_elements);
    
    gettimeofday(&start, NULL);
    omp_result = compute_using_openmp_v2(vector_a, vector_b, thread_count, num_elements);
    gettimeofday(&stop, NULL);
    
    fprintf(stderr, "Omp solution = %f\n", omp_result);
    fprintf(stderr, "Execution time = %fs\n",
	  (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Compute with omp using the lock construct */ 
    fprintf(stderr, "\nDot product of vectors with %d elements using omp with locks\n", num_elements);

    gettimeofday(&start, NULL);
    omp_result = compute_using_openmp_v3(vector_a, vector_b, thread_count, num_elements);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Omp solution = %f\n", omp_result);
    fprintf(stderr, "Execution time = %fs\n",
            (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Compute with omp using the barrier construct */ 
    fprintf(stderr, "\nDot product of vectors with %d elements using omp with barrier and master constructs\n", num_elements);
    gettimeofday(&start, NULL);
    omp_result = compute_using_openmp_v4(vector_a, vector_b, thread_count, num_elements);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Omp solution = %f\n", omp_result);
    fprintf(stderr, "Execution time = %fs\n",
	  (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Compute with omp using the reduction clause */ 
    fprintf(stderr, "\nDot product of vectors with %d elements using omp with the reduction clause\n", num_elements);
    gettimeofday(&start, NULL);
    omp_result = compute_using_openmp_v5(vector_a, vector_b, thread_count, num_elements);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Omp solution = %f\n", omp_result);
    fprintf(stderr, "Execution time = %fs\n",
	  (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

  free((void *)vector_a);
  free((void *)vector_b);

  exit(EXIT_SUCCESS);
}

/* Calculate reference soution using a single thread */
double compute_gold(float *vector_a, float *vector_b, int num_elements)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < num_elements; i++)
        sum += vector_a[i] * vector_b[i];
  
    return sum;
}

/* Calculate using multiple threads, using the "critical" construct */
double compute_using_openmp_v1(float *vector_a, float *vector_b, int thread_count, int num_elements)
{
    int i;
    double sum = 0.0;		/* Variable to hold final dot product */
    double local_sum;

#pragma omp parallel private(i, local_sum) shared(sum) num_threads(thread_count)
    {
        local_sum = 0.0;
        /* Parallelize the iterations of the for loop over the available threads */
#pragma omp for
        for (i = 0; i < num_elements; i++) 
            local_sum += vector_a[i] * vector_b[i];
        		
        /* The CRITICAL directive specifies a region of code that must be executed by only one thread at a time. If a thread is currently 
         * executing inside a CRITICAL region and another thread reaches that CRITICAL region and attempts to execute it, it will block 
         * until the first thread exits that CRITICAL region. 
         * */
#pragma omp critical
        {
            sum += local_sum;
            fprintf(stderr, "Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num(), local_sum, sum);
        } /* End critical region */
    } /* End parallel region */
  
    return sum;
}


/* Calculate dot product using the "atomic" construct which specifies that a specific memory location must be updated 
 * atomically, rather than letting multiple threads attempt to write to it. In essence, this directive 
 * provides a mini-critical section. This directive applies only to a single immediately following statement. */
double compute_using_openmp_v2(float *vector_a, float *vector_b, int thread_count, int num_elements)
{
    int i;
    double sum = 0.0;
    double local_sum;

#pragma omp parallel private(i, local_sum) shared(sum) num_threads(thread_count)
    {
        local_sum = 0.0;

#pragma omp for
        for (i = 0; i < num_elements; i++) 
            local_sum += vector_a[i] * vector_b[i];
        				
#pragma omp critical
        fprintf(stderr, "Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num(), local_sum, sum);
    
        /* Only a single statement can follow after the atomic construct */
#pragma omp atomic
        sum += local_sum;
    } /* End parallel region */
  
    return sum;
}

/* Calculate the dot product using the "lock" construct */
double compute_using_openmp_v3(float *vector_a, float *vector_b, int thread_count, int num_elements)
{
    int i;
    double sum = 0.0;
    omp_lock_t lock;            /* The lock variable */
    double local_sum;
    
    omp_init_lock(&lock);	    /* Initialize the lock */

#pragma omp parallel private(i, local_sum) shared(lock) num_threads(thread_count)
    {
        local_sum = 0.0;

#pragma omp for
        for (i = 0; i < num_elements; i++) 
            local_sum += vector_a[i] * vector_b[i];		
    
        omp_set_lock(&lock);	/* Attempt to gain the lock. Force the executing thread to block until the specified lock is available. */
        sum += local_sum;		/* Accumulate into the global sum */
        fprintf(stderr, "Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num(), local_sum, sum);
        omp_unset_lock(&lock);	    /* Release the lock */
    } /* End parallel region */

    omp_destroy_lock(&lock);	    /* Destroy the lock */

    return sum;
}

/* Calculate dot product using the "barrier" and "master" constructs */
double compute_using_openmp_v4(float *vector_a, float *vector_b, int thread_count, int num_elements)
{
  int i, j;
  double sum;
  double *local_sum = (double *)malloc(thread_count * sizeof(double));
  double psum = 0.0;
  int tid;

#pragma omp parallel num_threads(thread_count) private(tid, i, j) firstprivate(psum) shared(sum, local_sum)
  {
    tid = omp_get_thread_num();

    /* The MASTER directive specifies a region that is to be executed only by the master thread of the team. 
     * All other threads on the team skip this section of code. Note that there is no implied barrier associated 
     * with this directive. 
     * */
#pragma omp master
    {
      fprintf(stderr, "Master thread %d is performing some initialization\n", omp_get_thread_num());
      sum = 0.0;
      for (i = 0; i < thread_count; i++)
          local_sum[i] = 0.0;
    }
    
    /* Synchronize all threads in the team. When a BARRIER directive is reached, a thread will wait at that point until all other 
     * threads have reached that barrier. All threads then resume executing in parallel the code that follows the barrier. */
#pragma omp barrier

#pragma omp for
    for (i = 0; i < num_elements; i++)
        psum += vector_a[i] * vector_b[i];    
   
    local_sum[tid] = psum;
    
    fprintf(stderr, "Thread %d is at the barrier. local_sum = %f\n", tid, local_sum[tid]);

#pragma omp barrier  /* The barrier. Each barrier must be encountered by all threads or none at all */
    
    /* The master generates the final sum */
#pragma omp master
    {
      fprintf(stderr, "Master thread %d is computing the final sum\n", omp_get_thread_num());
      for (j = 0; j < thread_count; j++)
          sum += local_sum[j];
    }
  }	/* End parallel region */

  return sum;
}

/* Calculate dot product using reduction and schedule clauses */
double compute_using_openmp_v5 (float *vector_a, float *vector_b, int thread_count, int num_elements)
{
  int i;
  double sum = 0.0;

  omp_set_num_threads(thread_count);	/* Set the number of threads. */

  /* The REDUCTION clause performs a reduction on the variables that appear in its list. A private copy for each list variable 
   * is created for each thread. At the end of the reduction, the reduction variable is applied to all private copies of the shared 
   * variable, and the final result is written to the global shared variable. 
   *
   * The SCHEDULE clause describes how iterations of the loop are divided among the threads in the team. The default schedule is 
   * implementation dependent.
   * 
   * STATIC: Loop iterations are divided into pieces of size chunk and then statically assigned to threads. If chunk is not specified, 
   * the iterations are evenly (if possible) divided contiguously among the threads.
   *
   * DYNAMIC: Loop iterations are divided into pieces of size chunk, and dynamically scheduled among the threads; when a thread finishes 
   * one chunk, it is dynamically assigned another. The default chunk size is 1.
   *
   * GUIDED: Iterations are dynamically assigned to threads in blocks as threads request them until no blocks remain to be assigned. 
   * Similar to DYNAMIC except that the block size decreases each time a parcel of work is given to a thread. The size of the initial 
   * block is proportional to: number_of_iterations / number_of_threads. 
   * Subsequent blocks are proportional to number_of_iterations_remaining / number_of_threads
   * The chunk parameter defines the minimum block size. The default chunk size is 1.
   * */
#pragma omp parallel for reduction(+:sum) schedule(guided)
  for (i = 0; i < num_elements; i++)
      sum += vector_a[i] * vector_b[i];

  return sum;
}
