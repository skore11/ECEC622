/* A hello word program that uses OpenMP. 
 * 
 * For a good tutorial on OpenMP, please refer to 
 * computing.llnl.gov/tutorials/openMP
 *
 * Compile as follows: 
 * gcc -o omp_hello_world omp_hello_world_v1.c -fopenmp -std=c99 -O3 -Wall
 *
 * Author: Naga Kandasamy
 * Date created: April 15, 2011
 * Date modified: April 26, 2020
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void hello(void);

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s num-threads\n", argv[0]);
        fprintf(stderr, "num-threads: Number of threads to create\n");
        exit(EXIT_FAILURE);
    }
  
    int thread_count = atoi(argv[1]);	/* Number of threads to create */
    
    /* OpenMP block or parallel region here */
#pragma omp parallel num_threads(thread_count)
    {
        hello();
    } /* OpenMP enforces an implicit barrier sync at the end of the parallel construct */

    exit(EXIT_SUCCESS);
}

void hello(void)
{
    int tid = omp_get_thread_num();	/* Obtain thread ID */
    int thread_count = omp_get_num_threads();
    fprintf(stderr, "Hello from thread %d of %d threads\n", tid, thread_count);
  
    return;
}
