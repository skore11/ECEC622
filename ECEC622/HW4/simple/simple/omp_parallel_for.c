/* Parallelization of for loop using OpenMP. 
 *
 * Compile as follows: 
 * gcc -o omp_parallel_for omp_parallel_for.c -fopenmp -std=c99 -O3 -Wall
 *
 * Author: Naga Kandasamy
 * Date created: April 21, 2019
 * Date modified: April 26, 2020 
 *  */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <omp.h>

/* Function protoypes */
void foo(int);
void bar(int);

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s num-threads num-iterations\n", argv[0]);
        fprintf(stderr, "num-threads: Number of threads to create\n");
        fprintf(stderr, "num-iterations: Number of loop iterations to execute\n");
        exit(EXIT_FAILURE);
    }
  
    int thread_count = atoi(argv[1]);	/* Number of threads to create */
    int num_iterations = atoi(argv[2]); /* Number of loop iterations to execute */
    
#pragma omp parallel num_threads(thread_count)
    {
        int tid = omp_get_thread_num();	/* Obtain thread ID */
        foo(tid);  /* All threads execute foo() */

        /* The variable i, by virtue of being declared inside the omp construct, 
         * is private for each thread. That is, each thread gets a local or private copy of 
         * the variable. */
        int i;

        /* Paralellize for loop. Note that the for pragma does not create a new team of threads: it 
         * takes the team of threads that is active, and divides the loop iterations over them. */
#pragma omp for 
        for (i = 0; i < num_iterations; i++) {
            fprintf(stderr, "Iteration %d is executed by thread %d\n", i, tid);
            /* Loop body here. */
        }

        bar(tid); /* All threads execute the function bar() */
    } /* End omp construct */

    exit(EXIT_SUCCESS);
}

void foo(int tid)
{
    fprintf(stderr, "Thread %d executing foo()\n", tid);
    return;
}

void bar(int tid)
{
    fprintf(stderr, "Thread %d executing bar()\n", tid);
    return;
}
